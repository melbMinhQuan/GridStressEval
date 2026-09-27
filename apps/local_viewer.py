"""Run: python -m apps.local_viewer --open

Local-only dashboard; no framework or internet connection is required.
"""

import argparse
import base64
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from time import perf_counter
from urllib.parse import parse_qs, urlparse
import webbrowser

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

from gridstress.models.one_interval.exact import evaluate, exact_settings, exact_inputs
from gridstress.results import (
    load_results, type_settings,
    split_viewer_settings, overall_probability,
)
from gridstress.probabilities import TYPE_FIELDS, PROBABILITY_FIELDS
from gridstress.probabilities import normalize_settings


ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()

# Reuse published results only when they were calculated exactly.
SAVED = load_results()
DEFAULTS = exact_settings(SAVED)
TYPE_DEFAULTS = type_settings(SAVED)
TYPE_DEFAULTS = {key: round(value, 1) for key, value in TYPE_DEFAULTS.items()}
_remainder_key = max(TYPE_DEFAULTS, key=TYPE_DEFAULTS.get)
TYPE_DEFAULTS[_remainder_key] = round(TYPE_DEFAULTS[_remainder_key] + 100 - sum(TYPE_DEFAULTS.values()), 1)


@lru_cache(maxsize=8)
def render(settings_json):
    settings, type_probabilities = split_viewer_settings(
        normalize_settings(json.loads(settings_json)))
    started = perf_counter()
    saved_run = settings == DEFAULTS and SAVED['inputs'].get('method') == 'exact_binomial'
    rows = SAVED['results'] if saved_run else evaluate(settings)
    compute_seconds = perf_counter() - started
    figure = Figure(figsize=(10, 7.5), dpi=130)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111, projection="3d")
    points = axis.scatter(
        [row["S"] for row in rows], [row["M"] for row in rows],
        [row["L"] for row in rows],
        c=[row["violation_probability"] for row in rows],
        cmap=LinearSegmentedColormap.from_list("violation", ["blue", "red"]),
        vmin=0, vmax=1, s=max(4, min(24, 24000 / len(rows))), depthshade=False,
    )
    n = max(1, settings["customers"])
    axis.set(xlim=(-0.03*n, 1.03*n), ylim=(-0.03*n, 1.03*n),
             zlim=(-0.03*n, 1.03*n))
    for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.set_xlabel("Small customers (S)", labelpad=10)
    axis.set_ylabel("Medium customers (M)", labelpad=10)
    axis.text2D(-0.055, 0.5, "Large customers (L)", transform=axis.transAxes,
                rotation=90, va="center", ha="center")
    axis.view_init(elev=25, azim=45)
    colorbar = figure.colorbar(points, ax=axis, shrink=0.72, pad=0.12)
    colorbar.set_label("Conditional violation probability")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    figure.suptitle(
        f"Exact P(A > {settings['threshold']:,} | composition)  |  {settings['customers']} customers\n"
        f"Currents S/M/L: {settings['small']} / {settings['medium']} / {settings['large']} A\n"
        f"p(on) S/M/L: {settings['probability_small']}% / "
        f"{settings['probability_medium']}% / {settings['probability_large']}%",
        fontsize=12, y=0.96,
    )
    figure.subplots_adjust(left=0.10, right=0.91, bottom=0.09, top=0.86)
    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight", pad_inches=0.3)
    return {
        "inputs": dict(SAVED['inputs'] if saved_run else exact_inputs(settings),
                       p_type=type_probabilities, p_type_source="viewer_selection"),
        "overall": overall_probability(rows, type_probabilities),
        "results": rows,
        "source": "exact_binomial",
        "batch_inputs": SAVED['inputs'] if saved_run else None,
        "png": base64.b64encode(output.getvalue()).decode("ascii"),
        "calculation_seconds": round(compute_seconds, 4),
    }


class Handler(BaseHTTPRequestHandler):
    def send_body(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self.send_body((ROOT / "viewer.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/config":
            return self.send_body(json.dumps({**DEFAULTS, **TYPE_DEFAULTS}).encode(), "application/json")
        if url.path == "/api/evaluate":
            try:
                query = parse_qs(url.query, keep_blank_values=True)
                if any(len(values) != 1 for values in query.values()):
                    raise ValueError("Repeated input parameter")
                settings = {key: float(values[0]) if key in (*TYPE_FIELDS, *PROBABILITY_FIELDS) else int(values[0])
                            for key, values in query.items()}
                started = perf_counter()
                with LOCK:
                    response = dict(render(json.dumps(settings, sort_keys=True)))
                response["response_seconds"] = round(perf_counter() - started, 4)
                return self.send_body(json.dumps(response).encode(), "application/json")
            except (ValueError, TypeError) as error:
                return self.send_body(json.dumps({"error": str(error)}).encode(),
                                      "application/json", 400)
            except Exception:
                import traceback
                traceback.print_exc()
                return self.send_body(b'{"error":"Calculation failed; see the server log."}',
                                      "application/json", 500)
        return self.send_body(b"Not found", "text/plain", 404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the viewer in your browser")
    arguments = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), Handler)
    url = f"http://127.0.0.1:{arguments.port}"
    print(f"Viewer running at {url}. Press Ctrl+C to stop.", flush=True)
    # Warm the initial view so the browser's first request can use the cache.
    render(json.dumps(DEFAULTS, sort_keys=True))
    if arguments.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
