"""Run: python model/interactive_viewer.py --open

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

from interactive_engine import evaluate, SEED
from model_full import load_current, threshold, num_customers, p_on, iterations


ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
DEFAULTS = {"small": load_current["S"], "medium": load_current["M"],
            "large": load_current["L"], "customers": num_customers,
            "probability": round(p_on * 100), "threshold": threshold,
            "samples": iterations}

# Load the published batch run once at startup. Its pooled sample count is
# distinct from the per-seed count used for fresh interactive simulations.
SAVED = json.loads((ROOT / 'results.json').read_text())
SAVED_SETTINGS = {
    'small': SAVED['inputs']['load_current']['S'],
    'medium': SAVED['inputs']['load_current']['M'],
    'large': SAVED['inputs']['load_current']['L'],
    'customers': SAVED['inputs']['num_customers'],
    'probability': round(SAVED['inputs']['p_on'] * 100),
    'threshold': SAVED['inputs']['threshold'],
    'samples': SAVED['inputs'].get('samples_per_seed', SAVED['inputs']['iterations']),
}
DEFAULTS = dict(SAVED_SETTINGS)


@lru_cache(maxsize=8)
def render(settings_json):
    settings = json.loads(settings_json)
    started = perf_counter()
    saved_run = settings == SAVED_SETTINGS
    rows = SAVED['results'] if saved_run else evaluate(settings)
    sample_description = (
        f"{len(SAVED['inputs']['random_seeds'])} seeds x {settings['samples']:,} samples"
        if saved_run and SAVED['inputs'].get('random_seeds')
        else f"{settings['samples']:,} samples"
    )
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
    colorbar.set_label("Estimated violation probability")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    figure.suptitle(
        f"P(load > {settings['threshold']:,})  |  {settings['customers']} customers\n"
        f"Loads S/M/L: {settings['small']} / {settings['medium']} / {settings['large']}"
        f"  |  p(on): {settings['probability']}%  |  {sample_description}",
        fontsize=12, y=0.96,
    )
    figure.subplots_adjust(left=0.10, right=0.91, bottom=0.09, top=0.91)
    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight", pad_inches=0.3)
    return {
        "inputs": {**settings,
                   "random_seed": SAVED['inputs'].get('random_seed') if saved_run else SEED,
                   "random_seeds": SAVED['inputs'].get('random_seeds') if saved_run else [SEED]},
        "results": rows,
        "source": "saved_batch" if saved_run else "interactive_simulation",
        "batch_inputs": SAVED['inputs'] if saved_run else None,
        "png": base64.b64encode(output.getvalue()).decode("ascii"),
        "simulation_seconds": round(compute_seconds, 4),
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
            return self.send_body(json.dumps(DEFAULTS).encode(), "application/json")
        if url.path == "/api/evaluate":
            try:
                query = parse_qs(url.query)
                if any(len(values) != 1 for values in query.values()):
                    raise ValueError("Repeated input parameter")
                settings = {key: int(values[0]) for key, values in query.items()}
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
