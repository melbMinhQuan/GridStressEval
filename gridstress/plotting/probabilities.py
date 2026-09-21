"""Generate figures from the published Monte Carlo results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter


output_directory = Path(__file__).resolve().parent / "plot"
results_path = Path(__file__).resolve().parent / "results.json"


def plot_results(payload):
    """Save 3D, S/M and M/L views using the same probability color scale."""
    inputs, results = payload["inputs"], payload["results"]
    output_directory.mkdir(parents=True, exist_ok=True)
    coordinates = {key: [row[key] for row in results] for key in ("S", "M", "L")}
    probabilities = [row["violation_probability"] for row in results]
    names = {"S": "Small customers (S)", "M": "Medium customers (M)",
             "L": "Large customers (L)"}
    colormap = LinearSegmentedColormap.from_list("violation", ["blue", "red"])
    paths = []

    for x, y, z, filename in (
        ("S", "M", "L", "probability_3d.png"),
        ("S", "M", None, "probability_2d_s_vs_m.png"),
        ("M", "L", None, "probability_2d_m_vs_l.png"),
    ):
        figure = plt.figure(figsize=(11, 8))
        axis = figure.add_subplot(111, projection="3d" if z else None)
        args = [coordinates[x], coordinates[y]]
        if z:
            args.append(coordinates[z])
        options = {"depthshade": False} if z else {}
        points = axis.scatter(*args, c=probabilities, cmap=colormap,
                              vmin=0, vmax=1, s=22, **options)
        axis.set_xlabel(names[x], labelpad=10)
        axis.set_ylabel(names[y], labelpad=10)
        if z:
            # A 2D label stays inside the saved image at this viewing angle.
            axis.text2D(-0.06, 0.5, names[z], transform=axis.transAxes,
                        rotation=90, va="center", ha="center")
            axis.view_init(elev=25, azim=45)
        else:
            missing = ({"S", "M", "L"} - {x, y}).pop()
            axis.set_title(f"{missing} = {inputs['num_customers']} - {x} - {y}")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
            axis.set_axisbelow(True)
        colorbar = figure.colorbar(points, ax=axis, shrink=0.7, pad=0.12 if z else 0.04)
        colorbar.set_label("Estimated violation probability")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        sampling_label = f"{inputs['iterations']:,} samples per combination"
        if inputs.get("random_seeds"):
            sampling_label = (
                f"{len(inputs['random_seeds'])} seeds x {inputs['samples_per_seed']:,} trials"
                f" = {inputs['iterations']:,} per combination"
            )
        figure.suptitle(
            f"Monte Carlo: P(load > {inputs['threshold']})\n"
            f"p(on) = {inputs['p_on']:g}; {sampling_label}"
        )
        figure.subplots_adjust(left=0.08, right=0.88, bottom=0.13, top=0.85)
        path = output_directory / filename
        figure.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.3)
        plt.close(figure)
        paths.append(path)
    return paths


if __name__ == "__main__":
    if not results_path.exists():
        raise SystemExit("Run python model/model_full.py first to create results.json.")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    for path in plot_results(payload):
        print(f"Saved figure: {path}")
