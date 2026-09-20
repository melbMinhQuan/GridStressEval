from pathlib import Path

import matplotlib.pyplot as plt

from model_all_on import evaluate_all_combinations, num_customers, threshold


# All generated figures are stored in this folder.
output_directory = Path(__file__).resolve().parent / "plot"


def split_results(results):
    """Separate violating and non-violating combinations."""
    violations = [result for result in results if result["violation"]]
    non_violations = [result for result in results if not result["violation"]]
    return violations, non_violations


def plot_3d(results):
    """Create and save the 3D S/M/L combination plot."""
    violations, non_violations = split_results(results)
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    for points, color, label in (
        (non_violations, "blue", f"Non-violation (load <= {threshold})"),
        (violations, "red", f"Violation (load > {threshold})"),
    ):
        axis.scatter(
            [point["S"] for point in points],
            [point["M"] for point in points],
            [point["L"] for point in points],
            color=color,
            s=28,
            alpha=0.75,
            label=label,
        )

    axis.set_xlabel("Small customers (S)")
    axis.set_ylabel("Medium customers (M)")
    axis.set_zlabel("Large customers (L)")
    axis.set_title("All customers ON: threshold violations")
    axis.legend()
    figure.tight_layout()

    file_path = output_directory / "violations_3d.png"
    figure.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return file_path


def plot_small_vs_medium(results):
    """Create and save the 2D S-versus-M plot; L is implied."""
    violations, non_violations = split_results(results)
    figure, axis = plt.subplots(figsize=(9, 7))

    for points, color, label in (
        (non_violations, "blue", f"Non-violation (load <= {threshold})"),
        (violations, "red", f"Violation (load > {threshold})"),
    ):
        axis.scatter(
            [point["S"] for point in points],
            [point["M"] for point in points],
            color=color,
            s=28,
            alpha=0.75,
            label=label,
        )

    axis.set_xlabel("Small customers (S)")
    axis.set_ylabel("Medium customers (M)")
    axis.set_title(f"All customers ON (L = {num_customers} - S - M)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    file_path = output_directory / "violations_2d_s_vs_m.png"
    figure.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return file_path


def plot_medium_vs_large(results):
    """Create and save the 2D M-versus-L plot; S is implied."""
    violations, non_violations = split_results(results)
    figure, axis = plt.subplots(figsize=(9, 7))

    for points, color, label in (
        (non_violations, "blue", f"Non-violation (load <= {threshold})"),
        (violations, "red", f"Violation (load > {threshold})"),
    ):
        axis.scatter(
            [point["M"] for point in points],
            [point["L"] for point in points],
            color=color,
            s=28,
            alpha=0.75,
            label=label,
        )

    axis.set_xlabel("Medium customers (M)")
    axis.set_ylabel("Large customers (L)")
    axis.set_title(f"All customers ON (S = {num_customers} - M - L)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    file_path = output_directory / "violations_2d_m_vs_l.png"
    figure.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return file_path


if __name__ == "__main__":
    output_directory.mkdir(parents=True, exist_ok=True)
    combinations = evaluate_all_combinations()

    figure_3d = plot_3d(combinations)
    figure_small_vs_medium = plot_small_vs_medium(combinations)
    figure_medium_vs_large = plot_medium_vs_large(combinations)

    print(f"Saved 3D figure: {figure_3d}")
    print(f"Saved S-vs-M figure: {figure_small_vs_medium}")
    print(f"Saved M-vs-L figure: {figure_medium_vs_large}")
