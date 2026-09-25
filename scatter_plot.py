#!/usr/bin/env python3
"""scatter_plot.py - nuage de points de deux features similaires (projet dslr).

Repond a la question du sujet :

    « Quelles sont les deux features qui sont similaires ? »

Methode : on calcule la matrice de correlation de Pearson (implementee a la
main) entre toutes les paires de cours et on retient la paire dont la valeur
absolue de correlation est maximale. Deux features quasi identiques ont une
correlation proche de +/-1. La figure affiche ce nuage de points, colore par
maison, avec la droite de regression.

Usage :
    python3 scatter_plot.py datasets/dataset_train.csv
    python3 scatter_plot.py datasets/dataset_train.csv --no-show
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np

from dslr_utils import (
    HOUSES,
    HOUSE_COLORS,
    apply_style,
    correlation_matrix,
    finalize_figure,
    load_dataset,
    mean,
    parse_common_plot_args,
    prepare_plots_dir,
    require_plotting,
    std,
)


def most_correlated_pair(
    features: List[str], matrix: np.ndarray
) -> Tuple[int, int, float]:
    """Renvoie ``(i, j, r)`` de la paire de features la plus correlee (|r| max)."""
    correlation = correlation_matrix(matrix)
    best = (0, 1, 0.0)
    best_abs = -1.0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            value = correlation[i, j]
            if np.isnan(value):
                continue
            if abs(value) > best_abs:
                best_abs = abs(value)
                best = (i, j, float(value))
    return best


def regression_line(x: np.ndarray, y: np.ndarray, correlation: float) -> Tuple[float, float]:
    """Pente et ordonnee a l'origine de la droite des moindres carres (calculees a la main)."""
    sx = std(x, ddof=0)
    sy = std(y, ddof=0)
    if sx == 0:
        return 0.0, mean(y)
    slope = correlation * sy / sx
    intercept = mean(y) - slope * mean(x)
    return slope, intercept


def plot_pair(
    x_label: str,
    y_label: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    labels: np.ndarray,
    correlation: float,
    save_path: str,
    show: bool,
) -> None:
    """Trace le nuage de points de deux features, colore par maison."""
    require_plotting(show)
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    for house in HOUSES:
        mask = (labels == house) & ~np.isnan(x_values) & ~np.isnan(y_values)
        ax.scatter(x_values[mask], y_values[mask], s=20, alpha=0.62,
                   color=HOUSE_COLORS[house], label=house,
                   edgecolors="white", linewidths=0.4)

    valid = ~np.isnan(x_values) & ~np.isnan(y_values)
    if valid.any() and abs(correlation) > 1e-9:
        slope, intercept = regression_line(x_values[valid], y_values[valid], correlation)
        xs = np.array([np.nanmin(x_values), np.nanmax(x_values)])
        ax.plot(xs, slope * xs + intercept, color="#333333", linewidth=1.2,
                linestyle="--", label="regression lineaire (r = %.3f)" % correlation)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Features similaires : %s vs %s" % (x_label, y_label),
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best")
    fig.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nuage de points de 2 features (projet dslr).")
    parse_common_plot_args(parser, "scatter_plot.png")
    args = parser.parse_args()

    _, _, features, matrix, houses = load_dataset(args.dataset)
    i, j, r = most_correlated_pair(features, matrix)

    print("Paire de features la plus correlee (|r| maximal) :")
    print("  %s vs %s  ->  r = %.4f" % (features[i], features[j], r))
    print("\nReponse : %s et %s sont similaires." % (features[i], features[j]))

    if args.save:
        prepare_plots_dir("plots")
    plot_pair(features[i], features[j], matrix[:, i], matrix[:, j],
              np.asarray(houses), r, args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
