#!/usr/bin/env python3
"""pair_plot.py - matrice de nuages de points (projet dslr).

Repond a la question du sujet :

    « Quelles features allez-vous utiliser pour votre regression logistique ? »

Methode : on trace une matrice de nuages de points (scatter plot matrix) de
toutes les notes numeriques, coloree par maison. Un cours dont les nuages des
quatre maisons sont bien separes est discriminant ; un cours dont les nuages se
superposent (ou qui est parfaitement correle a un autre) est inutile.

Le classement des features utilise un ratio inter/intra maisons calcule a la
main sur des scores standardises : plus le ratio est grand, plus la feature
separe les maisons.

Usage :
    python3 pair_plot.py datasets/dataset_train.csv
    python3 pair_plot.py datasets/dataset_train.csv --no-show
"""

from __future__ import annotations

import argparse
import math
from typing import List, Sequence, Tuple

import numpy as np

from dslr_utils import (
    HOUSE_COLORS,
    HOUSE_COLORS_SOFT,
    HOUSES,
    apply_style,
    finalize_figure,
    load_dataset,
    mean,
    parse_common_plot_args,
    prepare_plots_dir,
    require_plotting,
    std,
)


def separation_rank(
    features: Sequence[str], matrix: np.ndarray, labels: Sequence[str]
) -> List[Tuple[str, float]]:
    """Classe les features par pouvoir discriminant (ratio inter/intra maisons).

    Les scores sont d'abord standardises (z-score manuel) pour rendre les ratios
    comparables d'un cours a l'autre. Le ratio ``between / within`` de chaque
    feature mesure a quel point les moyennes par maison s'ecartent.
    """
    labels = np.asarray(labels)
    standard = np.empty_like(matrix, dtype=float)
    for index in range(matrix.shape[1]):
        mu = mean(matrix[:, index])
        sd = std(matrix[:, index], ddof=0)
        if math.isnan(sd) or sd == 0:
            sd = 1.0
        standard[:, index] = (matrix[:, index] - mu) / sd

    scores: List[Tuple[str, float]] = []
    for index, feature in enumerate(features):
        column = standard[:, index]
        overall = mean(column)
        between = 0.0
        within = 0.0
        for house in HOUSES:
            group = column[(labels == house) & ~np.isnan(column)]
            if group.size == 0:
                continue
            group_mean = mean(group)
            between += group.size * (group_mean - overall) ** 2
            within += float(((group - group_mean) ** 2).sum())
        scores.append((feature, float("inf") if within <= 0 else between / within))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def plot_matrix(
    features: Sequence[str],
    matrix: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    show: bool,
) -> None:
    """Trace la matrice complete ``n x n`` de nuages de points (diagonale = hist)."""
    require_plotting(show)
    import matplotlib.pyplot as plt

    apply_style()
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(1.9 * n, 1.9 * n))

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            x_values = matrix[:, col]
            y_values = matrix[:, row]
            if row == col:
                finite = x_values[~np.isnan(x_values)]
                bins = np.linspace(finite.min(), finite.max(), 18) if finite.size else 10
                for house in HOUSES:
                    selected = x_values[(labels == house) & ~np.isnan(x_values)]
                    if selected.size:
                        ax.hist(selected, bins=bins, color=HOUSE_COLORS_SOFT[house],
                                alpha=0.85, edgecolor="white", linewidth=0.3)
            else:
                for house in HOUSES:
                    mask = (labels == house) & ~np.isnan(x_values) & ~np.isnan(y_values)
                    ax.scatter(x_values[mask], y_values[mask], s=2.5, alpha=0.5,
                               color=HOUSE_COLORS[house], edgecolors="none")

            ax.grid(False)
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if col != 0:
                ax.set_yticklabels([])
            if row != n - 1:
                ax.set_xticklabels([])
            if row == n - 1:
                ax.set_xlabel("\n".join(features[col].split(" ")), fontsize=6, labelpad=1)
            if col == 0:
                ax.set_ylabel(features[row], fontsize=6, labelpad=1)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                   color=HOUSE_COLORS[house], label=house)
        for house in HOUSES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=12,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Pair plot des notes par maison", fontsize=17, fontweight="bold", y=1.012)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    finalize_figure(fig, save_path=save_path, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrice de nuages de points (projet dslr).")
    parse_common_plot_args(parser, "pair_plot.png")
    args = parser.parse_args()

    _, _, features, matrix, houses = load_dataset(args.dataset)
    ranking = separation_rank(features, matrix, np.asarray(houses))

    print("Pouvoir discriminant des features (ratio inter/intra maisons, decroissant) :")
    for feature, score in ranking:
        print("  %-30s %.4f" % (feature, score))
    kept = [feature for feature, score in ranking if score >= 1.0]
    print("\nFeatures a retenir pour la regression logistique (ratio >= 1) :")
    print("  " + ", ".join(kept))

    if args.save:
        prepare_plots_dir("plots")
    plot_matrix(features, matrix, np.asarray(houses), args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
