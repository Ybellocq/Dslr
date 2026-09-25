#!/usr/bin/env python3
"""histogram.py - histogrammes par maison (projet dslr).

Repond a la question du sujet :

    « Quel cours de Poudlard a une distribution de scores homogene entre les
      quatre maisons ? »

Methode : pour chaque cours, on mesure la dispersion INTER-maisons des moyennes
et des ecarts-types de chaque maison, normalisee par l'ecart-type global du
cours. Le cours dont le score est le plus proche de zero est le plus homogene
(les quatre distributions se superposent). La figure affiche les quatre maisons
superposees pour chaque cours ; le cours gagnant est mis en evidence.

Usage :
    python3 histogram.py datasets/dataset_train.csv
    python3 histogram.py datasets/dataset_train.csv --no-show
"""

from __future__ import annotations

import argparse
import math
import textwrap
from typing import List, Sequence, Tuple

import numpy as np

from dslr_utils import (
    HOUSES,
    HOUSE_COLORS,
    apply_style,
    finalize_figure,
    load_dataset,
    mean,
    parse_common_plot_args,
    prepare_plots_dir,
    require_plotting,
    std,
)

WINNER_COLOR = "#1f7a44"


def homogeneity_score(column: Sequence[float], labels: Sequence[str]) -> float:
    """Score de dispersion inter-maisons d'un cours (0 = parfaitement homogene).

    Le score combine, en valeur relative par rapport a l'ecart-type global :

    * la dispersion des moyennes des quatre maisons ;
    * la dispersion des ecarts-types des quatre maisons.

    Plus le score est petit, plus les quatre distributions se ressemblent.
    """
    column = np.asarray(column, dtype=float)
    labels = np.asarray(labels)
    overall = std(column, ddof=1)
    if math.isnan(overall) or overall == 0:
        return float("inf")
    means = np.array([mean(column[labels == house]) for house in HOUSES])
    stds = np.array([std(column[labels == house], ddof=1) for house in HOUSES])
    mean_spread = std(means, ddof=0) / overall
    std_spread = std(stds, ddof=0) / overall
    return float(math.hypot(mean_spread, std_spread))


def rank_courses(
    features: Sequence[str], matrix: np.ndarray, labels: Sequence[str]
) -> List[Tuple[str, float]]:
    """Classe les cours du plus homogene au moins homogene."""
    scores = [
        (feature, homogeneity_score(matrix[:, index], labels))
        for index, feature in enumerate(features)
    ]
    return sorted(scores, key=lambda item: item[1])


def plot_histograms(
    features: Sequence[str],
    matrix: np.ndarray,
    labels: Sequence[str],
    winner: str,
    save_path: str,
    show: bool,
) -> None:
    """Trace une grille d'histogrammes (4 maisons superposees) pour chaque cours."""
    require_plotting(show)
    import matplotlib.pyplot as plt

    apply_style()
    labels = np.asarray(labels)
    n = len(features)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        index = features.index(feature)
        values = matrix[:, index]
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            ax.set_visible(False)
            continue
        bins = np.linspace(finite.min(), finite.max(), 24)
        for house in HOUSES:
            selected = values[(labels == house) & (~np.isnan(values))]
            if selected.size == 0:
                continue
            ax.hist(selected, bins=bins, color=HOUSE_COLORS[house], alpha=0.52,
                    edgecolor="white", linewidth=0.5, label=house)

        is_winner = feature == winner
        ax.set_title("\n".join(textwrap.wrap(feature, 20)), fontsize=9.5,
                     fontweight="bold" if is_winner else "semibold",
                     color=WINNER_COLOR if is_winner else "#1b1b1b")
        if is_winner:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(WINNER_COLOR)
                spine.set_linewidth(1.6)
            ax.text(0.5, -0.30, "distribution homogene", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color=WINNER_COLOR,
                    fontweight="semibold")
        ax.tick_params(labelsize=7.5)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=4,
                   fontsize=10.5, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(
        "Quel cours a un score homogene entre les 4 maisons ?  ->  %s" % winner,
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    finalize_figure(fig, save_path=save_path, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Histogrammes par maison (projet dslr).")
    parse_common_plot_args(parser, "histogram.png")
    args = parser.parse_args()

    _, _, features, matrix, labels = load_dataset(args.dataset)
    ranking = rank_courses(features, matrix, labels)
    winner = ranking[0][0]

    print("Cours les plus homogenes entre les maisons (score -> 0 = homogene) :")
    for feature, score in ranking[:5]:
        print("  %-30s %.6f" % (feature, score))
    print("\nReponse : le cours %s." % winner)

    save_path = args.save
    if save_path:
        prepare_plots_dir("plots")
    plot_histograms(features, matrix, labels, winner, save_path, show=not args.no_show)


if __name__ == "__main__":
    main()
