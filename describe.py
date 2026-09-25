#!/usr/bin/env python3
"""describe.py - statistiques descriptives manuelles (projet dslr).

Le programme prend un dataset en parametre et affiche, pour TOUTES les variables
numeriques, les huit statistiques demandees par le sujet :

    Count, Mean, Std, Min, 25%, 50%, 75%, Max

Aucune fonction de bibliotheque ne calcule ces valeurs : tout passe par les
fonctions ecrites a la main de ``dslr_utils``.

Usage :
    python3 describe.py datasets/dataset_train.csv
    python3 describe.py datasets/dataset_train.csv --exclude Divination,Flying
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from dslr_utils import STAT_KEYS, STAT_LABELS, describe_feature, load_dataset

#: Espacement (en caracteres) entre deux colonnes du tableau.
COLUMN_GAP = 2


def format_stats(features: List[str], stats: Dict[str, Dict[str, float]]) -> str:
    """Met en forme le tableau ``lignes = statistiques`` / ``colonnes = features``.

    Les valeurs sont affichees avec 6 decimales, comme dans l'exemple du sujet.
    Chaque colonne est alignee sur la largeur maximale entre son nom et ses
    valeurs formatees, puis separee de la suivante par :data:`COLUMN_GAP`.
    """
    label_width = max(len(label) for label in STAT_LABELS) + COLUMN_GAP
    columns = []
    for feature in features:
        values = ["%.6f" % stats[feature][key] for key in STAT_KEYS]
        width = max([len(feature)] + [len(value) for value in values]) + COLUMN_GAP
        columns.append((feature, values, width))

    header = " " * label_width + "".join(
        "%*s" % (width, name) for name, _, width in columns
    )
    lines = [header]
    for row_index, label in enumerate(STAT_LABELS):
        line = "%-*s" % (label_width, label)
        line += "".join("%*s" % (width, values[row_index]) for _, values, width in columns)
        lines.append(line)
    return "\n".join(lines)


def build_stats(features: List[str], matrix) -> Dict[str, Dict[str, float]]:
    """Calcule les statistiques de chaque colonne numerique de ``matrix``."""
    return {
        feature: describe_feature(matrix[:, index])
        for index, feature in enumerate(features)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistiques descriptives manuelles (projet dslr)."
    )
    parser.add_argument("dataset", help="chemin du CSV (ex. datasets/dataset_train.csv)")
    parser.add_argument(
        "--exclude",
        default="",
        help="colonnes numeriques a exclure, separees par des virgules",
    )
    args = parser.parse_args()

    exclude = [name.strip() for name in args.exclude.split(",") if name.strip()]
    _, _, features, matrix, _ = load_dataset(args.dataset, exclude=exclude)
    if not features:
        raise SystemExit("Aucune variable numerique trouvee dans %s" % args.dataset)

    stats = build_stats(features, matrix)
    print(format_stats(features, stats))


if __name__ == "__main__":
    main()
