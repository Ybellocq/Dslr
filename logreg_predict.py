#!/usr/bin/env python3
"""logreg_predict.py - prediction des maisons par la « Magic Hat » (dslr).

Le programme prend ``dataset_test.csv`` ET le fichier de poids produit par
``logreg_train.py`` en parametres. Il reconstruit exactement le meme
pretraitement (imputation par les moyennes d'entrainement, puis standardisation
avec les moyennes / ecarts-types d'entrainement), calcule la probabilite de
chaque maison via les quatre classifieurs one-vs-all, et retient la maison de
plus haute probabilite (argmax).

Il genere ``houses.csv`` au format impose par le sujet :

    Index,Hogwarts House
    0,Gryffindor
    1,Hufflepuff
    ...

Usage :
    python3 logreg_predict.py datasets/dataset_test.csv weights.json
    python3 logreg_predict.py datasets/dataset_test.csv weights.json --output houses.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np

from dslr_utils import HOUSES, load_dataset, sigmoid, standardize


def load_weights(path: str) -> Dict[str, object]:
    """Charge le fichier de poids JSON produit par :mod:`logreg_train`."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"houses", "features", "mu", "sd", "theta"}
    missing = required.difference(payload)
    if missing:
        raise ValueError("Fichier de poids incomplet, champs manquants : %s" % sorted(missing))
    return payload


def build_theta_matrix(payload: Dict[str, object]) -> Tuple[np.ndarray, List[str]]:
    """Reconstruit la matrice ``theta`` (n_features + 1, n_maisons) depuis le JSON."""
    houses = list(payload["houses"])
    columns = [np.asarray(payload["theta"][house], dtype=float) for house in houses]
    return np.column_stack(columns), houses


def predict(
    features: Sequence[str],
    X: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    theta: np.ndarray,
    houses: Sequence[str],
) -> List[str]:
    """Applique imputation + standardisation + sigmoide, puis argmax par ligne."""
    X_imputed = np.array(X, dtype=float, copy=True)
    for index in range(X_imputed.shape[1]):
        column = X_imputed[:, index]
        column[np.isnan(column)] = mu[index]

    X_standardized, _, _ = standardize(X_imputed, mu, sd)
    X_bias = np.hstack([np.ones((X_standardized.shape[0], 1)), X_standardized])
    scores = sigmoid(X_bias @ theta)
    indices = np.argmax(scores, axis=1)
    return [houses[index] for index in indices]


def write_predictions(path: str, indices: Sequence[str], predictions: Sequence[str]) -> None:
    """Ecrit le fichier de predictions au format exact ``Index,Hogwarts House``."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Index", "Hogwarts House"])
        for index, house in zip(indices, predictions):
            writer.writerow([index, house])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction des maisons (projet dslr).")
    parser.add_argument("dataset", help="chemin du CSV de test (ex. datasets/dataset_test.csv)")
    parser.add_argument("weights", help="fichier de poids produit par logreg_train.py")
    parser.add_argument("--output", default="houses.csv", help="fichier de predictions de sortie")
    args = parser.parse_args()

    payload = load_weights(args.weights)
    features = list(payload["features"])
    mu = np.asarray(payload["mu"], dtype=float)
    sd = np.asarray(payload["sd"], dtype=float)
    theta, houses = build_theta_matrix(payload)

    _, rows, _, X, _ = load_dataset(args.dataset, features=features)
    indices = [row.get("Index", str(position)) for position, row in enumerate(rows)]

    predictions = predict(features, X, mu, sd, theta, houses)

    counts = {house: predictions.count(house) for house in houses}
    write_predictions(args.output, indices, predictions)
    print("Predictions ecrites dans %s (%d lignes)" % (args.output, len(predictions)))
    print("Repartition : %s" % counts)


if __name__ == "__main__":
    main()
