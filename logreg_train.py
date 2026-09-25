#!/usr/bin/env python3
"""logreg_train.py - entrainement d'une regression logistique one-vs-all (dslr).

Le programme prend ``dataset_train.csv`` en parametre et entraine QUATRE
classifieurs binaires (un par maison) suivant la strategie « one-versus-all » :
pour chaque maison, on apprend un modele qui separe cette maison des trois
autres. L'optimisation se fait par DESCENTE DE GRADIENT (batch gradient
descent) - la technique imposee par le sujet.

Le programme produit un fichier de poids JSON contenant, pour chaque maison :
les thetas, ainsi que les parametres de pretraitement (moyennes et ecarts-types)
indispensables pour rejouer exactement la meme transformation sur le test.

Rappels mathematiques (sujet, annexe) :

    h_theta(x) = g(theta^T x),   g(z) = 1 / (1 + exp(-z))

    J(theta) = -1/m * sum_i [ y_i log(h) + (1 - y_i) log(1 - h) ]

    d/dtheta_j J(theta) = 1/m * sum_i ( h_theta(x_i) - y_i ) * x_i_j

Usage :
    python3 logreg_train.py datasets/dataset_train.csv
    python3 logreg_train.py datasets/dataset_train.csv --cv 5
    python3 logreg_train.py datasets/dataset_train.csv --lr 0.5 --epochs 8000 --output weights.json
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dslr_utils import (
    HOUSES,
    accuracy,
    impute_with_mean,
    load_dataset,
    one_hot,
    sigmoid,
    standardize,
    stratified_folds,
)

EPS = 1e-12


def cost_function(probabilities: np.ndarray, Y: np.ndarray) -> float:
    """Cout logistique moyen ``J(theta)`` sur un lot (log clampe pour la stabilite)."""
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    return float(-(Y * np.log(clipped) + (1.0 - Y) * np.log(1.0 - clipped)).sum()
                 / Y.shape[0])


def train_one_vs_all(
    X_standardized: np.ndarray,
    Y: np.ndarray,
    learning_rate: float,
    epochs: int,
    l2: float = 0.0,
    verbose: bool = True,
    log_every: int = 500,
) -> Tuple[np.ndarray, List[float]]:
    """Entraine les quatre classifieurs one-vs-all par descente de gradient batch.

    ``X_standardized`` ne contient PAS la colonne de biais : elle est ajoutee ici.
    Retourne ``(W, historique_du_cout)`` ou ``W`` a la forme ``(n_features + 1, 4)``.
    """
    m, n = X_standardized.shape
    X_bias = np.hstack([np.ones((m, 1)), X_standardized])
    theta = np.zeros((n + 1, Y.shape[1]))
    history: List[float] = []

    for epoch in range(epochs):
        probabilities = sigmoid(X_bias @ theta)
        gradient = X_bias.T @ (probabilities - Y) / m
        if l2:
            gradient = gradient + (l2 / m) * theta
        theta = theta - learning_rate * gradient

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            cost = cost_function(probabilities, Y)
            history.append(cost)
            print("  epoch %6d | cout J = %.6f" % (epoch, cost))
    return theta, history


def predict_scores(X_standardized: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Calcule la matrice des probabilites ``(n_lignes x 4)`` pour chaque maison."""
    X_bias = np.hstack([np.ones((X_standardized.shape[0], 1)), X_standardized])
    return sigmoid(X_bias @ theta)


def labels_from_scores(scores: np.ndarray, houses: Sequence[str] = HOUSES) -> List[str]:
    """Retourne la maison de plus haute probabilite pour chaque ligne (argmax)."""
    houses = list(houses)
    indices = np.argmax(scores, axis=1)
    return [houses[index] for index in indices]


def preprocess_train(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Impute puis standardise un jeu d'entrainement ; renvoie (X, mu, sd)."""
    X_imputed, _ = impute_with_mean(X)
    X_standardized, mu, sd = standardize(X_imputed)
    return X_standardized, mu, sd


def cross_validate(
    features: Sequence[str],
    X: np.ndarray,
    labels: Sequence[str],
    learning_rate: float,
    epochs: int,
    l2: float,
    k: int,
    seed: int,
) -> float:
    """Estime l'accuracy par validation croisee stratifiee (implementation manuelle)."""
    folds = stratified_folds(labels, k=k, seed=seed)
    labels = np.asarray(labels)
    scores: List[float] = []
    for fold_index in range(k):
        test_index = folds[fold_index]
        train_index = np.concatenate([folds[i] for i in range(k) if i != fold_index])
        X_train, mu, sd = preprocess_train(X[train_index])
        Y_train = one_hot(labels[train_index])
        theta, _ = train_one_vs_all(X_train, Y_train, learning_rate, epochs, l2, verbose=False)

        X_test_imputed, _ = impute_with_mean(X[test_index])
        X_test, _, _ = standardize(X_test_imputed, mu, sd)
        predictions = labels_from_scores(predict_scores(X_test, theta))
        fold_accuracy = accuracy(predictions, labels[test_index])
        scores.append(fold_accuracy)
        print("  pli %d/%d : accuracy = %.4f" % (fold_index + 1, k, fold_accuracy))
    return float(sum(scores) / len(scores))


def save_weights(
    path: str,
    features: Sequence[str],
    mu: np.ndarray,
    sd: np.ndarray,
    theta: np.ndarray,
    hyperparameters: Dict[str, float],
) -> None:
    """Serialise les poids et le pretraitement dans un fichier JSON."""
    payload = {
        "format": "dslr-logreg-v1",
        "houses": list(HOUSES),
        "features": list(features),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "theta": {house: theta[:, index].tolist() for index, house in enumerate(HOUSES)},
        "hyperparameters": hyperparameters,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("Poids enregistres dans %s" % path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrainement d'une regression logistique one-vs-all (projet dslr)."
    )
    parser.add_argument("dataset", help="chemin du CSV d'entrainement")
    parser.add_argument("--output", default="weights.json", help="fichier de poids de sortie")
    parser.add_argument("--features", default="",
                        help="liste de features a utiliser, separees par des virgules (defaut : toutes)")
    parser.add_argument("--lr", type=float, default=0.5, help="taux d'apprentissage")
    parser.add_argument("--epochs", type=int, default=8000, help="nombre d'iterations")
    parser.add_argument("--l2", type=float, default=0.0, help="regularisation L2 (0 = desactivee)")
    parser.add_argument("--cv", type=int, default=0,
                        help="nombre de plis de validation croisee (0 = desactivee)")
    parser.add_argument("--seed", type=int, default=42, help="graine aleatoire")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    features = [name.strip() for name in args.features.split(",") if name.strip()] or None
    _, _, features, X, labels = load_dataset(args.dataset, features=features)
    labels = np.asarray(labels)
    if any(label is None or label == "" for label in labels):
        raise SystemExit("Le jeu d'entrainement doit contenir la colonne 'Hogwarts House'.")

    print("Features utilisees (%d) : %s" % (len(features), ", ".join(features)))

    if args.cv > 0:
        print("\nValidation croisee stratifiee (%d plis) :" % args.cv)
        mean_accuracy = cross_validate(features, X, labels, args.lr, args.epochs,
                                       args.l2, args.cv, args.seed)
        print("Accuracy moyenne (%d plis) = %.4f" % (args.cv, mean_accuracy))

    print("\nEntrainement final sur l'integralite du jeu d'entrainement :")
    X_standardized, mu, sd = preprocess_train(X)
    Y = one_hot(labels)
    theta, _ = train_one_vs_all(X_standardized, Y, args.lr, args.epochs, args.l2)

    save_weights(args.output, features, mu, sd, theta,
                 {"lr": args.lr, "epochs": args.epochs, "l2": args.l2})


if __name__ == "__main__":
    main()
