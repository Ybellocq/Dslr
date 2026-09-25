"""Boite a outils commune au projet « Data Science x Logistic Regression » (dslr).

Ce module regroupe tout le code reutilisable par les six programmes du projet :

* ``describe.py``       : statistiques descriptives manuelles.
* ``histogram.py``      : histogrammes par maison.
* ``scatter_plot.py``   : nuage de points de deux features similaires.
* ``pair_plot.py``      : matrice de nuages de points.
* ``logreg_train.py``   : entrainement one-vs-all par descente de gradient.
* ``logreg_predict.py`` : prediction des maisons.

Conformement au sujet, AUCUNE fonction de bibliotheque ne fait le « gros du
travail » pour nous : ni ``pandas``, ni ``describe``, ni ``mean`` / ``std`` /
``percentile`` d'une bibliotheque. Toutes les statistiques sont reimplementees
ici a la main. Seuls ``numpy`` (algebre lineaire) et ``matplotlib`` (trace) sont
utilises.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depend de l'environnement
    raise SystemExit(
        "Erreur : numpy est introuvable pour cet interpreteur.\n"
        "  Interpreteur utilise : %s\n"
        "  Solutions :\n"
        "    - installer les dependances : python3 -m pip install -r requirements.txt\n"
        "    - ou lancer via 'make' (qui selectionne automatiquement un interpreteur\n"
        "      disposant de numpy ; verifiez avec 'make check').\n"
        "    - sous macOS, '/usr/bin/python3' dispose deja de numpy et matplotlib.\n"
        % sys.executable
    ) from exc

# --------------------------------------------------------------------------- #
# Constantes du domaine
# --------------------------------------------------------------------------- #

#: Les quatre maisons de Poudlard, dans un ordre stable et reutilisable.
HOUSES: List[str] = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

#: Palette principale des maisons : teintes sobres et harmonisees entre elles.
HOUSE_COLORS: Dict[str, str] = {
    "Gryffindor": "#b03a2e",  # rouge brique
    "Slytherin": "#1e7a52",   # vert emeraude
    "Ravenclaw": "#2e5f9e",   # bleu acier
    "Hufflepuff": "#d6a419",  # jaune dore
}

#: Variante adoucie (remplissages, aires, histogrammes).
HOUSE_COLORS_SOFT: Dict[str, str] = {
    "Gryffindor": "#e6a59c",
    "Slytherin": "#9ed3b8",
    "Ravenclaw": "#a9c4e6",
    "Hufflepuff": "#efd28c",
}

#: Encre et fond communs a toutes les figures (coherence visuelle).
FIGURE_BACKGROUND = "#fbfbf9"
FIGURE_EDGE = "#4a4a4a"
FIGURE_GRID = "#dcdcdc"

#: Colonnes du CSV qui ne sont pas des notes de cours numeriques.
NON_NUMERIC_COLUMNS = (
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
)

#: Libelles et cles des huit statistiques de ``describe``.
STAT_LABELS: List[str] = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
STAT_KEYS: List[str] = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

#: Statistiques supplementaires (bonus de ``describe.py``, option ``--extra``).
EXTRA_LABELS: List[str] = ["Variance", "Range", "IQR", "Missing", "Missing %", "Skewness"]
EXTRA_KEYS: List[str] = ["variance", "range", "iqr", "missing", "missing_pct", "skewness"]


# --------------------------------------------------------------------------- #
# Lecture des donnees (stdlib csv uniquement)
# --------------------------------------------------------------------------- #

def read_dataset(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Lit un CSV du projet et renvoie ``(entetes, lignes)``.

    ``entetes`` est la liste des noms de colonnes. ``lignes`` est une liste de
    dictionnaires ``colonne -> valeur brute (str)``. Le fichier est ouvert en
    ``utf-8-sig`` pour ignorer un eventuel BOM.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not header:
        raise ValueError("Fichier CSV vide ou illisible : %s" % path)
    return header, rows


def _parse_float(value: Optional[str]) -> float:
    """Convertit une cellule en ``float`` ; renvoie ``nan`` si vide/absent."""
    if value is None:
        return float("nan")
    value = value.strip()
    if value == "":
        return float("nan")
    return float(value)


def numeric_features(
    header: Sequence[str],
    rows: Sequence[Dict[str, str]],
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Determine la liste des colonnes numeriques (notes de cours).

    Une colonne est numerique si toutes ses valeurs non vides s'interprettent
    comme des nombres. Les colonnes de :data:`NON_NUMERIC_COLUMNS` et celles de
    ``exclude`` sont systematiquement ecartees.
    """
    excluded = set(NON_NUMERIC_COLUMNS)
    if exclude:
        excluded.update(exclude)
    features: List[str] = []
    for column in header:
        if column in excluded:
            continue
        numeric = False
        for row in rows:
            value = row.get(column)
            if value is None or value.strip() == "":
                continue
            try:
                float(value)
            except ValueError:
                numeric = False
                break
            numeric = True
        if numeric:
            features.append(column)
    return features


def to_matrix(rows: Sequence[Dict[str, str]], features: Sequence[str]) -> np.ndarray:
    """Construit la matrice ``(n_lignes x n_features)`` ; cellules vides -> NaN."""
    matrix = np.empty((len(rows), len(features)), dtype=float)
    for i, row in enumerate(rows):
        for j, feature in enumerate(features):
            matrix[i, j] = _parse_float(row.get(feature))
    return matrix


def load_dataset(
    path: str,
    features: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
):
    """Charge un dataset complet.

    Retourne ``(header, rows, features, X, houses)`` ou ``X`` est la matrice des
    notes (NaN pour les valeurs manquantes) et ``houses`` la liste des maisons
    (vide pour le jeu de test).
    """
    header, rows = read_dataset(path)
    if features is None:
        features = numeric_features(header, rows, exclude=exclude)
    else:
        features = list(features)
    X = to_matrix(rows, features)
    houses = [row.get("Hogwarts House") for row in rows]
    return header, rows, features, X, houses


# --------------------------------------------------------------------------- #
# Statistiques manuelles (aucune fonction de bibliotheque)
# --------------------------------------------------------------------------- #

def _finite(values: Sequence[float]) -> np.ndarray:
    """Renvoie les valeurs non-NaN (aplaties) d'une sequence."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def count(values: Sequence[float]) -> int:
    """Nombre de valeurs non manquantes."""
    return int(_finite(values).size)


def mean(values: Sequence[float]) -> float:
    """Moyenne arithmetique (ignore les NaN). NaN si aucune valeur."""
    data = _finite(values)
    if data.size == 0:
        return float("nan")
    return float(data.sum() / data.size)


def std(values: Sequence[float], ddof: int = 1) -> float:
    """Ecart-type (ignore les NaN).

    ``ddof=1`` : ecart-type d'echantillon (convention de ``describe``).
    ``ddof=0`` : ecart-type de population (convention de la standardisation).
    """
    data = _finite(values)
    n = data.size
    if n <= ddof:
        return float("nan")
    mu = data.sum() / n
    variance = ((data - mu) ** 2).sum() / (n - ddof)
    return float(math.sqrt(variance))


def minimum(values: Sequence[float]) -> float:
    """Valeur minimale (ignore les NaN)."""
    data = _finite(values)
    return float(data.min()) if data.size else float("nan")


def maximum(values: Sequence[float]) -> float:
    """Valeur maximale (ignore les NaN)."""
    data = _finite(values)
    return float(data.max()) if data.size else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    """q-ieme percentile (``q`` dans [0, 100]) par interpolation lineaire.

    Reproduit la methode par defaut de numpy/pandas : rang = q/100 * (n - 1).
    """
    data = np.sort(_finite(values))
    n = data.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(data[0])
    rank = (q / 100.0) * (n - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(data[low])
    frac = rank - low
    return float(data[low] * (1.0 - frac) + data[high] * frac)


def describe_feature(values: Sequence[float]) -> Dict[str, float]:
    """Renvoie les huit statistiques d'une colonne sous forme de dictionnaire."""
    return {
        "count": float(count(values)),
        "mean": mean(values),
        "std": std(values, ddof=1),
        "min": minimum(values),
        "25%": percentile(values, 25),
        "50%": percentile(values, 50),
        "75%": percentile(values, 75),
        "max": maximum(values),
    }


def describe_matrix(X: np.ndarray, features: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Applique :func:`describe_feature` a chaque colonne de ``X``."""
    return {feature: describe_feature(X[:, j]) for j, feature in enumerate(features)}


def variance(values: Sequence[float], ddof: int = 1) -> float:
    """Variance (ignore les NaN). ``ddof=1`` = echantillon, ``0`` = population."""
    data = _finite(values)
    n = data.size
    if n <= ddof:
        return float("nan")
    mu = data.sum() / n
    return float(((data - mu) ** 2).sum() / (n - ddof))


def skewness(values: Sequence[float]) -> float:
    """Asymetrie (coefficient de Fisher, moment centre d'ordre 3 normalise)."""
    data = _finite(values)
    n = data.size
    if n < 2:
        return float("nan")
    mu = data.sum() / n
    m2 = ((data - mu) ** 2).sum() / n
    if m2 == 0:
        return float("nan")
    m3 = ((data - mu) ** 3).sum() / n
    return float(m3 / (m2 ** 1.5))


def value_range(values: Sequence[float]) -> float:
    """Etendue : ``max - min`` (ignore les NaN)."""
    data = _finite(values)
    return float(data.max() - data.min()) if data.size else float("nan")


def iqr(values: Sequence[float]) -> float:
    """Ecart interquartile : ``Q3 - Q1``."""
    return float(percentile(values, 75) - percentile(values, 25))


def missing_count(values: Sequence[float]) -> int:
    """Nombre de valeurs manquantes (NaN) dans une colonne."""
    arr = np.asarray(values, dtype=float).ravel()
    return int(np.isnan(arr).sum())


def describe_feature_extra(values: Sequence[float]) -> Dict[str, float]:
    """Statistiques supplementaires du bonus (option ``--extra``)."""
    total = np.asarray(values, dtype=float).ravel().size
    missing = missing_count(values)
    ratio = (100.0 * missing / total) if total else float("nan")
    return {
        "variance": variance(values, ddof=1),
        "range": value_range(values),
        "iqr": iqr(values),
        "missing": float(missing),
        "missing_pct": ratio,
        "skewness": skewness(values),
    }


def describe_feature_full(values: Sequence[float]) -> Dict[str, float]:
    """Fusionne statistiques obligatoires et statistiques supplementaires."""
    merged = describe_feature(values)
    merged.update(describe_feature_extra(values))
    return merged


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Coefficient de correlation lineaire de Pearson (ignore les paires NaN)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = x.size
    if n < 2:
        return float("nan")
    mx = x.sum() / n
    my = y.sum() / n
    covariance = ((x - mx) * (y - my)).sum()
    sx = math.sqrt(((x - mx) ** 2).sum())
    sy = math.sqrt(((y - my) ** 2).sum())
    if sx == 0 or sy == 0:
        return float("nan")
    return float(covariance / (sx * sy))


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Matrice de correlation (Pearson) des colonnes de ``X``."""
    n = X.shape[1]
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = pearson(X[:, i], X[:, j])
            matrix[i, j] = r
            matrix[j, i] = r
    return matrix


# --------------------------------------------------------------------------- #
# Pretraitement
# --------------------------------------------------------------------------- #

def impute_with_mean(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remplace les NaN de chaque colonne par la moyenne de la colonne.

    Retourne ``(X_impute, moyennes)`` : la matrice completee et le vecteur des
    moyennes utilisees (indispensable pour rejouer le meme pretraitement sur le
    jeu de test).
    """
    X = np.asarray(X, dtype=float).copy()
    means = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        mu = mean(X[:, j])
        means[j] = mu
        column = X[:, j]
        column[np.isnan(column)] = mu
    return X, means


def standardize(
    X: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sd: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise (z-score) : ``(X - mu) / sd`` colonne par colonne.

    ``mu`` et ``sd`` peuvent etre fournis (pour le jeu de test) afin d'appliquer
    exactement la meme transformation qu'a l'entrainement. ``sd`` = ecart-type de
    population (``ddof=0``) et protege contre la division par zero.
    """
    X = np.asarray(X, dtype=float)
    if mu is None:
        mu = np.array([mean(X[:, j]) for j in range(X.shape[1])])
    if sd is None:
        sd = np.array([std(X[:, j], ddof=0) for j in range(X.shape[1])])
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd


# --------------------------------------------------------------------------- #
# Outils de classification
# --------------------------------------------------------------------------- #

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Fonction logistique ``g(z) = 1 / (1 + exp(-z))``, stable numeriquement."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    expz = np.exp(z[~positive])
    out[~positive] = expz / (1.0 + expz)
    return out


def one_hot(labels: Sequence[str], houses: Sequence[str] = HOUSES) -> np.ndarray:
    """Encode la maison de chaque ligne en vecteur binaire (one-vs-all)."""
    index = {house: k for k, house in enumerate(houses)}
    Y = np.zeros((len(labels), len(houses)))
    for i, label in enumerate(labels):
        if label in index:
            Y[i, index[label]] = 1.0
    return Y


def accuracy(predictions: Sequence[str], truth: Sequence[str]) -> float:
    """Taux de bonnes reponses (accuracy brute, sans bibliotheque)."""
    predictions = list(predictions)
    truth = list(truth)
    if not truth:
        return float("nan")
    correct = sum(1 for p, t in zip(predictions, truth) if p == t)
    return correct / len(truth)


def stratified_folds(labels: Sequence[str], k: int = 5, seed: int = 42) -> List[np.ndarray]:
    """Construit ``k`` plis stratifies par maison (indices), de facon deterministe.

    La stratification garantit que chaque pli contient une proportion similaire
    de chaque maison, ce qui stabilise l'estimation de l'accuracy.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    folds: List[List[int]] = [[] for _ in range(k)]
    for house in HOUSES:
        indices = np.where(labels == house)[0]
        rng.shuffle(indices)
        for position, value in enumerate(indices):
            folds[position % k].append(int(value))
    return [np.array(sorted(fold)) for fold in folds]


# --------------------------------------------------------------------------- #
# Aide au trace
# --------------------------------------------------------------------------- #

def apply_style() -> None:
    """Applique un style matplotlib sobre et coherent a toutes les figures.

    Police unique, grille discrete sous les traces, suppression des bordures
    haute et droite, fond legerement chaud. A appeler avant tout trace.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "figure.facecolor": FIGURE_BACKGROUND,
        "savefig.facecolor": FIGURE_BACKGROUND,
        "axes.facecolor": FIGURE_BACKGROUND,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "axes.labelcolor": "#2b2b2b",
        "axes.edgecolor": FIGURE_EDGE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": FIGURE_GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": "#4a4a4a",
        "ytick.color": "#4a4a4a",
        "text.color": "#1b1b1b",
    })
    matplotlib.use(matplotlib.get_backend())


def require_plotting(show: bool = False):
    """Importe matplotlib (backend adapte) ou explique comment installer les dependances."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depend de l'environnement
        raise SystemExit(
            "Erreur : matplotlib est introuvable pour cet interpreteur (%s).\n"
            "  Installez-le : python3 -m pip install -r requirements.txt\n"
            "  ou lancez via 'make' / 'make check'." % sys.executable
        ) from exc
    if not show:
        matplotlib.use("Agg")
    return matplotlib


def prepare_plots_dir(path: str = "plots") -> str:
    """Cree (si besoin) le dossier de sortie des figures et le renvoie."""
    os.makedirs(path, exist_ok=True)
    return path


def parse_common_plot_args(parser, default_name: str) -> None:
    """Ajoute les options communes aux scripts de visualisation."""
    parser.add_argument("dataset", help="chemin du CSV (ex. datasets/dataset_train.csv)")
    parser.add_argument("--save", default=os.path.join("plots", default_name),
                        help="chemin du fichier PNG de sortie")
    parser.add_argument("--no-show", action="store_true",
                        help="ne pas ouvrir de fenetre (execution sans ecran)")


def finalize_figure(fig, save_path: Optional[str] = None, show: bool = False) -> None:
    """Sauvegarde puis/ou affiche une figure, et libere la memoire."""
    import matplotlib.pyplot as plt

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Figure enregistree : %s" % save_path)
    if show:
        plt.show()
    plt.close(fig)
