# Projet DSLR - Statistiques Descriptives Manuelles

## 🌟 **Pour les débutants : Explication simple**

Imagine que tu es le **professeur McGonagall** à Poudlard. Tu as devant toi les **notes de tous tes élèves** dans différentes matières : Arithmancie, Astronomie, Botanique, Défense contre les Forces du Mal, etc.

Tu veux avoir une **vue d'ensemble rapide** des performances de ta classe. Au lieu de regarder les 1600 notes une par une, tu veux savoir :

- **Quelle est la moyenne** en Potions ?
- **Quelle est la meilleure note** en Sortilèges ?
- **Quelle est la note la plus fréquente** en Métamorphose ?
- **Y a-t-il beaucoup d'élèves** qui n'ont pas rendu leur devoir ?

C'est exactement ce que fait ce programme !

### 🎯 **À quoi ça ressemble ?**

Ton dataset (fichier CSV) ressemble à ça :

```
Index | Hogwarts House | First Name | Last Name | Arithmancy | Astronomy | Herbology | ...
------|----------------|------------|-----------|------------|-----------|-----------|---
0     | Ravenclaw      | Tamara     | Hsu       | 58384.0    | -487.89   | 5.73      | ...
1     | Slytherin      | Erich      | Paredes   | 67239.0    | -552.06   | -5.99     | ...
...   | ...            | ...        | ...       | ...        | ...       | ...       | ...
```

Le programme **analyse toutes les colonnes de notes** (Arithmancy, Astronomy, Herbology, etc.) et te donne un **tableau récapitulatif** :

```
              Arithmancy    Astronomy    Herbology    ...
Count        1600.000000  1600.000000  1600.000000  ...
Mean        50001.234567   -123.456789     5.123456  ...
Std          1234.567890     45.678901     2.345678  ...
Min           100.000000    -999.999999     0.000000  ...
25%          2500.000000    -500.000000     3.000000  ...
50%         50000.000000    -200.000000     5.000000  ...
75%         75000.000000    -100.000000     7.000000  ...
Max         99999.000000     888.888888    10.000000  ...
```

### 📊 **Que veulent dire ces chiffres ?**

| Statistique | Signification | Exemple |
|-------------|---------------|---------|
| **Count** | Nombre d'élèves qui ont une note | 1600 = tous les élèves ont une note |
| **Mean** | Moyenne de la classe | 5.12 = en moyenne, les élèves ont 5.12/10 |
| **Std** | Écart-type (comment les notes sont dispersées) | 2.35 = les notes varient beaucoup autour de la moyenne |
| **Min** | La pire note | 0.0 = au moins un élève a eu 0 |
| **25%** | 25% des élèves ont **au plus** cette note | 3.0 = un quart de la classe a ≤ 3/10 |
| **50%** | La **médiane** (la note du milieu) | 5.0 = la moitié a moins, la moitié a plus |
| **75%** | 75% des élèves ont **au plus** cette note | 7.0 = trois quarts de la classe a ≤ 7/10 |
| **Max** | La meilleure note | 10.0 = au moins un élève a eu 10/10 |

### 💡 **Pourquoi c'est utile ?**

1. **Déceler les problèmes** : Si la moyenne en Potions est de 2/10, il y a un souci !
2. **Comparer les matières** : En moyenne, les élèves réussissent mieux en Sortilèges qu'en Divination ?
3. **Identifier les outliers** : Un Min à 0 ou un Max à 100 peut indiquer une erreur de saisie
4. **Comprendre la distribution** : Si l'écart-type est très grand, les notes sont très dispersées

C'est un outil **fondamental en data science** pour **comprendre ses données avant de faire des analyses plus poussées**.

---

## 📋 **Sujet et Contexte**

Ce projet fait partie du module **DSLR** (Data Science x Logistic Regression). Il s'agit d'implémenter **manuellement** un outil de statistiques descriptives, similaire à `pandas.DataFrame.describe()`, mais **sans utiliser aucune fonction de bibliothèque** pour les calculs statistiques.

### Objectif principal

Le programme `describe.py` prend un dataset CSV en entrée et affiche, pour **toutes les variables numériques**, les huit statistiques obligatoires demandées par le sujet :

| Statistique | Description |
|-------------|-------------|
| **Count** | Nombre de valeurs non manquantes |
| **Mean** | Moyenne arithmétique |
| **Std** | Écart-type (échantillon, ddof=1) |
| **Min** | Valeur minimale |
| **25%** | Premier quartile |
| **50%** | Médiane |
| **75%** | Troisième quartile |
| **Max** | Valeur maximale |

### Option bonus

Avec le flag `--extra`, le programme ajoute six statistiques supplémentaires :

| Statistique | Description |
|-------------|-------------|
| **Variance** | Variance (échantillon) |
| **Range** | Étendue (Max - Min) |
| **IQR** | Écart interquartile (Q3 - Q1) |
| **Missing** | Nombre de valeurs manquantes |
| **Missing %** | Pourcentage de valeurs manquantes |
| **Skewness** | Coefficient d'asymétrie (Fisher) |

---

## 🏗️ **Architecture du Code et Choix de Conception**

### Structure du projet

```
Dslr/
├── describe.py          # Script principal (CLI)
├── dslr_utils.py        # Boîte à outils commune à 6 programmes
├── datasets/            # Données (dataset_train.csv, dataset_test.csv)
├── Makefile             # Automatisation (gestion des dépendances)
└── requirements.txt     # Dépendances (numpy, matplotlib)
```

### Pourquoi cette séparation ?

Le module `dslr_utils.py` est **partagé par les six programmes du projet** :
- `describe.py` (statistiques descriptives)
- `histogram.py` (histogrammes par maison)
- `scatter_plot.py` (nuage de points)
- `pair_plot.py` (matrice de nuages de points)
- `logreg_train.py` (entraînement par descente de gradient)
- `logreg_predict.py` (prédiction des maisons)

Cette architecture évite **la duplication de code** et garantit **la cohérence** des calculs entre tous les scripts.

---

## 🔍 **Analyse des Choix Techniques**

### 1. **Respect strict des contraintes du sujet**

La contrainte centrale du projet est :

> **AUCUNE fonction de bibliothèque ne doit faire le "gros du travail"**

Cela signifie :
- ❌ Pas de `pandas.read_csv()` → On utilise `csv.DictReader` (stdlib)
- ❌ Pas de `pandas.DataFrame.describe()` → On réimplémente tout
- ❌ Pas de `numpy.mean()`, `numpy.std()`, `numpy.percentile()` → On recalcule manuellement
- ✅ `numpy` est autorisé pour **l'algèbre linéaire** (matrices, opérations vectorielles)
- ✅ `matplotlib` est autorisé pour **le traçage de graphiques**

### 2. **Implémentation manuelle des statistiques**

Toutes les fonctions de calcul sont dans `dslr_utils.py` :

#### **Fonctions de base**
```python
def count(values) -> int:           # Compte les valeurs non-NaN
def mean(values) -> float:           # Moyenne arithmétique
def std(values, ddof=1) -> float:    # Écart-type (ddof=1 pour échantillon)
def minimum(values) -> float:        # Minimum
def maximum(values) -> float:        # Maximum
def percentile(values, q) -> float:  # Percentile avec interpolation linéaire
```

#### **Pourquoi réimplémenter `percentile` ?**

La fonction `numpy.percentile()` existe, mais le sujet **interdit son utilisation** pour calculer les statistiques. Nous devons donc :
1. Trier les données
2. Calculer le rang : `rank = (q/100) * (n - 1)` (méthode par défaut de numpy/pandas)
3. Appliquer une interpolation linéaire entre les deux valeurs adjacentes

**Alternative rejetée** : Utiliser `numpy.percentile()` directement → **Violation de la contrainte du sujet**.

#### **Gestion des NaN**

```python
def _finite(values) -> np.ndarray:
    """Renvoie les valeurs non-NaN (aplaties) d'une sequence."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[~np.isnan(arr)]
```

Cette fonction est **centrale** : elle filtre les valeurs manquantes avant tout calcul. Elle utilise `numpy.isnan()` car c'est une fonction d'**algèbre linéaire** (autorisée), pas une fonction statistique.

**Alternative rejetée** : Utiliser `pandas.isna()` → **Interdit** (pandas n'est pas autorisé).

### 3. **Combinaison des statistiques**

```python
def describe_feature(values) -> Dict[str, float]:
    """Les 8 statistiques obligatoires."""
    return {
        "count": count(values),
        "mean": mean(values),
        "std": std(values, ddof=1),
        "min": minimum(values),
        "25%": percentile(values, 25),
        "50%": percentile(values, 50),
        "75%": percentile(values, 75),
        "max": maximum(values),
    }

def describe_feature_extra(values) -> Dict[str, float]:
    """Les 6 statistiques bonus."""
    return {
        "variance": variance(values, ddof=1),
        "range": value_range(values),
        "iqr": iqr(values),
        "missing": float(missing_count(values)),
        "missing_pct": ratio,
        "skewness": skewness(values),
    }

def describe_feature_full(values) -> Dict[str, float]:
    """Fusionne obligatoires + bonus."""
    merged = describe_feature(values)
    merged.update(describe_feature_extra(values))
    return merged
```

**Pourquoi cette séparation ?**
- **Modularité** : On peut récupérer uniquement les stats obligatoires ou les bonus
- **Évolutivité** : Facile d'ajouter de nouvelles statistiques sans modifier le code existant
- **Performance** : On ne calcule les stats bonus que si `--extra` est activé

**Alternative rejetée** : Tout calculer dans une seule fonction → **Moins modulaire**, du code mort si `--extra` n'est pas utilisé.

### 4. **Chargement des données**

```python
def load_dataset(path, features=None, exclude=None) -> Tuple[...]:
    """Charge un dataset complet."""
    header, rows = read_dataset(path)  # csv.DictReader
    if features is None:
        features = numeric_features(header, rows, exclude=exclude)
    X = to_matrix(rows, features)  # np.ndarray avec NaN
    houses = [row.get("Hogwarts House") for row in rows]
    return header, rows, features, X, houses
```

**Pourquoi `csv.DictReader` et pas `pandas.read_csv()` ?**
- **Contrainte du sujet** : pandas est interdit
- **Portabilité** : `csv` fait partie de la bibliothèque standard de Python
- **Contrôle** : On maîtrise exactement comment les données sont lues et interprétées

**Alternative rejetée** : `pandas.read_csv()` → **Interdit par le sujet**.

### 5. **Identification des features numériques**

```python
def numeric_features(header, rows, exclude=None) -> List[str]:
    """Détermine la liste des colonnes numériques."""
    excluded = set(NON_NUMERIC_COLUMNS)  # Index, Hogwarts House, etc.
    if exclude:
        excluded.update(exclude)
    features = []
    for column in header:
        if column in excluded:
            continue
        # Vérifie que toutes les valeurs non vides sont des nombres
        numeric = True
        for row in rows:
            value = row.get(column)
            if value is None or value.strip() == "":
                continue
            try:
                float(value)
            except ValueError:
                numeric = False
                break
        if numeric:
            features.append(column)
    return features
```

**Pourquoi cette vérification manuelle ?**
- **Robustesse** : On s'assure que chaque colonne peut être convertie en `float`
- **Flexibilité** : On exclut les colonnes spécifiées via `--exclude`
- **Transparence** : Pas de magie noire, on voit exactement ce qui est numérisé

**Alternative rejetée** : `pandas.to_numeric()` → **Interdit** (pandas).

### 6. **Formatage de la sortie**

Dans `describe.py` :

```python
def format_stats(features, stats, labels, keys) -> str:
    """Met en forme le tableau lignes=statistiques / colonnes=features."""
    label_width = max(len(label) for label in labels) + COLUMN_GAP
    columns = []
    for feature in features:
        values = ["%.*f" % (DECIMALS, stats[feature][key]) for key in keys]
        width = max([len(feature)] + [len(value) for value in values]) + COLUMN_GAP
        columns.append((feature, values, width))
    
    # Construction du header et des lignes
    header = " " * label_width + "".join(...)
    lines = [header]
    for row_index, label in enumerate(labels):
        line = "%-*s" % (label_width, label)
        line += "".join("%*s" % (width, values[row_index]) for ...)
        lines.append(line)
    return "\n".join(lines)
```

**Pourquoi ce formatage manuel ?**
- **Précision** : Contrôle total sur l'affichage (6 décimales, alignement)
- **Adaptabilité** : La largeur des colonnes s'ajuste automatiquement au contenu
- **Simplicité** : Pas besoin de bibliothèque de formatage de tableaux

**Alternative rejetée** : Utiliser `tabulate` ou `pandas.DataFrame.to_string()` → **Bibliothèques externes interdites** ou **pandas interdit**.

---

## 📊 **Fonctionnement Concret**

### Flux de traitement

```
1. Parsing des arguments CLI (argparse)
   ├── dataset: chemin du CSV
   ├── --exclude: colonnes à exclure
   └── --extra: activer les stats bonus

2. Chargement des données (load_dataset)
   ├── Lecture du CSV (read_dataset)
   ├── Identification des features numériques (numeric_features)
   └── Construction de la matrice numpy (to_matrix)

3. Calcul des statistiques (build_stats)
   ├── Pour chaque feature:
   │   └── describe_feature_full() → Dict[str, float]
   └── Retourne: Dict[str, Dict[str, float]] (features → stats)

4. Formatage et affichage (format_stats)
   ├── Calcule les largeurs des colonnes
   ├── Construit le tableau texte
   └── Affiche le résultat
```

### Exemple d'exécution

```bash
# Statistiques obligatoires
python3 describe.py datasets/dataset_train.csv

# Statistiques obligatoires + bonus
python3 describe.py datasets/dataset_train.csv --extra

# Exclure certaines colonnes
python3 describe.py datasets/dataset_train.csv --exclude Divination,Flying

# Via Makefile (recommandé)
make describe
make describe-extra
```

### Sortie typique

```
                Arithmancy       Astronomy       Herbology  ...
Count        1600.000000     1600.000000     1600.000000  ...
Mean         5000.123456     -123.456789       5.123456  ...
Std           123.456789       45.678901        2.345678  ...
Min           100.000000      -999.999999       0.000000  ...
25%           250.000000      -500.000000       3.000000  ...
50%           500.000000      -200.000000       5.000000  ...
75%           750.000000      -100.000000       7.000000  ...
Max          9999.000000       888.888888      10.000000  ...

# Avec --extra:
Variance      15241.578945     2086.776859       5.501562  ...
Range         9899.000000     1888.888887      10.000000  ...
IQR           500.000000       400.000000       4.000000  ...
Missing          0.000000        0.000000       0.000000  ...
Missing %        0.000000        0.000000       0.000000  ...
Skewness        0.123456       -0.456789       0.789012  ...
```

---

## ⚖️ **Alternatives Envisagées et Rejetées**

| Choix | Alternative | Raison du rejet |
|-------|-------------|-----------------|
| `csv.DictReader` | `pandas.read_csv()` | **Interdit par le sujet** (pandas non autorisé) |
| Calcul manuel des stats | `numpy.mean()`, `numpy.std()`, etc. | **Interdit** (fonctions statistiques de bibliothèque) |
| `_finite()` avec `numpy.isnan()` | `pandas.isna()` | **Interdit** (pandas) |
| `percentile()` personnalisé | `numpy.percentile()` | **Interdit** (fonction statistique) |
| Formatage manuel | `tabulate` | **Bibliothèque externe non autorisée** |
| `describe_feature_full()` | Tout dans `describe.py` | **Duplication de code** entre les 6 programmes |
| `load_dataset()` dans chaque script | Copier-coller le code | **Violation du principe DRY** |

---

## 🎯 **Pourquoi Ces Fonctions Plutôt que D'autres ?**

### 1. **`describe_feature_full()` au lieu de calculs inline**

**Choix** : Centraliser tous les calculs dans `dslr_utils.py`

**Avantages** :
- ✅ **Réutilisable** : Utilisée par `describe.py` et potentiellement d'autres scripts
- ✅ **Testable** : Facile à tester unitairement
- ✅ **Maintenable** : Une seule place à modifier si la logique change
- ✅ **Lisible** : Le code de `describe.py` reste simple et focalisé sur l'I/O

**Inconvénient** :
- ⚠️ Couplage entre `describe.py` et `dslr_utils.py`

**Décision** : Le couplage est **accepté** car c'est le but du projet (module partagé).

### 2. **`build_stats()` dans `describe.py`**

```python
def build_stats(features, matrix) -> Dict[str, Dict[str, float]]:
    return {
        feature: describe_feature_full(matrix[:, index])
        for index, feature in enumerate(features)
    }
```

**Pourquoi pas dans `dslr_utils.py` ?**
- Cette fonction est **spécifique à `describe.py`** (format de sortie particulier)
- `dslr_utils.py` contient déjà `describe_matrix()` qui fait quelque chose de similaire
- On évite de polluer le module utilitaire avec des fonctions trop spécifiques

**Alternative envisagée** : Utiliser `describe_matrix()` directement → **Moins flexible** (ne gère pas le format exact attendu par `describe.py`).

### 3. **`format_stats()` dans `describe.py`**

**Pourquoi pas dans `dslr_utils.py` ?**
- Le formatage est **spécifique à l'affichage CLI** de `describe.py`
- D'autres scripts (`histogram.py`, etc.) ont leurs propres besoins de formatage
- On sépare **logique métier** (dans `dslr_utils.py`) et **présentation** (dans chaque script)

### 4. **Utilisation de `numpy` pour les matrices**

**Pourquoi numpy est autorisé ?**
- Le sujet autorise explicitement `numpy` pour **l'algèbre linéaire**
- Les opérations sur matrices (`X[:, j]`, `np.isnan()`, etc.) sont des opérations **bas niveau**
- `numpy` est **indispensable** pour la performance avec de grands datasets

**Limite** : On n'utilise **que** les fonctionnalités d'algèbre linéaire, **pas** les fonctions statistiques (`np.mean`, `np.std`, etc.)

### 5. **Gestion des NaN avec `float("nan")`**

```python
def _parse_float(value: Optional[str]) -> float:
    if value is None or value.strip() == "":
        return float("nan")
    return float(value)
```

**Pourquoi `float("nan")` et pas `None` ?**
- ✅ **Compatibilité numpy** : `numpy` gère nativement `np.nan`
- ✅ **Calculs numériques** : `np.nan` se propage correctement dans les opérations (`np.nan + 1 = np.nan`)
- ✅ **Filtrage facile** : `np.isnan()` fonctionne directement

**Alternative rejetée** : Utiliser `None` → **Incompatible** avec les opérations numpy.

---

## 🔧 **Détails d'Implémentation Clés**

### 1. **Calcul de la variance et de l'écart-type**

```python
def variance(values, ddof=1) -> float:
    data = _finite(values)
    n = data.size
    if n <= ddof:
        return float("nan")
    mu = data.sum() / n
    return float(((data - mu) ** 2).sum() / (n - ddof))

def std(values, ddof=1) -> float:
    return float(math.sqrt(variance(values, ddof)))
```

**Points importants** :
- `ddof=1` pour l'**échantillon** (convention de `describe`)
- `ddof=0` pour la **population** (utilisé dans `standardize()` pour la normalisation)
- On utilise `math.sqrt()` car c'est une fonction **mathématique de base**, pas une fonction statistique

### 2. **Calcul du percentile avec interpolation linéaire**

```python
def percentile(values, q) -> float:
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
```

**Pourquoi cette formule ?**
- `rank = (q/100) * (n - 1)` : Méthode par défaut de numpy/pandas (méthode 7)
- Interpolation linéaire entre `data[low]` et `data[high]`
- **Reproduit exactement** le comportement attendu par le sujet

### 3. **Calcul du coefficient d'asymétrie (Skewness)**

```python
def skewness(values) -> float:
    data = _finite(values)
    n = data.size
    if n < 2:
        return float("nan")
    mu = data.sum() / n
    m2 = ((data - mu) ** 2).sum() / n  # Moment centré d'ordre 2
    if m2 == 0:
        return float("nan")
    m3 = ((data - mu) ** 3).sum() / n  # Moment centré d'ordre 3
    return float(m3 / (m2 ** 1.5))    # Normalisation
```

**Formule** : Coefficient de Fisher = `m3 / (m2^(3/2))`
- Mesure l'**asymétrie** de la distribution
- > 0 : Distribution étirée vers la droite (queue à droite)
- < 0 : Distribution étirée vers la gauche (queue à gauche)
- ≈ 0 : Distribution symétrique

### 4. **Chargement et parsing du CSV**

```python
def read_dataset(path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return header, rows
```

**Points importants** :
- `encoding="utf-8-sig"` : Gère le BOM (Byte Order Mark) des fichiers Windows
- `csv.DictReader` : Retourne des dictionnaires `colonne -> valeur`
- **Pas de dépendance externe** : Utilise uniquement la bibliothèque standard

---

## 📈 **Performance et Optimisations**

### 1. **Vecteurisation avec numpy**

Toutes les opérations numériques utilisent **numpy** pour :
- ✅ **Performance** : Opérations vectorielles optimisées en C
- ✅ **Simplicité** : Code plus lisible et concis
- ✅ **Gestion des NaN** : `np.isnan()` et opérations sur tableaux

Exemple :
```python
# Calcul de la moyenne
mu = data.sum() / data.size  # Vectorisé, rapide

# vs. boucle Python
# mu = sum(data) / len(data)  # Plus lent pour grands tableaux
```

### 2. **Filtrage des NaN une seule fois**

Dans chaque fonction statistique :
```python
data = _finite(values)  # Filtre les NaN une fois
n = data.size
if n <= ddof:
    return float("nan")
# ... calculs sur data (sans NaN)
```

**Optimisation** : On filtre les NaN **une seule fois** par fonction, pas à chaque opération.

### 3. **Éviter les calculs inutiles**

Dans `describe.py` :
```python
if args.extra:
    labels += EXTRA_LABELS
    keys += EXTRA_KEYS

stats = build_stats(features, matrix)  # Calcule TOUTES les stats
print(format_stats(features, stats, labels=labels, keys=keys))
```

**Point à améliorer** : Actuellement, `build_stats` calcule **toutes** les statistiques (obligatoires + bonus), même si `--extra` n'est pas activé. On pourrait optimiser en :
```python
if args.extra:
    stats = build_stats_full(features, matrix)
else:
    stats = build_stats_basic(features, matrix)
```

**Pourquoi ce n'est pas fait ?**
- Le gain de performance est **minime** pour des datasets de taille raisonnable
- La complexité du code augmente
- Le sujet ne l'exige pas

---

## 🧪 **Testabilité et Robustesse**

### 1. **Gestion des edge cases**

Chaque fonction gère les cas particuliers :
- **Données vides** → Retourne `float("nan")`
- **Une seule valeur** → Retourne cette valeur (ou `nan` si indéfini, comme pour `std`)
- **Toutes les valeurs manquantes** → Retourne `float("nan")`
- **Division par zéro** → Protégé (ex: `m2 == 0` dans `skewness`)

### 2. **Typage fort**

```python
from typing import Dict, List, Optional, Sequence, Tuple

def describe_feature(values: Sequence[float]) -> Dict[str, float]:
    ...
```

**Avantages** :
- ✅ **Lisibilité** : On voit immédiatement les types attendus et retournés
- ✅ **Validation statique** : Compatible avec `mypy`
- ✅ **Documentation** : Les annotations servent de documentation

### 3. **Constantes centralisées**

```python
STAT_LABELS = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
STAT_KEYS = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
EXTRA_LABELS = ["Variance", "Range", "IQR", "Missing", "Missing %", "Skewness"]
EXTRA_KEYS = ["variance", "range", "iqr", "missing", "missing_pct", "skewness"]
```

**Pourquoi ?**
- ✅ **Cohérence** : Les mêmes labels sont utilisés partout
- ✅ **Maintenabilité** : Une seule place à modifier si un label change
- ✅ **Éviter les erreurs** : Pas de fautes de frappe possibles

---

## 📁 **Intégration dans le Projet Global**

### Makefile

Le `Makefile` simplifie l'exécution :

```bash
# Installer les dépendances automatiquement
make deps

# Exécuter describe
make describe

# Exécuter describe avec bonus
make describe-extra

# Vérifier l'environnement
make check
```

**Fonctionnement** :
1. `make deps` vérifie si numpy/matplotlib sont installés
2. Si non, crée un environnement virtuel `.venv` et installe les dépendances
3. Utilise toujours le bon interpréteur Python (avec numpy/matplotlib)

### Intégration avec les autres scripts

`dslr_utils.py` est utilisé par :
- `describe.py` → Statistiques descriptives
- `histogram.py` → Calcul des bins et histogrammes
- `scatter_plot.py` → Calcul des corrélations (Pearson)
- `pair_plot.py` → Matrice de corrélations
- `logreg_train.py` → Standardisation, descente de gradient
- `logreg_predict.py` → Chargement des données, prédictions

**Exemple** : La fonction `pearson()` de `dslr_utils.py` est utilisée par `scatter_plot.py` pour calculer le coefficient de corrélation entre deux features.

---

## 🎓 **Ce Que J'ai Appris**

1. **Respect des contraintes** : Un projet peut imposer des restrictions strictes qui forcent à réinventer la roue. C'est formateur !
2. **Modularité** : Séparer la logique métier (dans `dslr_utils.py`) de la présentation (dans chaque script) rend le code plus maintenable.
3. **Performance** : Même avec des contraintes, on peut optimiser (vecteurisation numpy, filtrage des NaN une seule fois).
4. **Robustesse** : Toujours gérer les edge cases (données vides, NaN, division par zéro).
5. **Documentation** : Les commentaires et les annotations de type sont cruciaux pour comprendre le code.
6. **Reproductibilité** : Le `Makefile` garantit que le projet fonctionne sur n'importe quelle machine.

---

## 📝 **Améliorations Possibles**

| Amélioration | Description | Priorité |
|--------------|-------------|----------|
| Optimiser `build_stats` | Ne calculer que les stats nécessaires | ⚠️ Moyenne |
| Ajouter des tests unitaires | Vérifier chaque fonction individuellement | ⭐ Haute |
| Gérer les très grands datasets | Lecture par chunks | ⚠️ Moyenne |
| Ajouter une option `--output` | Écrire dans un fichier | ⚠️ Moyenne |
| Support des CSV mal formés | Meilleure gestion des erreurs | ⚠️ Moyenne |

---

## 🚀 **Conclusion**

Ce projet illustre comment **réimplémenter des fonctionnalités de base** tout en respectant des contraintes strictes. Les choix techniques ont été guidés par :

1. **Le respect du sujet** : Aucune fonction de bibliothèque pour les calculs statistiques
2. **La modularité** : `dslr_utils.py` comme boîte à outils partagée
3. **La performance** : Utilisation optimale de numpy pour l'algèbre linéaire
4. **La robustesse** : Gestion complète des edge cases
5. **La maintenabilité** : Code bien structuré, documenté et typé

Le résultat est un outil **fonctionnel, performant et conforme aux exigences** du projet DSLR.
