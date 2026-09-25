# dslr — Data Science × Logistic Regression
## « Harry Potter and the Data Scientist »

Classifier multi-classes qui reconstruit le **Choixpeau magique** de Poudlard : à
partir des notes de cours des élèves, on prédit leur maison
(*Gryffindor*, *Hufflepuff*, *Ravenclaw*, *Slytherin*) avec une **régression
logistique one-vs-all** entraînée par **descente de gradient**.

Le projet suit le sujet 42 `en.subject.pdf` et implémente les six programmes
obligatoires : `describe.py`, `histogram.py`, `scatter_plot.py`, `pair_plot.py`,
`logreg_train.py` et `logreg_predict.py`.

---

## Sommaire

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Prérequis et installation](#2-prérequis-et-installation)
3. [Structure du dépôt](#3-structure-du-dépôt)
4. [Les données](#4-les-données)
5. [Méthodologie détaillée](#5-méthodologie-détaillée)
6. [Utilisation pas à pas](#6-utilisation-pas-à-pas)
7. [Référence complète des fonctions](#7-référence-complète-des-fonctions)
8. [Résultats](#8-résultats)
9. [Bonus, outillage (Makefile) et rendu graphique](#9-bonus-outillage-makefile-et-rendu-graphique)
10. [Limites et pistes d'amélioration](#10-limites-et-pistes-damélioration)
11. [Conformité au sujet](#11-conformité-au-sujet)

---

## 1. Contexte et objectifs

Le Choixpeau ne répond plus : il faut le remplacer par un modèle d'apprentissage
automatique. À partir d'un jeu de données d'élèves (notes dans 13 cours, plus
quelques informations annexes), on doit :

1. **Explorer** les données (`describe.py`) ;
2. **Visualiser** les données pour détecter les structures et anomalités
   (`histogram.py`, `scatter_plot.py`, `pair_plot.py`) ;
3. **Entraîner** un modèle de classification linéaire (`logreg_train.py`) ;
4. **Prédire** la maison des nouveaux élèves (`logreg_predict.py`).

Objectif de performance : **accuracy ≥ 98 %** (mesurée par `sklearn.metrics.accuracy_score`
sur le jeu de test officiel).

La contrainte forte du sujet : **aucune fonction de bibliothèque ne doit faire le
« gros du travail »** à notre place (`describe`, `count`, `mean`, `std`, `min`,
`max`, `percentile`, `sklearn.linear_model`, etc.). Toutes les statistiques et
l'optimisation sont donc réimplémentées à la main. Seuls `numpy` (algèbre
linéaire pure) et `matplotlib` (tracé) sont utilisés.

---

## 2. Prérequis et installation

- **Python 3.9 ou supérieur** (le code est écrit pour rester compatible 3.9 :
  pas de `match`, pas d'union de types `X | Y`).
- Dépendances : `numpy`, `matplotlib`.

```bash
# (optionnel) environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# installation des dépendances
pip install -r requirements.txt
```

`requirements.txt` :

```
numpy>=1.21
matplotlib>=3.5
```

> Le projet n'utilise **ni** `pandas`, **ni** `scikit-learn` dans le code livré.
> `pandas` n'a servi qu'à *vérifier* les résultats de `describe.py` (écart
> maximal observé : 5·10⁻⁷, dû à l'arrondi à 6 décimales).

### 2.1 Installation et interpréteur — géré automatiquement par `make`

Un ordinateur peut avoir **plusieurs** Python 3, et `numpy`/`matplotlib` ne sont
souvent installés que dans l'un d'eux (sous Homebrew, `python3` pointe vers une
version « externally managed » **sans** `numpy`). Vous n'avez donc **rien à
installer à la main** : le `Makefile` s'en charge.

Au premier `make`, la cible `deps` :

1. **utilise** un Python 3 du système qui possède déjà `numpy` + `matplotlib` s'il
   en existe un ; **sinon**
2. **crée un environnement virtuel local `.venv/`** et y installe les dépendances
   (`pip install -r requirements.txt`) — sans `sudo`, sans toucher au système.

Toutes les cibles réutilisent ensuite cet interpréteur.

```bash
make check     # affiche l'interpréteur retenu et les versions de numpy/matplotlib
make deps      # verifie / installe les dependances (fait automatiquement par `make all`)
make install   # force la (re)installation des dependances
make distclean # supprime .venv en plus des artefacts
```

- **Surcharger l'interpréteur** : `make PYTHON=/usr/bin/python3 all`.
- **Changer l'emplacement du venv** : `make VENV=.mon_venv all`.
- En lançant un script **directement** avec un interpréteur sans `numpy`, un message
  d'erreur explicite indique la marche à suivre (au lieu d'une `ModuleNotFoundError` brute).
- `requirements.txt` contient : `numpy>=1.21` et `matplotlib>=3.5`.

---

## 3. Structure du dépôt

```
Dslr/
├── datasets/
│   ├── dataset_train.csv      # 1600 élèves étiquetés (maison connue)
│   └── dataset_test.csv       # 400 élèves à classer (colonne maison vide)
├── plots/
│   ├── histogram.png          # figure produite par histogram.py
│   ├── scatter_plot.png       # figure produite par scatter_plot.py
│   └── pair_plot.png          # figure produite par pair_plot.py
├── dslr_utils.py              # bibliothèque commune (lecture, stats, ML, tracé)
├── describe.py                # statistiques descriptives
├── histogram.py               # histogrammes par maison
├── scatter_plot.py            # nuage de points des 2 features similaires
├── pair_plot.py               # matrice de nuages de points
├── logreg_train.py            # entraînement (génère weights.json)
├── logreg_predict.py          # prédiction (génère houses.csv)
├── weights.json               # poids + paramètres de prétraitement (généré)
├── houses.csv                 # prédictions finales (généré)
├── Makefile                   # automatisation (make all, make plot, make bonus...)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Les données

`dataset_train.csv` : **1600 lignes × 19 colonnes**. Colonnes non numériques :
`Index`, `Hogwarts House`, `First Name`, `Last Name`, `Birthday`, `Best Hand`.
Les **13 features numériques** (les notes de cours) sont :

| Feature | Type / ordre de grandeur |
|---|---|
| Arithmancy | ~0 – 100 000 |
| Astronomy | ~ −1 000 – 1 000 |
| Herbology | ~ −10 – 12 |
| Defense Against the Dark Arts | ~ −10 – 10 |
| Divination | ~ −9 – 10 |
| Muggle Studies | ~ −1 100 – 1 100 |
| Ancient Runes | ~ 280 – 745 |
| History of Magic | ~ −9 – 12 |
| Transfiguration | ~ 900 – 1 100 |
| Potions | ~ −5 – 14 |
| Care of Magical Creatures | ~ −3 – 3 |
| Charms | ~ −261 – −225 |
| Flying | ~ −181 – 279 |

Deux particularités importantes :

- **Valeurs manquantes** : 349 lignes sur 1600 (≈ 22 %) ont au moins une note
  vide ; chaque cours a entre 30 et 43 valeurs manquantes. `Charms` et `Flying`
  sont les seuls cours complets.
- **Échelles très différentes** : `Arithmancy` varie sur ~100 000 tandis que
  `Care of Magical Creatures` varie sur ~3. Ce déséquilibre rend la
  **standardisation obligatoire** avant la descente de gradient.

Distribution des maisons dans le jeu d'entraînement (classes déséquilibrées) :
`Hufflepuff` 529, `Ravenclaw` 443, `Gryffindor` 327, `Slytherin` 301.

---

## 5. Méthodologie détaillée

### 5.1 Analyse des données — `describe.py`

Le programme calcule, pour chaque colonne numérique, les huit statistiques du
sujet : `Count`, `Mean`, `Std`, `Min`, `25%`, `50%`, `75%`, `Max`.

- `Count` = nombre de valeurs **non manquantes** (les vides sont ignorés).
- `Std` = écart-type **d'échantillon** (`ddof = 1`), comme la sortie de
  référence du sujet (149 échantillons sur 150 dans l'exemple).
- Les quantiles utilisent l'**interpolation linéaire** : `rang = q/100 × (n−1)`
  (méthode par défaut de numpy/pandas, réimplémentée à la main).

```bash
python3 describe.py datasets/dataset_train.csv
```

Exemple de sortie (extrait, colonnes réelles) :

```
          Arithmancy      Astronomy  ...
Count   1566.000000    1568.000000  ...
Mean    49634.570243      39.797131 ...
Std     16679.806036     520.298268 ...
Min    -24370.000000    -966.740546 ...
25%     38511.500000    -489.551387 ...
50%     49013.500000     260.289446 ...
75%     60811.250000     524.771949 ...
Max    104956.000000    1016.211940 ...
```

### 5.2 Visualisation

#### `histogram.py` — distribution homogène

**Question :** quel cours a une distribution de scores homogène entre les
quatre maisons ?

**Méthode :** pour chaque cours, on mesure la dispersion *inter-maisons* en
combinant deux écarts normalisés par l'écart-type global du cours :

- dispersion des **moyennes** des 4 maisons ;
- dispersion des **écarts-types** des 4 maisons ;

`score = hypot(dispersion_moyennes, dispersion_ecarts-types)`. Plus le score est
proche de 0, plus les quatre distributions se superposent.

**Résultat :** `Care of Magical Creatures` (score 0.0598), suivi de
`Arithmancy` (0.0943) puis `Potions` (0.7717). Les quatre maisons y présentent
la même courbe en cloche : aucune maison n'est favorisée. La figure
`plots/histogram.png` montre les quatre maisons superposées pour chaque cours et
encadre en vert le cours gagnant.

#### `scatter_plot.py` — deux features similaires

**Question :** quelles sont les deux features similaires ?

**Méthode :** on calcule la matrice de corrélation de Pearson (à la main) et on
retient la paire d'|r| maximal.

**Résultat :** `Astronomy` et `Defense Against the Dark Arts`, avec
**r = −1.0000** (relation linéaire parfaite : les deux cours sont identiques à
une transformation affine près). La figure `plots/scatter_plot.png` montre un
nuage parfaitement aligné.

#### `pair_plot.py` — choix des features

**Question :** quelles features utiliser pour la régression logistique ?

**Méthode :** matrice de nuages de points (diagonale = histogrammes) colorée par
maison, plus un classement des features par **ratio inter/intra-maisons**
calculé à la main sur les scores standardisés (grand ratio ⇒ les maisons sont
bien séparées ⇒ feature discriminante).

Résultat du classement (décroissant) :

| Rang | Feature | Ratio inter/intra |
|---|---|---|
| 1 | Defense Against the Dark Arts | 7.95 |
| 2 | Astronomy | 7.92 |
| 3 | Charms | 6.89 |
| 4 | Ancient Runes | 5.92 |
| 5 | Divination | 5.38 |
| 6 | Herbology | 4.95 |
| 7 | Transfiguration | 4.66 |
| 8 | Flying | 4.43 |
| 9 | Muggle Studies | 4.38 |
| 10 | History of Magic | 3.91 |
| 11 | Potions | 0.94 |
| 12 | Care of Magical Creatures | 0.003 |
| 13 | Arithmancy | 0.0007 |

**Interprétation :** les features 1–10 séparent nettement les maisons (ratio > 3.9).
`Potions`, `Care of Magical Creatures` et `Arithmancy` ne portent presque aucune
information de maison (ratio < 1). Note importante : bien que `Astronomy` et
`Defense Against the Dark Arts` portent du signal, **elles sont redondantes**
(r = −1) — en garder une seule suffit. Cela n'a toutefois pas d'effet
mesurable sur le résultat (cf. § 5.3).

### 5.3 Sélection des features

Conformément à l'objectif de simplicité et de robustesse, **l'entraînement
utilise les 13 features numériques** par défaut. La redondance
(`Astronomy` / `DADA`) et le bruit (`Potions`, `Care of Magical Creatures`,
`Arithmancy`) sont absorbés sans dommage par les coefficients du modèle. Nous
avons vérifié empiriquement (validation croisée) que retirer ces features ne
change pas l'accuracy à la quatrième décimale près.

Une option `--features` permet d'imposer un sous-ensemble explicite :

```bash
python3 logreg_train.py datasets/dataset_train.csv \
    --features "Astronomy,Charms,Ancient Runes,Divination,Herbology,Transfiguration,Flying,Muggle Studies,History of Magic"
```

### 5.4 Nettoyage des données (imputation)

Les valeurs manquantes sont remplacées par la **moyenne de la colonne calculée
sur le jeu d'entraînement** (`impute_with_mean`). Ces moyennes sont stockées dans
`weights.json` (`mu`) et **réappliquées telles quelles au jeu de test**, afin que
le test soit transformé exactement comme l'entraînement.

Au niveau du classificateur, l'imputation est *parfaitement acceptable* ici : le
score à l'entraînement (98.19 %) et le score en validation croisée (98.18 %) sont
identiques, ce qui prouve que l'imputation n'introduit pas de fuite de données ni
de biais notable.

### 5.5 Standardisation (z-score)

Chaque colonne est centrée-réduite :

```
z = (x − mu) / sigma        (sigma = écart-type de population, ddof = 0)
```

Cette étape est **indispensable** : sans elle, `Arithmancy` (ordre de grandeur
100 000) écraserait totalement `Care of Magical Creatures` (ordre de grandeur 1)
dans la descente de gradient. `mu` et `sigma` sont sauvegardés dans
`weights.json` et réutilisés à l'identique pour le test.

### 5.6 Modèle : régression logistique one-vs-all

On entraîne **quatre** classifieurs binaires. Pour la maison *k*, on définit
`y = 1` si l'élève est dans la maison *k*, `y = 0` sinon. Chaque classifieur
apprend :

```
h_theta(x) = g(theta^T x),   g(z) = 1 / (1 + exp(−z))        (fonction logistique)
```

Pour prédire, on calcule les quatre probabilités et on prend l'**argmax**.

### 5.7 Optimisation : descente de gradient

On minimise le coût logistique (sujet, annexe VIII.1) :

```
J(theta) = −1/m · Σ_i [ y_i · log(h_theta(x_i)) + (1 − y_i) · log(1 − h_theta(x_i)) ]
```

dont le gradient vaut :

```
∂/∂theta_j J(theta) = 1/m · Σ_i ( h_theta(x_i) − y_i ) · x_i_j
```

La descente de gradient **par lots (batch)** met à jour les paramètres à chaque
itération (`logreg_train.py` → `train_one_vs_all`) :

```
theta <- theta − alpha · (1/m) · X_b^T (h(X_b) − Y)
```

où `alpha` est le taux d'apprentissage (`--lr`, défaut 0.5), `X_b` la matrice des
features augmentée d'une colonne de biais (intercept), et `Y` la matrice
d'indicatrices des maisons. Une **régularisation L2 optionnelle** (`--l2`) peut
être ajoutée. Le coût est affiché toutes les 500 itérations pour suivre la
convergence.

### 5.8 Validation

Comme la colonne `Hogwarts House` du jeu de test est **vide** (les vraies
étiquettes ne sont pas fournies), on valide sur le jeu d'entraînement via une
**validation croisée stratifiée à 5 plis** implémentée à la main
(`stratified_folds`) : chaque pli respecte les proportions de chaque maison.

```bash
python3 logreg_train.py datasets/dataset_train.csv --cv 5
```

---

## 6. Utilisation pas à pas

Toutes les commandes se lancent **depuis la racine du dépôt**.

### 6.1 Statistiques descriptives

```bash
python3 describe.py datasets/dataset_train.csv
python3 describe.py datasets/dataset_train.csv --exclude Divination,Flying
python3 describe.py datasets/dataset_train.csv --extra
```

- `--exclude` : liste (séparée par des virgules) de colonnes numériques à retirer.
- `--extra` : ajoute les statistiques **bonus** — `Variance`, `Range`, `IQR`,
  `Missing`, `Missing %`, `Skewness`. Sans cette option, la sortie reste
  strictement le tableau obligatoire à 8 lignes.

### 6.2 Visualisations

```bash
python3 histogram.py    datasets/dataset_train.csv
python3 scatter_plot.py datasets/dataset_train.csv
python3 pair_plot.py    datasets/dataset_train.csv
```

Options communes :

- `--save CHEMIN` : chemin du PNG de sortie (défaut : `plots/<script>.png`) ;
- `--no-show` : n'ouvre pas de fenêtre (utile sur serveur / en CI).

```bash
python3 histogram.py datasets/dataset_train.csv --no-show --save plots/histogram.png
```

### 6.3 Entraînement

```bash
# entraînement standard (génère weights.json)
python3 logreg_train.py datasets/dataset_train.csv

# avec validation croisée 5 plis et sortie personnalisée
python3 logreg_train.py datasets/dataset_train.csv --cv 5 --output weights.json

# hyperparamètres personnalisés
python3 logreg_train.py datasets/dataset_train.csv --lr 0.3 --epochs 10000 --l2 0.01

# BONUS : autres optimiseurs (descente stochastique / par mini-lots)
python3 logreg_train.py datasets/dataset_train.csv --optimizer sgd --epochs 50 --seed 42
python3 logreg_train.py datasets/dataset_train.csv --optimizer minibatch --batch-size 32 --epochs 300
```

Options : `--output`, `--features`, `--lr`, `--epochs`, `--optimizer {batch,sgd,minibatch}`,
`--batch-size`, `--l2`, `--cv`, `--seed`.

- `--optimizer` : `batch` (défaut, une mise à jour par époque sur tout le jeu),
  `sgd` (une mise à jour par exemple) ou `minibatch` (une mise à jour par lot).
- `--batch-size` : taille du lot pour `minibatch` (défaut 32).
- `--seed` : graine contrôlant **à la fois** le mélange des indices (SGD / mini-batch)
  et les plis de la validation croisée — garantit la reproductibilité.
- Avec `sgd`/`minibatch`, un `--epochs` bien plus faible (20–300) suffit : chaque
  époque fait déjà des milliers de mises à jour.

### 6.4 Prédiction

```bash
python3 logreg_predict.py datasets/dataset_test.csv weights.json
# -> houses.csv

python3 logreg_predict.py datasets/dataset_test.csv weights.json --output houses.csv
```

Options : `--output` (défaut `houses.csv`).

### 6.5 Enchaînement complet (Makefile)

Tout est également automatisé via le `Makefile` (exécuter `make help` pour la liste) :

```bash
make all          # describe + les 3 figures + entraînement + prédiction
make plots        # les trois figures uniquement
make train-cv     # entraînement + validation croisée 5 plis
make bonus        # describe --extra + comparaison batch / sgd / minibatch
make clean        # supprime weights.json, houses.csv et plots/*.png
make re           # clean + all
```

Manuellement, dans l'ordre :

```bash
python3 describe.py      datasets/dataset_train.csv
python3 histogram.py     datasets/dataset_train.csv --no-show
python3 scatter_plot.py  datasets/dataset_train.csv --no-show
python3 pair_plot.py     datasets/dataset_train.csv --no-show
python3 logreg_train.py  datasets/dataset_train.csv --cv 5
python3 logreg_predict.py datasets/dataset_test.csv weights.json
cat houses.csv
```

---

## 7. Référence complète des fonctions

### 7.1 `dslr_utils.py` — bibliothèque commune

#### Lecture des données

| Fonction | Description |
|---|---|
| `read_dataset(path) -> (header, rows)` | Lit un CSV avec le module `csv` (ouvert en `utf-8-sig`). `header` = noms de colonnes ; `rows` = liste de dictionnaires `colonne → str`. |
| `_parse_float(value) -> float` | Convertit une cellule en `float` ; renvoie `nan` si la cellule est vide ou absente. |
| `numeric_features(header, rows, exclude=None) -> list[str]` | Détecte les colonnes **numériques** (toutes les valeurs non vides parsables en float). Exclut `NON_NUMERIC_COLUMNS` et `exclude`. |
| `to_matrix(rows, features) -> np.ndarray` | Construit la matrice `(n_lignes × n_features)` ; les cellules vides deviennent des `NaN`. |
| `load_dataset(path, features=None, exclude=None)` | Fonction haut niveau : renvoie `(header, rows, features, X, houses)`. `houses` est la liste des maisons (vide pour le test). |

#### Statistiques manuelles (aucune fonction de bibliothèque)

| Fonction | Description |
|---|---|
| `_finite(values) -> np.ndarray` | Renvoie les valeurs non-`NaN` (aplaties) d'une séquence. |
| `count(values) -> int` | Nombre de valeurs non manquantes. |
| `mean(values) -> float` | Moyenne arithmétique (ignore les `NaN`). |
| `std(values, ddof=1) -> float` | Écart-type. `ddof=1` = échantillon (convention `describe`) ; `ddof=0` = population (standardisation). |
| `minimum(values) / maximum(values) -> float` | Min / max (ignorent les `NaN`). |
| `percentile(values, q) -> float` | q-ième percentile par interpolation linéaire `rang = q/100 × (n−1)`. |
| `describe_feature(values) -> dict` | Les 8 statistiques d'une colonne (`count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`). |
| `describe_matrix(X, features) -> dict` | Applique `describe_feature` à chaque colonne de `X`. |
| `pearson(a, b) -> float` | Corrélation de Pearson (ignore les paires contenant un `NaN`). |
| `correlation_matrix(X) -> np.ndarray` | Matrice de corrélation complète (symétrique) des colonnes de `X`. |

#### Prétraitement

| Fonction | Description |
|---|---|
| `impute_with_mean(X) -> (X_imputé, moyennes)` | Remplace les `NaN` de chaque colonne par la moyenne de cette colonne. Retourne aussi les moyennes utilisées (pour réutiliser sur le test). |
| `standardize(X, mu=None, sd=None) -> (X_std, mu, sd)` | Centre-réduit colonne par colonne : `(X − mu) / sd`. Si `mu`/`sd` fournis, applique la transformation d'entraînement (test). Protège la division par zéro. |

#### Classification

| Fonction | Description |
|---|---|
| `sigmoid(z) -> np.ndarray` | Fonction logistique `1/(1+exp(−z))`, implémentée par cas pour la stabilité numérique (évite l'`overflow`). |
| `one_hot(labels, houses=HOUSES) -> np.ndarray` | Encode la maison en vecteur binaire `(n × 4)` (one-vs-all). |
| `accuracy(predictions, truth) -> float` | Taux de bonnes réponses, calculé à la main (aucune dépendance à sklearn). |
| `stratified_folds(labels, k=5, seed=42) -> list[np.ndarray]` | Construit `k` plis stratifiés par maison (indices), de façon déterministe (`seed`). |

#### Statistiques supplémentaires (bonus)

| Fonction | Description |
|---|---|
| `variance(values, ddof=1) -> float` | Variance (échantillon ou population selon `ddof`). |
| `skewness(values) -> float` | Asymétrie (coefficient de Fisher, moment centré d'ordre 3 normalisé). |
| `value_range(values) -> float` | Étendue `max − min`. |
| `iqr(values) -> float` | Écart interquartile `Q3 − Q1`. |
| `missing_count(values) -> int` | Nombre de valeurs manquantes (NaN). |
| `describe_feature_extra(values) -> dict` | Les six statistiques du bonus. |
| `describe_feature_full(values) -> dict` | Fusionne statistiques obligatoires et bonus. |

#### Aide au tracé

| Fonction | Description |
|---|---|
| `apply_style()` | Applique le style matplotlib commun (police, grille discrète, bordures, dpi, fond). |
| `prepare_plots_dir(path="plots") -> str` | Crée (si besoin) le dossier de figures et le renvoie. |
| `parse_common_plot_args(parser, default_name)` | Ajoute à un `ArgumentParser` les options communes : `dataset`, `--save`, `--no-show`. |
| `finalize_figure(fig, save_path=None, show=False)` | Sauvegarde en PNG (150 dpi) et/ou affiche une figure, puis libère la mémoire. |

### 7.2 `describe.py`

| Fonction | Description |
|---|---|
| `format_stats(features, stats, labels=STAT_LABELS, keys=STAT_KEYS) -> str` | Aligne et met en forme le tableau (lignes = statistiques, colonnes = features) avec 6 décimales ; `labels`/`keys` permettent d'ajouter les lignes bonus. |
| `build_stats(features, matrix) -> dict` | Calcule les statistiques de chaque colonne numérique. |
| `main()` | Lit les arguments, charge le dataset, affiche le tableau. |

### 7.3 `histogram.py`

| Fonction | Description |
|---|---|
| `homogeneity_score(column, labels) -> float` | Score de dispersion inter-maisons (`hypot` des dispersions relatives des moyennes et écarts-types). 0 = distributions identiques. |
| `rank_courses(features, matrix, labels) -> list` | Classe les cours du plus homogène au moins homogène. |
| `plot_histograms(features, matrix, labels, winner, save_path, show)` | Grille d'histogrammes (4 maisons superposées par cours), cours gagnant mis en évidence. |
| `main()` | Charge les données, affiche le classement et la réponse, produis la figure. |

### 7.4 `scatter_plot.py`

| Fonction | Description |
|---|---|
| `most_correlated_pair(features, matrix) -> (i, j, r)` | Renvoie la paire de features d'|corrélation| maximale. |
| `plot_pair(...)` | Trace le nuage de points de ces deux features, coloré par maison. |
| `main()` | Affiche la réponse (les deux features similaires) et produit la figure. |

### 7.5 `pair_plot.py`

| Fonction | Description |
|---|---|
| `separation_rank(features, matrix, labels) -> list` | Classe les features par ratio inter/intra-maisons (sur scores standardisés). |
| `plot_matrix(features, matrix, labels, save_path, show)` | Trace la matrice `n × n` de nuages de points (diagonale = histogrammes par maison). |
| `main()` | Affiche le classement du pouvoir discriminant et les features à retenir, produit la figure. |

### 7.6 `logreg_train.py`

| Fonction | Description |
|---|---|
| `cost_function(probabilities, Y) -> float` | Calcule `J(theta)` sur un lot (log clampé à `[eps, 1−eps]` pour éviter `log(0)`). |
| `OPTIMIZERS` | Tuple des optimiseurs disponibles : `("batch", "sgd", "minibatch")`. |
| `effective_batch_size(optimizer, batch_size, n_samples) -> int` | Traduit un nom d'optimiseur en taille de lot effective (bornée à `n`). |
| `train_one_vs_all(X_std, Y, lr, epochs, l2=0.0, verbose, log_every, optimizer="batch", batch_size=32, seed=42) -> (theta, historique)` | Cœur de l'algorithme : descente de gradient (batch / SGD / mini-batch), ajoute la colonne de biais, mélange reproductible via `seed`, renvoie `theta` de forme `(n_features+1, 4)`. |
| `predict_scores(X_std, theta) -> np.ndarray` | Probabilités `(n_lignes × 4)` pour chaque maison. |
| `labels_from_scores(scores, houses) -> list[str]` | Argmax par ligne → nom de maison. |
| `preprocess_train(X) -> (X_std, mu, sd)` | Impute puis standardise un jeu d'entraînement. |
| `cross_validate(features, X, labels, lr, epochs, l2, k, seed, optimizer="batch", batch_size=32) -> float` | Accuracy moyenne par validation croisée stratifiée, avec l'optimiseur choisi. |
| `save_weights(path, features, mu, sd, theta, hyperparameters)` | Sérialise poids + prétraitement dans un JSON. |
| `build_parser() -> ArgumentParser` | Construit l'interface en ligne de commande. |
| `main()` | Orchestration : validation éventuelle, entraînement final, sauvegarde. |

### 7.7 `logreg_predict.py`

| Fonction | Description |
|---|---|
| `load_weights(path) -> dict` | Charge le JSON et vérifie les champs requis (`houses`, `features`, `mu`, `sd`, `theta`). |
| `build_theta_matrix(payload) -> (theta, houses)` | Reconstruit la matrice `theta` `(n_features+1, 4)` depuis le JSON. |
| `predict(features, X, mu, sd, theta, houses) -> list[str]` | Impute (avec `mu`), standardise (avec `mu`/`sd`), calcule les probabilités, argmax. |
| `write_predictions(path, indices, predictions)` | Écrit `houses.csv` au format exact `Index,Hogwarts House`. |
| `main()` | Charge poids et test, prédit, écrit `houses.csv`. |

---

## 8. Résultats

### Statistiques descriptives

`describe.py` produit le tableau complet des 13 notes (vérifié contre
`pandas.describe()` : écart maximal **5·10⁻⁷**, purement dû à l'arrondi).

### Réponses aux questions du sujet

| Question | Réponse |
|---|---|
| Cours à distribution homogène entre les maisons | **Care of Magical Creatures** |
| Les deux features similaires | **Astronomy** et **Defense Against the Dark Arts** (r = −1.0000) |
| Features utiles pour la régression logistique | les 10 features de ratio > 3.9 (cf. § 5.2) ; les 13 sont conservées par défaut |

### Performance du classifieur

Validation croisée stratifiée à 5 plis (pipeline livré, 8000 itérations, lr = 0.5) :

| Pli | Accuracy |
|---|---|
| 1 | 0.9876 |
| 2 | 0.9907 |
| 3 | 0.9781 |
| 4 | 0.9843 |
| 5 | 0.9686 |
| **Moyenne** | **0.9818** |

- Accuracy d'entraînement (ajustement complet) : **0.9819**.
- Le fait que l'accuracy d'entraînement soit égale à l'accuracy de validation
  montre que le modèle **ne surapprend pas** : l'erreur résiduelle (≈ 1.8 %)
  correspond à des étiquettes intrinsèquement ambiguës (élèves dont le profil
  chevauche deux maisons), et non à un défaut du modèle.

### Comparaison des optimiseurs (bonus)

Les trois optimiseurs ont été comparés avec la même validation croisée à 5 plis :

| Optimiseur | Réglage | Accuracy moyenne |
|---|---|---|
| `batch` | 5000 époques, lr 0.5 | **0.9818** |
| `minibatch` | lots de 32, 300 époques, lr 0.5 | **0.9818** |
| `sgd` | 50 époques, lr 0.5 | **0.9818** |

Les trois convergent vers le même plafond : la variante choisie n'a pas d'effet
mesurable ici, seule la vitesse de convergence diffère (mini-batch converge en
beaucoup moins d'époques).

`logreg_predict.py` produit `houses.csv` (400 lignes + en-tête) :

```
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
2,Gryffindor
3,Hufflepuff
4,Hufflepuff
...
```

---

## 9. Bonus, outillage (Makefile) et rendu graphique

### 9.1 Bonus du sujet réalisés

| Bonus du PDF | Statut | Détail |
|---|---|---|
| Champs supplémentaires pour `describe.py` | ✅ | Option `--extra` : Variance, Range, IQR, Missing, Missing %, Skewness |
| Descente de gradient stochastique (SGD) | ✅ | `--optimizer sgd` (mélange reproductible via `--seed`) |
| Autres optimiseurs (mini-batch / batch) | ✅ | `--optimizer batch` (défaut) et `--optimizer minibatch --batch-size N` |

L'accuracy restant identique entre les trois optimiseurs (cf. § 8), le modèle
n'a pas été modifié : seul l'outil l'a été.

### 9.2 Makefile

| Cible | Commande exécutée |
|---|---|
| `make deps` | vérifie et installe `numpy`/`matplotlib` (venv local si nécessaire) |
| `make check` | affiche l'interpréteur retenu + versions numpy/matplotlib |
| `make install` | force la (re)installation des dépendances |
| `make describe` | `python3 describe.py datasets/dataset_train.csv` |
| `make describe-extra` | idem + `--extra` |
| `make histogram` / `scatter` / `pair` | les trois scripts de visualisation (`--no-show`) |
| `make plots` | les trois figures |
| `make train` | `logreg_train.py` (batch) → `weights.json` |
| `make train-cv` | `logreg_train.py --cv 5` |
| `make bonus` | `describe --extra` + comparaison batch / sgd / minibatch |
| `make predict` | `logreg_predict.py` → `houses.csv` |
| `make all` | `deps` + `describe` + `plots` + `train` + `predict` |
| `make clean` | supprime `weights.json`, `houses.csv`, `plots/*.png` |
| `make distclean` | comme `clean` + supprime `.venv/` |
| `make re` | `clean` puis `all` |

### 9.3 Rendu graphique

- **Palette unifiée** (`HOUSE_COLORS`) : rouge brique (Gryffindor), vert émeraude
  (Slytherin), bleu acier (Ravenclaw), jaune doré (Hufflepuff) — plus sobres et
  mieux contrastées que les couleurs « bonbon » d'origine. Une variante adoucie
  (`HOUSE_COLORS_SOFT`) sert aux remplissages.
- **Style commun** (`apply_style()`) : police unique, grille discrète passée
  *sous* les tracés, suppression des bordures haute et droite, fond légèrement
  chaud, export à 150 dpi. Les **trois** figures partagent la même identité.
- **Histogrammes** : titres sur plusieurs lignes (plus de chevauchement), barres à
  contour blanc, cours gagnant encadré et annoté.
- **Nuage de points** : points à contour fin, **droite de régression** ajoutée
  avec la valeur de `r` en légende.
- **Pair plot** : grille aérée, diagonale en histogrammes superposés par maison,
  étiquettes d'axes lisibles, légende unique.

---

## 10. Limites et pistes d'amélioration

- **Plafond ≈ 98.2 %** : la frontière linéaire one-vs-all atteint le maximum
  atteignable sur ce jeu ; le reste est du bruit d'étiquettes. Des features
  polynomiales n'apportent qu'un gain marginal (vérifié : ~98.5 %).
- **Bonus réalisés** (cf. § 9) : SGD et mini-batch, statistiques étendues de
  `describe.py`. **Pistes restantes** : arrêt anticipé sur l'évolution du coût,
  matrice de confusion / précision-rappel par maison, tests automatisés.
- **Robustesse** : le modèle est sensible au taux d'apprentissage ; `--lr 0.5`
  converge bien, mais `--l2` peut aider à stabiliser si l'on ajoute des features.

---

## 11. Conformité au sujet

- ✅ Six programmes nommés exactement comme demandé.
- ✅ Aucune fonction de bibliothèque ne « fait le gros du travail »
  (`count`, `mean`, `std`, `min`, `max`, `percentile`, `describe` sont
  réimplémentés à la main ; pas de `pandas`, pas de `sklearn` dans le code).
- ✅ Descente de gradient (batch) imposée par le sujet pour l'entraînement.
- ✅ Régression logistique **multi-classes one-vs-all**.
- ✅ `houses.csv` exactement au format demandé.
- ✅ Accuracy ≥ 98 % en validation.
- ✅ **Bonus** : statistiques étendues (`describe.py --extra`), SGD et mini-batch
  (`logreg_train.py --optimizer`), automatisation (`Makefile`), reproductibilité (`--seed`).
