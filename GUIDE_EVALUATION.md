# 📚 GUIDE COMPLET POUR L'ÉVALUATION DSLR

*Prépare-toi à répondre à toutes les questions de l'évaluateur avec confiance !*

---

## 🎯 **SOMMAIRE**

1. [📋 Préparation avant l'évaluation](#-préparation-avant-lévaluation)
2. [💻 Questions sur `describe`](#-questions-sur-describe)
3. [📊 Questions sur les visualisations](#-questions-sur-les-visualisations)
4. [🧠 Questions théoriques sur la régression logistique](#-questions-théoriques-sur-la-régression-logistique)
5. [⚡ Questions sur l'exécution](#-questions-sur-l'exécution)
6. [💡 Réponses aux questions pièges](#-réponses-aux-questions-pièges)

---

## 📋 **PRÉPARATION AVANT L'ÉVALUATION**

### ✅ Vérifie que tout fonctionne

```bash
# Tester toutes les commandes obligatoires
make describe
make histogram
make scatter
make pair
make train
make predict

# Tester les bonus
make describe-extra
make bonus

# Vérifier l'environnement
make check
```

### ✅ Nettoie ton dépôt

```bash
# Supprime les fichiers temporaires
make clean

# Vérifie que seul le nécessaire est dans le dépôt
git status
```

### ✅ Prépare tes explications

- **Relis ce guide** plusieurs fois
- **Entraîne-toi à expliquer** les concepts à voix haute
- **Teste avec un ami** pour simuler l'évaluation

---

## 💻 **QUESTIONS SUR `describe`**

### ❓ Question 1 : "Exécutez `describe` avec `dataset_train.csv`"

**Commande :**
```bash
python3 describe.py datasets/dataset_train.csv
# ou
make describe
```

**Ce que l'évaluateur voit :**
```
              Arithmancy       Astronomy       Herbology  ...
Count        1600.000000     1600.000000     1600.000000  ...
Mean        5000.123456     -123.456789       5.123456  ...
Std           123.456789       45.678901        2.345678  ...
Min           100.000000      -999.999999       0.000000  ...
25%           250.000000      -500.000000       3.000000  ...
50%           500.000000      -200.000000       5.000000  ...
75%           750.000000      -100.000000       7.000000  ...
Max          9999.000000       888.888888      10.000000  ...
```

**✅ Réponse :**
> "Oui, la sortie contient bien les **8 statistiques obligatoires** : Count, Mean, Std, Min, 25%, 50%, 75%, Max, pour toutes les features numériques du dataset."

---

### ❓ Question 2 : "Montrez-moi le code source de `describe`"

**Fichier :** `describe.py` + `dslr_utils.py`

**Ce que l'évaluateur vérifie :**
- Pas de fonctions tierces (`numpy.mean()`, `pandas.describe()`, etc.)
- Tout est implémenté manuellement

**✅ Réponse :**
> "Voici le code. Toutes les fonctions statistiques sont dans `dslr_utils.py` :
> - `count()` compte les valeurs non-NaN
> - `mean()` calcule la moyenne arithmétique
> - `std()` calcule l'écart-type avec ddof=1 (échantillon)
> - `minimum()` et `maximum()` trouvent les extrêmes
> - `percentile()` implémente l'interpolation linéaire
> 
> Aucune fonction de bibliothèque ne fait le travail à notre place. On utilise `numpy` uniquement pour l'algèbre linéaire (matrices, opérations vectorielles), pas pour les statistiques."

---

### ❓ Question 3 : "Expliquez ces notions"

#### 🔹 Qu'est-ce que la **moyenne (mean)** ?

**✅ Réponse complète (5 points) :**
> "La **moyenne** (ou mean en anglais) est la **valeur centrale** d'un ensemble de données. 
> 
> **Formule :** mean = (somme de toutes les valeurs) / (nombre de valeurs)
> 
> **Exemple concret :** Si j'ai les notes [10, 12, 14] en Potions, la moyenne est (10+12+14)/3 = 12. 
> 
> **Utilité :** 
> - Elle donne une **idée générale** de la tendance centrale
> - Elle permet de **comparer** différentes distributions
> - C'est la valeur autour de laquelle les données sont "équilibrées"
> 
> **Attention :** La moyenne est sensible aux **valeurs extrêmes** (outliers). Par exemple, si un élève a 20/20 et les autres ont 10/20, la moyenne sera tirée vers le haut."

---

#### 🔹 Qu'est-ce que l'**écart-type (std)** ?

**✅ Réponse complète (5 points) :**
> "L'**écart-type** (standard deviation) mesure **la dispersion** des données autour de la moyenne. 
> 
> **Formule :** std = √(variance) = √[Σ(xi - mean)² / (n - 1)]  (pour un échantillon)
> 
> **Exemple concret :** 
> - Notes [10, 10, 10] → écart-type = 0 (toutes les valeurs sont identiques)
> - Notes [8, 10, 12] → écart-type ≈ 2 (valeurs un peu dispersées)
> - Notes [0, 10, 20] → écart-type ≈ 10 (valeurs très dispersées)
> 
> **Interprétation :**
> - **std = 0** : Toutes les valeurs sont identiques
> - **std petit** : Les valeurs sont proches de la moyenne
> - **std grand** : Les valeurs sont très dispersées
> 
> **Utilité :**
> - Comprendre la **variabilité** des données
> - Comparer la dispersion de différentes distributions
> - En régression logistique, la **normalisation** (z-score) utilise la moyenne et l'écart-type"

---

#### 🔹 Qu'est-ce qu'un **quartile** (25%, 50%, 75%) ?

**✅ Réponse complète (5 points) :**
> "Les **quartiles** divisent les données **ordonnées** en 4 parties égales.
> 
> - **25% (Q1)** : 25% des données sont **inférieures ou égales** à cette valeur
> - **50% (Q2 = Médiane)** : 50% des données sont inférieures ou égales (sépare en deux moitiés)
> - **75% (Q3)** : 75% des données sont inférieures ou égales
> 
> **Exemple concret :** Notes [5, 7, 8, 10, 12, 15, 20]
> - Q1 (25%) = 7 (25% des notes ≤ 7)
> - Q2 (50%) = 10 (50% des notes ≤ 10)
> - Q3 (75%) = 15 (75% des notes ≤ 15)
> 
> **Utilité :**
> - **IQR (Interquartile Range) = Q3 - Q1** : Mesure la dispersion des 50% centraux
> - **Boîte à moustaches** : Utilise Q1, médiane, Q3
> - **Détecter les outliers** : Valeurs < Q1 - 1.5*IQR ou > Q3 + 1.5*IQR
> - **Robuste aux outliers** : Contrairement à la moyenne, les quartiles ne sont pas affectés par les valeurs extrêmes"

---

---

## 📊 **QUESTIONS SUR LES VISUALISATIONS**

### ❓ **Histogram : "Quel cours a une distribution homogène entre les 4 maisons ?"**

**Commande :**
```bash
make histogram
# ou
python3 histogram.py datasets/dataset_train.csv
```

**Ce que l'évaluateur voit :**
- Un histogramme avec 4 couleurs (Gryffindor, Hufflepuff, Ravenclaw, Slytherin)
- Chaque sous-graphique = un cours
- Le cours avec la distribution la plus homogène est **mis en évidence**

**✅ Réponse :**
> "Mon script `histogram.py` répond à cette question en **calculant un score d'homogénéité** pour chaque cours.
> 
> **Méthode :**
> 1. Pour chaque cours, je calcule la **moyenne** et l'**écart-type** des notes **par maison**
> 2. Je mesure la **dispersion INTER-maisons** :
>    - Dispersion des moyennes entre les 4 maisons
>    - Dispersion des écarts-types entre les 4 maisons
> 3. Je normalise ces dispersions par l'écart-type **global** du cours
> 4. Je combine les deux avec `hypot()` (racine carrée de la somme des carrés)
> 5. Le cours avec le **score le plus proche de 0** est le plus homogène
> 
> **Interprétation :**
> - Si le score = 0 : Les 4 maisons ont **exactement la même distribution** (même moyenne, même écart-type)
> - Si le score > 0 : Les distributions diffèrent entre les maisons
> 
> **Exemple :** Si le cours `Charms` a un score de 0.1 et `Potions` a un score de 0.8, alors **Charms** a une distribution plus homogène.
> 
> **La visualisation aide car :** On voit directement que les 4 histogrammes se **superposent** pour le cours gagnant, alors qu'ils sont décalés ou ont des formes différentes pour les autres."

---

### ❓ **Scatter Plot : "Quelles sont les 2 features similaires ?"**

**Commande :**
```bash
make scatter
# ou
python3 scatter_plot.py datasets/dataset_train.csv
```

**Ce que l'évaluateur voit :**
- Un nuage de points avec 4 couleurs (les maisons)
- Une droite de régression
- Le titre indique les 2 features comparées

**✅ Réponse :**
> "Mon script `scatter_plot.py` répond en **calculant la matrice de corrélation de Pearson** entre toutes les paires de features.
> 
> **Méthode :**
> 1. Je calcule la corrélation pour **toutes les paires** de cours
> 2. Je cherche la paire avec la **valeur absolue de corrélation maximale** (proche de ±1)
> 3. Deux features **parfaitement similaires** ont une corrélation de +1 ou -1
> 4. J'affiche le nuage de points de cette paire, avec la droite de régression
> 
> **Interprétation de la corrélation :**
> - **r = +1** : Les deux features varient **dans le même sens** (quand l'une augmente, l'autre augmente)
> - **r = -1** : Les deux features varient **en sens inverse** (quand l'une augmente, l'autre diminue)
> - **r ≈ 0** : Aucune relation linéaire
> 
> **Pourquoi une seule réponse est évidente :**
> Dans le dataset, il y a une paire de cours dont les notes sont **presque identiques** (corrélation très proche de 1). C'est cette paire que mon script identifie.
> 
> **La visualisation aide car :** On voit que les points **s'alignent parfaitement** sur la droite de régression, ce qui confirme la forte corrélation."

---

### ❓ **Pair Plot : "Quelles caractéristiques allez-vous utiliser pour la régression logistique ?"**

**Commande :**
```bash
make pair
# ou
python3 pair_plot.py datasets/dataset_train.csv
```

**Ce que l'évaluateur voit :**
- Une matrice de scatter plots (toutes les paires de features)
- Diagonale = histogrammes
- 4 couleurs pour les maisons

**✅ Réponse :**
> "Mon script `pair_plot.py` répond en **calculant un ratio discriminant** pour chaque feature.
> 
> **Méthode :**
> 1. Pour chaque feature, je calcule un **ratio inter/intra maisons**
> 2. **Standardisation** : Je normalise d'abord les données (z-score) pour comparer les features
> 3. **Variance inter-classe (between)** : Mesure à quel point les moyennes des 4 maisons sont éloignées
> 4. **Variance intra-classe (within)** : Mesure à quel point les données sont dispersées **au sein de chaque maison**
> 5. **Ratio = between / within** : Plus ce ratio est grand, plus la feature **sépare bien les maisons**
> 
> **Interprétation :**
> - **Ratio ≈ 0** : Les moyennes des maisons sont proches → la feature **ne sépare pas** les maisons
> - **Ratio > 1** : La variance entre les maisons est plus grande que la variance au sein des maisons → **bonne feature pour la classification**
> - **Ratio très grand** : Excellente feature discriminante
> 
> **Sélection des features :**
> Je retiens toutes les features avec un **ratio ≥ 1**, car elles permettent de bien distinguer les maisons.
> 
> **La visualisation aide car :**
> - Si les 4 couleurs (maisons) sont **bien séparées** dans un scatter plot, les deux features sont discriminantes
> - Si les points de toutes les couleurs se **superposent**, la feature ne permet pas de distinguer les maisons
> - Les histogrammes sur la diagonale montrent la **distribution par maison** pour chaque feature"

---

---

## 🧠 **QUESTIONS THÉORIQUES SUR LA RÉGRESSION LOGISTIQUE**

### ❓ Question 1 : "Comment fonctionne la régression logistique vs régression linéaire ?"

**✅ Réponse complète :**

> **Régression linéaire :**
> - **But** : Prédire une **valeur continue** (ex: prix d'une maison)
> - **Fonction** : hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ (équation linéaire)
> - **Sortie** : Un nombre réel (ℝ)
> - **Fonction de coût** : Erreur quadratique moyenne (MSE)
> - **Exemple** : Prédire le prix en fonction de la surface
> 
> **Régression logistique :**
> - **But** : Prédire une **classe binaire** (0 ou 1, ex: Gryffindor ou pas Gryffindor)
> - **Fonction** : hθ(x) = **sigmoid**(θ₀ + θ₁x₁ + ... + θₙxₙ)
> - **Sigmoid** : g(z) = 1 / (1 + e⁻ᶻ) → **Compresse la sortie entre 0 et 1**
> - **Sortie** : Une **probabilité** (P(y=1|x)) entre 0 et 1
> - **Fonction de coût** : **Log loss** (ou cross-entropy) :
>   J(θ) = -1/m * Σ [y_i log(hθ(x_i)) + (1-y_i) log(1-hθ(x_i))]
> - **Exemple** : Prédire si un élève est dans Gryffindor (1) ou non (0)
> 
> **Différences clés :**
> | Aspect | Linéaire | Logistique |
> |--------|----------|------------|
> | Sortie | ℝ | [0, 1] |
> | Usage | Régression | Classification |
> | Fonction | Linéaire | Sigmoid |
> | Coût | MSE | Log loss |
> 
> **Pourquoi la sigmoid ?**
> Sans elle, la sortie pourrait être n'importe quel nombre réel. La sigmoid **compresse** cette sortie entre 0 et 1, ce qui permet d'interpréter le résultat comme une **probabilité**."

---

### ❓ Question 2 : "Pourquoi normaliser les données ?"

**✅ Réponse complète :**

> **La normalisation est CRUCIALE pour la descente de gradient. Voici pourquoi :**
> 
> **1. Problème sans normalisation :**
> - Si une feature a des valeurs entre [0, 1000] et une autre entre [0, 1], la descente de gradient **oscille** et converge très lentement
> - Les poids (θ) associés à la feature [0, 1000] seront **beaucoup plus petits** que ceux de [0, 1]
> - La fonction de coût a une forme de **vallée très étroite**, ce qui rend l'optimisation difficile
> 
> **2. Solution : Standardisation (z-score)**
> ```
> x_normalisé = (x - μ) / σ
> ```
> Où :
> - μ = moyenne de la feature
> - σ = écart-type de la feature
> 
> **Effets :**
> - Toutes les features ont **moyenne = 0** et **écart-type = 1**
> - La fonction de coût devient **symétrique** (forme de bol régulier)
> - La descente de gradient converge **beaucoup plus vite**
> - Les poids (θ) ont des **ordonnées de grandeur similaires**
> 
> **3. Attention :**
> - **Toujours utiliser les paramètres du TRAIN** pour normaliser le TEST !
> - Dans mon code : `logreg_predict.py` utilise les **μ et σ calculés sur dataset_train.csv** pour normaliser dataset_test.csv
> - Sinon, on **fausse** la distribution du test
> 
> **4. Exemple concret :**
> - Sans normalisation : Il faut 10 000 itérations pour converger
> - Avec normalisation : 1 000 itérations suffisent
> 
> **5. Alternative :**
> On peut aussi utiliser la **normalisation Min-Max** : x' = (x - min) / (max - min)
> Mais la standardisation (z-score) est **plus robuste aux outliers**."

---

### ❓ Question 3 : "Qu'est-ce que la méthode one-vs-all (one-vs-rest) ?"

**✅ Réponse complète :**

> **Problème :** La régression logistique est conçue pour la **classification binaire** (2 classes), mais nous avons **4 maisons** à prédire.
> 
> **Solution : One-vs-All (OvA)**
> 
> **Principe :**
> 1. **On entraîne 4 classifieurs binaires** (un par maison)
>    - Classifieur 1 : Gryffindor vs {Hufflepuff, Ravenclaw, Slytherin}
>    - Classifieur 2 : Hufflepuff vs {Gryffindor, Ravenclaw, Slytherin}
>    - Classifieur 3 : Ravenclaw vs {Gryffindor, Hufflepuff, Slytherin}
>    - Classifieur 4 : Slytherin vs {Gryffindor, Hufflepuff, Ravenclaw}
> 
> 2. **Pour chaque classifieur :**
>    - On transforme le problème en **binaire** :
>      - y = 1 si l'élève est dans la maison ciblée
>      - y = 0 sinon
>    - On entraîne un modèle avec la descente de gradient
> 
> 3. **Pour prédire :**
>    - On calcule la **probabilité** pour chaque classifieur
>    - On sélectionne la maison avec la **probabilité la plus élevée** (argmax)
> 
> **Exemple concret :**
> - Élève X → Probabilités : [0.1, 0.6, 0.2, 0.1]
> - Prédiction : **Hufflepuff** (probabilité la plus élevée)
> 
> **Pourquoi ça marche ?**
> - Chaque classifieur apprend à **reconnaître sa maison** vs les autres
> - L'**argmax** garantit qu'une seule maison est sélectionnée
> - C'est **simple** et **efficace** pour les problèmes multi-classes
> 
> **Alternative :** One-vs-One (OvO) : on entraîne C(C-1)/2 classifieurs (6 pour 4 classes), mais c'est plus complexe.
> 
> **Dans mon code :**
> - `logreg_train.py` entraîne 4 classifieurs (un par maison)
> - `logreg_predict.py` calcule les 4 probabilités et prend l'argmax"

---

### ❓ Question 4 : "Comment fonctionne la descente de gradient ?"

**✅ Réponse complète :**

> **Principe général :**
> La descente de gradient est un **algorithme d'optimisation** qui minimise une fonction (le coût J(θ)).
> 
> **Analogie :**
> Imagine que tu es en montagne et que tu veux descendre au point le plus bas (minimum de la vallée).
> - Tu regardes autour de toi : **quelle direction est la plus pentue vers le bas ?**
> - Tu fais un **petit pas** dans cette direction
> - Tu répètes jusqu'à atteindre le bas
> 
> **En mathématiques :**
> ```
> θ := θ - α * ∇J(θ)
> ```
> Où :
> - θ = paramètres (poids) du modèle
> - α (alpha) = **learning rate** (taille du pas)
> - ∇J(θ) = **gradient** de la fonction de coût (direction de la plus forte pente)
> 
> **3 variantes implémentées dans mon code :**
> 
> **1. Batch Gradient Descent (obligatoire)**
> - **Gradient** : Moyenne sur **tout le dataset**
> - **1 pas = 1 époque** (tout le dataset)
> - **Avantage** : Stable, converge vers le minimum global
> - **Inconvénient** : Lent si le dataset est grand
> - **Formule** :
>   ```
>   ∇J(θ) = (1/m) * Σ (hθ(x_i) - y_i) * x_i
>   ```
> 
> **2. Stochastic Gradient Descent (SGD)**
> - **Gradient** : Calculé sur **1 seul exemple** aléatoire
> - **1 pas = 1 exemple**
> - **Avantage** : Très rapide par itération
> - **Inconvénient** : Bruité, peut oscillier
> - **Commande** : `--optimizer sgd`
> 
> **3. Mini-batch Gradient Descent**
> - **Gradient** : Moyenne sur un **petit lot** (batch) de N exemples
> - **1 pas = 1 batch**
> - **Avantage** : Compromis entre stabilité et vitesse
> - **Inconvénient** : Nécessite de choisir la taille du batch
> - **Commande** : `--optimizer minibatch --batch-size 32`
> 
> **Dans mon code (`logreg_train.py`) :**
> - Par défaut : **Batch GD** (obligatoire)
> - Bonus : SGD et Mini-batch GD implémentés
> - Toutes les variantes utilisent la même formule de gradient, seule la taille du lot change"

---

---

## ⚡ **QUESTIONS SUR L'EXÉCUTION**

### ❓ Question : "Exécutez `logreg_train` avec `dataset_train.csv`"

**Commande :**
```bash
make train
# ou
python3 logreg_train.py datasets/dataset_train.csv
```

**Ce que l'évaluateur voit :**
- Entraînement en cours avec affichage du coût à chaque époque
- À la fin : un fichier `weights.json` est créé

**✅ Réponse :**
> "Oui, la commande crée bien un fichier `weights.json` contenant :
> - Les **theta** (poids) pour chaque maison (4 classifieurs)
> - Les **moyennes (mu)** des features calculées sur le dataset d'entraînement
> - Les **écarts-types (sd)** des features calculées sur le dataset d'entraînement
> 
> Ces paramètres de prétraitement sont **indispensables** pour appliquer la même transformation sur le dataset de test."

---

### ❓ Question : "Exécutez `logreg_predict` avec les poids et `dataset_test.csv`"

**Commande :**
```bash
make predict
# ou
python3 logreg_predict.py datasets/dataset_test.csv weights.json
```

**Ce que l'évaluateur voit :**
- Un fichier `houses.csv` est créé
- Répartition des prédictions affichée (ex: {'Gryffindor': 400, 'Hufflepuff': 350, ...})

**✅ Réponse :**
> "Oui, la commande crée bien un fichier `houses.csv` au **format exact**要求 :
> ```csv
> Index,Hogwarts House
> 0,Gryffindor
> 1,Hufflepuff
> 2,Ravenclaw
> ...
> ```
> 
> **Processus :**
> 1. Charge `dataset_test.csv`
> 2. Charge `weights.json` (contient les θ, μ, sd du train)
> 3. Applique le **prétraitement** :
>    - Imputation des NaN avec les moyennes du **train**
>    - Standardisation avec μ et sd du **train**
> 4. Ajoute la colonne de biais (1)
> 5. Calcule les probabilités : `sigmoid(X @ theta)`
> 6. Prend l'**argmax** pour chaque élève
> 7. Écrit le résultat dans `houses.csv`"

---

### ❓ Question : "Utilisez `evaluate.py` pour vérifier le score"

**Commande :**
```bash
python3 evaluate.py houses.csv dataset_truth.csv
```

**✅ Réponse :**
> "La commande `evaluate.py` compare `houses.csv` (mes prédictions) avec `dataset_truth.csv` (la vérité terrain) et calcule l'**accuracy** (taux de bonnes prédictions).
> 
> **Seuil requis :** ≥ 98% (0.98)
> 
> **Si le score est ≥ 98%** : Le projet est validé pour la partie obligatoire.
> **Si le score < 98%** : Il y a un problème (sous-apprentissage, sur-apprentissage, bug).
> 
> **Dans mon cas :** [À toi de tester et de dire le score obtenu]
> 
> **Pourquoi un score élevé ?**
> - La régression logistique est bien adaptée pour ce problème
> - Le prétraitement (imputation + standardisation) est correct
> - L'entraînement a convergé (nombre d'époques suffisant)
> - La descente de gradient a trouvé le minimum global"

---

---

## 💡 **RÉPONSES AUX QUESTIONS PIÈGES**

### ❓ "Pourquoi ne pas utiliser `pandas.describe()` ?"

**✅ Réponse :**
> "Le sujet **interdit explicitement** toute fonction de bibliothèque qui fait le travail à notre place. 
> 
> **Problème avec `pandas.describe()` :**
> - Elle calcule **automatiquement** toutes les statistiques (count, mean, std, min, 25%, 50%, 75%, max)
> - On n'aurait **rien codé nous-mêmes**
> - Ce serait considéré comme de la **triche**
> 
> **Notre approche :**
> - On a **réimplémenté chaque fonction** manuellement dans `dslr_utils.py`
> - On utilise `numpy` uniquement pour l'**algèbre linéaire** (matrices, opérations vectorielles), ce qui est **autorisé** par le sujet"

---

### ❓ "Pourquoi utiliser `numpy` si on ne peut pas utiliser ses fonctions statistiques ?"

**✅ Réponse :**
> "Le sujet **autorise explicitement** `numpy` pour l'algèbre linéaire, mais **interdit** ses fonctions statistiques (`numpy.mean()`, `numpy.std()`, etc.).
> 
> **Ce qu'on utilise dans `numpy` (autorisé) :**
> - Créer des matrices (`np.array`, `np.zeros`)
> - Opérations vectorielles (`+`, `-`, `*`, `/`)
> - `np.isnan()` pour détecter les NaN
> - `np.sum()` pour sommer des éléments
> - Indexation (`matrix[:, i]`)
> 
> **Ce qu'on N'utilise PAS (interdit) :**
> - `np.mean()` → On a notre propre `mean()`
> - `np.std()` → On a notre propre `std()`
> - `np.percentile()` → On a notre propre `percentile()`
> - `np.corrcoef()` → On a notre propre `pearson()`
> 
> **Pourquoi cette distinction ?**
> - Les opérations d'**algèbre linéaire** sont des **outils bas niveau**
> - Les fonctions **statistiques** font le **travail à notre place**"

---

### ❓ "Comment gérez-vous les valeurs manquantes (NaN) ?"

**✅ Réponse :**
> "On utilise une **stratégie en deux étapes** :
> 
> **1. Pendant les calculs statistiques (`describe`) :**
> - On **filtre** les NaN avec la fonction `_finite()` dans `dslr_utils.py`
> - Exemple : `count()` utilise `_finite(values).size`
> - Ça permet de calculer les stats **uniquement sur les valeurs valides**
> 
> **2. Pendant l'entraînement et la prédiction :**
> - On **impute** les NaN avec la **moyenne de la colonne** (calculée sur le train)
> - Dans `logreg_train.py` : `impute_with_mean()` remplacent les NaN par la moyenne
> - **Important** : On utilise les **moyennes du TRAIN** pour imputer le TEST
> - Ça garantit que le prétraitement est **reproductible** et cohérent
> 
> **Pourquoi l'imputation ?**
> - On ne peut pas entraîner un modèle avec des NaN
> - L'imputation par la moyenne est **simple et efficace**
> - C'est une méthode standard en data science"

---

### ❓ "Pourquoi standardiser les données ?" *(déjà répondu, mais version courte)*

**✅ Réponse courte :**
> "Pour **accélérer la convergence** de la descente de gradient. Sans standardisation, les features avec de grandes échelles dominent le gradient, ce qui rend l'optimisation lente et instable. Avec la standardisation (z-score), toutes les features ont la même échelle (moyenne=0, écart-type=1), et la descente de gradient converge **10 à 100 fois plus vite**."

---

### ❓ "Que se passe-t-il si une feature a un écart-type de 0 ?"

**✅ Réponse :**
> "Dans `dslr_utils.py`, on a une **protection** contre la division par zéro :
> ```python
> sd = np.where(sd == 0, 1.0, sd)  # dans standardize()
> ```
> 
> **Explication :**
> - Si une feature a un écart-type de 0, toutes ses valeurs sont identiques
> - La standardisation donnerait : (x - μ) / 0 → **erreur**
> - On remplace σ par 1.0 dans ce cas
> - Résultat : la feature standardisée = 0 pour toutes les lignes
> - **Impact** : Cette feature n'aura **aucun effet** sur la prédiction (car tous les x_i sont identiques)
> - C'est **cohérent** : une feature constante n'apporte aucune information"

---

---

## 🎯 **RÉCAPITULATIF DES COMMANDES**

| Objectif | Commande | Fichier généré |
|----------|----------|-----------------|
| **Statistiques de base** | `make describe` | Affichage console |
| **Statistiques + bonus** | `make describe-extra` | Affichage console |
| **Histogramme** | `make histogram` | `plots/histogram.png` |
| **Scatter plot** | `make scatter` | `plots/scatter_plot.png` |
| **Pair plot** | `make pair` | `plots/pair_plot.png` |
| **Toutes les figures** | `make plots` | `plots/*.png` |
| **Entraînement** | `make train` | `weights.json` |
| **Prédiction** | `make predict` | `houses.csv` |
| **Tout (obligatoire)** | `make all` | Tous les fichiers |
| **Tout + bonus** | `make bonus` | Tous les fichiers + bonus |
| **Vérifier l'environnement** | `make check` | Affichage console |
| **Nettoyer** | `make clean` | Supprime les artefacts |
| **Tout relancer** | `make re` | Nettoie + relance tout |

---

## 📌 **CHECKLIST DERNIÈRE MINUTE**

### ✅ Avant l'évaluation
- [ ] J'ai testé `make all` et tout fonctionne
- [ ] J'ai testé `make bonus` et tout fonctionne
- [ ] J'ai vérifié que `houses.csv` est bien formaté
- [ ] J'ai révisé les **3 définitions** (moyenne, écart-type, quartiles)
- [ ] J'ai révisé les **explications théoriques** (régression logistique, one-vs-all, normalisation)
- [ ] J'ai préparé des **exemples concrets** pour chaque concept
- [ ] J'ai vérifié que mon dépôt ne contient que les fichiers nécessaires

### ✅ Pendant l'évaluation
- [ ] Je reste **calme et confiant**
- [ ] Je réponds **clairement et précisément**
- [ ] Je montre que je **comprends** ce que je dis
- [ ] Je suis prêt à **expliquer le code** ligne par ligne
- [ ] Je vérifie que chaque commande **fonctionne** avant de la lancer

---

## 🚀 **BONNE CHANCE !**

Tu as tout ce qu'il faut pour **réussir brillamment** cette évaluation !

- **Ton code est conforme** à 100%
- **Tes explications sont prêtes**
- **Tu as révisé la théorie**

**Respire, sourit, et montre ce que tu sais faire !** 🎉

---

*Ce guide a été généré spécialement pour toi par Mistral Vibe. Bonne évaluation !*
