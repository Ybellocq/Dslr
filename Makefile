# Makefile du projet dslr (Data Science x Logistic Regression).
#
# Objectif : `make` doit fonctionner sur N'IMPORTE QUELLE machine, meme si
# numpy/matplotlib ne sont pas installes. La cible `deps` s'en charge :
#
#   1. si un Python 3 du systeme possede deja numpy + matplotlib -> utilise tel quel ;
#   2. sinon -> creation d'un environnement virtuel local `.venv` et installation
#      des dependances dedans (aucun sudo, aucune modification du systeme).
#
# L'interpreteur retenu est reutilise par toutes les cibles. Verifiez avec : make check
# Surcharge possible : make PYTHON=/chemin/vers/python3 ...   et   make VENV=.mon_venv ...

BASE_PYTHON ?= $(shell sh -c 'for p in python3 python3.13 python3.12 python3.11 python3.10 python3.9 /usr/bin/python3; do command -v $$p >/dev/null 2>&1 && { echo $$p; exit 0; }; done; echo python3')
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
STAMP := .deps.stamp

# Interpreteur utilise par les cibles : le venv s'il existe, sinon un Python systeme
# deja dote de numpy+matplotlib, sinon le venv (qui sera cree par `make deps`).
PYTHON ?= $(shell sh -c 'if [ -x "$(VENV_PY)" ]; then echo "$(VENV_PY)"; exit 0; fi; for p in python3 python3.13 python3.12 python3.11 python3.10 python3.9 /usr/bin/python3; do if command -v $$p >/dev/null 2>&1 && $$p -c "import numpy, matplotlib" >/dev/null 2>&1; then echo $$p; exit 0; fi; done; echo "$(VENV_PY)"')

DATASET_TRAIN ?= datasets/dataset_train.csv
DATASET_TEST ?= datasets/dataset_test.csv
WEIGHTS ?= weights.json
OUTPUT ?= houses.csv
PLOTS ?= plots

.PHONY: all deps install check describe describe-extra histogram scatter pair plots train train-cv bonus predict clean distclean re help

all: deps describe plots train predict  ## chaine complete : deps + stats + figures + entrainement + prediction

# --------------------------------------------------------------------------- #
# Dependances
# --------------------------------------------------------------------------- #
deps: $(STAMP)  ## verifie (et installe si besoin) numpy et matplotlib

$(STAMP): requirements.txt
	@echo ">>> Verification des dependances (numpy, matplotlib)..."
	@if $(PYTHON) -c "import numpy, matplotlib" >/dev/null 2>&1; then \
		echo "    OK : dependances deja disponibles ($(PYTHON))"; \
		touch $(STAMP); \
	elif [ -x "$(VENV_PY)" ]; then \
		echo "    Installation dans l'environnement virtuel existant $(VENV)..."; \
		$(VENV_PY) -m pip install --quiet --upgrade pip; \
		$(VENV_PY) -m pip install --quiet -r requirements.txt && touch $(STAMP); \
	else \
		echo "    Aucun interpréteur avec numpy : creation de $(VENV) avec $(BASE_PYTHON)..."; \
		$(BASE_PYTHON) -m venv $(VENV) && \
		$(VENV_PY) -m pip install --quiet --upgrade pip && \
		$(VENV_PY) -m pip install --quiet -r requirements.txt && touch $(STAMP); \
	fi

install:  ## force la (re)installation des dependances
	@rm -f $(STAMP)
	@$(MAKE) --no-print-directory deps

check: deps  ## affiche l'interpreteur retenu et les versions
	@$(PYTHON) -c "import sys; print('Interpreteur      :', sys.executable)"
	@$(PYTHON) -c "import numpy, matplotlib; print('numpy / matplotlib:', numpy.__version__, '/', matplotlib.__version__)"

# --------------------------------------------------------------------------- #
# Programmes du sujet
# --------------------------------------------------------------------------- #
describe: deps  ## statistiques descriptives (partie obligatoire)
	$(PYTHON) describe.py $(DATASET_TRAIN)

describe-extra: deps  ## statistiques descriptives + bonus (Variance, IQR, Missing, Skewness...)
	$(PYTHON) describe.py $(DATASET_TRAIN) --extra

histogram: deps  ## histogrammes par maison (ecrit $(PLOTS)/histogram.png)
	$(PYTHON) histogram.py $(DATASET_TRAIN) --no-show

scatter: deps  ## nuage de points des 2 features similaires (ecrit $(PLOTS)/scatter_plot.png)
	$(PYTHON) scatter_plot.py $(DATASET_TRAIN) --no-show

pair: deps  ## matrice de nuages de points (ecrit $(PLOTS)/pair_plot.png)
	$(PYTHON) pair_plot.py $(DATASET_TRAIN) --no-show

plots: histogram scatter pair  ## genere les trois figures

train: deps  ## entraine le modele (descente de gradient batch) -> $(WEIGHTS)
	$(PYTHON) logreg_train.py $(DATASET_TRAIN) --output $(WEIGHTS)

train-cv: deps  ## entraine + validation croisee 5 plis -> $(WEIGHTS)
	$(PYTHON) logreg_train.py $(DATASET_TRAIN) --cv 5 --output $(WEIGHTS)

bonus: deps  ## BONUS : compare les 3 optimiseurs (batch/sgd/minibatch) + describe --extra
	@echo '=== Bonus 1 : statistiques etendues ==='
	$(PYTHON) describe.py $(DATASET_TRAIN) --extra
	@echo ''
	@echo '=== Bonus 2 : comparaison des optimiseurs (validation croisee 5 plis) ==='
	$(PYTHON) logreg_train.py $(DATASET_TRAIN) --optimizer batch --epochs 5000 --cv 5 --output $(WEIGHTS)
	$(PYTHON) logreg_train.py $(DATASET_TRAIN) --optimizer minibatch --batch-size 32 --epochs 300 --cv 5 --output $(WEIGHTS)
	$(PYTHON) logreg_train.py $(DATASET_TRAIN) --optimizer sgd --epochs 50 --cv 5 --output $(WEIGHTS)

predict: deps  ## predit les maisons du jeu de test -> $(OUTPUT)
	$(PYTHON) logreg_predict.py $(DATASET_TEST) $(WEIGHTS) --output $(OUTPUT)

# --------------------------------------------------------------------------- #
# Nettoyage et aide
# --------------------------------------------------------------------------- #
clean:  ## supprime les artefacts generes (garde l'environnement virtuel)
	rm -f $(WEIGHTS) $(OUTPUT) $(STAMP)
	rm -f $(PLOTS)/*.png

distclean: clean  ## comme clean + supprime l'environnement virtuel
	rm -rf $(VENV)

re: clean all  ## relance tout depuis zero

help:  ## liste les cibles disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  %-16s %s\n", $$1, $$2}'
