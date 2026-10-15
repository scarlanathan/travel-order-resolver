# Travel Order Resolver

Système complet pour résoudre des demandes de voyage en français :
1. **NLP** : Extraction automatique d'origine et destination (spaCy, CamemBERT, Baseline)
2. **Pathfinding** : Calcul d'itinéraire optimal (Dijkstra, A*, moins de correspondances)
3. **API SNCF** : Données des gares françaises (fuzzy matching, gare la plus proche)

## Contexte

Ce projet combine **Named Entity Recognition (NER)** et **algorithme de Dijkstra** pour transformer une requête naturelle en itinéraire de train :

```
Input:  "Je voudrais aller de Paris à Lyon"
Output: Route Paris -> Lyon (1h57, 392 km, TGV direct)
```

### Pipeline complète

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Phrase NL     │────>│  NLP (spaCy /       │────>│   Pathfinding       │
│                 │     │  CamemBERT /        │     │   (Dijkstra / A* /  │
│ "Paris à Lyon"  │     │  Baseline)          │     │   min-changes)      │
│                 │     │  origin: Paris      │     │                     │
│                 │     │  dest: Lyon         │     │   Route optimale    │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
```

### Modèles NLP implémentés

| Modèle | Accuracy (villes connues) | Accuracy (villes inconnues) | Inférence | Description |
|--------|---------------------------|-----------------------------|-----------|-------------|
| **CamemBERT 1k + gazetteer** | **98.33%** | **98.75%** | 65 ms | Transformers fine-tuné : le plus robuste |
| spaCy blank 1k + gazetteer | 98.73% | 83.41% | 2.5 ms | Très rapide, généralise moins bien |
| Baseline | 99.93% | 0.00% | 0.3 ms | Gazetteer + mots-indices (monde fermé) |

Détails, méthodologie et limites : voir [Résultats des modèles](#résultats-des-modèles).

### Réseau ferroviaire

- **51 gares** principales françaises
- **142 connexions** (TGV, Intercités, TER)
- Algorithme **Dijkstra** pour le plus court chemin (durée)
- Algorithme **A*** avec heuristique Haversine/TGV
- Stratégie **moins de correspondances**
- **Fuzzy matching** des noms de villes (Levenshtein)
- **Gare la plus proche** pour les villes sans gare

## Installation

```bash
# Cloner le repository
git clone https://github.com/scarlanathan/travel-order-resolver.git
cd travel-order-resolver

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Variables d'environnement (clé API SNCF optionnelle, l'API publique fonctionne sans)
cp .env.example .env

# Télécharger les modèles spaCy français (optionnel, pour les modèles sm/md/lg)
python -m spacy download fr_core_news_sm
python -m spacy download fr_core_news_md
python -m spacy download fr_core_news_lg
```

> **Important :** les dossiers `models/`, `data/` et `outputs/` sont exclus du dépôt
> (`.gitignore`, ~4 Go de modèles). Après un clone, il faut régénérer le dataset
> puis entraîner les modèles (voir [Commandes](#commandes)). Sans modèle entraîné,
> `travel_resolver.py` bascule automatiquement sur le modèle **Baseline**.
>
> La liste des gares est récupérée en direct depuis l'API Open Data SNCF : une
> connexion Internet est nécessaire.

## Structure du projet

```
travel-order-resolver/
├── configs/
│   └── config.yaml              # Configuration générale
├── data/
│   ├── datasets/
│   │   └── full_dataset.csv     # Dataset complet (10k phrases)
│   └── training/
│       ├── train.csv            # Données d'entraînement (7k)
│       ├── val.csv              # Données de validation (1.5k)
│       └── test.csv             # Données de test (1.5k)
├── docs/
│   ├── critique_qualitative.md  # Analyse comparative des modèles
│   └── spacy_pipeline.md        # Documentation pipeline spaCy
├── models/
│   ├── camembert_ner/           # Modèles CamemBERT entraînés
│   │   ├── camembert_ner_1000/
│   │   ├── camembert_ner_3000/  # Meilleur CamemBERT (98.03% F1)
│   │   ├── camembert_ner_5000/
│   │   ├── camembert_ner_full/
│   │   ├── camembert_lr1e-05/   # Variation learning rate
│   │   ├── camembert_lr3e-05/
│   │   ├── camembert_lr5e-05/
│   │   ├── camembert_bs8/       # Variation batch size
│   │   ├── camembert_bs16/
│   │   ├── camembert_bs32/
│   │   ├── camembert_sched_linear/  # Variation scheduler
│   │   └── camembert_sched_cosine/
│   └── spacy_ner/               # Modèles spaCy entraînés
│       ├── ner_blank_1000/
│       ├── ner_blank_3000/
│       ├── ner_blank_5000/      # Meilleur spaCy (95.98% F1)
│       ├── ner_blank_full/
│       ├── ner_sm_full/
│       ├── ner_md_full/
│       └── ner_lg_full/
├── outputs/
│   └── evaluation/
│       ├── baseline_test_report.csv
│       ├── spacy_experiments.json
│       ├── camembert_experiments.json
│       └── spacy_loss_curves.png
├── scripts/
│   ├── generate_dataset.py          # Génération du dataset
│   ├── generate_dataset_simple.py
│   ├── evaluate_baseline.py         # Évaluation baseline (rapport détaillé)
│   ├── evaluate_models.py           # Comparaison de tous les modèles (villes connues / inconnues)
│   ├── train_spacy_models.py        # Entraînement spaCy
│   ├── train_camembert.py           # Entraînement CamemBERT
│   ├── travel_resolver.py           # Pipeline complète NLP + Pathfinding
│   └── travel_resolver_voice.py     # Pipeline vocale (Whisper + pyttsx3)
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── sncf_client.py       # Client API SNCF (gares, connexions)
│   ├── pathfinder/
│   │   ├── __init__.py
│   │   └── dijkstra.py          # Algorithme Dijkstra (NetworkX)
│   ├── speech/
│   │   ├── transcriber.py       # Speech-to-text (Whisper)
│   │   └── synthesizer.py       # Text-to-speech (pyttsx3)
│   ├── data_generation/
│   │   ├── sentence_generator.py
│   │   └── templates.py
│   └── nlp/
│       ├── evaluator.py         # Classe d'évaluation
│       ├── gazetteer.py         # Villes connues, alignement des entités, post-traitement
│       └── models/
│           ├── baseline_model.py    # Modèle baseline
│           ├── spacy_ner.py         # Modèle spaCy
│           └── camembert_ner.py     # Modèle CamemBERT
├── tests/
│   └── test_nlp.py              # Tests unitaires (gazetteer, baseline, évaluateur)
├── docker/
│   ├── Dockerfile.nlp
│   └── docker-compose.yml
├── docs/ & *.pdf                # Sujet, kickoff, bootstrap, rapports
├── test.py                      # Génère les graphiques comparatifs spaCy/CamemBERT (*.png)
├── Makefile
├── requirements.txt
└── README.md
```

## Commandes

### 1. Génération du dataset

```bash
# Générer le dataset complet
python scripts/generate_dataset_simple.py

# Le dataset sera créé dans data/datasets/full_dataset.csv
# et divisé en train/val/test dans data/training/
```

### 2. Modèle Baseline

```bash
# Évaluer le modèle baseline (gazetteer + mots-indices)
python scripts/evaluate_baseline.py

# Résultats dans outputs/evaluation/baseline_test_report.csv
```

### 3. Modèle spaCy NER

```bash
# Entraîner tous les modèles spaCy (différentes tailles et bases)
python scripts/train_spacy_models.py --all-experiments --n-iter 30

# Entraîner un modèle spécifique
python scripts/train_spacy_models.py --train-size 1000 --n-iter 30

# Options disponibles:
#   --all-experiments : Entraîner tous les modèles (blank_1k, blank_3k, blank_5k, blank_full, sm_full, md_full, lg_full)
#   --train-size      : Nombre de samples (omis = dataset complet)
#   --n-iter          : Nombre d'itérations d'entraînement (défaut: 30)
#   --base-model      : Modèle de base (blank, fr_core_news_sm, fr_core_news_md, fr_core_news_lg)
#   --output          : Fichier de sortie des résultats

# Résultats dans outputs/evaluation/spacy_experiments.json
# Modèles dans models/spacy_ner/
```

### 4. Modèle CamemBERT

```bash
# Entraîner tous les modèles CamemBERT (1000, 3000, 5000, full)
python scripts/train_camembert.py --all-experiments --epochs 3

# Entraîner un modèle spécifique
python scripts/train_camembert.py --train-size 1000 --epochs 3

# Options disponibles:
#   --all-experiments : Entraîner tous les modèles (1k, 3k, 5k, full + variations LR/batch/scheduler)
#   --train-size      : Nombre de samples d'entraînement
#   --epochs          : Nombre d'époques
#   --batch-size      : Taille du batch (défaut: 16)
#   --learning-rate   : Learning rate (défaut: 5e-5)
#   --lr-scheduler    : Type de scheduler ('linear', 'cosine')
#   --output          : Fichier de sortie des résultats

# Résultats dans outputs/evaluation/camembert_experiments.json
# Modèles dans models/camembert_ner/
```

> L'entraînement CamemBERT nécessite `accelerate` (inclus dans `requirements.txt`).

### 5. Comparaison des modèles

```bash
# Évalue le baseline et tous les modèles de models/ (bruts et + gazetteer)
# sur le test set et sur un test set de villes inconnues (généré via l'API SNCF)
python scripts/evaluate_models.py

# Modèles choisis, sans le test villes inconnues (hors ligne)
python scripts/evaluate_models.py --models baseline models/spacy_ner/ner_blank_1000 --no-unseen

# Résultats dans outputs/evaluation/model_comparison.json

# Tests unitaires
python -m pytest tests/ -v
```

### 6. Travel Resolver (Pipeline complète)

```bash
# Résoudre une demande de voyage (NLP + Pathfinding)
python scripts/travel_resolver.py "Je voudrais aller de Paris à Marseille"

# Mode interactif
python scripts/travel_resolver.py --interactive

# Afficher les routes alternatives
python scripts/travel_resolver.py -a "Lyon Bordeaux"

# Utiliser le modèle baseline au lieu de spaCy
python scripts/travel_resolver.py -m baseline "Nantes Strasbourg"

# Utiliser le modèle CamemBERT
python scripts/travel_resolver.py -m camembert "Nantes Strasbourg"

# Sans argument : démo sur 5 phrases d'exemple
python scripts/travel_resolver.py
```

> Modèles chargés par défaut : `models/spacy_ner/ner_blank_1000` (`-m spacy`) et
> `models/camembert_ner/camembert_ner_1000` (`-m camembert`, repli sur
> `camembert_ner_full`). Si le modèle est absent, le Baseline est utilisé.

**Exemple de sortie :**
```
Input: "Je voudrais aller de Paris à Marseille"
Extracted: origin=Paris, destination=Marseille

Route: Paris -> Marseille
Duration: 2h48
Distance: 666.1 km
Changes: 0
----------------------------------------
1. Paris -> Avignon (TGV, 148 min, 582.3 km)
2. Avignon -> Marseille (TGV, 20 min, 83.8 km)
```

### 7. Utilisation des modèles en Python

```python
# NLP - Extraction origine/destination (spaCy)
from src.nlp.models.spacy_ner import SpacyNERModel
model = SpacyNERModel("models/spacy_ner/ner_blank_1000")
result = model.predict("Un billet Marseille Bordeaux")
print(f"Origine: {result.origin}, Destination: {result.destination}")

# NLP - Extraction avec CamemBERT
# (post-traitement gazetteer actif par défaut, use_gazetteer=False pour la sortie brute)
from src.nlp.models.camembert_ner import CamembertNERModel
model = CamembertNERModel("models/camembert_ner/camembert_ner_1000")
result = model.predict("Je veux aller de Paris à Lyon")
print(f"Origine: {result.origin}, Destination: {result.destination}")

# Pathfinding - Trouver un itinéraire (optimisé par durée)
from src.pathfinder.dijkstra import TrainGraph
graph = TrainGraph()
route = graph.find_shortest_path("Paris", "Lyon")        # Dijkstra
route_astar = graph.find_path_astar("Paris", "Lyon")     # A*
route_min = graph.find_path_min_changes("Paris", "Lyon")  # Min correspondances
print(route)

# API SNCF - Accès aux données des gares (avec fuzzy matching)
from src.api.sncf_client import SNCFClient
client = SNCFClient()
client.load_data()
paris = client.find_station_by_city("Pari")  # Fuzzy matching
print(f"{paris.name} ({paris.code})")

# Pipeline complète
from scripts.travel_resolver import TravelResolver
resolver = TravelResolver(nlp_model="camembert")  # ou un chemin vers un modèle
route = resolver.resolve("Je veux aller de Lyon à Nice")
print(route)
```

## Résultats des modèles

### Emplacement des résultats

| Type | Emplacement |
|------|-------------|
| Comparaison de tous les modèles | `outputs/evaluation/model_comparison.json` (`scripts/evaluate_models.py`) |
| Rapport détaillé Baseline | `outputs/evaluation/baseline_test_report.csv` |
| Entraînements spaCy / CamemBERT | `outputs/evaluation/spacy_experiments.json`, `camembert_experiments.json` |
| Analyse qualitative | `docs/critique_qualitative.md` |

### Méthodologie

- **Accuracy** : une phrase est correcte si l'origine **et** la destination sont justes.
  Pour une phrase invalide (`INVALID`), le modèle doit ne rien renvoyer.
- **Villes connues** : `data/training/test.csv` (1500 phrases dont 215 invalides).
  Toutes ses villes apparaissent aussi dans le train : ce test mesure la performance
  sur des gares connues.
- **Villes inconnues** : les 1284 phrases valides du test, dont les villes sont
  remplacées par des gares SNCF absentes du train (`data/training/test_unseen_cities.csv`).
  Ce test mesure la capacité à généraliser à un nom jamais vu.

### Résultats (mesurés le 15/10/2026, sur CPU)

| Modèle | Villes connues : accuracy | F1 | Villes inconnues : accuracy | F1 | Inférence |
|--------|------------|-----|------------|-----|-----------|
| Baseline (gazetteer + mots-indices) | **99.93%** | 99.98% | 0.00% | 0.00% | 0.3 ms |
| spaCy blank 1k (brut) | 91.53% | 96.72% | 83.41% | 94.32% | 2.3 ms |
| spaCy blank 1k + gazetteer | 98.73% | 99.59% | 83.41% | 91.99% | 2.5 ms |
| spaCy blank full (brut) | 89.00% | 96.12% | 48.99% | 81.06% | 2.2 ms |
| spaCy blank full + gazetteer | 98.87% | 99.61% | 48.99% | 68.88% | 2.6 ms |
| CamemBERT 1k (brut) | 96.73% | 99.22% | **98.75%** | **99.51%** | 66 ms |
| **CamemBERT 1k + gazetteer** | 98.33% | 99.45% | **98.75%** | **99.51%** | 65 ms |

Les modèles spaCy `ner_blank_3000`, `ner_blank_5000`, `ner_sm_full` et `ner_md_full`
ont été entraînés le 30/01, avant la régénération du dataset (12/02). Ils n'obtiennent
que 26 à 37 % d'accuracy sur le test actuel et doivent être réentraînés.

### Ce qui a fait progresser l'accuracy

| Correction | Effet mesuré |
|------------|--------------|
| **Évaluateur** : une phrase invalide correctement rejetée (`None`) était comptée fausse face au label `"INVALID"` | Jusqu'à +14,3 points sur tous les modèles (215 phrases sur 1500). Les anciennes accuracies (67 à 75 %) étaient sous-estimées. |
| **Baseline réécrit** sur un gazetteer (villes du train + gares SNCF) tolérant la casse, les accents, les tirets et les fautes, avec un rôle origine/destination déduit des mots-indices (`de`, `depuis`, `vers`, `à destination de`…) | 4,2 % → **99,9 %**. L'ancien gazetteer venait de l'API SNCF live, dont les noms ne correspondaient pas au dataset ; 227 erreurs sur 278 étaient des inversions origine/destination. |
| **CamemBERT : décodage par mot** (le premier sous-mot décide du label de tout le mot) | 94,6 % → 96,7 % sans réentraînement. Plus d'entités coupées comme « Musée d' » ou « l » pour « L'Isle-Jourdain ». |
| **Post-traitement gazetteer** des modèles NER : prolonge un span tronqué jusqu'à la ville connue, complète l'entité manquante, rejette les phrases à une seule ville (« Quel temps fait-il à X ? ») | spaCy 1k : 91,5 → 98,7 % ; CamemBERT : 96,7 → 98,3 %. **Aucune perte** sur les villes inconnues. |
| **Alignement des annotations d'entraînement** : les entités étaient cherchées par correspondance exacte du label. 12 % des phrases valides avaient donc une ville non annotée (sans accent, avec une faute, espace au lieu d'un tiret), apprise comme « pas une ville ». | Annotation : 87,8 % → 100 % des phrases. Explique l'erreur principale de spaCy (« une seule entité extraite »). **Effet à mesurer après réentraînement.** |

**Le mode « fuzzy » reste désactivé par défaut.** `SpacyNERModel(..., fuzzy_cities=True)`
(idem pour CamemBERT) corrige aussi les fautes vers la ville connue la plus proche.
Mesuré : +1 point sur les villes connues, mais −13 à −28 points sur les villes inconnues
(« Vosves » devient « Voves »).

### Recommandation NLP

- **CamemBERT 1k + gazetteer** : meilleur choix global. C'est le seul modèle qui reste
  à ~99 % sur des villes jamais vues, pour ~65 ms par phrase sur CPU.
- **spaCy blank 1k + gazetteer** : ~25× plus rapide et excellent sur les gares connues,
  mais perd ~15 points sur les villes inconnues (à re-mesurer après réentraînement).
- **Baseline** : parfait si toutes les gares sont dans le gazetteer (monde fermé),
  inutilisable sinon. C'est le principal argument en faveur des modèles NER.

### Pistes pour progresser encore

1. **Réentraîner spaCy et CamemBERT** avec l'alignement corrigé (`make train`). C'est
   le gain attendu le plus important pour spaCy sur les villes inconnues.
2. **Rendre le dataset plus exigeant** : réserver une partie des gares au test
   (aujourd'hui, toutes les villes du test sont dans le train), ajouter des homonymes
   prénom/ville (Albert, Florence) et des phrases à trois villes ou plus.
3. **Élargir le gazetteer** à la liste complète des gares SNCF via l'API, pour que le
   baseline et le post-traitement couvrent toutes les gares réelles.
4. **Erreurs restantes de CamemBERT** sur les villes inconnues : surtout des frontières
   d'entités sur les noms composés (« Saint-Jean-de-Maurienne Arvan »). Davantage de
   noms longs dans le dataset aideraient.

<details>
<summary>Historique : résultats annoncés précédemment (ancien évaluateur, modèles non disponibles)</summary>

Ces chiffres ont été mesurés avec l'ancien évaluateur, qui comptait faux toutes les
phrases invalides. Les modèles correspondants (CamemBERT 3k/5k/full, variations de LR,
de batch et de scheduler) ne sont plus disponibles.

| Modèle | Accuracy | F1 Score | Inférence |
|--------|----------|----------|-----------|
| CamemBERT 3k | 75.42% | 98.03% | 108 ms |
| CamemBERT lr=1e-5 | 75.07% | 98.00% | 78 ms |
| spaCy blank 5k | 69.38% | 95.98% | 9.5 ms |
| spaCy blank 3k | 70.00% | 95.86% | 5.2 ms |
| Baseline (ancienne version) | 67.13% | 88.41% | 17.38 ms |

</details>

## Module Pathfinding

### Algorithmes disponibles

Le module `src/pathfinder/dijkstra.py` utilise **NetworkX** pour calculer les itinéraires :

| Algorithme | Méthode | Description |
|-----------|---------|-------------|
| **Dijkstra** | `find_shortest_path()` | Plus court chemin par durée |
| **A*** | `find_path_astar()` | Heuristique Haversine / vitesse TGV |
| **Min-changes** | `find_path_min_changes()` | Minimise les correspondances |
| **Alternatives** | `find_all_paths()` | K plus courts chemins |

### Fonctionnalités SNCF

- **Fuzzy matching** : Correspondance approximative des noms de villes (Levenshtein, seuil 0.8)
- **Gare la plus proche** : Pour les villes sans gare, trouve la gare la plus proche par distance Haversine
- **Vérification géographique** : Détection des villes hors France métropolitaine

### Gares disponibles

51 gares principales : Paris, Lyon, Marseille, Toulouse, Bordeaux, Lille, Nice, Nantes, Strasbourg, Montpellier, Rennes, Grenoble, Dijon, Tours, Reims, Avignon, Cannes, Monaco...

### Exemples de routes

| Trajet | Durée | Distance | Correspondances |
|--------|-------|----------|-----------------|
| Paris → Marseille | 2h48 | 666 km | 0 (TGV) |
| Lyon → Bordeaux | 3h22 | 657 km | 2 |
| Lille → Nice | 5h30 | 1100 km | 1 |

## Mode interactif

### Baseline
```bash
# Une phrase
python scripts/travel_resolver.py -m baseline "Je veux aller de Paris à Lyon"

# Interactif
python scripts/travel_resolver.py -m baseline --interactive
```

### spaCy
```bash
# Une requête
python scripts/travel_resolver.py "Je veux aller de Paris à Marseille"

# Mode interactif
python scripts/travel_resolver.py --interactive
```

### CamemBERT
```bash
# Une requête
python scripts/travel_resolver.py -m camembert "Je veux aller de Paris à Lyon"

# Mode interactif
python scripts/travel_resolver.py -m camembert --interactive
```

## Module vocal (Speech-to-Text / Text-to-Speech)

Le script `scripts/travel_resolver_voice.py` ajoute une entrée vocale (**Whisper**)
et une réponse vocale (**pyttsx3**) au-dessus de la pipeline `TravelResolver`.

```bash
# Dépendances système (ffmpeg, portaudio, espeak-ng) - Linux/macOS
make install-speech

# Mode vocal interactif
python scripts/travel_resolver_voice.py

# Saisie texte + réponse vocale
python scripts/travel_resolver_voice.py --text

# Options
#   --nlp-model, -m   : spacy (défaut) ou baseline
#   --optimize, -o    : duration (défaut) ou distance
#   --model-size      : taille Whisper (tiny, base, small, medium)
```

Raccourcis Makefile : `make voice`, `make voice-text`, `make voice-small`,
`make test-stt`, `make test-tts`, `make demo-voice`.

## Docker

```bash
make docker-build   # construit les conteneurs (docker/)
make docker-up      # Neo4j :7474, Jupyter :8888, MLflow :5000
make docker-down
```

## État actuel et limitations connues

Vérifié le 15/10/2026 (Python 3.13, spaCy 3.8, transformers 5.1, torch 2.10, CPU) :

| Élément | État |
|---------|------|
| `travel_resolver.py` (baseline, spaCy 1k, CamemBERT 1k) | ✅ 5/5 phrases de démo résolues |
| `travel_resolver.py -a` (routes alternatives) | ✅ |
| `evaluate_models.py`, `evaluate_baseline.py` | ✅ |
| `pytest tests/` | ✅ 22 tests |
| `travel_resolver_voice.py --help` | ✅ (STT/TTS non testés) |

- **Test set en monde fermé** : toutes les villes du test apparaissent dans le train.
  Le test « villes inconnues » de `evaluate_models.py` compense en partie.
- **Sans dataset** (clone frais), le gazetteer ne contient que les 51 gares principales.
  Générer le dataset (`python scripts/generate_dataset_simple.py`) avant d'évaluer.
- **Modèles périmés** : `ner_blank_3000`, `ner_blank_5000`, `ner_sm_full` et `ner_md_full`
  ont été entraînés sur un ancien dataset et sont à réentraîner.
- **Nombre de correspondances** : `Changes` compte les changements de *type de
  ligne* (TGV → Intercités…), pas le nombre de trains. Un trajet 100 % TGV en
  4 étapes affiche donc `Changes: 0`.
- **Makefile** : les cibles pointent désormais vers les scripts existants. Non testé
  sous Windows, où `make` n'est pas installé par défaut.
