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

| Modèle | Accuracy | F1 Score | Temps inférence | Description |
|--------|----------|----------|-----------------|-------------|
| **CamemBERT 3k** | **75.42%** | **98.03%** | 108 ms | Transformers fine-tuné (meilleure précision) |
| spaCy blank 5k | 69.38% | 95.98% | 9.5 ms | Meilleur modèle spaCy |
| spaCy blank 3k | 70.00% | 95.86% | 5.2 ms | Excellent compromis performance/vitesse |
| Baseline | 67.13% | 88.41% | 17.38 ms | Règles + prépositions + gazetteer |

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
git clone <repository-url>
cd T-AIA-911-PAR_13

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les modèles spaCy français (optionnel, pour les modèles sm/md/lg)
python -m spacy download fr_core_news_sm
python -m spacy download fr_core_news_md
python -m spacy download fr_core_news_lg
```

## Structure du projet

```
T-AIA-911-PAR_13/
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
│   ├── evaluate_baseline.py         # Évaluation baseline
│   ├── train_spacy_models.py        # Entraînement spaCy
│   ├── train_camembert.py           # Entraînement CamemBERT
│   ├── reevaluate_camembert.py      # Réévaluation de tous les modèles CamemBERT
│   ├── analyze_camembert_errors.py  # Analyse des erreurs de prédiction
│   └── travel_resolver.py           # Pipeline complète NLP + Pathfinding
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── sncf_client.py       # Client API SNCF (gares, connexions)
│   ├── pathfinder/
│   │   ├── __init__.py
│   │   └── dijkstra.py          # Algorithme Dijkstra (NetworkX)
│   ├── data_generation/
│   │   ├── sentence_generator.py
│   │   └── templates.py
│   └── nlp/
│       ├── evaluator.py         # Classe d'évaluation
│       └── models/
│           ├── baseline_model.py    # Modèle baseline
│           ├── spacy_ner.py         # Modèle spaCy
│           └── camembert_ner.py     # Modèle CamemBERT
├── docker/
│   ├── Dockerfile.nlp
│   └── docker-compose.yml
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
# Évaluer le modèle baseline (règles + prépositions)
python scripts/evaluate_baseline.py

# Résultats dans outputs/evaluation/baseline_test_report.csv
```

### 3. Modèle spaCy NER

```bash
# Entraîner tous les modèles spaCy (différentes tailles et bases)
python scripts/train_spacy_models.py --all-experiments --n-iter 30

# Entraîner un modèle spécifique
python scripts/train_spacy_models.py --size 1000 --n-iter 30

# Options disponibles:
#   --all-experiments : Entraîner tous les modèles (blank_1k, blank_3k, blank_5k, blank_full, sm_full, md_full, lg_full)
#   --size            : Nombre de samples (1000, 3000, 5000, ou 'full')
#   --n-iter          : Nombre d'itérations d'entraînement (défaut: 30)
#   --base            : Modèle de base ('blank', 'sm', 'md', 'lg')
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

# Réévaluer tous les modèles CamemBERT (après un fix par exemple)
python scripts/reevaluate_camembert.py

# Analyser les erreurs de prédiction sur un modèle
python scripts/analyze_camembert_errors.py --model-path models/camembert_ner/camembert_ner_1000 --num-samples 30
```

### 5. Travel Resolver (Pipeline complète)

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
```

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

### 6. Utilisation des modèles en Python

```python
# NLP - Extraction origine/destination (spaCy)
from src.nlp.models.spacy_ner import SpacyNERModel
model = SpacyNERModel("models/spacy_ner/ner_blank_3000")
result = model.predict("Un billet Marseille Bordeaux")
print(f"Origine: {result.origin}, Destination: {result.destination}")

# NLP - Extraction avec CamemBERT
from src.nlp.models.camembert_ner import CamembertNERModel
model = CamembertNERModel("models/camembert_ner/camembert_ner_3000")
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
resolver = TravelResolver(nlp_model="camembert")
route = resolver.resolve("Je veux aller de Lyon à Nice")
print(route)
```

## Résultats des modèles

### Emplacement des résultats

| Type | Emplacement |
|------|-------------|
| Métriques Baseline | `outputs/evaluation/baseline_test_report.csv` |
| Métriques spaCy | `outputs/evaluation/spacy_experiments.json` |
| Métriques CamemBERT | `outputs/evaluation/camembert_experiments.json` |
| Analyse qualitative | `docs/critique_qualitative.md` |

### Résultats comparatifs (meilleurs modèles)

| Modèle | Accuracy | F1 Score | Origin F1 | Dest F1 | Inférence |
|--------|----------|----------|-----------|---------|-----------|
| **CamemBERT 3k** | **75.42%** | **98.03%** | 97.89% | 98.17% | 108 ms |
| CamemBERT lr=1e-5 | 75.07% | 98.00% | 97.94% | 98.06% | 78 ms |
| spaCy blank 5k | 69.38% | 95.98% | 96.27% | 95.68% | 9.5 ms |
| spaCy blank 3k | 70.00% | 95.86% | 96.33% | 95.38% | 5.2 ms |
| Baseline | 67.13% | 88.41% | - | - | 17.38 ms |

### Tous les modèles disponibles

#### Modèles spaCy (7 variantes)

| Modèle | Base | Samples | Accuracy | F1 Score | Origin F1 | Dest F1 | Inférence |
|--------|------|---------|----------|----------|-----------|---------|-----------|
| ner_blank_5000 | blank | 4945 | 69.38% | **95.98%** | 96.27% | 95.68% | 9.5 ms |
| ner_blank_full | blank | 8352 | 69.29% | 95.92% | 96.04% | 95.80% | 5.4 ms |
| ner_blank_3000 | blank | 2965 | 70.00% | 95.86% | 96.33% | 95.38% | 5.2 ms |
| ner_blank_1000 | blank | 989 | 67.72% | 94.92% | 96.04% | 93.81% | 5.3 ms |
| ner_md_full | fr_core_news_md | 8352 | 54.94% | 90.55% | 90.91% | 90.19% | 13.1 ms |
| ner_lg_full | fr_core_news_lg | 8352 | 55.03% | 90.26% | 90.26% | 90.26% | 8.5 ms |
| ner_sm_full | fr_core_news_sm | 8352 | 52.32% | 86.93% | 89.33% | 84.53% | 11.9 ms |

#### Modèles CamemBERT (12 variantes)

**Impact de la taille du dataset :**

| Modèle | Samples | Accuracy | F1 Score | Origin F1 | Dest F1 | Inférence |
|--------|---------|----------|----------|-----------|---------|-----------|
| **camembert_ner_3000** | 3000 | **75.42%** | **98.03%** | 97.89% | 98.17% | 108 ms |
| camembert_ner_5000 | 5000 | 72.88% | 97.34% | 97.20% | 97.49% | 112 ms |
| camembert_ner_full | 8462 | 72.00% | 97.06% | 97.09% | 97.03% | 96 ms |
| camembert_ner_1000 | 1000 | 71.48% | 96.53% | 96.45% | 96.62% | 161 ms |

**Impact du learning rate (dataset full) :**

| Modèle | LR | Accuracy | F1 Score | Inférence |
|--------|-----|----------|----------|-----------|
| camembert_lr1e-05 | 1e-5 | **75.07%** | **98.00%** | 78 ms |
| camembert_lr3e-05 | 3e-5 | 72.18% | 96.88% | 91 ms |
| camembert_lr5e-05 | 5e-5 | 71.92% | 97.03% | 80 ms |

**Impact du batch size (dataset full) :**

| Modèle | Batch | Accuracy | F1 Score | Inférence |
|--------|-------|----------|----------|-----------|
| camembert_bs32 | 32 | **72.35%** | **97.14%** | 80 ms |
| camembert_bs16 | 16 | 71.83% | 96.91% | 81 ms |
| camembert_bs8 | 8 | 70.87% | 96.65% | 78 ms |

**Impact du scheduler (dataset full) :**

| Modèle | Scheduler | Accuracy | F1 Score | Inférence |
|--------|-----------|----------|----------|-----------|
| camembert_sched_cosine | cosine | 71.74% | 96.88% | 134 ms |
| camembert_sched_linear | linear | 71.13% | 96.65% | 135 ms |

### Recommandation NLP

**Deux modèles recommandés selon votre cas d'usage :**

**CamemBERT 3k** - Pour la meilleure précision :
- Performance exceptionnelle (98.03% F1, 75.42% accuracy)
- Extraction très précise des entités (97.89% origin, 98.17% destination)
- Modèle transformers robuste
- Temps d'inférence : ~108 ms (acceptable pour API)

**spaCy blank 3k** - Pour la rapidité :
- Excellente performance (95.86% F1, 70.00% accuracy)
- Temps d'inférence ultra-rapide (5.2 ms, **~20x plus rapide**)
- Modèle léger (~10 MB vs ~400 MB pour CamemBERT)
- Idéal pour production à fort volume

### Observations clés

1. **3000 samples est le sweet spot** : Pour spaCy et CamemBERT, 3000 samples donne les meilleurs résultats. Au-delà, les gains sont marginaux voire négatifs.

2. **Les modèles blank spaCy battent les pré-entraînés** : Les modèles sm/md/lg (52-55% accuracy) sous-performent largement les modèles blank (67-70% accuracy). Les entités NER pré-entraînées interfèrent avec l'apprentissage de ORIGIN/DESTINATION.

3. **Un LR plus faible améliore CamemBERT** : lr=1e-5 (98.00% F1) surpasse lr=5e-5 (97.03% F1) sur le dataset complet.

4. **Le post-processing est crucial** : La méthode `_clean_entity()` dans CamemBERT nettoie les espaces et ponctuation parasites, améliorant significativement les métriques.

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
