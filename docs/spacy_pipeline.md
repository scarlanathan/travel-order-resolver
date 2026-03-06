# Pipeline spaCy NER pour l'extraction Origine/Destination

## Architecture de la Pipeline

La pipeline spaCy utilisée pour l'extraction NER (Named Entity Recognition) comprend les composants suivants :

```
Input Text → Tokenizer → Tok2Vec → NER → Output Entities
```

### 1. Tokenizer

**Rôle** : Découpe le texte en tokens (mots, ponctuations).

```python
# Exemple
"Je vais de Paris à Lyon" → ["Je", "vais", "de", "Paris", "à", "Lyon"]
```

**Configuration** : Tokenizer français standard de spaCy avec règles pour :
- Contractions (l', d', qu')
- Tirets dans noms composés (Saint-Étienne)
- Apostrophes

### 2. Tok2Vec (Token to Vector)

**Rôle** : Convertit chaque token en vecteur numérique dense.

**Architecture utilisée** :

```
MultiHashEmbed → MaxoutWindowEncoder
```

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| width | 96 | Dimension des vecteurs |
| depth | 4 | Profondeur de l'encodeur |
| window_size | 1 | Fenêtre contextuelle |
| maxout_pieces | 3 | Pièces maxout pour non-linéarité |

**Embedding** (MultiHashEmbed) :
- `ORTH` : Forme du mot (5000 lignes)
- `SHAPE` : Pattern du mot (2500 lignes)
- `PREFIX` : Préfixe (1000 lignes)
- `SUFFIX` : Suffixe (1000 lignes)

### 3. NER (Named Entity Recognition)

**Rôle** : Identifie et classifie les entités nommées.

**Architecture** : TransitionBasedParser avec état NER

| Paramètre | Valeur |
|-----------|--------|
| hidden_width | 64 |
| maxout_pieces | 2 |
| use_upper | true |

**Labels entraînés** :
- `ORIGIN` : Ville de départ
- `DESTINATION` : Ville d'arrivée

**Fonctionnement** :
Le modèle utilise un système de transitions (BIO tagging) :
- `B-ORIGIN` : Début d'une entité origine
- `I-ORIGIN` : Continuation d'une entité origine
- `B-DESTINATION` : Début d'une entité destination
- `I-DESTINATION` : Continuation d'une entité destination
- `O` : Token hors entité

## Comparaison des Modèles de Base

| Modèle | Vocabulaire | Vecteurs | Taille |
|--------|-------------|----------|--------|
| `spacy.blank("fr")` | Vide | Non | ~1 MB |
| `fr_core_news_sm` | 500k | Non | 16 MB |
| `fr_core_news_md` | 500k | 50k vecteurs | 46 MB |
| `fr_core_news_lg` | 500k | 500k vecteurs | 560 MB |

### Résultats expérimentaux

#### Impact de la taille du dataset (modèle blank)

| Modèle | Samples | Accuracy | F1 Score | Origin F1 | Dest F1 | Inférence |
|--------|---------|----------|----------|-----------|---------|-----------|
| blank_5000 | 4945 | 69.38% | **95.98%** | 96.27% | 95.68% | 9.5 ms |
| blank_full | 8352 | 69.29% | 95.92% | 96.04% | 95.80% | 5.4 ms |
| blank_3000 | 2965 | **70.00%** | 95.86% | 96.33% | 95.38% | 5.2 ms |
| blank_1000 | 989 | 67.72% | 94.92% | 96.04% | 93.81% | 5.3 ms |

#### Impact du modèle de base (dataset complet ~8350 samples)

| Modèle | Base | Accuracy | F1 Score | Origin F1 | Dest F1 | Inférence |
|--------|------|----------|----------|-----------|---------|-----------|
| blank_full | blank | **69.29%** | **95.92%** | 96.04% | 95.80% | 5.4 ms |
| lg_full | fr_core_news_lg | 55.03% | 90.26% | 90.26% | 90.26% | 8.5 ms |
| md_full | fr_core_news_md | 54.94% | 90.55% | 90.91% | 90.19% | 13.1 ms |
| sm_full | fr_core_news_sm | 52.32% | 86.93% | 89.33% | 84.53% | 11.9 ms |

**Conclusions** :
- Le modèle **blank avec 3000-5000 échantillons** offre le meilleur compromis performance/temps
- Les modèles pré-entraînés (sm, md, lg) **sous-performent** les modèles blank de ~15% en accuracy et ~5% en F1. Les entités NER pré-existantes (PER, LOC, ORG) interfèrent avec l'apprentissage de ORIGIN/DESTINATION
- Au-delà de 3000 samples, les gains sont marginaux (plateau de performance)

## Entraînement

### Processus d'entraînement

```python
# 1. Chargement des données
train_data = [
    ("Je vais de Paris à Lyon", {"entities": [(11, 16, "ORIGIN"), (19, 23, "DESTINATION")]}),
    ...
]

# 2. Création du modèle
nlp = spacy.blank("fr")
ner = nlp.add_pipe("ner")
ner.add_label("ORIGIN")
ner.add_label("DESTINATION")

# 3. Entraînement
for epoch in range(n_iter):
    for batch in minibatch(train_data):
        examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch]
        nlp.update(examples, drop=0.35)
```

### Hyperparamètres

| Paramètre | Valeur | Impact |
|-----------|--------|--------|
| n_iter | 20-30 | Plus = meilleure convergence |
| dropout | 0.35 | Régularisation |
| batch_size | 4→32 | Compounding pour stabilité |
| learn_rate | 0.001 | Vitesse d'apprentissage |

## Inférence

```python
from src.nlp.models.spacy_ner import SpacyNERModel

# Chargement du modèle (recommandé : blank_3000 ou blank_5000)
model = SpacyNERModel("models/spacy_ner/ner_blank_3000")

# Prédiction
result = model.predict("je voudrais aller de marseille a nice")
print(f"Origine: {result.origin}")        # marseille
print(f"Destination: {result.destination}") # nice
```

## Limitations

1. **Entités non vues** : Difficulté avec les villes non présentes dans le dataset d'entraînement
2. **Structures inhabituelles** : Performances réduites sur des formulations très différentes des templates
3. **Homonymes** : Confusion possible avec les villes ayant un nom commun (Albert, Orange)

## Fichiers

- `src/nlp/models/spacy_ner.py` : Module d'entraînement et d'inférence
- `scripts/train_spacy_models.py` : Script d'entraînement
- `models/spacy_ner/` : Modèles entraînés
- `outputs/evaluation/spacy_experiments.json` : Résultats des expériences
