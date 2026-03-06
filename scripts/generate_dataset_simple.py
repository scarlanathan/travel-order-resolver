#!/usr/bin/env python
"""
Simple dataset generator for NER model.
Uses SNCF API data for city names to ensure consistency with pathfinding.
"""

import csv
import random
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# TEMPLATES

SIMPLE_TEMPLATES = [
    "Je voudrais un billet {origin} {destination}.",
    "Je veux aller à {destination} depuis {origin}.",
    "Un billet pour {destination} au départ de {origin}.",
    "Donnez-moi un billet {origin} {destination}.",
    "Je prends le train de {origin} à {destination}.",
    "Réservez-moi {origin} {destination}.",
    "Je pars de {origin} pour {destination}.",
    "De {origin} à {destination}, s'il vous plaît.",
    "Un aller simple {origin} {destination}.",
    "Un aller-retour de {origin} à {destination}.",
    "Billet {origin} {destination} pour une personne.",
    "Je réserve un trajet de {origin} à {destination}.",
    "Voyage de {origin} à {destination}.",
    "Train {origin} {destination}.",
    "Je voudrais partir de {origin} et arriver à {destination}.",
    "Trajet de {origin} vers {destination}.",
]

POLITE_TEMPLATES = [
    "Je souhaiterais me rendre à {destination} depuis {origin}.",
    "Je voudrais me rendre à {destination} en partant de {origin}.",
    "Pourriez-vous me réserver un billet de {origin} à {destination} ?",
    "Auriez-vous un train de {origin} vers {destination} ?",
    "Je souhaite voyager de {origin} à {destination}.",
    "J'aimerais réserver un trajet {origin} {destination}.",
    "Serait-il possible d'avoir un billet pour {destination} depuis {origin} ?",
    "Pourriez-vous m'indiquer un train de {origin} à {destination} ?",
    "Je souhaiterais réserver une place de {origin} à {destination}.",
    "Auriez-vous l'amabilité de me trouver un train de {origin} vers {destination} ?",
    "Je désirerais voyager de {origin} jusqu'à {destination}.",
    "Merci de me réserver un billet de {origin} à {destination}.",
    "S'il vous plaît, un train de {origin} à {destination}.",
    "Je voudrais savoir les horaires de {origin} à {destination}.",
    "Excusez-moi, je cherche un train de {origin} vers {destination}.",
]

QUESTION_TEMPLATES = [
    "Comment me rendre à {destination} depuis {origin} ?",
    "Comment aller à {destination} en partant de {origin} ?",
    "Quel train prendre de {origin} à {destination} ?",
    "Y a-t-il un train de {origin} vers {destination} ?",
    "À quelle heure y a-t-il des trains vers {destination} en partance de {origin} ?",
    "Quels sont les horaires de train de {origin} à {destination} ?",
    "Comment puis-je aller de {origin} à {destination} ?",
    "Est-ce qu'il y a des trains entre {origin} et {destination} ?",
    "Existe-t-il une liaison ferroviaire entre {origin} et {destination} ?",
    "Pouvez-vous me dire comment aller de {origin} à {destination} ?",
    "Quand part le prochain train de {origin} pour {destination} ?",
    "Quel est le meilleur itinéraire de {origin} à {destination} ?",
    "Est-il possible de voyager de {origin} à {destination} en train ?",
    "Combien de temps pour aller de {origin} à {destination} ?",
    "Y a-t-il des TGV de {origin} à {destination} ?",
    "Le train de {origin} va-t-il jusqu'à {destination} ?",
]

TIME_TEMPLATES = [
    "Je veux partir de {origin} pour {destination} demain matin.",
    "Un train de {origin} à {destination} cet après-midi.",
    "Je dois aller de {origin} à {destination} ce soir.",
    "Quel est le prochain train de {origin} vers {destination} ?",
    "Y a-t-il un train tôt le matin de {origin} à {destination} ?",
    "Je cherche un train de {origin} à {destination} en fin de journée.",
    "Je dois être à {destination} demain, je pars de {origin}.",
    "Train de nuit de {origin} à {destination} ?",
    "Je voudrais partir de {origin} vers {destination} à 8h.",
    "Un train du matin de {origin} pour {destination}.",
    "Départ de {origin} à midi pour {destination}.",
    "Je pars ce week-end de {origin} pour {destination}.",
    "Train de {origin} à {destination} vendredi soir.",
    "Je dois arriver à {destination} avant midi, départ de {origin}.",
    "Premier train de {origin} à {destination} le matin.",
    "Dernier train de {origin} vers {destination} ce soir.",
]

COMPLEX_TEMPLATES = [
    "Je dois me rendre à {destination} depuis {origin} pour le travail.",
    "Je voyage de {origin} à {destination} pour voir ma famille.",
    "Avec mes bagages, je vais de {origin} à {destination}.",
    "Je pars en vacances de {origin} vers {destination}.",
    "Pour mon rendez-vous, je dois aller de {origin} à {destination}.",
    "Je me déplace de {origin} à {destination} pour affaires.",
    "Pour les fêtes, je vais de {origin} à {destination}.",
    "En voyage d'affaires, je dois aller de {origin} à {destination}.",
    "Je rentre chez moi de {origin} à {destination}.",
    "Pour mon entretien, je dois être à {destination} en partant de {origin}.",
    "Je vais voir des amis à {destination} depuis {origin}.",
    "Pour le concert, je prends le train de {origin} à {destination}.",
    "Je dois assister à une réunion à {destination}, départ de {origin}.",
    "En déplacement professionnel de {origin} vers {destination}.",
    "Je me rends à un mariage à {destination} depuis {origin}.",
]

NAMES_TEMPLATES = [
    "Je veux aller à {destination} voir mon ami {name} en partant de {origin}.",
    "Avec mes amis {name1} et {name2}, je voudrais aller de {origin} à {destination}.",
    "Mon ami {name} et moi voulons aller de {origin} à {destination}.",
    "Je retrouve {name} à {destination}, je pars de {origin}.",
    "Je vais chercher {name} à {destination}, départ de {origin}.",
    "{name} m'attend à {destination}, je pars de {origin}.",
    "Je voyage avec {name} de {origin} à {destination}.",
    "Je rejoins {name} à {destination} en partant de {origin}.",
    "Mon collègue {name} et moi partons de {origin} pour {destination}.",
    "{name1} et {name2} veulent m'accompagner de {origin} à {destination}.",
    "Je vais rendre visite à {name} à {destination} depuis {origin}.",
    "Avec ma soeur {name}, nous allons de {origin} à {destination}.",
]

REVERSE_TEMPLATES = [
    "Pour aller à {destination}, je pars de {origin}.",
    "Direction {destination} au départ de {origin}.",
    "Vers {destination} depuis {origin}.",
    "Destination {destination}, départ {origin}.",
    "En direction de {destination} en provenance de {origin}.",
    "Pour {destination}, départ prévu de {origin}.",
    "À destination de {destination}, je pars de {origin}.",
    "Arrivée à {destination}, départ de {origin}.",
    "Cap sur {destination} depuis {origin}.",
    "Je vise {destination} au départ de {origin}.",
    "Ma destination est {destination}, mon point de départ est {origin}.",
    "Objectif {destination}, départ de {origin}.",
]

PREPOSITION_TEMPLATES = [
    "Train partant de {origin} allant à {destination}.",
    "Trajet en provenance de {origin} à destination de {destination}.",
    "Départ depuis {origin} arrivée vers {destination}.",
    "Voyage à partir de {origin} jusqu'à {destination}.",
    "En partance de {origin} pour {destination}.",
    "Au départ de {origin} en direction de {destination}.",
    "Depuis {origin} vers {destination}.",
    "De {origin} jusqu'à {destination}.",
    "En provenance de {origin}, à destination de {destination}.",
    "Partant de {origin}, arrivant à {destination}.",
    "Liaison {origin} - {destination}.",
    "Connexion de {origin} vers {destination}.",
]

INFORMAL_TEMPLATES = [
    "Faut que j'aille de {origin} à {destination}.",
    "J'dois aller à {destination} depuis {origin}.",
    "Besoin d'un billet {origin} {destination}.",
    "Faut que je parte de {origin} pour {destination}.",
    "Je file de {origin} à {destination}.",
    "J'me tire de {origin} pour {destination}.",
    "Hop, un train de {origin} à {destination}.",
    "Vite, {origin} {destination}.",
    "Je bouge de {origin} vers {destination}.",
    "Allez, de {origin} à {destination}.",
    "Go de {origin} à {destination}.",
    "Je trace de {origin} vers {destination}.",
    "Direct {origin} {destination}.",
    "Fastoche, de {origin} à {destination}.",
]

ALL_TEMPLATES = (
    SIMPLE_TEMPLATES +
    POLITE_TEMPLATES +
    QUESTION_TEMPLATES +
    TIME_TEMPLATES +
    COMPLEX_TEMPLATES +
    NAMES_TEMPLATES +
    REVERSE_TEMPLATES +
    PREPOSITION_TEMPLATES +
    INFORMAL_TEMPLATES
)

INVALID_TEMPLATES = [
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Quel temps fait-il à {city} ?",
    "Je voudrais un café s'il vous plaît.",
    "Où se trouve la gare ?",
    "Combien coûte un billet ?",
    "Quelle est l'heure ?",
    "Merci beaucoup pour votre aide.",
    "Au revoir et bonne journée.",
    "Je cherche un restaurant.",
    "Pouvez-vous m'aider ?",
    "C'est trop cher.",
    "Je ne comprends pas.",
    "Parlez-vous anglais ?",
    "J'ai perdu mon billet.",
    "Où sont les toilettes ?",
    "Le train est en retard.",
    "Je veux annuler ma réservation.",
    "À quelle heure ferme la gare ?",
    "Quel est le prix du billet ?",
    "Y a-t-il un distributeur de billets ?",
    "Où puis-je acheter un sandwich ?",
    "Je cherche la sortie.",
    "Le wifi fonctionne-t-il dans le train ?",
    "Combien de bagages puis-je emporter ?",
    "Où est le quai numéro 5 ?",
    "Je suis en retard.",
    "Mon train a été annulé.",
    "Y a-t-il une correspondance ?",
    "Je ne trouve pas ma place.",
    "Le contrôleur est-il passé ?",
    "C'est quoi ce bruit ?",
    "Il fait chaud dans ce wagon.",
    "Pouvez-vous baisser le son ?",
    "Je voudrais changer de place.",
    "À quelle heure arrive-t-on ?",
    "On est encore loin ?",
    "Je me suis trompé de train.",
    "Excusez-moi, cette place est-elle libre ?",
    "Bonjour, je cherche la voiture 12.",
    "J'aime voyager en train.",
    "{city} est une belle ville.",
    "Il pleut à {city} aujourd'hui.",
    "Je connais bien {city}.",
    "Mon ami habite à {city}.",
]


# CITIES

MAJOR_CITIES = [
    "Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux",
    "Lille", "Nice", "Nantes", "Strasbourg", "Montpellier",
    "Rennes", "Reims", "Tours", "Saint-Étienne", "Le Havre",
    "Grenoble", "Dijon", "Angers", "Nîmes", "Clermont-Ferrand",
    "Aix-en-Provence", "Brest", "Le Mans", "Amiens", "Limoges",
    "Perpignan", "Metz", "Besançon", "Orléans", "Rouen",
    "Mulhouse", "Caen", "Nancy", "Avignon", "Poitiers",
    "Toulon", "Cannes", "Antibes", "Saint-Nazaire", "Valence",
    "Chambéry", "Annecy", "Bayonne", "Pau", "Tarbes",
    "La Baule", "Vannes", "Quimper", "Lorient", "Saint-Brieuc",
    "Chartres", "Blois", "Bourges", "Vichy", "Aurillac",
    "Rodez", "Albi", "Carcassonne", "Béziers", "Sète",
    "Arles", "Hyères", "Fréjus", "Menton", "Monaco",
    "Colmar", "Épinal", "Troyes", "Châlons-en-Champagne", "Charleville-Mézières",
    "Dunkerque", "Calais", "Boulogne-sur-Mer", "Valenciennes", "Douai",
]

TRICKY_CITIES = [
    "Albert", "Orange", "Cognac", "Port-Boulet", "Bourg-en-Bresse",
    "Château-Thierry", "La Rochelle", "Saint-Malo", "Mont-de-Marsan",
    "Lourdes", "Vic-sur-Cère", "Dax", "Agen", "Auch",
    "Gap", "Briançon", "Digne-les-Bains", "Sisteron", "Manosque",
    "Salon-de-Provence", "Martigues", "Istres", "La Ciotat", "Cassis",
    "Florence", "Milan", "Genève", "Bruxelles", "Luxembourg",
]

ALL_CITIES = MAJOR_CITIES + TRICKY_CITIES


def get_sncf_cities():
    """Get city names from SNCF API data."""
    try:
        from src.api.sncf_client import SNCFClient
        client = SNCFClient()
        client.load_data(use_api=False)
        stations = client.get_all_stations()
        return list(set(s.city for s in stations))
    except ImportError:
        return ALL_CITIES


# Use SNCF cities by default
CITIES = get_sncf_cities()

FRENCH_NAMES = [
    "Albert", "Marie", "Pierre", "Florence", "Jacques",
    "Sophie", "Louis", "Julie", "Paul", "Emma",
    "Lucas", "Léa", "Hugo", "Chloé", "Thomas",
    "Camille", "Nathan", "Clara", "Antoine", "Sarah",
    "Mathis", "Inès", "Raphaël", "Manon", "Gabriel",
    "Charlotte", "Nicolas", "Alice", "Victor", "Louise",
]

# HELPER FUNCTIONS

def remove_accents(text):
    """Remove French accents."""
    accents = {
        'à': 'a', 'â': 'a', 'ä': 'a',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'î': 'i', 'ï': 'i',
        'ô': 'o', 'ö': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'ÿ': 'y', 'ç': 'c',
        'À': 'A', 'Â': 'A', 'Ä': 'A',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'Î': 'I', 'Ï': 'I',
        'Ô': 'O', 'Ö': 'O',
        'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'Ÿ': 'Y', 'Ç': 'C',
    }
    for accent, replacement in accents.items():
        text = text.replace(accent, replacement)
    return text


def add_misspellings(text):
    """Add random misspellings."""
    words = text.split()
    if len(words) < 3:
        return text

    idx = random.randint(1, len(words) - 1)
    word = words[idx]

    if len(word) > 3:
        # Swap two letters
        pos = random.randint(0, len(word) - 2)
        word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
        words[idx] = word

    return ' '.join(words)


def generate_valid_sentence(templates, cities, names):
    """Generate a valid travel order sentence."""
    template = random.choice(templates)

    origin = random.choice(cities)
    destination = random.choice([c for c in cities if c != origin])

    if "{name}" in template or "{name1}" in template:
        name = random.choice(names)
        name1 = random.choice(names)
        name2 = random.choice([n for n in names if n != name1])
        sentence = template.format(
            origin=origin,
            destination=destination,
            name=name,
            name1=name1,
            name2=name2,
        )
    else:
        sentence = template.format(origin=origin, destination=destination)

    return sentence, origin, destination


def generate_invalid_sentence(templates, cities):
    """Generate an invalid sentence."""
    template = random.choice(templates)

    if "{city}" in template:
        city = random.choice(cities)
        sentence = template.format(city=city)
    else:
        sentence = template

    return sentence, "INVALID", "INVALID"


def generate_dataset(num_sentences=2000, seed=42):
    """Generate complete dataset."""
    random.seed(seed)

    data = []
    sentence_id = 1

    # 85% valid, 15% invalid
    num_valid = int(num_sentences * 0.85)
    num_invalid = num_sentences - num_valid

    # Generate valid sentences with variations
    valid_count = 0
    while valid_count < num_valid:
        sentence, origin, destination = generate_valid_sentence(
            ALL_TEMPLATES, ALL_CITIES, FRENCH_NAMES
        )

        # Original
        data.append({
            'sentence_id': sentence_id,
            'sentence': sentence,
            'origin': origin,
            'destination': destination,
        })
        sentence_id += 1
        valid_count += 1

        if valid_count >= num_valid:
            break

        # Variation: lowercase (30% chance)
        if random.random() < 0.3:
            data.append({
                'sentence_id': sentence_id,
                'sentence': sentence.lower(),
                'origin': origin,
                'destination': destination,
            })
            sentence_id += 1
            valid_count += 1

            if valid_count >= num_valid:
                break

        # Variation: no accents (20% chance)
        if random.random() < 0.2:
            data.append({
                'sentence_id': sentence_id,
                'sentence': remove_accents(sentence),
                'origin': origin,
                'destination': destination,
            })
            sentence_id += 1
            valid_count += 1

            if valid_count >= num_valid:
                break

        # Variation: misspellings (10% chance)
        if random.random() < 0.1:
            data.append({
                'sentence_id': sentence_id,
                'sentence': add_misspellings(sentence),
                'origin': origin,
                'destination': destination,
            })
            sentence_id += 1
            valid_count += 1

    # Generate invalid sentences
    for _ in range(num_invalid):
        sentence, origin, destination = generate_invalid_sentence(
            INVALID_TEMPLATES, ALL_CITIES
        )
        data.append({
            'sentence_id': sentence_id,
            'sentence': sentence,
            'origin': origin,
            'destination': destination,
        })
        sentence_id += 1

    # Shuffle
    random.shuffle(data)

    # Reset IDs
    for i, item in enumerate(data):
        item['sentence_id'] = i + 1

    return data


def split_dataset(data, train_ratio=0.7, val_ratio=0.15):
    """Split dataset into train/val/test."""
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


def save_csv(data, filepath):
    """Save data to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sentence_id', 'sentence', 'origin', 'destination'])
        writer.writeheader()
        writer.writerows(data)

    print(f"Saved {len(data)} sentences to {filepath}")


def main():
    NUM_SENTENCES = 10000

    print("=" * 60)
    print("GENERATING TRAINING DATASET (10,000 sentences)")
    print("=" * 60)

    # Generate
    data = generate_dataset(num_sentences=NUM_SENTENCES)

    # Save full dataset
    save_csv(data, "data/datasets/full_dataset.csv")

    # Split
    train_data, val_data, test_data = split_dataset(data)

    save_csv(train_data, "data/training/train.csv")
    save_csv(val_data, "data/training/val.csv")
    save_csv(test_data, "data/training/test.csv")

    # Statistics
    valid_count = sum(1 for d in data if d['origin'] != 'INVALID')
    invalid_count = len(data) - valid_count

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total sentences:    {len(data)}")
    print(f"  - Valid orders:   {valid_count}")
    print(f"  - Invalid orders: {invalid_count}")
    print(f"\nSplit:")
    print(f"  - Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  - Val:   {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  - Test:  {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
    print("\nSample sentences:")
    print("-" * 60)
    for item in data[:10]:
        print(f"[{item['sentence_id']:4d}] {item['sentence'][:60]}...")
        print(f"       -> {item['origin']} -> {item['destination']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
