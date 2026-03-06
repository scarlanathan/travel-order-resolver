"""
Sentence Templates for French Train Travel Orders

This module contains ~50+ diverse grammatical structures for generating
training data for the NLP model.

Uses SNCF API data for city names to ensure consistency with pathfinding.
"""

from typing import List, Dict, Optional


# Simple direct order templates
SIMPLE_TEMPLATES = [
    "Je voudrais un billet {origin} {destination}.",
    "Je veux aller à {destination} depuis {origin}.",
    "Un billet pour {destination} au départ de {origin}.",
    "Donnez-moi un billet {origin} {destination}.",
    "Je prends le train de {origin} à {destination}.",
    "Réservez-moi {origin} {destination}.",
    "Je pars de {origin} pour {destination}.",
    "De {origin} à {destination}, s'il vous plaît.",
]

# Polite/formal templates
POLITE_TEMPLATES = [
    "Je souhaiterais me rendre à {destination} depuis {origin}.",
    "Je voudrais me rendre à {destination} en partant de {origin}.",
    "Pourriez-vous me réserver un billet de {origin} à {destination} ?",
    "Auriez-vous un train de {origin} vers {destination} ?",
    "Je souhaite voyager de {origin} à {destination}.",
    "J'aimerais réserver un trajet {origin} {destination}.",
    "Serait-il possible d'avoir un billet pour {destination} depuis {origin} ?",
]

# Question templates
QUESTION_TEMPLATES = [
    "Comment me rendre à {destination} depuis {origin} ?",
    "Comment aller à {destination} en partant de {origin} ?",
    "Quel train prendre de {origin} à {destination} ?",
    "Y a-t-il un train de {origin} vers {destination} ?",
    "À quelle heure y a-t-il des trains vers {destination} en partance de {origin} ?",
    "Quels sont les horaires de train de {origin} à {destination} ?",
    "Comment puis-je aller de {origin} à {destination} ?",
    "Est-ce qu'il y a des trains entre {origin} et {destination} ?",
]

# Time-specific templates
TIME_TEMPLATES = [
    "Je veux partir de {origin} pour {destination} demain matin.",
    "Un train de {origin} à {destination} cet après-midi.",
    "Je dois aller de {origin} à {destination} ce soir.",
    "Quel est le prochain train de {origin} vers {destination} ?",
    "Y a-t-il un train tôt le matin de {origin} à {destination} ?",
    "Je cherche un train de {origin} à {destination} en fin de journée.",
]

# Complex with additional information
COMPLEX_TEMPLATES = [
    "Je dois me rendre à {destination} depuis {origin} pour le travail.",
    "Je voyage de {origin} à {destination} pour voir ma famille.",
    "Avec mes bagages, je vais de {origin} à {destination}.",
    "Je pars en vacances de {origin} vers {destination}.",
    "Pour mon rendez-vous, je dois aller de {origin} à {destination}.",
]

# Templates with person names (tricky cases)
NAMES_TEMPLATES = [
    "Je veux aller à {destination} voir mon ami {name} en partant de {origin}.",
    "Avec mes amis {name1} et {name2}, je voudrais aller de {origin} à {destination}.",
    "Mon ami {name} et moi voulons aller de {origin} à {destination}.",
    "Je retrouve {name} à {destination}, je pars de {origin}.",
]

# Reverse order templates (destination before origin)
REVERSE_TEMPLATES = [
    "Pour aller à {destination}, je pars de {origin}.",
    "Direction {destination} au départ de {origin}.",
    "Vers {destination} depuis {origin}.",
    "Destination {destination}, départ {origin}.",
    "En direction de {destination} en provenance de {origin}.",
]

# Preposition variations
PREPOSITION_TEMPLATES = [
    "Train partant de {origin} allant à {destination}.",
    "Trajet en provenance de {origin} à destination de {destination}.",
    "Départ depuis {origin} arrivée vers {destination}.",
    "Voyage à partir de {origin} jusqu'à {destination}.",
]

# Informal/spoken templates
INFORMAL_TEMPLATES = [
    "Faut que j'aille de {origin} à {destination}.",
    "J'dois aller à {destination} depuis {origin}.",
    "Besoin d'un billet {origin} {destination}.",
    "Faut que je parte de {origin} pour {destination}.",
]

# SMS/text message templates (informal, abbreviated)
SMS_TEMPLATES = [
    "jsuis a {origin} jveux aller {destination}",
    "slt billet {origin} {destination} svp",
    "go {origin} {destination}",
    "jdois aller {destination} depuis {origin}",
    "train {origin} {destination} stp",
    "cc ya un train {origin} {destination} ?",
    "bjr je cherche {origin} {destination}",
    "pk {origin} a {destination} cbn de tps ?",
    "wsh un billet {origin} {destination}",
    "hey billet {origin} {destination} plz",
]

# Templates with intermediate stops (bonus)
INTERMEDIATE_TEMPLATES = [
    "Je voudrais aller de {origin} à {destination} en passant par {intermediate}.",
    "De {origin} à {destination} avec arrêt à {intermediate}.",
    "Trajet {origin} {destination} via {intermediate}.",
]

# Ambiguous OOD templates (close to domain but not travel orders)
AMBIGUOUS_TEMPLATES = [
    "J'ai visité {city} hier c'était super.",
    "Mon ami habite à {city} depuis longtemps.",
    "Le train de {city} était en retard ce matin.",
    "J'ai perdu mes bagages à la gare de {city}.",
    "Il fait très beau à {city} en ce moment.",
    "La gare de {city} est en travaux actuellement.",
    "Je me souviens de mon voyage à {city} l'été dernier.",
    "Est-ce que {city} est une belle ville ?",
]

# All templates combined
ALL_TEMPLATES = (
    SIMPLE_TEMPLATES +
    POLITE_TEMPLATES +
    QUESTION_TEMPLATES +
    TIME_TEMPLATES +
    COMPLEX_TEMPLATES +
    NAMES_TEMPLATES +
    REVERSE_TEMPLATES +
    PREPOSITION_TEMPLATES +
    INFORMAL_TEMPLATES +
    SMS_TEMPLATES
)

# INVALID SENTENCE TEMPLATES

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
]

# FRENCH CITIES & STATIONS

MAJOR_CITIES = [
    "Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux",
    "Lille", "Nice", "Nantes", "Strasbourg", "Montpellier",
    "Rennes", "Reims", "Tours", "Saint-Étienne", "Le Havre",
    "Grenoble", "Dijon", "Angers", "Nîmes", "Clermont-Ferrand",
    "Aix-en-Provence", "Brest", "Le Mans", "Amiens", "Limoges",
    "Perpignan", "Metz", "Besançon", "Orléans", "Rouen",
    "Mulhouse", "Caen", "Nancy", "Argenteuil", "Montreuil",
]

# Tricky city names (homonyms, common words, person names)
TRICKY_CITIES = [
    "Albert", "Paris", "Florence", "Lourdes", "Orange",
    "Cognac", "Port-Boulet", "Bourg-en-Bresse", "Château-Thierry",
]

# Common French first names (for tricky templates)
FRENCH_NAMES = [
    "Albert", "Marie", "Pierre", "Florence", "Jacques",
    "Sophie", "Louis", "Julie", "Paul", "Emma",
]

# All cities combined
ALL_CITIES = MAJOR_CITIES + TRICKY_CITIES


# HELPER FUNCTIONS

_sncf_cities_cache: Optional[List[str]] = None


def get_sncf_cities() -> List[str]:
    """Get city names from SNCF API data."""
    global _sncf_cities_cache

    if _sncf_cities_cache is not None:
        return _sncf_cities_cache

    try:
        from src.api.sncf_client import SNCFClient
        client = SNCFClient()
        client.load_data(use_api=True)  # Use online API
        stations = client.get_all_stations()
        _sncf_cities_cache = list(set(s.city for s in stations))
        return _sncf_cities_cache
    except Exception:
        return ALL_CITIES


def get_all_templates() -> List[str]:
    """Return all valid templates."""
    return ALL_TEMPLATES


def get_invalid_templates() -> List[str]:
    """Return all invalid templates (static + ambiguous)."""
    return INVALID_TEMPLATES + AMBIGUOUS_TEMPLATES


def get_cities(use_sncf: bool = True) -> List[str]:
    """
    Return all city names.

    Args:
        use_sncf: If True, use SNCF API data. If False, use static list.

    Returns:
        List of city names
    """
    if use_sncf:
        sncf_cities = get_sncf_cities()
        if sncf_cities:
            return sncf_cities
    return ALL_CITIES


def get_names() -> List[str]:
    """Return all person names."""
    return FRENCH_NAMES


def get_template_categories() -> Dict[str, List[str]]:
    """Return templates organized by category."""
    return {
        "simple": SIMPLE_TEMPLATES,
        "polite": POLITE_TEMPLATES,
        "question": QUESTION_TEMPLATES,
        "time": TIME_TEMPLATES,
        "complex": COMPLEX_TEMPLATES,
        "names": NAMES_TEMPLATES,
        "reverse": REVERSE_TEMPLATES,
        "preposition": PREPOSITION_TEMPLATES,
        "informal": INFORMAL_TEMPLATES,
        "sms": SMS_TEMPLATES,
        "intermediate": INTERMEDIATE_TEMPLATES,
        "invalid": INVALID_TEMPLATES,
        "ambiguous": AMBIGUOUS_TEMPLATES,
    }
