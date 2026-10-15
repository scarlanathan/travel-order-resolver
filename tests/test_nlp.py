"""
Unit tests for the gazetteer, the baseline model and the evaluator.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.evaluator import NLPEvaluator
from src.nlp.gazetteer import CityGazetteer, find_entity_span, normalize
from src.nlp.models.baseline_model import BaselineModel

CITIES = [
    "Paris", "Lyon", "Marseille", "Nice", "Nice Saint-Augustin", "Marigny",
    "Le Havre", "L'Isle-Jourdain", "Musée d'Orsay", "Saint-Étienne",
    "Vougeot - Gilly-lès-Cîteaux", "Toulouse", "Chambéry",
]


@pytest.fixture(scope="module")
def gazetteer():
    return CityGazetteer(cities=CITIES, train_path="does_not_exist.csv")


@pytest.fixture(scope="module")
def baseline(gazetteer):
    return BaselineModel(gazetteer=gazetteer)


def test_normalize():
    assert normalize("Saint-Étienne") == "saint etienne"
    assert normalize("L'Isle-Jourdain") == "l isle jourdain"
    assert normalize("Vougeot - Gilly-lès-Cîteaux") == "vougeot gilly les citeaux"


@pytest.mark.parametrize("text, entity, expected", [
    ("Je vais à saint etienne demain", "Saint-Étienne", "saint etienne"),
    ("Un billet pour Maringy", "Marigny", "Maringy"),
    ("De Paris à Lyon", "Lyon", "Lyon"),
])
def test_find_entity_span(text, entity, expected):
    span = find_entity_span(text, entity)
    assert span is not None
    assert text[span[0]:span[1]] == expected


def test_find_entity_span_exclude():
    text = "Aller de Paris à Paris"
    first = find_entity_span(text, "Paris")
    second = find_entity_span(text, "Paris", exclude=[first])
    assert first != second


def test_find_mentions_longest_match(gazetteer):
    mentions = gazetteer.find_mentions("Je pars de Nice Saint-Augustin")
    assert [m.city for m in mentions] == ["Nice Saint-Augustin"]


def test_find_mentions_misspelled(gazetteer):
    mentions = gazetteer.find_mentions("Un billet de Le Hvre à Paris")
    assert [m.city for m in mentions] == ["Le Havre", "Paris"]


@pytest.mark.parametrize("sentence, origin, destination", [
    ("Je voudrais aller de Paris à Lyon", "Paris", "Lyon"),
    ("Vers Marseille depuis Toulouse.", "Toulouse", "Marseille"),
    ("Comment me rendre à Chambéry depuis Paris ?", "Paris", "Chambéry"),
    ("Destination Nice, départ Lyon", "Lyon", "Nice"),
    ("En dirrection de Lyon en provenance de Paris.", "Paris", "Lyon"),
    ("Un billet Lyon Marseille", "Lyon", "Marseille"),
    ("trajet en provenance de lyon à destination de saint etienne.", "Lyon", "Saint-Étienne"),
    ("Est-ce qu'il y a des trains entre Toulouse et Paris ?", "Toulouse", "Paris"),
])
def test_baseline_valid(baseline, sentence, origin, destination):
    result = baseline.predict(sentence)
    assert result.is_valid
    assert (result.origin, result.destination) == (origin, destination)


@pytest.mark.parametrize("sentence", [
    "Quel temps fait-il à Paris ?",
    "Bonjour, comment allez-vous ?",
])
def test_baseline_invalid(baseline, sentence):
    result = baseline.predict(sentence)
    assert not result.is_valid
    assert result.origin is None and result.destination is None


def test_refine_extends_truncated_span(gazetteer):
    assert gazetteer.refine("Je veux aller à Musée d'Orsay depuis Paris", "Paris", "Musée d'") \
        == ("Paris", "Musée d'Orsay")


def test_refine_fills_missing_entity(gazetteer):
    assert gazetteer.refine("Je veux aller à Lyon depuis Paris", None, "Lyon") == ("Paris", "Lyon")


def test_refine_keeps_unknown_city(gazetteer):
    # "Vosves" is not in the gazetteer: the NER prediction is kept
    assert gazetteer.refine("Vers Vosves depuis Paris", "Paris", "Vosves") == ("Paris", "Vosves")


def test_refine_rejects_single_city(gazetteer):
    assert gazetteer.refine("Quel temps fait-il à Lyon ?", None, "Lyon") == (None, None)


def test_evaluator_counts_rejected_invalid_as_correct(baseline):
    data = [{"sentence": "Quel temps fait-il à Paris ?", "origin": "INVALID", "destination": "INVALID"}]
    result = NLPEvaluator(verbose=False).evaluate(baseline, data)
    assert result.accuracy == 1.0
