"""
City Gazetteer for Travel Order NLP

Shared tooling used by every model:
1. Accent/case/hyphen-insensitive normalization that keeps character offsets
2. City mention detection in a sentence (exact + fuzzy for misspellings)
3. Robust entity span alignment (used to annotate training data)
4. Origin/destination role assignment from French cue words
5. Post-processing of NER predictions (canonical names, missing entity
   completion, invalid request detection)
"""

import csv
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "training" / "train.csv"

WORD_RE = re.compile(r"\w+")

# Cue words (normalized) preceding a city mention
ORIGIN_CUES = [
    "en provenance de", "au depart de", "en partance de", "a partir de",
    "partant de", "partir de", "pars de", "part de", "parte de",
    "depart depuis", "depart de", "depart", "depuis", "de", "d", "du", "des",
    "entre", "provenance",
]
DESTINATION_CUES = [
    "a destination de", "en direction de", "direction de", "destination de",
    "arrivee vers", "arrivee a", "arrivee", "jusqu a", "jusque",
    "destination", "direction", "allant a", "vers", "pour", "a", "au", "aux",
    "et",
]
# Frequent words of travel requests, never fuzzy-matched to a city (completed
# by the template vocabulary learned from the training set)
COMMON_WORDS = set("""
je tu il elle on nous vous ils me te se moi mon ma mes un une des le la les du de
au aux a et ou en dans par pour vers sur sous avec sans depuis entre chez est sont
pars part parte partir partant aller vais va voudrais veux voulons souhaite
souhaiterais aimerais faut dois rendre arriver arrivee depart train trains billet
billets trajet voyage voyager prendre reserver reservez donnez besoin quel quelle
quels quelles comment combien prochain demain matin soir journee aujourd hui merci
bonjour possible serait auriez avez avoir allant destination direction provenance
partance gare heure horaire horaires famille ami amis amie mere pere frere soeur
""".split())
CUE_WORDS = sorted({w for cue in ORIGIN_CUES + DESTINATION_CUES for w in cue.split() if len(w) >= 5})


def normalize_char(c: str) -> str:
    """Normalize a single character (lowercase, no accent, separators -> space)."""
    if c in "-'’`_":
        return " "
    base = unicodedata.normalize("NFD", c)[0]
    return base.lower() if base.isalnum() else (" " if not base.strip() else base)


def normalize(text: str) -> str:
    """Normalize text for matching: lowercase, no accents, words separated by one space."""
    return " ".join(WORD_RE.findall("".join(normalize_char(c) for c in text)))


def normalize_with_offsets(text: str) -> Tuple[str, List[int]]:
    """
    Normalize text keeping a map from each normalized char to its original index.

    Returns:
        (normalized_text, offsets) with len(offsets) == len(normalized_text)
    """
    chars: List[str] = []
    offsets: List[int] = []
    for i, c in enumerate(text):
        n = normalize_char(c)
        if not n.isalnum():
            n = " "
        if n == " ":
            if not chars or chars[-1] == " ":
                continue
        chars.append(n)
        offsets.append(i)
    while chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


@dataclass
class CityMention:
    """A city found in a sentence."""
    city: str        # Canonical city name
    start: int       # Start char offset in the original sentence
    end: int         # End char offset in the original sentence
    fuzzy: bool = False


def find_entity_span(
    text: str,
    entity: str,
    exclude: Sequence[Tuple[int, int]] = (),
    min_ratio: float = 0.8,
) -> Optional[Tuple[int, int]]:
    """
    Find the character span of an entity in text, tolerant to case, accents,
    hyphens/spaces and small misspellings.

    Args:
        text: Sentence
        entity: Entity as written in the labels (canonical city name)
        exclude: Spans already taken (another entity)
        min_ratio: Minimum similarity for fuzzy matching

    Returns:
        (start, end) char offsets in text, or None
    """
    norm_text, offsets = normalize_with_offsets(text)
    norm_entity = normalize(entity)
    if not norm_entity:
        return None

    def overlaps(s: int, e: int) -> bool:
        return any(not (e <= xs or s >= xe) for xs, xe in exclude)

    def to_original(ns: int, ne: int) -> Tuple[int, int]:
        return offsets[ns], offsets[ne - 1] + 1

    # Exact normalized match on word boundaries
    for m in re.finditer(r"(?<![a-z0-9])" + re.escape(norm_entity) + r"(?![a-z0-9])", norm_text):
        s, e = to_original(m.start(), m.end())
        if not overlaps(s, e):
            return s, e

    # Fuzzy match over word windows of similar length
    words = list(WORD_RE.finditer(norm_text))
    n_words = len(norm_entity.split())
    best, best_ratio = None, min_ratio
    for size in {max(1, n_words - 1), n_words, n_words + 1}:
        for i in range(len(words) - size + 1):
            ns, ne = words[i].start(), words[i + size - 1].end()
            ratio = SequenceMatcher(None, norm_text[ns:ne], norm_entity).ratio()
            if ratio > best_ratio:
                s, e = to_original(ns, ne)
                if not overlaps(s, e):
                    best, best_ratio = (s, e), ratio
    return best


class CityGazetteer:
    """
    Gazetteer of known cities (training data cities + SNCF stations).
    """

    MAX_NGRAM = 8

    def __init__(
        self,
        cities: Optional[Sequence[str]] = None,
        train_path: Optional[str] = None,
        use_sncf: bool = True,
        fuzzy_cutoff: float = 0.8,
    ):
        """
        Args:
            cities: Explicit list of cities (skips automatic loading)
            train_path: Training CSV used to collect cities and template vocabulary
            use_sncf: Add SNCF stations (static list, no network call)
            fuzzy_cutoff: Similarity cutoff for misspelled city names
        """
        self.fuzzy_cutoff = fuzzy_cutoff
        train_file = Path(train_path) if train_path else DEFAULT_TRAIN_PATH
        train_rows = self._read_rows(train_file)

        if cities is None:
            found: Set[str] = set()
            for row in train_rows:
                found.update(c for c in (row["origin"], row["destination"]) if c and c != "INVALID")
            if use_sncf:
                found.update(self._sncf_cities())
            cities = sorted(found)

        self.cities = list(cities)
        self.lookup: Dict[str, str] = {}
        for city in self.cities:
            key = normalize(city)
            if key:
                # Prefer the shortest canonical form for a given key
                if key not in self.lookup or len(city) < len(self.lookup[key]):
                    self.lookup[key] = city
        self.keys = list(self.lookup)
        self.city_words = {w for k in self.keys for w in k.split()}
        self.template_words = self._template_vocabulary(train_rows)

    @staticmethod
    def _read_rows(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _sncf_cities() -> List[str]:
        try:
            from src.api.sncf_client import SNCFClient
            client = SNCFClient()
            client.load_data(use_api=False)
            return [s.city for s in client.get_all_stations()]
        except Exception:
            return []

    def _template_vocabulary(self, rows: List[Dict[str, str]]) -> Set[str]:
        """Words used by sentence templates (never fuzzy-matched as cities)."""
        vocab: Set[str] = set()
        for row in rows:
            sentence = normalize(row["sentence"])
            for c in (row["origin"], row["destination"]):
                if c and c != "INVALID":
                    sentence = sentence.replace(normalize(c), " ")
            vocab.update(sentence.split())
        return (vocab | COMMON_WORDS) - self.city_words

    def canonicalize(self, name: Optional[str], fuzzy: bool = True) -> Optional[str]:
        """Map a city name (possibly misspelled if fuzzy) to its canonical form."""
        if not name:
            return name
        key = normalize(name)
        if key in self.lookup:
            return self.lookup[key]
        if not fuzzy:
            return name
        match = get_close_matches(key, self.keys, n=1, cutoff=self.fuzzy_cutoff)
        return self.lookup[match[0]] if match else name

    def find_mentions(self, sentence: str, fuzzy: bool = True) -> List[CityMention]:
        """
        Find city mentions in a sentence (greedy longest match, left to right).
        """
        norm, offsets = normalize_with_offsets(sentence)
        words = list(WORD_RE.finditer(norm))
        mentions: List[CityMention] = []
        covered = [False] * len(words)

        i = 0
        while i < len(words):
            for size in range(min(self.MAX_NGRAM, len(words) - i), 0, -1):
                key = norm[words[i].start():words[i + size - 1].end()]
                if key in self.lookup:
                    mentions.append(CityMention(
                        self.lookup[key],
                        offsets[words[i].start()],
                        offsets[words[i + size - 1].end() - 1] + 1,
                    ))
                    for j in range(i, i + size):
                        covered[j] = True
                    i += size
                    break
            else:
                i += 1

        if fuzzy:
            mentions.extend(self._fuzzy_mentions(norm, offsets, words, covered))
        mentions.sort(key=lambda m: m.start)
        return mentions

    def _fuzzy_mentions(self, norm, offsets, words, covered) -> List[CityMention]:
        """Find misspelled cities among words not covered by exact matches."""
        found: List[CityMention] = []
        i = 0
        while i < len(words):
            best = None
            for size in range(min(4, len(words) - i), 0, -1):
                span = words[i:i + size]
                if any(covered[i:i + size]):
                    continue
                tokens = [w.group() for w in span]
                if any(t in self.template_words for t in tokens):
                    continue
                if any(len(t) < 3 and t not in self.city_words for t in (tokens[0], tokens[-1])):
                    continue
                key = " ".join(tokens)
                if len(key) < 4:
                    continue
                match = get_close_matches(key, self.keys, n=1, cutoff=self.fuzzy_cutoff)
                if match:
                    best = (size, match[0])
                    break
            if best:
                size, key = best
                found.append(CityMention(
                    self.lookup[key],
                    offsets[words[i].start()],
                    offsets[words[i + size - 1].end() - 1] + 1,
                    fuzzy=True,
                ))
                i += size
            else:
                i += 1
        return found

    # ------------------------------------------------------------------
    # Role assignment
    # ------------------------------------------------------------------

    @staticmethod
    def cue_role(sentence: str, start: int) -> Optional[str]:
        """
        Guess the role of a mention from the words just before it.

        Returns:
            "ORIGIN", "DESTINATION" or None
        """
        before = normalize(sentence[:start]).split()[-3:]
        # Tolerate misspelled cue words ("dirrection", "provenence")
        before = [
            (get_close_matches(w, CUE_WORDS, n=1, cutoff=0.8) or [w])[0] if len(w) >= 5 else w
            for w in before
        ]
        context = " ".join(before)
        best_role, best_len = None, 0
        for role, cues in (("DESTINATION", DESTINATION_CUES), ("ORIGIN", ORIGIN_CUES)):
            for cue in cues:
                if (context == cue or context.endswith(" " + cue)) and len(cue) > best_len:
                    best_role, best_len = role, len(cue)
        return best_role

    def assign_roles(
        self,
        sentence: str,
        mentions: List[CityMention],
    ) -> Tuple[Optional[CityMention], Optional[CityMention]]:
        """
        Pick the origin and destination among city mentions.

        Strategy: cue words first ("de", "depuis" -> origin; "à", "vers",
        "pour" -> destination), then word order (origin before destination).
        """
        if not mentions:
            return None, None
        roles = [(m, self.cue_role(sentence, m.start)) for m in mentions]
        origins = [m for m, r in roles if r == "ORIGIN"]
        dests = [m for m, r in roles if r == "DESTINATION"]
        unknown = [m for m, r in roles if r is None]

        origin = origins[0] if origins else None
        destination = dests[-1] if dests else None
        if destination is not None and destination is origin:
            destination = None

        def remaining():
            return [m for m in mentions if m is not origin and m is not destination]

        # Fill missing roles: prefer mentions without cue, then any other one
        if origin is None:
            pool = [m for m in unknown if m is not destination] or remaining()
            if destination is not None:
                before = [m for m in pool if m.start < destination.start]
                pool = before or pool
            origin = pool[0] if pool else None
        if destination is None:
            pool = [m for m in unknown if m is not origin] or remaining()
            if origin is not None:
                after = [m for m in pool if m.start > origin.start]
                pool = after or pool
            destination = pool[-1] if pool else None
        return origin, destination

    # ------------------------------------------------------------------
    # NER post-processing
    # ------------------------------------------------------------------

    def refine(
        self,
        sentence: str,
        origin: Optional[str],
        destination: Optional[str],
        fuzzy: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Post-process an NER prediction.

        1. Extend a truncated span to the known city that contains it
           ("Musée d'" -> "Musée d'Orsay"), split a span merging two cities
        2. Fill a missing entity with another known city of the sentence
        3. Return (None, None) when two distinct cities cannot be found

        The NER prediction is kept for unknown cities. With fuzzy=True,
        misspelled names are also mapped to the closest known city: better
        when every city is in the gazetteer, harmful for unknown cities
        (e.g. "Vosves" would become "Voves").
        """
        mentions = self.find_mentions(sentence, fuzzy=fuzzy)
        used: List[CityMention] = []
        extra: List[CityMention] = []

        def snap(pred: Optional[str], role: str) -> Optional[str]:
            if not pred:
                return None
            span = find_entity_span(sentence, pred, exclude=[(m.start, m.end) for m in used])
            if span:
                start, end = span
                inside = [m for m in mentions
                          if m not in used and m.start >= start and m.end <= end]
                if len(inside) >= 2:
                    # One predicted span covering two cities ("challans chantenay")
                    chosen = inside[0] if role == "ORIGIN" else inside[-1]
                    used.append(chosen)
                    extra.extend(m for m in inside if m is not chosen)
                    return chosen.city
                for m in mentions:
                    overlaps = not (m.end <= start or m.start >= end)
                    if m not in used and overlaps and m.end - m.start >= end - start:
                        used.append(m)
                        return m.city
            return self.canonicalize(pred, fuzzy=fuzzy)

        new_origin = snap(origin, "ORIGIN")
        new_destination = snap(destination, "DESTINATION")

        if new_origin is None or new_destination is None:
            free = extra + [m for m in mentions if m not in used and m not in extra]
            if new_origin is None and new_destination is None:
                o, d = self.assign_roles(sentence, sorted(free, key=lambda m: m.start))
                new_origin = o.city if o else None
                new_destination = d.city if d else None
            elif new_origin is None and free:
                pool = [m for m in free if self.cue_role(sentence, m.start) != "DESTINATION"] or free
                new_origin = pool[0].city
            elif new_destination is None and free:
                pool = [m for m in free if self.cue_role(sentence, m.start) != "ORIGIN"] or free
                new_destination = pool[-1].city

        if not new_origin or not new_destination or new_origin == new_destination:
            return None, None
        return new_origin, new_destination
