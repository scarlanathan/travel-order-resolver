"""
SNCF Open Data API Client

Fetches French train stations and connections from SNCF Open Data.
Falls back to static data if API is unavailable.
"""

import os
import json
import math
import unicodedata
import difflib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import requests

# Load .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


@dataclass
class Station:
    """Represents a train station."""
    code: str
    name: str
    city: str
    latitude: float
    longitude: float

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        if isinstance(other, Station):
            return self.code == other.code
        return False


@dataclass
class Connection:
    """Represents a connection between two stations."""
    origin_code: str
    destination_code: str
    duration_minutes: int
    distance_km: float
    line_type: str  # TGV, TER, Intercités par exemple


class SNCFClient:
    """Client for SNCF Open Data API with static fallback."""

    # SNCF Open Data API v2.1
    API_BASE_URL = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets"
    DATASET_GARES = "gares-de-voyageurs"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SNCF_API_KEY", "")
        self.stations: Dict[str, Station] = {}
        self.connections: List[Connection] = []
        self._loaded = False

    def load_data(self, use_api: bool = True) -> None:
        """Load stations and connections data."""
        if self._loaded:
            return

        if use_api and self.api_key:
            try:
                self._load_from_api()
                self._loaded = True
                print(f"Loaded {len(self.get_all_stations())} stations from SNCF API")
                return
            except Exception as e:
                print(f"API error, falling back to static data: {e}")

        self._load_static_data()
        self._loaded = True

    def _extract_city_from_name(self, station_name: str) -> str:
        """Extract city name from station name."""
        # Common suffixes to remove
        suffixes = [
            " Gare de Lyon", " Montparnasse", " Gare du Nord", " Saint-Lazare",
            " Part-Dieu", " Saint-Charles", " Matabiau", " Saint-Jean",
            " Flandres", " Ville", " Saint-Roch", " Châteaucreux",
            " Saint-Laud", " Bénédictins", " Rive Droite", " TGV",
            " Loché TGV", " Aéroport", " Centre",
        ]
        city = station_name
        for suffix in suffixes:
            if city.endswith(suffix):
                city = city[:-len(suffix)]
                break
        return city.strip()

    def _load_from_api(self) -> None:
        """Load stations from SNCF API v2.1."""
        # First load static major stations to ensure connectivity
        self._load_static_major_stations()

        url = f"{self.API_BASE_URL}/{self.DATASET_GARES}/records"

        # Fetch stations in batches (API limit is 100 per request)
        offset = 0
        limit = 100

        while True:
            params = {
                "limit": limit,
                "offset": offset,
                "apikey": self.api_key,
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            for record in results:
                # Extract fields from API v2.1 response
                code = record.get("codes_uic", "")
                name = record.get("nom", "")
                city = self._extract_city_from_name(name)

                # Get coordinates
                geo = record.get("position_geographique", {})
                if isinstance(geo, dict):
                    lat = geo.get("lat", 0.0)
                    lon = geo.get("lon", 0.0)
                else:
                    lat, lon = 0.0, 0.0

                if code and name and lat != 0.0:
                    station = Station(
                        code=str(code),
                        name=name,
                        city=city,
                        latitude=float(lat),
                        longitude=float(lon),
                    )
                    # Don't overwrite major stations
                    if city.lower() not in self.stations:
                        self.stations[city.lower()] = station
                    if name.lower() not in self.stations:
                        self.stations[name.lower()] = station

            offset += limit
            # Limit to first 500 major stations to keep graph manageable
            if offset >= 500:
                break

        self._generate_connections()

    def _load_static_major_stations(self) -> None:
        """Load static major French stations for connectivity."""
        static_stations = [
            ("87686006", "Paris Gare de Lyon", "Paris", 48.8448, 2.3735),
            ("87391003", "Paris Montparnasse", "Paris", 48.8408, 2.3188),
            ("87271007", "Paris Gare du Nord", "Paris", 48.8809, 2.3553),
            ("87113001", "Paris Saint-Lazare", "Paris", 48.8763, 2.3247),
            ("87547000", "Lyon Part-Dieu", "Lyon", 45.7606, 4.8593),
            ("87751008", "Marseille Saint-Charles", "Marseille", 43.3026, 5.3803),
            ("87611004", "Toulouse Matabiau", "Toulouse", 43.6114, 1.4535),
            ("87581009", "Bordeaux Saint-Jean", "Bordeaux", 44.8256, -0.5561),
            ("87223263", "Lille Flandres", "Lille", 50.6365, 3.0700),
            ("87756056", "Nice Ville", "Nice", 43.7044, 7.2619),
            ("87481002", "Nantes", "Nantes", 47.2173, -1.5418),
            ("87212027", "Strasbourg", "Strasbourg", 48.5850, 7.7350),
            ("87765008", "Montpellier Saint-Roch", "Montpellier", 43.6047, 3.8815),
            ("87471003", "Rennes", "Rennes", 48.1035, -1.6721),
            ("87574004", "Reims", "Reims", 49.2583, 4.0242),
            ("87571000", "Tours", "Tours", 47.3894, 0.6925),
            ("87726000", "Saint-Étienne Châteaucreux", "Saint-Étienne", 45.4428, 4.3997),
            ("87413013", "Le Havre", "Le Havre", 49.4938, 0.1077),
            ("87747006", "Grenoble", "Grenoble", 45.1914, 5.7149),
            ("87713040", "Dijon Ville", "Dijon", 47.3233, 5.0272),
            ("87484006", "Angers Saint-Laud", "Angers", 47.4648, -0.5563),
            ("87775007", "Nîmes", "Nîmes", 43.8328, 4.3651),
            ("87183004", "Avignon TGV", "Avignon", 43.9217, 4.7863),
            ("87318964", "Aix-en-Provence TGV", "Aix-en-Provence", 43.4553, 5.3174),
            ("87681007", "Valence TGV", "Valence", 44.9780, 4.9020),
            ("87734004", "Cannes", "Cannes", 43.5528, 7.0170),
            ("87141002", "Orléans", "Orléans", 47.9083, 1.9050),
            ("87592006", "Poitiers", "Poitiers", 46.5826, 0.3331),
            ("87545194", "Mâcon Loché TGV", "Mâcon", 46.2881, 4.7936),
            ("87673004", "Chambéry", "Chambéry", 45.5713, 5.9178),
            ("87734327", "Antibes", "Antibes", 43.5858, 7.1074),
            ("87286005", "Amiens", "Amiens", 49.8900, 2.3050),
            ("87444000", "Caen", "Caen", 49.1783, -0.3478),
            ("87485003", "Le Mans", "Le Mans", 47.9956, 0.1922),
            ("87342006", "Clermont-Ferrand", "Clermont-Ferrand", 45.7789, 3.1000),
            ("87214056", "Metz Ville", "Metz", 49.1097, 6.1775),
            ("87191007", "Nancy", "Nancy", 48.6897, 6.1747),
            ("87172007", "Mulhouse Ville", "Mulhouse", 47.7417, 7.3425),
            ("87411017", "Rouen Rive Droite", "Rouen", 49.4486, 1.0939),
            ("87765800", "Perpignan", "Perpignan", 42.6975, 2.8792),
            ("87673608", "Annecy", "Annecy", 45.9022, 6.1222),
            ("87491001", "Saint-Nazaire", "Saint-Nazaire", 47.2867, -2.2064),
            ("87476002", "Lorient", "Lorient", 47.7475, -3.3597),
            ("87474007", "Brest", "Brest", 48.3878, -4.4800),
            ("87594002", "La Rochelle", "La Rochelle", 46.1522, -1.1500),
            ("87586008", "Angoulême", "Angoulême", 45.6531, 0.1603),
            ("87722025", "Limoges Bénédictins", "Limoges", 45.8361, 1.2667),
            ("87615013", "Pau", "Pau", 43.3000, -0.3672),
            ("87615039", "Bayonne", "Bayonne", 43.5000, -1.4653),
            ("87613000", "Biarritz", "Biarritz", 43.4667, -1.5500),
            ("87734160", "Monaco Monte-Carlo", "Monaco", 43.7400, 7.4200),
        ]

        for code, name, city, lat, lon in static_stations:
            station = Station(
                code=code,
                name=name,
                city=city,
                latitude=lat,
                longitude=lon,
            )
            self.stations[city.lower()] = station
            self.stations[name.lower()] = station

    def _load_static_data(self) -> None:
        """Load static French station data."""
        # Major French stations with approximate coordinates
        static_stations = [
            ("87686006", "Paris Gare de Lyon", "Paris", 48.8448, 2.3735),
            ("87391003", "Paris Montparnasse", "Paris", 48.8408, 2.3188),
            ("87271007", "Paris Gare du Nord", "Paris", 48.8809, 2.3553),
            ("87113001", "Paris Saint-Lazare", "Paris", 48.8763, 2.3247),
            ("87547000", "Lyon Part-Dieu", "Lyon", 45.7606, 4.8593),
            ("87751008", "Marseille Saint-Charles", "Marseille", 43.3026, 5.3803),
            ("87611004", "Toulouse Matabiau", "Toulouse", 43.6114, 1.4535),
            ("87581009", "Bordeaux Saint-Jean", "Bordeaux", 44.8256, -0.5561),
            ("87223263", "Lille Flandres", "Lille", 50.6365, 3.0700),
            ("87756056", "Nice Ville", "Nice", 43.7044, 7.2619),
            ("87481002", "Nantes", "Nantes", 47.2173, -1.5418),
            ("87212027", "Strasbourg", "Strasbourg", 48.5850, 7.7350),
            ("87765008", "Montpellier Saint-Roch", "Montpellier", 43.6047, 3.8815),
            ("87471003", "Rennes", "Rennes", 48.1035, -1.6721),
            ("87574004", "Reims", "Reims", 49.2583, 4.0242),
            ("87571000", "Tours", "Tours", 47.3894, 0.6925),
            ("87726000", "Saint-Étienne Châteaucreux", "Saint-Étienne", 45.4428, 4.3997),
            ("87413013", "Le Havre", "Le Havre", 49.4938, 0.1077),
            ("87747006", "Grenoble", "Grenoble", 45.1914, 5.7149),
            ("87713040", "Dijon Ville", "Dijon", 47.3233, 5.0272),
            ("87484006", "Angers Saint-Laud", "Angers", 47.4648, -0.5563),
            ("87775007", "Nîmes", "Nîmes", 43.8328, 4.3651),
            ("87183004", "Avignon TGV", "Avignon", 43.9217, 4.7863),
            ("87318964", "Aix-en-Provence TGV", "Aix-en-Provence", 43.4553, 5.3174),
            ("87681007", "Valence TGV", "Valence", 44.9780, 4.9020),
            ("87734004", "Cannes", "Cannes", 43.5528, 7.0170),
            ("87141002", "Orléans", "Orléans", 47.9083, 1.9050),
            ("87592006", "Poitiers", "Poitiers", 46.5826, 0.3331),
            ("87545194", "Mâcon Loché TGV", "Mâcon", 46.2881, 4.7936),
            ("87673004", "Chambéry", "Chambéry", 45.5713, 5.9178),
            ("87734327", "Antibes", "Antibes", 43.5858, 7.1074),
            ("87286005", "Amiens", "Amiens", 49.8900, 2.3050),
            ("87444000", "Caen", "Caen", 49.1783, -0.3478),
            ("87485003", "Le Mans", "Le Mans", 47.9956, 0.1922),
            ("87342006", "Clermont-Ferrand", "Clermont-Ferrand", 45.7789, 3.1000),
            ("87214056", "Metz Ville", "Metz", 49.1097, 6.1775),
            ("87191007", "Nancy", "Nancy", 48.6897, 6.1747),
            ("87172007", "Mulhouse Ville", "Mulhouse", 47.7417, 7.3425),
            ("87411017", "Rouen Rive Droite", "Rouen", 49.4486, 1.0939),
            ("87765800", "Perpignan", "Perpignan", 42.6975, 2.8792),
            ("87673608", "Annecy", "Annecy", 45.9022, 6.1222),
            ("87491001", "Saint-Nazaire", "Saint-Nazaire", 47.2867, -2.2064),
            ("87476002", "Lorient", "Lorient", 47.7475, -3.3597),
            ("87474007", "Brest", "Brest", 48.3878, -4.4800),
            ("87594002", "La Rochelle", "La Rochelle", 46.1522, -1.1500),
            ("87586008", "Angoulême", "Angoulême", 45.6531, 0.1603),
            ("87722025", "Limoges Bénédictins", "Limoges", 45.8361, 1.2667),
            ("87615013", "Pau", "Pau", 43.3000, -0.3672),
            ("87615039", "Bayonne", "Bayonne", 43.5000, -1.4653),
            ("87613000", "Biarritz", "Biarritz", 43.4667, -1.5500),
            ("87734160", "Monaco Monte-Carlo", "Monaco", 43.7400, 7.4200),
        ]

        for code, name, city, lat, lon in static_stations:
            station = Station(
                code=code,
                name=name,
                city=city,
                latitude=lat,
                longitude=lon,
            )
            self.stations[city.lower()] = station
            self.stations[name.lower()] = station

        self._generate_connections()

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in km (Haversine formula)."""
        import math
        R = 6371  

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def _generate_connections(self) -> None:
        """Generate connections between major stations."""
        # TGV connections (high-speed, ~300km/h average)
        tgv_routes = [
            # Paris - Lyon - Marseille
            ("Paris", "Lyon", "TGV"),
            ("Lyon", "Marseille", "TGV"),
            ("Lyon", "Montpellier", "TGV"),
            ("Montpellier", "Marseille", "TGV"),
            # Paris - Bordeaux
            ("Paris", "Bordeaux", "TGV"),
            ("Bordeaux", "Toulouse", "TGV"),
            # Paris - Lille
            ("Paris", "Lille", "TGV"),
            # Paris - Strasbourg
            ("Paris", "Strasbourg", "TGV"),
            # Paris - Rennes - Nantes
            ("Paris", "Rennes", "TGV"),
            ("Paris", "Nantes", "TGV"),
            ("Rennes", "Nantes", "TER"),
            # Lyon connections
            ("Lyon", "Grenoble", "TER"),
            ("Lyon", "Saint-Étienne", "TER"),
            ("Lyon", "Dijon", "TGV"),
            ("Dijon", "Paris", "TGV"),
            # Côte d'Azur
            ("Marseille", "Nice", "TGV"),
            ("Nice", "Cannes", "TER"),
            ("Nice", "Monaco", "TER"),
            ("Cannes", "Antibes", "TER"),
            # Sud-Ouest
            ("Toulouse", "Montpellier", "Intercités"),
            ("Bordeaux", "Bayonne", "TGV"),
            ("Bayonne", "Biarritz", "TER"),
            ("Bayonne", "Pau", "TER"),
            # Bretagne
            ("Rennes", "Brest", "TGV"),
            ("Rennes", "Lorient", "TER"),
            ("Nantes", "La Rochelle", "TER"),
            ("Nantes", "Saint-Nazaire", "TER"),
            # Nord
            ("Lille", "Amiens", "TER"),
            ("Amiens", "Rouen", "TER"),
            ("Rouen", "Le Havre", "TER"),
            ("Paris", "Rouen", "Intercités"),
            ("Paris", "Le Havre", "Intercités"),
            ("Paris", "Caen", "Intercités"),
            # Est
            ("Strasbourg", "Mulhouse", "TER"),
            ("Strasbourg", "Metz", "TER"),
            ("Metz", "Nancy", "TER"),
            ("Nancy", "Paris", "TGV"),
            # Centre
            ("Paris", "Tours", "TGV"),
            ("Tours", "Bordeaux", "TGV"),
            ("Paris", "Orléans", "Intercités"),
            ("Orléans", "Tours", "TER"),
            ("Tours", "Poitiers", "TGV"),
            ("Poitiers", "Bordeaux", "TGV"),
            ("Poitiers", "La Rochelle", "TER"),
            ("Bordeaux", "Angoulême", "TGV"),
            ("Angoulême", "Poitiers", "TGV"),
            # Auvergne
            ("Lyon", "Clermont-Ferrand", "Intercités"),
            ("Clermont-Ferrand", "Paris", "Intercités"),
            # Alpes
            ("Lyon", "Chambéry", "TER"),
            ("Chambéry", "Annecy", "TER"),
            ("Chambéry", "Grenoble", "TER"),
            # Avignon TGV
            ("Paris", "Avignon", "TGV"),
            ("Avignon", "Marseille", "TGV"),
            ("Avignon", "Montpellier", "TGV"),
            ("Avignon", "Nîmes", "TER"),
            ("Nîmes", "Montpellier", "TER"),
            # Perpignan
            ("Montpellier", "Perpignan", "TGV"),
            ("Perpignan", "Toulouse", "Intercités"),
            # Valence
            ("Lyon", "Valence", "TGV"),
            ("Valence", "Avignon", "TGV"),
            # Limoges
            ("Paris", "Limoges", "Intercités"),
            ("Limoges", "Toulouse", "Intercités"),
            ("Limoges", "Bordeaux", "Intercités"),
            # Le Mans - Angers
            ("Paris", "Le Mans", "TGV"),
            ("Le Mans", "Rennes", "TGV"),
            ("Le Mans", "Nantes", "TGV"),
            ("Le Mans", "Angers", "TER"),
            ("Angers", "Nantes", "TER"),
            # Reims
            ("Paris", "Reims", "TGV"),
            # Aix-en-Provence
            ("Marseille", "Aix-en-Provence", "TER"),
            ("Aix-en-Provence", "Avignon", "TGV"),
        ]

        # Speed by line type (km/h)
        speeds = {
            "TGV": 280,
            "Intercités": 160,
            "TER": 100,
        }

        unique_stations = set()
        for city in self.stations:
            unique_stations.add(self.stations[city].city.lower())

        for origin_city, dest_city, line_type in tgv_routes:
            origin_key = origin_city.lower()
            dest_key = dest_city.lower()

            if origin_key not in self.stations or dest_key not in self.stations:
                continue

            origin = self.stations[origin_key]
            dest = self.stations[dest_key]

            distance = self._calculate_distance(
                origin.latitude, origin.longitude,
                dest.latitude, dest.longitude
            )

            # Estimate duration based on distance and line type
            speed = speeds.get(line_type, 100)
            duration = int((distance / speed) * 60)  # minutes

            # Some overhead for stops 
            duration = int(duration * 1.2)

            connection = Connection(
                origin_code=origin.code,
                destination_code=dest.code,
                duration_minutes=duration,
                distance_km=round(distance, 1),
                line_type=line_type,
            )
            self.connections.append(connection)

           #Reverse connection
            reverse = Connection(
                origin_code=dest.code,
                destination_code=origin.code,
                duration_minutes=duration,
                distance_km=round(distance, 1),
                line_type=line_type,
            )
            self.connections.append(reverse)

    def get_station(self, city_name: str) -> Optional[Station]:
        """Get station by city name."""
        self.load_data(use_api=True)
        return self.stations.get(city_name.lower())

    def get_all_stations(self) -> List[Station]:
        """Get all stations."""
        self.load_data(use_api=True)
        seen = set()
        result = []
        for station in self.stations.values():
            if station.code not in seen:
                seen.add(station.code)
                result.append(station)
        return result

    def get_connections(self) -> List[Connection]:
        """Get all connections."""
        self.load_data(use_api=True)
        return self.connections

    def get_connections_from(self, station_code: str) -> List[Connection]:
        """Get all connections from a specific station."""
        self.load_data(use_api=True)
        return [c for c in self.connections if c.origin_code == station_code]

    # Coordinates of major French cities without train stations
    CITY_COORDINATES = {
        "versailles": (48.8014, 2.1301),
        "argenteuil": (48.9472, 2.2467),
        "montreuil": (48.8638, 2.4483),
        "saint-denis": (48.9362, 2.3574),
        "boulogne-billancourt": (48.8397, 2.2399),
        "vitry-sur-seine": (48.7872, 2.3928),
        "créteil": (48.7900, 2.4628),
        "nanterre": (48.8924, 2.2071),
        "courbevoie": (48.8966, 2.2525),
        "colombes": (48.9232, 2.2542),
        "aubervilliers": (48.9148, 2.3828),
        "asnières-sur-seine": (48.9119, 2.2886),
        "rueil-malmaison": (48.8769, 2.1869),
        "champigny-sur-marne": (48.8177, 2.5156),
        "ivry-sur-seine": (48.8153, 2.3839),
    }

    # France metropolitan bounding box
    FRANCE_LAT_MIN = 41.0
    FRANCE_LAT_MAX = 51.5
    FRANCE_LON_MIN = -5.5
    FRANCE_LON_MAX = 10.0

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a name by removing accents, hyphens, and lowercasing."""
        name = name.lower().strip()
        name = name.replace("-", " ").replace("'", " ")
        # Remove accents
        nfkd = unicodedata.normalize("NFKD", name)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def _is_in_france(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within France metropolitan bounds."""
        return (self.FRANCE_LAT_MIN <= lat <= self.FRANCE_LAT_MAX and
                self.FRANCE_LON_MIN <= lon <= self.FRANCE_LON_MAX)

    def find_station_by_city(self, city: str) -> Optional[Station]:
        """
        Find station by city name with fuzzy matching (Levenshtein).

        Uses exact match first, then partial match, then difflib fuzzy matching
        with normalized names (no accents, hyphens).
        """
        self.load_data(use_api=True)
        city_lower = city.lower().strip()

        # Direct match
        if city_lower in self.stations:
            station = self.stations[city_lower]
            if self._is_in_france(station.latitude, station.longitude):
                return station
            else:
                print(f"Warning: Station '{city}' is outside France ({station.latitude}, {station.longitude})")
                return station

        # Partial match
        for key, station in self.stations.items():
            if city_lower in key or key in city_lower:
                return station

        # Fuzzy matching using difflib (Levenshtein-like)
        normalized_city = self._normalize_name(city)
        station_keys = list(self.stations.keys())
        normalized_keys = {self._normalize_name(k): k for k in station_keys}

        matches = difflib.get_close_matches(
            normalized_city, normalized_keys.keys(), n=1, cutoff=0.8
        )
        if matches:
            original_key = normalized_keys[matches[0]]
            print(f"Fuzzy match: '{city}' -> '{self.stations[original_key].city}'")
            return self.stations[original_key]

        return None

    def find_nearest_station(self, city_name: str) -> Optional[Tuple[Station, float]]:
        """
        Find the nearest station to a city that has no direct station.

        First tries to find the city in CITY_COORDINATES, then searches for
        the closest station by Haversine distance.

        Args:
            city_name: Name of the city

        Returns:
            Tuple of (nearest Station, distance in km) or None
        """
        self.load_data(use_api=True)

        city_key = city_name.lower().strip()

        # Get city coordinates
        lat, lon = None, None
        normalized = self._normalize_name(city_name)
        for key, coords in self.CITY_COORDINATES.items():
            if self._normalize_name(key) == normalized:
                lat, lon = coords
                break

        if lat is None:
            # Try fuzzy match on city coordinates
            coord_keys = list(self.CITY_COORDINATES.keys())
            norm_coord_keys = {self._normalize_name(k): k for k in coord_keys}
            matches = difflib.get_close_matches(
                normalized, norm_coord_keys.keys(), n=1, cutoff=0.7
            )
            if matches:
                orig_key = norm_coord_keys[matches[0]]
                lat, lon = self.CITY_COORDINATES[orig_key]

        if lat is None:
            return None

        # Find nearest station
        best_station = None
        best_distance = float("inf")

        seen_codes = set()
        for station in self.stations.values():
            if station.code in seen_codes:
                continue
            seen_codes.add(station.code)

            dist = self._calculate_distance(lat, lon, station.latitude, station.longitude)
            if dist < best_distance:
                best_distance = dist
                best_station = station

        if best_station:
            return best_station, round(best_distance, 1)
        return None


if __name__ == "__main__":
    client = SNCFClient()
    client.load_data(use_api=True)

    print(f"Loaded {len(client.get_all_stations())} stations")
    print(f"Loaded {len(client.connections)} connections")

    # Test finding stations
    paris = client.find_station_by_city("Paris")
    lyon = client.find_station_by_city("Lyon")

    if paris and lyon:
        print(f"\nParis: {paris.name} ({paris.code})")
        print(f"Lyon: {lyon.name} ({lyon.code})")

        # Find connections from Paris
        paris_connections = client.get_connections_from(paris.code)
        print(f"\nConnections from Paris: {len(paris_connections)}")
        for conn in paris_connections[:5]:
            dest = next((s for s in client.get_all_stations() if s.code == conn.destination_code), None)
            if dest:
                print(f"  -> {dest.city}: {conn.duration_minutes} min, {conn.distance_km} km ({conn.line_type})")
