"""
Train Route Pathfinding with Dijkstra Algorithm

Uses NetworkX to find optimal routes between train stations.
Supports optimization by duration or distance.
"""

import sys
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.api.sncf_client import SNCFClient, Station, Connection


@dataclass
class RouteStep:
    """A single step in a route."""
    origin: Station
    destination: Station
    duration_minutes: int
    distance_km: float
    line_type: str


@dataclass
class Route:
    """Complete route between two cities."""
    origin: Station
    destination: Station
    steps: List[RouteStep] = field(default_factory=list)
    total_duration_minutes: int = 0
    total_distance_km: float = 0.0
    num_changes: int = 0

    def __str__(self) -> str:
        hours = self.total_duration_minutes // 60
        minutes = self.total_duration_minutes % 60
        lines = [
            f"Route: {self.origin.city} -> {self.destination.city}",
            f"Duration: {hours}h{minutes:02d}",
            f"Distance: {self.total_distance_km:.1f} km",
            f"Changes: {self.num_changes}",
            "-" * 40,
        ]
        for i, step in enumerate(self.steps, 1):
            lines.append(
                f"{i}. {step.origin.city} -> {step.destination.city} "
                f"({step.line_type}, {step.duration_minutes} min, {step.distance_km} km)"
            )
        return "\n".join(lines)


class TrainGraph:
    """Graph representation of train network for pathfinding."""

    def __init__(self, sncf_client: Optional[SNCFClient] = None):
        self.client = sncf_client or SNCFClient()
        self.client.load_data(use_api=False)
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Build NetworkX graph from SNCF connections."""
        #  nodes (stations)
        for station in self.client.get_all_stations():
            self.graph.add_node(
                station.code,
                name=station.name,
                city=station.city,
                lat=station.latitude,
                lon=station.longitude,
                station=station,
            )

        #  edges (connections)
        for conn in self.client.get_connections():
            self.graph.add_edge(
                conn.origin_code,
                conn.destination_code,
                duration=conn.duration_minutes,
                distance=conn.distance_km,
                line_type=conn.line_type,
                connection=conn,
            )

    def find_shortest_path(
        self,
        origin_city: str,
        destination_city: str,
    ) -> Optional[Route]:
        """
        Find shortest path between two cities based on duration.

        Args:
            origin_city: Name of origin city
            destination_city: Name of destination city

        Returns:
            Route object or None if no path found
        """
        origin_station = self.client.find_station_by_city(origin_city)
        dest_station = self.client.find_station_by_city(destination_city)

        if not origin_station:
            print(f"Station not found for city: {origin_city}")
            return None

        if not dest_station:
            print(f"Station not found for city: {destination_city}")
            return None

        if origin_station.code == dest_station.code:
            return Route(
                origin=origin_station,
                destination=dest_station,
                steps=[],
                total_duration_minutes=0,
                total_distance_km=0.0,
                num_changes=0,
            )

        # Always optimize by duration (shortest travel time)
        try:
            path = nx.dijkstra_path(
                self.graph,
                origin_station.code,
                dest_station.code,
                weight="duration"
            )
        except nx.NetworkXNoPath:
            print(f"No path found between {origin_city} and {destination_city}")
            return None

        # Build route from path
        steps = []
        total_duration = 0
        total_distance = 0.0

        for i in range(len(path) - 1):
            edge_data = self.graph.edges[path[i], path[i+1]]
            origin_node = self.graph.nodes[path[i]]
            dest_node = self.graph.nodes[path[i+1]]

            step = RouteStep(
                origin=origin_node["station"],
                destination=dest_node["station"],
                duration_minutes=edge_data["duration"],
                distance_km=edge_data["distance"],
                line_type=edge_data["line_type"],
            )
            steps.append(step)
            total_duration += edge_data["duration"]
            total_distance += edge_data["distance"]

        # Count changes
        num_changes = 0
        for i in range(1, len(steps)):
            if steps[i].line_type != steps[i-1].line_type:
                num_changes += 1

        return Route(
            origin=origin_station,
            destination=dest_station,
            steps=steps,
            total_duration_minutes=total_duration,
            total_distance_km=round(total_distance, 1),
            num_changes=num_changes,
        )

    def find_all_paths(
        self,
        origin_city: str,
        destination_city: str,
        max_paths: int = 3,
    ) -> List[Route]:
        """
        Find multiple alternative paths between two cities based on duration.

        Args:
            origin_city: Name of origin city
            destination_city: Name of destination city
            max_paths: Maximum number of paths to return

        Returns:
            List of Route objects
        """
        origin_station = self.client.find_station_by_city(origin_city)
        dest_station = self.client.find_station_by_city(destination_city)

        if not origin_station or not dest_station:
            return []

        # Always optimize by duration (shortest travel time)
        try:
            paths_gen = nx.shortest_simple_paths(
                self.graph,
                origin_station.code,
                dest_station.code,
                weight="duration"
            )

            routes = []
            for path in paths_gen:
                if len(routes) >= max_paths:
                    break

                steps = []
                total_duration = 0
                total_distance = 0.0

                for i in range(len(path) - 1):
                    edge_data = self.graph.edges[path[i], path[i+1]]
                    origin_node = self.graph.nodes[path[i]]
                    dest_node = self.graph.nodes[path[i+1]]

                    step = RouteStep(
                        origin=origin_node["station"],
                        destination=dest_node["station"],
                        duration_minutes=edge_data["duration"],
                        distance_km=edge_data["distance"],
                        line_type=edge_data["line_type"],
                    )
                    steps.append(step)
                    total_duration += edge_data["duration"]
                    total_distance += edge_data["distance"]

                num_changes = 0
                for i in range(1, len(steps)):
                    if steps[i].line_type != steps[i-1].line_type:
                        num_changes += 1

                route = Route(
                    origin=origin_station,
                    destination=dest_station,
                    steps=steps,
                    total_duration_minutes=total_duration,
                    total_distance_km=round(total_distance, 1),
                    num_changes=num_changes,
                )
                routes.append(route)

            return routes

        except nx.NetworkXNoPath:
            return []

    @staticmethod
    def _haversine_time_estimate(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Haversine distance converted to estimated travel time in minutes.
        Uses TGV average speed (280 km/h) as optimistic estimate for A* admissibility.
        """
        R = 6371  # Earth radius in km
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_km = R * c
        # TGV speed as upper bound => admissible heuristic
        return (distance_km / 280) * 60

    def find_path_astar(
        self,
        origin_city: str,
        destination_city: str,
    ) -> Optional[Route]:
        """
        Find path using A* with geographic heuristic (Haversine / TGV speed).

        Args:
            origin_city: Name of origin city
            destination_city: Name of destination city

        Returns:
            Route object or None if no path found
        """
        origin_station = self.client.find_station_by_city(origin_city)
        dest_station = self.client.find_station_by_city(destination_city)

        if not origin_station or not dest_station:
            if not origin_station:
                print(f"Station not found for city: {origin_city}")
            if not dest_station:
                print(f"Station not found for city: {destination_city}")
            return None

        if origin_station.code == dest_station.code:
            return Route(origin=origin_station, destination=dest_station)

        dest_lat = dest_station.latitude
        dest_lon = dest_station.longitude

        def heuristic(node_a, node_b):
            a_data = self.graph.nodes[node_a]
            return self._haversine_time_estimate(
                a_data["lat"], a_data["lon"], dest_lat, dest_lon
            )

        try:
            path = nx.astar_path(
                self.graph,
                origin_station.code,
                dest_station.code,
                heuristic=heuristic,
                weight="duration",
            )
        except nx.NetworkXNoPath:
            print(f"No path found between {origin_city} and {destination_city}")
            return None

        return self._build_route(path, origin_station, dest_station)

    def find_path_min_changes(
        self,
        origin_city: str,
        destination_city: str,
        max_candidates: int = 10,
    ) -> Optional[Route]:
        """
        Find path with minimum number of line changes (correspondances).

        Iterates over the K shortest simple paths and selects the one
        with the fewest changes in line_type.

        Args:
            origin_city: Name of origin city
            destination_city: Name of destination city
            max_candidates: Maximum candidate paths to evaluate

        Returns:
            Route with minimum changes, or None if no path found
        """
        origin_station = self.client.find_station_by_city(origin_city)
        dest_station = self.client.find_station_by_city(destination_city)

        if not origin_station or not dest_station:
            return None

        if origin_station.code == dest_station.code:
            return Route(origin=origin_station, destination=dest_station)

        try:
            paths_gen = nx.shortest_simple_paths(
                self.graph,
                origin_station.code,
                dest_station.code,
                weight="duration",
            )

            best_route = None
            best_changes = float("inf")

            for i, path in enumerate(paths_gen):
                if i >= max_candidates:
                    break

                route = self._build_route(path, origin_station, dest_station)
                if route.num_changes < best_changes:
                    best_changes = route.num_changes
                    best_route = route

                # Can't do better than 0 changes
                if best_changes == 0:
                    break

            return best_route

        except nx.NetworkXNoPath:
            return None

    def _build_route(self, path: List[str], origin_station: Station, dest_station: Station) -> Route:
        """Build a Route object from a list of node codes."""
        steps = []
        total_duration = 0
        total_distance = 0.0

        for i in range(len(path) - 1):
            edge_data = self.graph.edges[path[i], path[i + 1]]
            origin_node = self.graph.nodes[path[i]]
            dest_node = self.graph.nodes[path[i + 1]]

            step = RouteStep(
                origin=origin_node["station"],
                destination=dest_node["station"],
                duration_minutes=edge_data["duration"],
                distance_km=edge_data["distance"],
                line_type=edge_data["line_type"],
            )
            steps.append(step)
            total_duration += edge_data["duration"]
            total_distance += edge_data["distance"]

        num_changes = 0
        for i in range(1, len(steps)):
            if steps[i].line_type != steps[i - 1].line_type:
                num_changes += 1

        return Route(
            origin=origin_station,
            destination=dest_station,
            steps=steps,
            total_duration_minutes=total_duration,
            total_distance_km=round(total_distance, 1),
            num_changes=num_changes,
        )

    def get_direct_connections(self, city: str) -> List[Tuple[Station, int, float, str]]:
        """
        Get all direct connections from a city.

        Returns:
            List of (destination_station, duration, distance, line_type) tuples
        """
        station = self.client.find_station_by_city(city)
        if not station:
            return []

        connections = []
        for _, dest_code, edge_data in self.graph.out_edges(station.code, data=True):
            dest_station = self.graph.nodes[dest_code]["station"]
            connections.append((
                dest_station,
                edge_data["duration"],
                edge_data["distance"],
                edge_data["line_type"],
            ))

        return sorted(connections, key=lambda x: x[1]) 

    def get_network_stats(self) -> Dict:
        """Get statistics about the train network."""
        return {
            "num_stations": self.graph.number_of_nodes(),
            "num_connections": self.graph.number_of_edges(),
            "avg_connections_per_station": self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
            "is_connected": nx.is_weakly_connected(self.graph),
        }


def find_route(
    origin: str,
    destination: str,
) -> Optional[Route]:
    """
    Convenience function to find a route between two cities based on duration.

    Args:
        origin: Origin city name
        destination: Destination city name

    Returns:
        Route object or None
    """
    graph = TrainGraph()
    return graph.find_shortest_path(origin, destination)


if __name__ == "__main__":
    print("=" * 60)
    print("TRAIN PATHFINDING - DIJKSTRA ALGORITHM")
    print("=" * 60)

    graph = TrainGraph()
    stats = graph.get_network_stats()
    print(f"\nNetwork: {stats['num_stations']} stations, {stats['num_connections']} connections")

    # Test routes
    test_routes = [
        ("Paris", "Marseille"),
        ("Lyon", "Bordeaux"),
        ("Lille", "Nice"),
        ("Nantes", "Strasbourg"),
        ("Toulouse", "Paris"),
    ]

    for origin, destination in test_routes:
        print(f"\n{'='*60}")
        route = graph.find_shortest_path(origin, destination)
        if route:
            print(route)
        else:
            print(f"No route found: {origin} -> {destination}")

    # Show direct connections from Paris
    print(f"\n{'='*60}")
    print("Direct connections from Paris:")
    for dest, duration, distance, line_type in graph.get_direct_connections("Paris"):
        print(f"  -> {dest.city}: {duration} min ({line_type})")
