from __future__ import annotations

import argparse
import gzip
import math
import os
import pickle
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import osmium
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

try:
    import osmnx as ox
except ImportError:  # pragma: no cover - optional fast snapping helper
    ox = None

PBF_PATH = "DATA/western-zone-260518.osm.pbf"
GRAPH_CACHE = "DATA/mumbai_pune_route_graph_v1.pkl.gz"
WAYS_CACHE = "DATA/mumbai_pune_route_ways_v1.pkl.gz"
ROUTE_PLOT_PATH = "DATA/route_mumbai_pune.png"

# Corridor covering the road network between Mumbai and Pune.
# Format: (min_lat, max_lat, min_lon, max_lon)
ROUTE_CORRIDOR_BBOX = (18.15, 19.45, 72.55, 74.25)

PRESET_LOCATIONS = {
    "Mumbai Warehouse": (19.0740, 72.8800),
    "Mumbai": (19.0760, 72.8777),
    "Mumbai CST": (18.9398, 72.8355),
    "Dadar": (19.0183, 72.8424),
    "Bandra": (19.0544, 72.8406),
    "Andheri": (19.1197, 72.8464),
    "Borivali": (19.2290, 72.8561),
    "Thane": (19.2183, 72.9781),
    "Navi Mumbai": (19.0330, 73.0297),
    "Panvel": (18.9894, 73.1175),
    "Khopoli": (18.7854, 73.3451),
    "Lonavala": (18.7547, 73.4068),
    "Talegaon": (18.7357, 73.6763),
    "Wakad": (18.5987, 73.7625),
    "Hinjewadi": (18.5918, 73.7389),
    "Baner": (18.5600, 73.7760),
    "Aundh": (18.5635, 73.8075),
    "Kothrud": (18.5074, 73.8077),
    "Shivajinagar": (18.5308, 73.8477),
    "Pune Station": (18.5287, 73.8747),
    "Hadapsar": (18.4965, 73.9219),
    "Kharadi": (18.5516, 73.9418),
    "Pune": (18.5204, 73.8567),
}

DEFAULT_VRP_DEPOT = "Mumbai Warehouse"
DEFAULT_VRP_DELIVERY_NAMES = [
    "Dadar",
    "Bandra",
    "Andheri",
    "Borivali",
    "Thane",
    "Navi Mumbai",
    "Panvel",
    "Khopoli",
    "Lonavala",
    "Talegaon",
    "Wakad",
    "Hinjewadi",
    "Baner",
    "Aundh",
    "Kothrud",
    "Shivajinagar",
    "Pune Station",
    "Hadapsar",
    "Kharadi",
    "Pune",
]


@dataclass(frozen=True)
class RouteResult:
    source_name: str
    destination_name: str
    source_node: int
    destination_node: int
    path: list[int]
    distance_m: float


@dataclass(frozen=True)
class VehicleRoute:
    vehicle_id: int
    stops: list[str]
    location_indices: list[int]
    distance_m: float
    node_path: list[int]


@dataclass(frozen=True)
class VRPResult:
    depot_name: str
    depot_index: int
    deliveries: list[str]
    routes: list[VehicleRoute]
    total_distance_m: float


def point_in_corridor(lat: float, lon: float) -> bool:
    min_lat, max_lat, min_lon, max_lon = ROUTE_CORRIDOR_BBOX
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def parse_location(value: str) -> tuple[str, tuple[float, float]]:
    raw = value.strip()
    if raw in PRESET_LOCATIONS:
        return raw, PRESET_LOCATIONS[raw]

    if "," in raw:
        lat_text, lon_text = raw.split(",", 1)
        return raw, (float(lat_text.strip()), float(lon_text.strip()))

    raise ValueError(
        f"Unknown location '{value}'. Use a preset name like 'Mumbai CST' or a 'lat,lon' pair."
    )


def oneway_direction(tags: dict[str, str]) -> str:
    value = str(tags.get("oneway", "")).strip().lower()
    if value in {"yes", "true", "1"}:
        return "forward"
    if value in {"-1", "reverse"}:
        return "reverse"
    return "both"


def load_gzip_pickle(path: str):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def save_gzip_pickle(path: str, value) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(value, handle)


class CorridorWayHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.ways: list[dict[str, object]] = []
        self.highway_count = 0

    def way(self, way):
        if "highway" not in way.tags:
            return

        self.highway_count += 1
        if self.highway_count % 100_000 == 0:
            print(f"Scanned highway ways: {self.highway_count:,}")

        nodes: list[tuple[int, float, float]] = []
        touches_corridor = False
        for node_ref in way.nodes:
            if not node_ref.location or not node_ref.location.valid():
                continue

            lat = float(node_ref.location.lat)
            lon = float(node_ref.location.lon)
            nodes.append((int(node_ref.ref), lat, lon))
            if point_in_corridor(lat, lon):
                touches_corridor = True

        if touches_corridor and len(nodes) >= 2:
            self.ways.append({
                "id": int(way.id),
                "nodes": nodes,
                "tags": dict(way.tags),
            })


def load_or_scan_ways(force_rebuild: bool = False) -> list[dict[str, object]]:
    if os.path.exists(WAYS_CACHE) and not force_rebuild:
        print("Loading cached Mumbai/Pune corridor ways...")
        try:
            ways = load_gzip_pickle(WAYS_CACHE)
            print(f"Loaded {len(ways):,} cached ways")
            return ways
        except (EOFError, OSError, pickle.UnpicklingError) as exc:
            print(f"Cache read failed for ways ({exc}); rebuilding...")
            try:
                os.remove(WAYS_CACHE)
            except OSError:
                pass

    print("Scanning PBF for Mumbai/Pune corridor ways...")
    handler = CorridorWayHandler()
    handler.apply_file(PBF_PATH, locations=True)
    ways = handler.ways

    save_gzip_pickle(WAYS_CACHE, ways)
    print(f"Saved {len(ways):,} corridor ways to cache")
    return ways


def load_or_build_graph(force_rebuild: bool = False) -> nx.DiGraph:
    if os.path.exists(GRAPH_CACHE) and not force_rebuild:
        print("Loading cached route graph...")
        try:
            graph = load_gzip_pickle(GRAPH_CACHE)
            graph.graph.setdefault("crs", "epsg:4326")
            print(f"Loaded graph with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges")
            return graph
        except (EOFError, OSError, pickle.UnpicklingError) as exc:
            print(f"Cache read failed for graph ({exc}); rebuilding...")
            try:
                os.remove(GRAPH_CACHE)
            except OSError:
                pass

    ways = load_or_scan_ways(force_rebuild=force_rebuild)

    print("Building route graph...")
    graph = nx.DiGraph()
    graph.graph["crs"] = "epsg:4326"
    edge_count = 0

    for index, way in enumerate(ways, start=1):
        nodes = way["nodes"]
        tags = way["tags"]

        for node_id, lat, lon in nodes:
            if node_id not in graph:
                graph.add_node(node_id, y=lat, x=lon)

        direction = oneway_direction(tags)
        for i in range(len(nodes) - 1):
            u_id, u_lat, u_lon = nodes[i]
            v_id, v_lat, v_lon = nodes[i + 1]
            length_m = haversine_m(u_lat, u_lon, v_lat, v_lon)
            graph.add_edge(u_id, v_id, length=length_m)
            edge_count += 1
            if direction == "both":
                graph.add_edge(v_id, u_id, length=length_m)
                edge_count += 1
            elif direction == "reverse":
                graph.add_edge(v_id, u_id, length=length_m)
                edge_count += 1

        if index % 50_000 == 0:
            print(
                f"Processed {index:,} ways, graph now has {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges"
            )

    save_gzip_pickle(GRAPH_CACHE, graph)
    print(f"Saved graph cache with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges")
    return graph


def nearest_node(graph: nx.DiGraph, lat: float, lon: float) -> int:
    best_node = None
    best_distance = float("inf")

    for node_id, data in graph.nodes(data=True):
        distance = haversine_m(lat, lon, float(data["y"]), float(data["x"]))
        if distance < best_distance:
            best_distance = distance
            best_node = node_id

    if best_node is None:
        raise RuntimeError("Could not find any node in the graph.")

    return int(best_node)


def nearest_nodes_for_locations(graph: nx.DiGraph, locations: list[tuple[str, tuple[float, float]]]) -> list[int]:
    if ox is not None:
        lons = [coords[1] for _, coords in locations]
        lats = [coords[0] for _, coords in locations]
        try:
            nearest = ox.distance.nearest_nodes(graph, lons, lats)
        except (ImportError, KeyError, ValueError):
            nearest = None
        else:
            if isinstance(nearest, list):
                return [int(node_id) for node_id in nearest]
            return [int(node_id) for node_id in list(nearest)]

    return [nearest_node(graph, lat, lon) for _, (lat, lon) in locations]


def shortest_route(graph: nx.DiGraph, source_name: str, source: tuple[float, float], destination_name: str, destination: tuple[float, float]) -> RouteResult:
    search_started = time.perf_counter()
    source_node = nearest_node(graph, source[0], source[1])
    destination_node = nearest_node(graph, destination[0], destination[1])

    snap_elapsed = time.perf_counter() - search_started

    print(f"Source '{source_name}' snapped to node {source_node}")
    print(f"Destination '{destination_name}' snapped to node {destination_node}")
    print(f"Snapping took {snap_elapsed:.2f} seconds")

    try:
        distance_m, path = nx.bidirectional_dijkstra(graph, source_node, destination_node, weight="length")
    except nx.NetworkXNoPath as exc:
        raise RuntimeError(
            "No road path was found between the selected source and destination inside the Mumbai/Pune corridor."
        ) from exc

    path_elapsed = time.perf_counter() - search_started - snap_elapsed
    print(f"Shortest path search took {path_elapsed:.2f} seconds")

    return RouteResult(
        source_name=source_name,
        destination_name=destination_name,
        source_node=source_node,
        destination_node=destination_node,
        path=path,
        distance_m=distance_m,
    )


def path_between_nodes(
    graph: nx.DiGraph,
    source_node: int,
    destination_node: int,
    route_cache: dict[tuple[int, int], tuple[float, list[int]]],
) -> tuple[float, list[int]]:
    cache_key = (source_node, destination_node)
    if cache_key in route_cache:
        return route_cache[cache_key]

    distance_m, path = nx.bidirectional_dijkstra(graph, source_node, destination_node, weight="length")
    route_cache[cache_key] = (float(distance_m), list(path))
    return route_cache[cache_key]


def build_distance_matrix_for_vrp(
    graph: nx.DiGraph,
    snapped_nodes: list[int],
) -> tuple[list[list[int]], dict[tuple[int, int], tuple[float, list[int]]]]:
    route_cache: dict[tuple[int, int], tuple[float, list[int]]] = {}
    size = len(snapped_nodes)
    matrix: list[list[int]] = [[0] * size for _ in range(size)]
    total_pairs = size * (size - 1) // 2
    processed_pairs = 0

    print(f"Building VRP distance matrix for {size} locations ({total_pairs} shortest-path pairs)...")

    for i in range(size):
        for j in range(i + 1, size):
            processed_pairs += 1
            distance_m, path = path_between_nodes(graph, snapped_nodes[i], snapped_nodes[j], route_cache)
            rounded_distance = max(1, int(round(distance_m)))
            matrix[i][j] = rounded_distance
            matrix[j][i] = rounded_distance
            route_cache[(snapped_nodes[j], snapped_nodes[i])] = (distance_m, list(reversed(path)))

            if processed_pairs == 1 or processed_pairs % max(1, total_pairs // 10) == 0:
                print(f"  Distance pair progress: {processed_pairs}/{total_pairs}")

    return matrix, route_cache


def solve_vrp(
    graph: nx.DiGraph,
    depot_name: str,
    depot: tuple[float, float],
    deliveries: list[tuple[str, tuple[float, float]]],
    vehicle_count: int,
    vehicle_capacity: int,
) -> VRPResult:
    if not deliveries:
        raise ValueError("At least one delivery location is required for VRP mode.")

    all_locations = [(depot_name, depot)] + deliveries
    print(f"Snapping {len(all_locations)} VRP locations to the road graph...")
    snapped_nodes = nearest_nodes_for_locations(graph, all_locations)
    print("Snapping complete. Building travel-time/distance matrix...")
    distance_matrix, route_cache = build_distance_matrix_for_vrp(graph, snapped_nodes)

    print(f"Depot '{depot_name}' snapped to node {snapped_nodes[0]}")
    for index, (delivery_name, _) in enumerate(deliveries, start=1):
        print(f"Delivery '{delivery_name}' snapped to node {snapped_nodes[index]}")

    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.AddDimension(
        transit_callback_index,
        0,
        50_000_000,
        True,
        "Distance",
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    demands = [0] + [1] * len(deliveries)
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        lambda from_index: demands[manager.IndexToNode(from_index)]
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [vehicle_capacity] * vehicle_count,
        True,
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(15)

    print(f"Solving VRP with {vehicle_count} trucks, capacity {vehicle_capacity} deliveries per truck...")
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("VRP solver could not find a feasible assignment for the provided trucks and deliveries.")

    print("VRP solve complete. Extracting routes...")

    routes: list[VehicleRoute] = []
    total_distance_m = 0.0
    delivery_names = [name for name, _ in deliveries]

    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)
        location_indices = [0]
        stops: list[str] = []
        route_distance_m = 0.0
        node_path: list[int] = []

        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            from_location = manager.IndexToNode(index)
            to_location = manager.IndexToNode(next_index)
            route_distance_m += float(distance_matrix[from_location][to_location])
            location_indices.append(to_location)

            if to_location != 0:
                stops.append(delivery_names[to_location - 1])

            if from_location != to_location:
                _, segment_path = route_cache[(snapped_nodes[from_location], snapped_nodes[to_location])]
                if not node_path:
                    node_path.extend(segment_path)
                else:
                    node_path.extend(segment_path[1:])
            elif not node_path:
                node_path.append(snapped_nodes[from_location])

            index = next_index

        total_distance_m += route_distance_m
        routes.append(
            VehicleRoute(
                vehicle_id=vehicle_id,
                stops=stops,
                location_indices=location_indices,
                distance_m=route_distance_m,
                node_path=node_path,
            )
        )
        print(f"  Truck {vehicle_id + 1} route ready: {len(stops)} stops, {route_distance_m / 1000:.2f} km")

    return VRPResult(
        depot_name=depot_name,
        depot_index=0,
        deliveries=delivery_names,
        routes=routes,
        total_distance_m=total_distance_m,
    )


def plot_route(graph: nx.DiGraph, route: RouteResult) -> None:
    xs = [float(graph.nodes[node_id]["x"]) for node_id in route.path]
    ys = [float(graph.nodes[node_id]["y"]) for node_id in route.path]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(xs, ys, color="#e53935", linewidth=3.0, zorder=3, label="Shortest path")
    ax.scatter(xs[0], ys[0], s=90, color="#2e7d32", zorder=4, label=route.source_name)
    ax.scatter(xs[-1], ys[-1], s=90, color="#1565c0", zorder=4, label=route.destination_name)
    ax.scatter(xs[1:-1], ys[1:-1], s=10, color="#ffb300", alpha=0.75, zorder=3)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max((max_x - min_x) * 0.08, 0.01)
    pad_y = max((max_y - min_y) * 0.08, 0.01)

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_title(
        f"Route Optimization Engine: {route.source_name} to {route.destination_name}\n"
        f"Shortest distance: {route.distance_m / 1000:.2f} km"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(ROUTE_PLOT_PATH), exist_ok=True)
    fig.savefig(ROUTE_PLOT_PATH, dpi=180)
    print(f"Saved route visualization to {ROUTE_PLOT_PATH}")
    plt.show()


def plot_vrp_routes(graph: nx.DiGraph, vrp_result: VRPResult, route_lookup: dict[str, tuple[float, float]]) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    colors = ["#e53935", "#1565c0", "#2e7d32", "#6d4c41", "#8e24aa", "#00897b"]

    depot_lat, depot_lon = route_lookup[vrp_result.depot_name]
    ax.scatter(depot_lon, depot_lat, s=180, marker="*", color="#ffb300", edgecolor="black", zorder=5, label=vrp_result.depot_name)

    stop_labels = {name: idx for idx, name in enumerate(vrp_result.deliveries, start=1)}
    for stop_name, (lat, lon) in route_lookup.items():
        if stop_name == vrp_result.depot_name:
            continue
        ax.scatter(lon, lat, s=35, color="#455a64", zorder=4)
        ax.text(lon, lat, str(stop_labels.get(stop_name, "")), fontsize=8, ha="left", va="bottom")

    all_xs: list[float] = [depot_lon]
    all_ys: list[float] = [depot_lat]

    for route in vrp_result.routes:
        color = colors[route.vehicle_id % len(colors)]
        xs = [float(graph.nodes[node_id]["x"]) for node_id in route.node_path]
        ys = [float(graph.nodes[node_id]["y"]) for node_id in route.node_path]
        if len(xs) >= 2:
            ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9, label=f"Truck {route.vehicle_id + 1} ({route.distance_m / 1000:.1f} km)")
            all_xs.extend(xs)
            all_ys.extend(ys)

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)
    pad_x = max((max_x - min_x) * 0.06, 0.01)
    pad_y = max((max_y - min_y) * 0.06, 0.01)

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_title(
        f"Route Optimization Engine: VRP for {vrp_result.depot_name}\n"
        f"{len(vrp_result.deliveries)} deliveries, {len(vrp_result.routes)} trucks, total distance {vrp_result.total_distance_m / 1000:.2f} km"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig("DATA/vrp_routes_mumbai_pune.png", dpi=180)
    print("Saved VRP visualization to DATA/vrp_routes_mumbai_pune.png")
    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route Optimization Engine for the Mumbai-Pune corridor")
    parser.add_argument(
        "--mode",
        choices=["single", "vrp"],
        default="vrp",
        help="single = one source and one destination, vrp = warehouse + multiple trucks + deliveries",
    )
    parser.add_argument(
        "--source",
        default="Mumbai CST",
        help="Source location preset name or 'lat,lon' pair. Default: Mumbai CST",
    )
    parser.add_argument(
        "--destination",
        default="Pune Station",
        help="Destination location preset name or 'lat,lon' pair. Default: Pune Station",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore caches and rebuild the reduced graph from the PBF file.",
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        default=3,
        help="Vehicle count for VRP mode. Default: 3",
    )
    parser.add_argument(
        "--vehicle-capacity",
        type=int,
        default=8,
        help="Per-truck delivery capacity for VRP mode. Default: 8",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    print("Loading route optimization graph...")
    graph = load_or_build_graph(force_rebuild=args.rebuild)

    if args.mode == "single":
        source_name, source = parse_location(args.source)
        destination_name, destination = parse_location(args.destination)

        print("Finding shortest route...")
        route = shortest_route(graph, source_name, source, destination_name, destination)

        print("\nRoute Optimization Engine Summary")
        print(f"Source: {route.source_name} ({source[0]:.6f}, {source[1]:.6f})")
        print(f"Destination: {route.destination_name} ({destination[0]:.6f}, {destination[1]:.6f})")
        print(f"Shortest path nodes: {len(route.path):,}")
        print(f"Shortest path distance: {route.distance_m / 1000:.2f} km")

        print("\nGenerating route visualization...")
        plot_route(graph, route)
        return

    depot_name, depot = parse_location(DEFAULT_VRP_DEPOT)
    deliveries = [parse_location(name) for name in DEFAULT_VRP_DELIVERY_NAMES]

    print("Solving VRP for depot + multiple deliveries...")
    vrp_result = solve_vrp(
        graph=graph,
        depot_name=depot_name,
        depot=depot,
        deliveries=deliveries,
        vehicle_count=args.vehicles,
        vehicle_capacity=args.vehicle_capacity,
    )

    location_lookup = {depot_name: depot}
    location_lookup.update(dict(deliveries))

    print("\nRoute Optimization Engine Summary")
    print(f"Depot: {vrp_result.depot_name} ({depot[0]:.6f}, {depot[1]:.6f})")
    print(f"Deliveries: {len(vrp_result.deliveries)}")
    print(f"Trucks: {len(vrp_result.routes)}")
    print(f"Total optimized distance: {vrp_result.total_distance_m / 1000:.2f} km")

    for route in vrp_result.routes:
        if route.stops:
            stop_text = " -> ".join([vrp_result.depot_name, *route.stops, vrp_result.depot_name])
        else:
            stop_text = f"{vrp_result.depot_name} -> {vrp_result.depot_name}"
        print(f"Truck {route.vehicle_id + 1}: {route.distance_m / 1000:.2f} km | {stop_text}")

    print("\nGenerating VRP visualization...")
    plot_vrp_routes(graph, vrp_result, location_lookup)


if __name__ == "__main__":
    main()