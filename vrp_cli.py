"""Interactive VRP CLI for the Mumbai-Pune route engine.

Usage:
  py vrp_cli.py

Prompts for:
  - number of trucks
  - warehouse (preset or lat,lon)
  - number of deliveries and each delivery location (preset or lat,lon)

Then runs the VRP solver in `main.py` and shows a static matplotlib plot and
an interactive `DATA/vrp_map.html` (if `folium` is installed).
"""
from __future__ import annotations

import sys
import os
import argparse
from typing import List, Tuple

try:
    import folium
except Exception:
    folium = None

import main as engine


def prompt_int(prompt: str, default: int) -> int:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            n = int(val)
            if n > 0:
                return n
        except ValueError:
            pass
        print("Please enter a positive integer.")


def prompt_location(prompt: str) -> Tuple[str, Tuple[float, float]]:
    presets = sorted(engine.PRESET_LOCATIONS.keys())
    print(f"Available presets: {', '.join(presets)}")
    while True:
        raw = input(f"{prompt}: ").strip()
        if not raw:
            print("Please enter a preset name or a lat,lon pair.")
            continue
        try:
            name, coords = engine.parse_location(raw)
            return name, coords
        except Exception as e:
            print(f"Invalid location: {e}")


def prompt_deliveries(count: int) -> List[Tuple[str, Tuple[float, float]]]:
    deliveries = []
    print(f"Enter {count} delivery locations (preset or lat,lon).")
    for i in range(count):
        name, coords = prompt_location(f"Delivery {i+1}")
        deliveries.append((name, coords))
    return deliveries


def build_and_solve(depot_name: str, depot: Tuple[float, float], deliveries, vehicles: int, capacity: int):
    print("Loading/creating the route graph (may take time on first run)...")
    graph = engine.load_or_build_graph(force_rebuild=False)

    print("Running VRP solver...")
    vrp = engine.solve_vrp(graph=graph, depot_name=depot_name, depot=depot, deliveries=deliveries, vehicle_count=vehicles, vehicle_capacity=capacity)

    lookup = {depot_name: depot}
    lookup.update(dict(deliveries))

    print("Rendering static plot (matplotlib)...")
    engine.plot_vrp_routes(graph, vrp, lookup)

    if folium is not None:
        print("Rendering interactive map (folium)...")
        # Use explicit tile layer and ensure map centers on all points
        m = folium.Map(location=[depot[0], depot[1]], zoom_start=9, tiles="OpenStreetMap")
        # depot marker
        folium.Marker(location=[depot[0], depot[1]], popup=depot_name, icon=folium.Icon(color="orange", icon="info-sign")).add_to(m)

        # collect non-empty routes and their coords
        plotted_routes = []
        all_coords = []
        for route in vrp.routes:
            coords = []
            for node_id in route.node_path:
                node = graph.nodes[node_id]
                coords.append((node["y"], node["x"]))
            if len(coords) >= 2:
                plotted_routes.append((route, coords))
                all_coords.extend(coords)

        def generate_colors(n: int) -> List[str]:
            # generate n visually distinct colors using HSV
            colors = []
            for i in range(n):
                h = i / max(1, n)
                s = 0.7
                v = 0.9
                # HSV to RGB
                import colorsys

                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
            return colors

        num = max(1, len(plotted_routes))
        palette = generate_colors(num)

        for idx, (route, coords) in enumerate(plotted_routes):
            # draw each route as individual segments with varying colors
            for seg_idx in range(len(coords) - 1):
                seg = [coords[seg_idx], coords[seg_idx + 1]]
                seg_color = palette[(idx + seg_idx) % len(palette)]
                folium.PolyLine(seg, color=seg_color, weight=5, opacity=0.9).add_to(m)

            # add small numbered markers at each node along the route
            for j, (lat, lon) in enumerate(coords):
                folium.CircleMarker(location=[lat, lon], radius=3, color="#222", fill=True, fill_color=palette[(idx + j) % len(palette)], fill_opacity=0.9, popup=f"Truck {route.vehicle_id + 1} - point {j+1}").add_to(m)

        for name, (lat, lon) in lookup.items():
            folium.CircleMarker(location=[lat, lon], radius=4, color="#333", fill=True, fill_opacity=0.7, popup=name).add_to(m)

        if all_coords:
            # fit map to all plotted coordinates
            lats = [c[0] for c in all_coords]
            lons = [c[1] for c in all_coords]
            sw = (min(lats), min(lons))
            ne = (max(lats), max(lons))
            m.fit_bounds([sw, ne])

        out_html = os.path.join("DATA", "vrp_map.html")
        os.makedirs(os.path.dirname(out_html), exist_ok=True)
        m.save(out_html)
        print(f"Saved interactive map to {out_html}")
    else:
        print("Skipping interactive map: folium not installed. Install with 'py -m pip install folium'")


def main():
    parser = argparse.ArgumentParser(description="Interactive VRP CLI for the route engine")
    parser.add_argument("--vehicles", type=int, default=3, help="Number of trucks")
    parser.add_argument("--capacity", type=int, default=8, help="Per-truck capacity")
    parser.add_argument("--depot", default=engine.DEFAULT_VRP_DEPOT, help="Depot preset or 'lat,lon'")
    parser.add_argument("--deliveries", type=int, default=20, help="Number of deliveries to prompt for")
    args = parser.parse_args()

    try:
        depot_name, depot_coords = engine.parse_location(args.depot)
    except Exception:
        depot_name, depot_coords = prompt_location("Depot")

    print(f"Depot: {depot_name} at {depot_coords}")

    # First ask how many deliveries (stops) the user wants to add, then collect them.
    num_deliveries = prompt_int("Number of deliveries (stops)", args.deliveries)
    deliveries = prompt_deliveries(num_deliveries)

    # After collecting stops, ask for number of trucks.
    vehicles = prompt_int("Number of trucks", args.vehicles)

    build_and_solve(depot_name, depot_coords, deliveries, vehicles, args.capacity)


if __name__ == "__main__":
    main()
