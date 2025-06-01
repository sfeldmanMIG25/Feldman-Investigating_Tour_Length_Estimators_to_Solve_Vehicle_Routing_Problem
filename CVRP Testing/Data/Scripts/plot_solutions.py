#!/usr/bin/env python3
"""
plot_solutions.py

For each CVRP solution file in "../solutions", read the matching instance from "../instances",
and produce a node‐colored plot with:
  - Depot as a red star
  - Each customer node colored according to its assigned vehicle (no connecting lines)
  - Title: "<basename> — Cost: <objective_value>"
Save each figure as PNG under "../solutions/Visuals/<basename>.png".

Directory structure (relative to this script):
  project/
    instances/       # contains all .vrp files
    solutions/       # contains all .sol files
      Visuals/       # (will be created) where .png outputs go
    Scripts/         # this script lives here

Usage:
  cd Scripts
  python3 plot_solutions.py
"""

import os
import matplotlib.pyplot as plt

def parse_vrp(vrp_path):
    """
    Parse a TSPLIB CVRP file to extract:
      - depot_id
      - coords: dict[node_id] = (x, y)
    """
    coords = {}
    depot_id = None
    mode = None

    with open(vrp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "NODE_COORD_SECTION":
                mode = "coords"
                continue
            if line == "DEPOT_SECTION":
                mode = "depot"
                continue

            if mode == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    node = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[node] = (x, y)
                continue

            if mode == "depot":
                node = int(line)
                if node == -1:
                    break
                depot_id = node
                continue

    if depot_id is None:
        raise ValueError(f"No DEPOT_SECTION found in {vrp_path}")
    return depot_id, coords

def parse_sol(sol_path):
    """
    Parse a solution file to extract:
      - routes: list of lists of node IDs (int)
      - cost: float or int
    """
    routes = []
    cost = None

    with open(sol_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Route"):
                # e.g. "Route #1: 98 46 15 73"
                parts = line.split(":")
                if len(parts) >= 2:
                    node_str = parts[1].strip()
                    if node_str:
                        node_ids = [int(x) for x in node_str.split()]
                        routes.append(node_ids)
                continue
            if line.startswith("Cost"):
                # e.g. "Cost 29888"
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        cost = float(parts[1])
                    except ValueError:
                        cost = parts[1]
                continue

    if cost is None:
        raise ValueError(f"No Cost found in {sol_path}")
    return routes, cost

def plot_solution(basename, depot_id, coords, routes, cost, output_path):
    """
    Create a matplotlib plot:
      - Depot as red star
      - Each customer node plotted as a colored dot according to route assignment
      - Title includes basename and cost
    Save to output_path (PNG).
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Extract depot coordinates and plot it
    x0, y0 = coords[depot_id]
    ax.scatter(x0, y0, c='red', marker='*', s=100)

    # Choose a colormap with enough distinct colors
    cmap = plt.get_cmap('tab20')
    num_routes = len(routes)

    # Plot each customer node in the color of its assigned vehicle
    for k, route in enumerate(routes, start=1):
        xs = []
        ys = []
        for node in route:
            x_i, y_i = coords[node]
            xs.append(x_i)
            ys.append(y_i)
        color = cmap((k - 1) % 20)
        ax.scatter(xs, ys, c=[color], s=20)

    # Title
    ax.set_title(f"{basename}  —  Cost: {cost}", fontsize=12)

    # Equal aspect ratio
    ax.set_aspect('equal', 'box')

    # Axis labels (optional)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    instances_dir = os.path.join(script_dir, os.pardir, "instances")
    solutions_dir = os.path.join(script_dir, os.pardir, "solutions")
    visuals_dir = os.path.join(solutions_dir, "Visuals")

    os.makedirs(visuals_dir, exist_ok=True)

    # Loop over all .sol files in solutions/
    for filename in os.listdir(solutions_dir):
        if not filename.lower().endswith(".sol"):
            continue
        basename = os.path.splitext(filename)[0]
        sol_path = os.path.join(solutions_dir, filename)
        vrp_path = os.path.join(instances_dir, f"{basename}.vrp")

        if not os.path.isfile(vrp_path):
            print(f"Warning: instance file not found for {filename}, skipping.")
            continue

        try:
            # Parse instance and solution
            depot_id, coords = parse_vrp(vrp_path)
            routes, cost = parse_sol(sol_path)

            # Output PNG path
            output_path = os.path.join(visuals_dir, f"{basename}.png")

            # Plot and save
            plot_solution(basename, depot_id, coords, routes, cost, output_path)
            print(f"Plotted {basename} → Visuals/{basename}.png")
        except Exception as e:
            print(f"Error processing {basename}: {e}")

if __name__ == "__main__":
    main()
