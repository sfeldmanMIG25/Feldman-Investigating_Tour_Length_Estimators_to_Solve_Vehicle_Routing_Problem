#!/usr/bin/env python3
"""
xml_to_dat.py

Convert all TSPLIB‐format CVRP instance files (.vrp) in "../instances" into AMPL‐ready .dat files in "../Input".
Number of vehicles K is inferred by counting “Route #” lines in the matching ".sol" file under "../solutions".

Directory structure (relative to this script):
  project/
    instances/       # contains all .vrp files (e.g., XML100_1111_01.vrp, etc.)
    solutions/       # contains corresponding .sol files (e.g., XML100_1111_01.sol)
    Scripts/         # this script lives here
    Input/           # output folder for generated .dat files

Usage:
  cd Scripts
  python3 xml_to_dat.py

Output:
  For each basename.vrp in instances/, produce Input/basename.dat
"""

import os
import sys

def parse_vrp(vrp_path):
    """
    Parse a TSPLIB CVRP file at vrp_path.
    Returns:
      capacity (int),
      depot_id (int),
      coords (dict: node_id -> (x, y)),
      demands (dict: node_id -> demand)
    """
    capacity = None
    coords = {}
    demands = {}
    depot_id = None

    mode = None
    with open(vrp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header: look for CAPACITY
            if line.startswith("CAPACITY"):
                # Format: "CAPACITY : 4"
                parts = line.replace(':', ' ').split()
                capacity = int(parts[-1])
                continue

            # Mark section changes
            if line == "NODE_COORD_SECTION":
                mode = "coords"
                continue
            if line == "DEMAND_SECTION":
                mode = "demand"
                continue
            if line == "DEPOT_SECTION":
                mode = "depot"
                continue

            # Parse NODE_COORD_SECTION
            if mode == "coords":
                parts = line.split()
                # Expect: node_id  x  y
                if len(parts) >= 3:
                    node = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[node] = (x, y)
                continue

            # Parse DEMAND_SECTION
            if mode == "demand":
                parts = line.split()
                # Expect: node_id  demand
                if len(parts) >= 2:
                    node = int(parts[0])
                    d = int(parts[1])
                    demands[node] = d
                continue

            # Parse DEPOT_SECTION
            if mode == "depot":
                node = int(line)
                if node == -1:
                    break
                depot_id = node
                continue

    if capacity is None:
        raise ValueError(f"No CAPACITY found in {vrp_path}")
    if depot_id is None:
        raise ValueError(f"No DEPOT_SECTION found in {vrp_path}")
    return capacity, depot_id, coords, demands

def count_routes(sol_path):
    """
    Count the number of routes by counting lines that start with "Route".
    Returns K (int).
    """
    K = 0
    with open(sol_path, 'r') as f:
        for line in f:
            if line.strip().startswith("Route"):
                K += 1
    return K

def write_dat_file(basename, capacity, depot_id, coords, demands, K, output_dir):
    """
    Write an AMPL .dat file for basename into output_dir.
    """
    # Customers are all nodes except depot
    customer_ids = sorted([nid for nid in coords.keys() if nid != depot_id])

    # Depot coordinates
    x0, y0 = coords[depot_id]

    dat_path = os.path.join(output_dir, f"{basename}.dat")
    with open(dat_path, 'w') as f:
        f.write("data;\n\n")

        # Vehicle capacity and number of vehicles
        f.write(f"param Q := {capacity};\n")
        f.write(f"param K := {K};\n\n")

        # Depot coordinates
        f.write(f"param x0 := {x0};\n")
        f.write(f"param y0 := {y0};\n\n")

        # Customer set
        f.write("set N := ")
        for nid in customer_ids:
            f.write(f"{nid} ")
        f.write(";\n\n")

        # Customer x‐coordinates
        f.write("param x :=\n")
        for nid in customer_ids:
            x_val = coords[nid][0]
            f.write(f"{nid} {x_val}\n")
        f.write(";\n\n")

        # Customer y‐coordinates
        f.write("param y :=\n")
        for nid in customer_ids:
            y_val = coords[nid][1]
            f.write(f"{nid} {y_val}\n")
        f.write(";\n\n")

        # Customer demands
        f.write("param d :=\n")
        for nid in customer_ids:
            d_val = demands.get(nid, 0)
            f.write(f"{nid} {d_val}\n")
        f.write(";\n")

def main():
    # Directories relative to this script
    script_dir = os.path.abspath(os.path.dirname(__file__))
    instances_dir = os.path.join(script_dir, os.pardir, "instances")
    solutions_dir = os.path.join(script_dir, os.pardir, "solutions")
    input_dir = os.path.join(script_dir, os.pardir, "Input")

    os.makedirs(input_dir, exist_ok=True)

    # Iterate over all .vrp files in instances/
    for filename in os.listdir(instances_dir):
        if not filename.lower().endswith(".vrp"):
            continue
        basename = os.path.splitext(filename)[0]
        vrp_path = os.path.join(instances_dir, filename)
        sol_path = os.path.join(solutions_dir, f"{basename}.sol")

        if not os.path.isfile(sol_path):
            print(f"Warning: no solution file found for {filename}, skipping.")
            continue

        try:
            capacity, depot_id, coords, demands = parse_vrp(vrp_path)
            K = count_routes(sol_path)
            write_dat_file(basename, capacity, depot_id, coords, demands, K, input_dir)
            print(f"Converted {filename} → {basename}.dat (K={K}, Q={capacity})")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
