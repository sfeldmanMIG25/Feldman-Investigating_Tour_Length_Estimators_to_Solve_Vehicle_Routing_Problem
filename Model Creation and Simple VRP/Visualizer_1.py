# Visualizer_1.py

import matplotlib.pyplot as plt
from pathlib import Path

def parse_assignments(solution_file):
    assignments = {}
    lines = solution_file.read_text().splitlines()

    # 1) Find the line that starts with 'assign'
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("assign"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find 'assign' section in solution file.")

    # 2) The actual data rows begin two lines after 'assign'
    data_idx = start_idx + 2

    # 3) Read until the line that is just a semicolon (';')
    for line in lines[data_idx:]:
        stripped = line.strip()
        if stripped == ";" or stripped == "":
            break
        parts = stripped.split()
        # parts[0] = node index, parts[1..m] = binary values for each vehicle
        node = int(parts[0])
        # remaining columns correspond to vehicle 1, 2, ..., K
        for vehicle_idx, val in enumerate(parts[1:], start=1):
            if val == "1":
                assignments[node] = vehicle_idx
                break

    return assignments

def parse_coordinates(dat_file):
    x_coords = {}
    y_coords = {}
    depot = None

    lines = dat_file.read_text().splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # Parse x block
        if stripped.startswith("param x :="):
            i += 1
            while i < len(lines) and lines[i].strip() != ";":
                parts = lines[i].split()
                if len(parts) == 2:
                    idx = int(parts[0])
                    val = float(parts[1])
                    x_coords[idx] = val
                i += 1
        # Parse y block
        elif stripped.startswith("param y :="):
            i += 1
            while i < len(lines) and lines[i].strip() != ";":
                parts = lines[i].split()
                if len(parts) == 2:
                    idx = int(parts[0])
                    val = float(parts[1])
                    y_coords[idx] = val
                i += 1
        else:
            i += 1

    # Extract depot (index 0), if present
    if 0 in x_coords and 0 in y_coords:
        depot = (x_coords.pop(0), y_coords.pop(0))
    return x_coords, y_coords, depot

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    solution_file = script_dir / "heuristic_solution.txt"
    dat_file      = script_dir / "cheap_problem.dat"

    if not solution_file.exists():
        raise FileNotFoundError(f"Cannot find {solution_file}")
    if not dat_file.exists():
        raise FileNotFoundError(f"Cannot find {dat_file}")

    # 1) Read assignments and coordinates
    assignments = parse_assignments(solution_file)
    x_coords, y_coords, depot = parse_coordinates(dat_file)

    # 2) Plot
    plt.figure(figsize=(8, 6))

    vehicles = sorted(set(assignments.values()))
    cmap = plt.get_cmap("tab10", len(vehicles))

    # Plot each customer, colored by its assigned vehicle
    for node, veh in assignments.items():
        plt.scatter(
            x_coords[node],
            y_coords[node],
            color=cmap(veh - 1),
            s=50,
            label=f"Vehicle {veh}" if node == min(n for n, v in assignments.items() if v == veh) else ""
        )

    # Plot depot as a black star
    if depot is not None:
        plt.scatter(depot[0], depot[1], color="black", marker="*", s=150, label="Depot")

    plt.title("VRP Customer Assignments (Heuristic)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
