#!/usr/bin/env python3
"""
run_with_heuristic.py  (Debug‐enhanced)

Selects up to 100 random .dat files from Data/Input/, then for each:
  1. Computes Çavdar–Sokol’s TSP estimate (g_full) from Data/instances/<basename>.vrp.
  2. Invokes AMPL with Models/CVRP_Cheap_1.mod and the .dat, setting Gurobi’s Cutoff=g_full and TimeLimit=18000.
  3. Captures ALL AMPL+Gurobi output into Solution_Heuristic/<basename>.txt.
  4. Parses out “assign[i,k]” and “TotalCost” from the AMPL display.
     – If parsing fails or return code ≠ 0, extra debugging info is printed.
  5. If solve succeeded, plots node‐colored solution as before.

Directory structure (relative to this script, which lives in Data/Scripts/):
  project/
    Data/
      Input/          # contains *.dat
      Scripts/        # this script
      instances/      # contains *.vrp
    Models/
      CVRP_Cheap_1.mod
    Solution_Heuristic/
      Visual/         # (will be created) for PNGs

Usage:
  cd Data/Scripts
  python3 run_with_heuristic.py
"""

import os
import random
import subprocess
import re
import math
import matplotlib.pyplot as plt

# --- PARAMETERS ---
NUM_SAMPLES = 100
TIME_LIMIT = 5 * 3600  # 5 hours in seconds

# --- HELPERS FOR ÇAVDAR–SOKOL ’G’ COMPUTATION ---
def compute_g_full(vrp_path, alpha=2.791, beta=0.2669):
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
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[nid] = (x, y)
                continue
            if mode == "depot":
                nid = int(line)
                if nid == -1:
                    break
                depot_id = nid
                continue

    if depot_id is None:
        raise ValueError(f"No DEPOT_SECTION in {vrp_path}")

    x0, y0 = coords[depot_id]
    all_coords = [(x0, y0)] + [coords[i] for i in coords if i != depot_id]
    n = len(all_coords)

    sum_x = sum(pt[0] for pt in all_coords)
    sum_x2 = sum(pt[0]**2 for pt in all_coords)
    sum_y = sum(pt[1] for pt in all_coords)
    sum_y2 = sum(pt[1]**2 for pt in all_coords)

    mu_x = sum_x / n
    mu_y = sum_y / n

    var_x = sum_x2 / n - mu_x**2
    var_y = sum_y2 / n - mu_y**2
    stdev_x = math.sqrt(max(var_x, 0.0))
    stdev_y = math.sqrt(max(var_y, 0.0))

    abs_devs_x = [abs(pt[0] - mu_x) for pt in all_coords]
    sum_abs_x = sum(abs_devs_x)
    bar_c_x = sum_abs_x / n

    abs_devs_y = [abs(pt[1] - mu_y) for pt in all_coords]
    sum_abs_y = sum(abs_devs_y)
    bar_c_y = sum_abs_y / n

    sum_abs_x2 = sum(dev**2 for dev in abs_devs_x)
    cstdev_x = math.sqrt(max(sum_abs_x2 / n - bar_c_x**2, 0.0))

    sum_abs_y2 = sum(dev**2 for dev in abs_devs_y)
    cstdev_y = math.sqrt(max(sum_abs_y2 / n - bar_c_y**2, 0.0))

    xs = [pt[0] for pt in all_coords]
    ys = [pt[1] for pt in all_coords]
    x_max, x_min = max(xs), min(xs)
    y_max, y_min = max(ys), min(ys)
    area_rect = (x_max - x_min) * (y_max - y_min)

    term1 = alpha * math.sqrt(n * (cstdev_x * cstdev_y))
    term2 = beta * math.sqrt(n * (stdev_x * stdev_y) * (area_rect / (bar_c_x * bar_c_y + 1e-12)))
    g_full = term1 + term2
    return g_full


# --- PARSING ASSIGN & TOTALCOST FROM AMPL OUTPUT ---
def parse_ampl_output(lines):
    assign = {}
    cluster_size = {}
    total_cost = None

    assign_pattern = re.compile(r'^\s*\[(\d+),\s*(\d+)\]\s+([01])')
    csize_pattern = re.compile(r'^\s*cluster_size\[(\d+)\]\s*=\s*(\d+)')
    cost_pattern = re.compile(r'^\s*TotalCost\s*=\s*([0-9.+-eE]+)')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = assign_pattern.match(line)
        if m:
            i = int(m.group(1))
            k = int(m.group(2))
            val = int(m.group(3))
            assign[(i, k)] = val
            continue

        m2 = csize_pattern.match(line)
        if m2:
            k = int(m2.group(1))
            val = int(m2.group(2))
            cluster_size[k] = val
            continue

        m3 = cost_pattern.match(line)
        if m3:
            total_cost = float(m3.group(1))
            continue

    return assign, cluster_size, total_cost


# --- PLOTTING FUNCTION (re‐used) ---
def plot_solution(basename, vrp_path, assign, total_cost, output_path):
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
        raise ValueError(f"No depot found in {vrp_path}")

    clusters = {}
    for (i, k), val in assign.items():
        if val == 1:
            clusters.setdefault(k, []).append(i)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    x0, y0 = coords[depot_id]
    ax.scatter(x0, y0, c='red', marker='*', s=100)

    cmap = plt.get_cmap('tab20')
    for k, nodes in clusters.items():
        xs = [coords[i][0] for i in nodes]
        ys = [coords[i][1] for i in nodes]
        color = cmap((k - 1) % 20)
        ax.scatter(xs, ys, c=[color], s=20)

    ax.set_title(f"{basename}  —  Cost: {total_cost}", fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Coord')
    ax.set_ylabel('Y Coord')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# --- MAIN SCRIPT ---
def main():
    # 1) Determine script_dir and all related directories (absolute paths)
    script_dir = os.path.abspath(os.path.dirname(__file__))

    data_dir        = os.path.abspath(os.path.join(script_dir, os.pardir))
    data_input_dir  = os.path.join(data_dir, "Input")
    instances_dir   = os.path.join(data_dir, "instances")
    models_dir      = os.path.abspath(os.path.join(data_dir, os.pardir, "Models"))
    solheur_dir     = os.path.abspath(os.path.join(data_dir, os.pardir, "Solution_Heuristic"))
    visuals_dir     = os.path.join(solheur_dir, "Visual")
    cvrp_mod        = os.path.join(models_dir, "CVRP_Cheap_1.mod")

    # 1a) Debug: print out all paths
    print("===== Path Configuration =====")
    print(f"  script_dir        = {script_dir}")
    print(f"  data_dir          = {data_dir}")
    print(f"  Input dir         = {data_input_dir}")
    print(f"  instances_dir     = {instances_dir}")
    print(f"  models_dir        = {models_dir}")
    print(f"  CVRP mod file     = {cvrp_mod}")
    print(f"  Solution_Heuristic = {solheur_dir}")
    print(f"  Visual subdir     = {visuals_dir}")
    print("================================\n")

    # 2) Create output folders if missing
    os.makedirs(solheur_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    # 3) List all .dat files under Data/Input/
    try:
        all_dat = [f for f in os.listdir(data_input_dir) if f.lower().endswith(".dat")]
    except FileNotFoundError:
        print(f"ERROR: Cannot find directory {data_input_dir}. Exiting.")
        return

    if not all_dat:
        print("No .dat files found in Data/Input/. Exiting.")
        return

    # 4) Sample up to NUM_SAMPLES
    sample_files = random.sample(all_dat, min(NUM_SAMPLES, len(all_dat)))

    # 5) Process each sampled .dat
    for dat_file in sample_files:
        basename = os.path.splitext(dat_file)[0]
        dat_path = os.path.join(data_input_dir, dat_file)
        vrp_path = os.path.join(instances_dir, f"{basename}.vrp")

        print(f"\n--- Processing {basename} ---")
        print(f"  Looking for VRP: {vrp_path}")

        if not os.path.isfile(vrp_path):
            print(f"  [ERROR] Missing instance VRP: {vrp_path}. Skipping.")
            continue

        # 5a) Compute Çavdar–Sokol g_full
        try:
            g_full = compute_g_full(vrp_path)
            print(f"  Computed g_full (TSP estimate) = {g_full:.4f}")
        except Exception as e:
            print(f"  [ERROR] Failed to compute g_full for {basename}: {e}")
            continue

        # 5b) Build an AMPL run file (.run)
        run_contents = f"""
model "{cvrp_mod}";
data "{dat_path}";
option solver gurobi;
# Set Gurobi cutoff and time limit:
option gurobi_options "Cutoff={g_full:.6f} TimeLimit={TIME_LIMIT}";
solve;
display assign;
display cluster_size;
display TotalCost;
quit;
"""
        run_file = os.path.join(script_dir, f"run_{basename}.run")
        with open(run_file, "w") as rf:
            rf.write(run_contents)

        # 5c) Call AMPL, capture all output
        txt_outpath = os.path.join(solheur_dir, f"{basename}.txt")
        with open(txt_outpath, "w") as outfile:
            print(f"  → Launching AMPL for {basename} ...")
            proc = subprocess.Popen(
                ["ampl", run_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            all_lines = []
            for line in proc.stdout:
                outfile.write(line)
                all_lines.append(line.rstrip("\n"))
            proc.wait()
            print(f"  → AMPL finished with return code {proc.returncode}")

        # 5d) Parse assign, cluster_size, TotalCost
        assign_dict, cluster_size_dict, total_cost = parse_ampl_output(all_lines)

        # If no TotalCost or return code != 0, dump debug info
        if total_cost is None or proc.returncode != 0:
            print(f"  [ERROR] Could not parse TotalCost or nonzero return code for {basename}.")
            print("  --- RUN FILE CONTENTS ---")
            print(run_contents.strip())
            print("  --- AMPL OUTPUT (first 50 lines) ---")
            for ln in all_lines[:50]:
                print("    " + ln)
            print("  (See full log in", txt_outpath + ")\n")
            # Skip plotting for this instance
            continue

        print(f"  Parsed TotalCost = {total_cost:.4f}")

        # 5e) Plot solution
        visu_outpath = os.path.join(visuals_dir, f"{basename}.png")
        try:
            plot_solution(basename, vrp_path, assign_dict, total_cost, visu_outpath)
            print(f"  → Saved visualization at {visu_outpath}")
        except Exception as e:
            print(f"  [ERROR] Plotting failed for {basename}: {e}")

        # 5f) Clean up run file
        try:
            os.remove(run_file)
        except OSError:
            pass

    print("\nAll done.")


if __name__ == "__main__":
    main()
