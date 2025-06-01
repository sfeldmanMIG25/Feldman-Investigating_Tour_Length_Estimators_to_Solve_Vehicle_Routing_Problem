#!/usr/bin/env python3
"""
run_with_heuristic_K_means.py

Same as run_with_heuristic.py, but uses a capacity‐aware K‐means heuristic to generate an initial CVRP solution,
then sums Çavdar–Sokol’s TSP estimates over those K clusters (including depot) to form a tighter cutoff bound.
Additionally, writes all decision variables and the objective into separate temporary files from AMPL,
then appends them to a single <basename>.txt (along with the solver log). Prints execution time for each instance.

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
  python3 run_with_heuristic_K_means.py
"""

import os
import random
import subprocess
import re
import math
import time
import matplotlib.pyplot as plt

# --- PARAMETERS ---
NUM_SAMPLES = 100
TIME_LIMIT = 5 * 3600  # 5 hours in seconds
MAX_KMEANS_ITERS = 50
CONVERGENCE_TOL = 1e-4

# --- HELPER: PARSE VRP FOR COORDS & DEMANDS & DEPOT ---
def parse_vrp_coords_demands(vrp_path):
    coords = {}
    demands = {}
    depot_id = None
    mode = None

    with open(vrp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "NODE_COORD_SECTION":
                mode = "coords"; continue
            if line == "DEMAND_SECTION":
                mode = "demand"; continue
            if line == "DEPOT_SECTION":
                mode = "depot"; continue

            if mode == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    nid = int(parts[0]); x = float(parts[1]); y = float(parts[2])
                    coords[nid] = (x, y)
                continue
            if mode == "demand":
                parts = line.split()
                if len(parts) >= 2:
                    nid = int(parts[0]); d = int(parts[1])
                    demands[nid] = d
                continue
            if mode == "depot":
                nid = int(line)
                if nid == -1: break
                depot_id = nid
                continue

    if depot_id is None:
        raise ValueError(f"No DEPOT_SECTION in {vrp_path}")
    return coords, demands, depot_id

# --- ÇAVDAR–SOKOL g(TSP) FOR A SET OF COORDS ---
def compute_tsp_estimate(node_list, coords):
    pts = [coords[nid] for nid in node_list]
    n = len(pts)
    if n == 0:
        return 0.0

    sum_x = sum(pt[0] for pt in pts)
    sum_x2 = sum(pt[0]**2 for pt in pts)
    sum_y = sum(pt[1] for pt in pts)
    sum_y2 = sum(pt[1]**2 for pt in pts)

    mu_x = sum_x / n
    mu_y = sum_y / n

    var_x = sum_x2 / n - mu_x**2
    var_y = sum_y2 / n - mu_y**2
    stdev_x = math.sqrt(max(var_x, 0.0))
    stdev_y = math.sqrt(max(var_y, 0.0))

    abs_devs_x = [abs(pt[0] - mu_x) for pt in pts]
    sum_abs_x = sum(abs_devs_x)
    bar_c_x = sum_abs_x / n

    abs_devs_y = [abs(pt[1] - mu_y) for pt in pts]
    sum_abs_y = sum(abs_devs_y)
    bar_c_y = sum_abs_y / n

    sum_abs_x2 = sum(dev**2 for dev in abs_devs_x)
    cstdev_x = math.sqrt(max(sum_abs_x2 / n - bar_c_x**2, 0.0))

    sum_abs_y2 = sum(dev**2 for dev in abs_devs_y)
    cstdev_y = math.sqrt(max(sum_abs_y2 / n - bar_c_y**2, 0.0))

    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    x_max, x_min = max(xs), min(xs)
    y_max, y_min = max(ys), min(ys)
    area_rect = (x_max - x_min) * (y_max - y_min)

    alpha = 2.791
    beta = 0.2669
    term1 = alpha * math.sqrt(n * (cstdev_x * cstdev_y))
    term2 = beta * math.sqrt(n * (stdev_x * stdev_y) * (area_rect / (bar_c_x * bar_c_y + 1e-12)))
    return term1 + term2

# --- K‐MEANS CLUSTERING (CAPACITY‐AWARE) ---
def kmeans_capacity(coords, demands, depot_id, K, Q):
    customers = [nid for nid in coords if nid != depot_id]
    centers = random.sample([coords[i] for i in customers], K)

    for iteration in range(MAX_KMEANS_ITERS):
        clusters = {k: [] for k in range(K)}
        for i in customers:
            x_i, y_i = coords[i]
            best_k = min(range(K), key=lambda k: (x_i - centers[k][0])**2 + (y_i - centers[k][1])**2)
            clusters[best_k].append(i)

        new_centers = []
        for k in range(K):
            pts = [coords[i] for i in clusters[k]]
            if pts:
                mean_x = sum(pt[0] for pt in pts) / len(pts)
                mean_y = sum(pt[1] for pt in pts) / len(pts)
                new_centers.append((mean_x, mean_y))
            else:
                rand_cust = random.choice(customers)
                new_centers.append(coords[rand_cust])

        max_shift = max(
            math.hypot(centers[k][0] - new_centers[k][0], centers[k][1] - new_centers[k][1]) 
            for k in range(K)
        )
        centers = new_centers
        if max_shift < CONVERGENCE_TOL:
            break

    cluster_demands = {k: sum(demands[i] for i in clusters[k]) for k in range(K)}
    over_loaded = [k for k, total in cluster_demands.items() if total > Q]
    while over_loaded:
        for k in over_loaded:
            members = sorted(clusters[k], key=lambda i: demands[i], reverse=True)
            for i in members:
                for j in range(K):
                    if j == k:
                        continue
                    if cluster_demands[j] + demands[i] <= Q:
                        clusters[k].remove(i)
                        clusters[j].append(i)
                        cluster_demands[k] -= demands[i]
                        cluster_demands[j] += demands[i]
                        break
                if cluster_demands[k] <= Q:
                    break
        over_loaded = [k for k, total in cluster_demands.items() if total > Q]

    return clusters

# --- COMPUTE K‐MEANS BOUND ---
def compute_kmeans_bound(vrp_path, K):
    coords, demands, depot_id = parse_vrp_coords_demands(vrp_path)
    Q = None
    with open(vrp_path, 'r') as f:
        for line in f:
            if line.strip().startswith("CAPACITY"):
                parts = line.replace(":", " ").split()
                Q = int(parts[-1])
                break
    if Q is None:
        raise ValueError(f"No CAPACITY found in {vrp_path}")

    clusters = kmeans_capacity(coords, demands, depot_id, K, Q)
    total_est = 0.0
    for k in range(K):
        node_list = [depot_id] + clusters.get(k, [])
        total_est += compute_tsp_estimate(node_list, coords)
    return total_est

# --- PARSING ASSIGN & TOTALCOST FROM TEMP FILES ---
def parse_temp_assign(file_path):
    """
    Parses a file with lines: i k val
    Returns dict[(i,k)] = val
    """
    assign = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                i = int(parts[0]); k = int(parts[1]); val = int(parts[2])
                assign[(i, k)] = val
    return assign

def parse_temp_cost(file_path):
    """
    Parses a file containing "TotalCost = <val>"
    Returns float
    """
    with open(file_path, 'r') as f:
        for line in f:
            if "TotalCost" in line:
                parts = line.replace("=", " ").split()
                for p in parts:
                    try:
                        return float(p)
                    except:
                        continue
    return None

# --- PLOTTING FUNCTION ---
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
                mode = "coords"; continue
            if line == "DEPOT_SECTION":
                mode = "depot"; continue
            if mode == "coords":
                parts = line.split()
                if len(parts) >= 3:
                    node = int(parts[0]); x = float(parts[1]); y = float(parts[2])
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
    ax.set_xlabel('X Coord'); ax.set_ylabel('Y Coord')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# --- MAIN SCRIPT ---
def main():
    script_dir     = os.path.abspath(os.path.dirname(__file__))
    data_dir       = os.path.abspath(os.path.join(script_dir, os.pardir))
    data_input_dir = os.path.join(data_dir, "Input")
    instances_dir  = os.path.join(data_dir, "instances")
    models_dir     = os.path.abspath(os.path.join(data_dir, os.pardir, "Models"))
    solheur_dir    = os.path.abspath(os.path.join(data_dir, os.pardir, "Solution_Heuristic"))
    visuals_dir    = os.path.join(solheur_dir, "Visual")
    cvrp_mod       = os.path.join(models_dir, "CVRP_Cheap_1.mod")

    # Debug: print paths
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

    os.makedirs(solheur_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    try:
        all_dat = [f for f in os.listdir(data_input_dir) if f.lower().endswith(".dat")]
    except FileNotFoundError:
        print(f"ERROR: Cannot find directory {data_input_dir}. Exiting.")
        return

    if not all_dat:
        print("No .dat files found in Data/Input/. Exiting.")
        return

    sample_files = random.sample(all_dat, min(NUM_SAMPLES, len(all_dat)))

    for dat_file in sample_files:
        basename = os.path.splitext(dat_file)[0]
        dat_path = os.path.join(data_input_dir, dat_file)
        vrp_path = os.path.join(instances_dir, f"{basename}.vrp")

        print(f"\n--- Processing {basename} ---")
        print(f"  Looking for VRP: {vrp_path}")

        if not os.path.isfile(vrp_path):
            print(f"  [ERROR] Missing instance VRP: {vrp_path}. Skipping.")
            continue

        # Read K from .dat
        K = None
        with open(dat_path, 'r') as f:
            for line in f:
                if line.strip().startswith("param K"):
                    parts = line.replace(";", "").split()
                    if ":=" in parts:
                        idx = parts.index(":=")
                        K = int(parts[idx + 1])
                    break
        if K is None:
            print(f"  [ERROR] Couldn't read K from {dat_path}. Skipping.")
            continue
        print(f"  Read K = {K} from .dat")

        # 1) Compute K‐means cutoff bound
        try:
            kmeans_bound = compute_kmeans_bound(vrp_path, K)
            print(f"  Computed K‐means bound = {kmeans_bound:.4f}")
        except Exception as e:
            print(f"  [ERROR] compute_kmeans_bound failed for {basename}: {e}")
            continue

        # Prepare temp file paths
        temp_assign = os.path.join(solheur_dir, f"{basename}_assign.txt")
        temp_csize  = os.path.join(solheur_dir, f"{basename}_csize.txt")
        temp_cost   = os.path.join(solheur_dir, f"{basename}_cost.txt")
        main_txt    = os.path.join(solheur_dir, f"{basename}.txt")

        # 2) Build AMPL run file with WRITE commands
        run_contents = f"""
model "{cvrp_mod}";
data "{dat_path}";
option solver gurobi;
option gurobi_options "Cutoff={kmeans_bound:.6f} TimeLimit={TIME_LIMIT}";
solve;
write assign > "{temp_assign}";
write cluster_size > "{temp_csize}";
print "TotalCost = ", TotalCost > "{temp_cost}";
quit;
"""
        run_file = os.path.join(script_dir, f"run_{basename}.run")
        with open(run_file, "w") as rf:
            rf.write(run_contents)

        # 3) Call AMPL, capture solver log, measure time
        print(f"  → Launching AMPL for {basename} ...")
        start_time = time.time()
        proc = subprocess.Popen(
            ["ampl", run_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        solver_log = []
        with open(main_txt, "w") as outfile:
            for line in proc.stdout:
                outfile.write(line)
                solver_log.append(line.rstrip("\n"))
        proc.wait()
        elapsed = time.time() - start_time
        print(f"  → AMPL finished with return code {proc.returncode} (Elapsed: {elapsed:.1f} sec)")

        # 4) Read temp files for variables & cost
        if proc.returncode != 0:
            print(f"  [ERROR] AMPL returned code {proc.returncode}. Dumping full solver log below:")
            for ln in solver_log:
                print("    " + ln)
            print(f"  (Full log in {main_txt})\n")
            continue

        # Parse assign
        try:
            assign_dict = parse_temp_assign(temp_assign)
        except Exception as e:
            print(f"  [ERROR] Failed to parse assign file: {e}")
            continue

        # Parse cluster_size (optional, but read anyway)
        cluster_size_dict = {}
        try:
            with open(temp_csize, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].startswith("cluster_size"):
                        # format: cluster_size[k] <value>
                        inside = parts[0].strip("cluster_size[]")
                        k = int(inside)
                        val = int(parts[1])
                        cluster_size_dict[k] = val
        except:
            pass

        # Parse cost
        total_cost = None
        try:
            total_cost = parse_temp_cost(temp_cost)
        except Exception as e:
            print(f"  [ERROR] Failed to parse cost file: {e}")

        if total_cost is None:
            print(f"  [ERROR] Couldn't read TotalCost from {temp_cost}.")
            continue
        print(f"  Parsed TotalCost = {total_cost:.4f}")

        # 5) Append decision variables & cost to main_txt
        with open(main_txt, "a") as outfile:
            outfile.write("\n=== ASSIGN ===\n")
            with open(temp_assign, 'r') as fa:
                outfile.write(fa.read())
            outfile.write("\n=== CLUSTER_SIZE ===\n")
            with open(temp_csize, 'r') as fc:
                outfile.write(fc.read())
            outfile.write("\n=== TOTAL_COST ===\n")
            with open(temp_cost, 'r') as ft:
                outfile.write(ft.read())

        # 6) Plot solution
        visu_outpath = os.path.join(visuals_dir, f"{basename}.png")
        try:
            plot_solution(basename, vrp_path, assign_dict, total_cost, visu_outpath)
            print(f"  → Saved visualization at {visu_outpath}")
        except Exception as e:
            print(f"  [ERROR] Plotting failed for {basename}: {e}")

        # 7) Clean up run file and temp files
        for fpath in [run_file, temp_assign, temp_csize, temp_cost]:
            try:
                os.remove(fpath)
            except OSError:
                pass

    print("\nAll done.")

if __name__ == "__main__":
    main()
 