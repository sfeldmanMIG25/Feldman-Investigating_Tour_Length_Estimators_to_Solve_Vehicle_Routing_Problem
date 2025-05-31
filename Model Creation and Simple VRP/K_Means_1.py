# K_Means_1.py
#!/usr/bin/env python3

import random
import math
import time
from pathlib import Path
import matplotlib.pyplot as plt

def parse_data(dat_file):
    lines = dat_file.read_text().splitlines()
    x_coords = {}
    y_coords = {}
    K = None
    x0 = y0 = None

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
        # Parse x0
        elif stripped.startswith("param x0"):
            parts = stripped.replace("param", "").replace(";", "").split(":=")
            if len(parts) == 2:
                try:
                    x0 = float(parts[1].strip())
                except:
                    x0 = None
        # Parse y0
        elif stripped.startswith("param y0"):
            parts = stripped.replace("param", "").replace(";", "").split(":=")
            if len(parts) == 2:
                try:
                    y0 = float(parts[1].strip())
                except:
                    y0 = None
        # Parse K
        elif stripped.startswith("param K"):
            parts = stripped.replace("param", "").replace(";", "").split(":=")
            if len(parts) == 2:
                K = int(parts[1].strip())
        i += 1

    # Build coords dict for customers only (exclude idx=0 if present)
    coords = {}
    for idx, xv in x_coords.items():
        if idx != 0:
            coords[idx] = (xv, y_coords.get(idx, 0.0))

    # If depot (idx 0) is explicitly defined, use it; otherwise compute centroid of customers
    if (0 in x_coords) and (0 in y_coords) and (x0 is not None) and (y0 is not None):
        depot = (x_coords[0], y_coords[0])
    else:
        # Compute centroid of all customer coordinates
        xs = [coords[n][0] for n in coords]
        ys = [coords[n][1] for n in coords]
        depot = (sum(xs) / len(xs), sum(ys) / len(ys))

    return coords, depot, K

def kmeans(coords, K, max_iters=100, tol=1e-4):

    nodes = list(coords.keys())
    # Random initial centroids: pick K distinct customer nodes
    initial_centers = random.sample(nodes, K)
    centroids = [coords[n] for n in initial_centers]

    labels = {node: None for node in nodes}

    for iteration in range(max_iters):
        # Assignment step
        clusters = {k: [] for k in range(K)}
        changed = False
        for node in nodes:
            x, y = coords[node]
            dists = [math.hypot(x - cx, y - cy) for (cx, cy) in centroids]
            min_idx = dists.index(min(dists))
            clusters[min_idx].append(node)
            if labels[node] != min_idx:
                changed = True
            labels[node] = min_idx

        # If no assignment changed, we’re done
        if not changed and iteration > 0:
            break

        # Update step: recompute centroid of each cluster
        new_centroids = []
        for k in range(K):
            members = clusters[k]
            if len(members) == 0:
                # If cluster is empty, pick a random customer
                rand_node = random.choice(nodes)
                new_centroids.append(coords[rand_node])
            else:
                avg_x = sum(coords[n][0] for n in members) / len(members)
                avg_y = sum(coords[n][1] for n in members) / len(members)
                new_centroids.append((avg_x, avg_y))

        # Check movement of centroids
        move = max(
            math.hypot(new_centroids[k][0] - centroids[k][0],
                       new_centroids[k][1] - centroids[k][1])
            for k in range(K)
        )
        centroids = new_centroids
        if move < tol:
            break

    # Convert labels to 1..K
    final_labels = {node: labels[node] + 1 for node in nodes}
    return final_labels, centroids

def cavdar_sokol_estimate(cluster_nodes, coords):
    n = len(cluster_nodes)
    x_list = [coords[n][0] for n in cluster_nodes]
    y_list = [coords[n][1] for n in cluster_nodes]

    # Mean of x and y
    mean_x = sum(x_list) / n
    mean_y = sum(y_list) / n

    # Standard deviations (raw)
    var_x = sum((x - mean_x)**2 for x in x_list) / n
    var_y = sum((y - mean_y)**2 for y in y_list) / n
    stdev_x = math.sqrt(var_x)
    stdev_y = math.sqrt(var_y)

    # Mean absolute deviations
    cbar_x = sum(abs(x - mean_x) for x in x_list) / n
    cbar_y = sum(abs(y - mean_y) for y in y_list) / n

    # “Centered” standard deviation of absolute deviations
    cstdev_x = math.sqrt((sum((abs(x - mean_x))**2 for x in x_list) / n) - cbar_x**2)
    cstdev_y = math.sqrt((sum((abs(y - mean_y))**2 for y in y_list) / n) - cbar_y**2)

    # Bounding-rectangle area
    xmin, xmax = min(x_list), max(x_list)
    ymin, ymax = min(y_list), max(y_list)
    area = (xmax - xmin) * (ymax - ymin)

    alpha = 2.791
    beta = 0.2669

    term1 = alpha * math.sqrt(n * (cstdev_x * cstdev_y))
    # In Çağdar & Sokol’s formula, the area term is in numerator:
    term2 = beta * math.sqrt(n * (stdev_x * stdev_y) * (area / (cbar_x * cbar_y)))

    return term1 + term2

def euclidean(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    dat_file      = script_dir / "cheap_problem.dat"

    # 1) Parse input data
    coords, depot, K = parse_data(dat_file)
    if K is None:
        raise ValueError("Could not parse K from data file.")

    # 2) Perform K-means clustering on customer-only coords
    start_time = time.time()
    random.seed(42)
    labels, centroids = kmeans(coords, K)
    elapsed = time.time() - start_time
    print(f"K-means clustering done in {elapsed:.2f}s\n")

    # 3) Build clusters dictionary: cluster_index -> [customer nodes]
    clusters = {k: [] for k in range(1, K+1)}
    for node, cl in labels.items():
        clusters[cl].append(node)

    # 4) Compute heuristic per cluster (include depot in each cluster for g(S))
    total_combined = 0.0
    print("Cluster breakdown and heuristic computations:")
    for k in range(1, K+1):
        cust_nodes = clusters[k]
        # Include depot (node 0) in the cluster_nodes for g(S)
        cluster_nodes = [0] + cust_nodes

        # Create a coords_with_depot dict for cavdar_sokol_estimate
        coords_with_depot = {0: depot, **coords}

        g_val = cavdar_sokol_estimate(cluster_nodes, coords_with_depot)

        # Compute cluster centroid (from K-means result)
        centroid = centroids[k-1]

        # Distance from centroid to depot
        dist_to_depot = euclidean(centroid, depot)

        combined = g_val + dist_to_depot
        total_combined += combined

        print(f"\nCluster {k}:")
        print(f"  Customer nodes: {cust_nodes}")
        print(f"  g(S) (with depot)  : {g_val:.4f}")
        print(f"  Centroid (cust-only): ({centroid[0]:.2f}, {centroid[1]:.2f})")
        print(f"  Depot location     : ({depot[0]:.2f}, {depot[1]:.2f})")
        print(f"  Dist(Centroid,Depot): {dist_to_depot:.4f}")
        print(f"  Combined (g + dist): {combined:.4f}")

    print(f"\nTotal combined value: {total_combined:.4f}\n")

    # 5) Visualization
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10", K)

    # Plot customers, colored by their cluster
    for node, cl in labels.items():
        x, y = coords[node]
        plt.scatter(
            x,
            y,
            color=cmap(cl - 1),
            s=50,
            label=f"Vehicle {cl}" if node == min(n for n, c in labels.items() if c == cl) else ""
        )

    # Plot centroids as black 'x'
    for idx, (cx, cy) in enumerate(centroids, start=1):
        plt.scatter(cx, cy, color="black", marker="x", s=100,
                    label="Centroid" if idx == 1 else "")

    # Plot the depot as a distinct red star
    plt.scatter(depot[0], depot[1], color="red", marker="*", s=150, label="Depot")

    plt.title("K-Means Clustering & Heuristic Tour Estimates")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
