"""
File 1: Data Generation for VRP Tour Estimator (Revised for Multi-Model Output)

This script performs a one-time, comprehensive data generation process. It
iterates through VRP instances, generates sample tours, and calculates an
expanded set of geometric and topological features.

REVISIONS:
- Calculates new high-impact features (NN-stats, expanded MST/Hull, PCA).
- Outputs a single Excel file with two tabs:
  1. 'alpha_only_data': Prepared for training a single-output alpha model.
  2. 'alpha_beta_data': Prepared for training a multi-output alpha-beta model.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm
import concurrent.futures
import time

# Ensure vrp_utils can be imported
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

try:
    from vrp_utils import parse_vrp, parse_sol, generate_single_greedy_route, solve_tsp_lkh, _calculate_mst_length
except ImportError:
    print("FATAL: Could not import from vrp_utils.py.")
    sys.exit(1)

# --- Configuration ---
DATA_DIR = SCRIPT_DIR.parent
INSTANCE_DIR = DATA_DIR / "instances"
SOLUTION_DIR = DATA_DIR / "solutions"
OUTPUT_DIR = SCRIPT_DIR / "ML_model"
OUTPUT_EXCEL = OUTPUT_DIR / "vrp_tour_dataset.xlsx" # <-- Changed to .xlsx

SAMPLES_PER_INSTANCE = 50
NUM_WORKERS = max(1, os.cpu_count() - 2)

# --- Feature Engineering ---

def calculate_tour_features(coords_list):
    """
    Calculates an expanded set of descriptive features for a tour's coordinates.
    The depot is assumed to be the first element.
    """
    if len(coords_list) < 4: return {} # Need enough points for stable features
    coords = np.array(coords_list)
    features = {'n': len(coords)}
    dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    
    # --- Basic, Hull, and Bounding Box Features ---
    try:
        hull = ConvexHull(coords)
        features['convex_hull_area'] = hull.volume
        features['convex_hull_perimeter'] = hull.area
        features['hull_vertex_count'] = len(hull.vertices) # TIER 1
        features['hull_ratio'] = features['hull_vertex_count'] / features['n'] # TIER 1
    except Exception:
        features.update({'convex_hull_area': 0, 'convex_hull_perimeter': 0, 'hull_vertex_count': 0, 'hull_ratio': 0})
    
    min_c, max_c = coords.min(axis=0), coords.max(axis=0)
    width, height = max_c[0] - min_c[0], max_c[1] - min_c[1]
    features['bounding_box_area'] = width * height
    
    # --- Nearest-Neighbor (NN) Statistics (TIER 1) ---
    np.fill_diagonal(dist_matrix, np.inf) # Exclude self-distance
    one_nn_dists = np.min(dist_matrix, axis=1)
    features['one_nn_dist_mean'] = one_nn_dists.mean()
    features['one_nn_dist_std'] = one_nn_dists.std()
    
    # --- PCA for Anisotropy (TIER 2) ---
    try:
        pca = PCA(n_components=2).fit(coords)
        eigenvalues = pca.explained_variance_
        features['pca_eigenvalue_ratio'] = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 0
    except Exception:
        features['pca_eigenvalue_ratio'] = 1.0 # Default to isotropic if PCA fails
        
    # --- Expanded MST Features (TIER 1) ---
    np.fill_diagonal(dist_matrix, 0) # Restore diagonal for MST
    mst = minimum_spanning_tree(dist_matrix).toarray()
    degrees = np.count_nonzero(mst + mst.T, axis=1)
    features['mst_degree_mean'] = degrees.mean()
    features['mst_degree_max'] = degrees.max() # NEW
    features['mst_degree_std'] = degrees.std()
    features['mst_leaf_nodes_fraction'] = np.sum(degrees == 1) / features['n'] # NEW

    # --- Other existing features ---
    features['coord_std_dev_x'], features['coord_std_dev_y'] = coords[:, 0].std(), coords[:, 1].std()
    depot_coord, customer_coords = coords[0], coords[1:]
    if len(customer_coords) > 0:
        dists_from_depot = np.linalg.norm(customer_coords - depot_coord, axis=1)
        features['avg_dist_from_depot'], features['max_dist_from_depot'] = dists_from_depot.mean(), dists_from_depot.max()

    return features

def process_instance(instance_path, partition_map):
    """Worker function to process a single VRP instance."""
    instance_basename = instance_path.stem
    sol_path = SOLUTION_DIR / f"{instance_basename}.sol"
    if not sol_path.exists():
        return []

    try:
        capacity, depot_id, coords, demands = parse_vrp(instance_path)
        optimal_routes, _ = parse_sol(sol_path)
        
        # --- FIX: Correctly create all_customer_data from coords and demands dicts ---
        all_customer_data = {
            nid: {'coords': c, 'demand': demands.get(nid, 0)} 
            for nid, c in coords.items()
        }
        # --- END FIX ---
        
        customer_nodes = [nid for nid in all_customer_data if nid != depot_id]

        tours_to_process = optimal_routes[:]
        for _ in range(SAMPLES_PER_INSTANCE):
            route = generate_single_greedy_route(customer_nodes, all_customer_data, capacity)
            if route and len(route) > 2:
                tours_to_process.append(route)
        
        instance_results = []
        for tour in tours_to_process:
            node_ids = [depot_id] + sorted(list(tour))
            coords_for_tour = [coords[nid] for nid in node_ids]
            mst_length = _calculate_mst_length(np.array(coords_for_tour))
            true_tsp_cost = solve_tsp_lkh(node_ids, all_customer_data)
            
            if mst_length > 0 and true_tsp_cost > 0:
                alpha = true_tsp_cost / mst_length
                if not (0.9 < alpha < 3.0):
                    continue
                
                features = calculate_tour_features(coords_for_tour)
                if not features:
                    continue
                
                record = {
                    'instance': instance_basename,
                    'partition': partition_map.get(instance_basename, 'unknown'),
                    'alpha': alpha,
                    'true_tsp_cost': true_tsp_cost,
                    'mst_length': mst_length,
                    **features
                }
                instance_results.append(record)
        return instance_results
    except Exception as e:
        # Catching the error here provides more context
        print(f"Warning: Failed to process {instance_basename}. Error: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for full traceback
        return []

def main():
    """Main execution function."""
    print("--- VRP Tour Dataset Generation (Expanded Features for Multi-Model) ---")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    all_instance_paths = sorted([f for f in INSTANCE_DIR.glob('XML100_*.vrp')])
    train_val_names, test_names = train_test_split([p.stem for p in all_instance_paths], test_size=0.15, random_state=42)
    train_names, val_names = train_test_split(train_val_names, test_size=(0.15/0.85), random_state=42)
    partition_map = {name: 'train' for name in train_names}
    partition_map.update({name: 'validation' for name in val_names})
    partition_map.update({name: 'test' for name in test_names})

    print(f"Discovered {len(all_instance_paths)} instances to process.")

    all_results = []
    # --- FIX: Use a more robust and readable parallel processing loop ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Create a list of futures, one for each instance
        futures = {executor.submit(process_instance, path, partition_map): path for path in all_instance_paths}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_instance_paths), desc="Processing Instances"):
            result = future.result()
            if result:
                all_results.extend(result)
    # --- END FIX ---
    
    if not all_results:
        print("\nFATAL: No data was generated. Check warnings above.")
        return

    df = pd.DataFrame(all_results)
    print(f"\nSuccessfully generated {len(df)} total tour samples.")
    
    # Create Data for the Two Tabs
    df_alpha_only = df.copy()
    df_alpha_beta = df.copy()
    base_alpha = df_alpha_beta[df_alpha_beta['partition'] == 'train']['alpha'].mean()
    df_alpha_beta['target_alpha'] = df_alpha_beta['alpha']
    df_alpha_beta['target_beta'] = df_alpha_beta['true_tsp_cost'] - (base_alpha * df_alpha_beta['mst_length'])
    
    # Save to a single Excel file with two tabs
    print(f"Saving datasets to Excel file: {OUTPUT_EXCEL}")
    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        df_alpha_only.to_excel(writer, sheet_name='alpha_only_data', index=False)
        df_alpha_beta.to_excel(writer, sheet_name='alpha_beta_data', index=False)
    
    print("\nProcess complete.")

if __name__ == "__main__":
    main()