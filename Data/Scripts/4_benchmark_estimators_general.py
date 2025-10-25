# File: 4_benchmark_estimators_general.py
# Purpose: DOE setup to benchmark various TSP approximation methods and models
#          against a set of generated instances.

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.spatial import ConvexHull
from itertools import combinations
import math
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.multioutput import MultiOutputRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed
import openpyxl

inf = float('inf')

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
GENERATED_INSTANCES_DIR = DATA_DIR / "Generated_TSPs"
ANALYSIS_OUTPUT_DIR = DATA_DIR / "Generalized_TSP_Analysis"
ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)
ML_MODELS_DIR = SCRIPT_DIR / "ML_model"

# Ensure vrp_utils can be imported
sys.path.append(str(SCRIPT_DIR))

# Import LKH_EXECUTABLE_PATH and relevant functions from vrp_utils.
from vrp_utils import LKH_EXECUTABLE_PATH, estimate_tsp_tour_length, _calculate_mst_length, parse_vrp, _calculate_cavdar, _calculate_vinel, _calculate_held_karp, calculate_features_and_mst_length

# Load the ML models
try:
    ALPHA_MODEL = joblib.load(ML_MODELS_DIR / "alpha_predictor_model.joblib")
except FileNotFoundError:
    ALPHA_MODEL = None
    print("Warning: ALPHA_MODEL not found. ML_Alpha_Est will be skipped.")

try:
    ALPHA_BETA_MODEL = joblib.load(ML_MODELS_DIR / "alpha_beta_predictor_model.joblib")
except FileNotFoundError:
    ALPHA_BETA_MODEL = None
    print("Warning: ALPHA_BETA_MODEL not found. ML_Alpha_Beta_Est will be skipped.")

# --- Parsing and Helper Functions ---
def parse_vrp_instance(file_path):
    """
    Parses a .vrp file to extract instance data and LKH-3 cost.
    Assumes the format created by Full_Graph_Generator.py.
    """
    data = {}
    coords = {}
    node_ids = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    parsing_coords = False
    parsing_demands = False
    
    for line in lines:
        line = line.strip()
        if line.startswith("NAME :"):
            data['name'] = line.split(":")[1].strip()
        elif line.startswith("DIMENSION :"):
            data['dimension'] = int(line.split(":")[1].strip())
        elif line.startswith("CAPACITY :"):
            data['capacity'] = int(line.split(":")[1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            parsing_coords = True
            continue
        elif line.startswith("DEMAND_SECTION"):
            parsing_coords = False
            parsing_demands = True
            continue
        elif line.startswith("DEPOT_SECTION"):
            parsing_demands = False
        elif line.startswith("TOUR_LENGTH:"):
            data['lkh_cost'] = float(line.split(":")[1].strip())
        elif parsing_coords and line:
            parts = line.split()
            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            coords[node_id] = np.array([x, y])
            node_ids.append(node_id)
    
    data['node_ids'] = node_ids
    data['coords'] = coords
    return data

def run_single_benchmark(file_path):
    """Helper function to run benchmarks for a single instance, for parallelization."""
    instance_data = parse_vrp_instance(file_path)
    true_lkh_cost = instance_data['lkh_cost']
    coords = np.array([instance_data['coords'][i] for i in instance_data['node_ids']])
    
    features, mst_length = calculate_features_and_mst_length(coords)
    
    # Extract distribution type and parameters from the instance name
    name_parts = instance_data['name'].split('-')
    dist_type = name_parts[1]
    
    params = {}
    if dist_type == 'clustered':
        for part in name_parts[2:]:
            if part.startswith('c'):
                params['clust_n'] = int(part[1:])
            elif part.startswith('r'):
                params['clust_rad'] = float(part[1:]) / 100
    
    vinel_est = estimate_tsp_tour_length(coords, mode='vinel')
    cavdar_est = estimate_tsp_tour_length(coords, mode='cavdar')
    composite_est = estimate_tsp_tour_length(coords, mode='composite')
    
    ml_alpha_cost = np.nan
    if ALPHA_MODEL is not None:
        ml_alpha_cost = estimate_tsp_tour_length(coords, mode='RegressionTree', bounding_stats=ALPHA_MODEL)
    
    ml_alpha_beta_cost = np.nan
    if ALPHA_BETA_MODEL is not None:
        ml_alpha_beta_cost = estimate_tsp_tour_length(coords, mode='RegressionTree', bounding_stats=ALPHA_BETA_MODEL)
    
    feldman_composite_cost = estimate_tsp_tour_length(coords, mode='composite_mst', bounding_stats=ALPHA_MODEL)
    
    result = {
        'Instance': instance_data['name'],
        'N': instance_data['dimension'],
        'Distribution': dist_type,
        'clust_n': params.get('clust_n'),
        'clust_rad': params.get('clust_rad'),
        'True_LKH_Cost': true_lkh_cost,
        'Vinel_Est': vinel_est,
        'Cavdar_Est': cavdar_est,
        'Composite_Est': composite_est,
        'ML_Alpha_Est': ml_alpha_cost,
        'ML_Alpha_Beta_Est': ml_alpha_beta_cost,
        'Feldman_Est': feldman_composite_cost
    }
    return result

# --- Main Benchmarking Function ---

def run_benchmarks():
    """Main function to run all TSP estimation methods on generated instances."""
    print("--- Running TSP Benchmarks ---")
    all_vrp_files = list(GENERATED_INSTANCES_DIR.glob('*.vrp'))
    if not all_vrp_files:
        print("FATAL: No .vrp files found in the generated instances directory.")
        return

    benchmark_results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(run_single_benchmark, file_path): file_path for file_path in all_vrp_files}
        
        for future in tqdm(as_completed(futures), total=len(all_vrp_files), desc="Running Solvers"):
            benchmark_results.append(future.result())

    # Convert to DataFrame and save to XLSX
    results_df = pd.DataFrame(benchmark_results)
    
    # Calculate percentage differences
    results_df['Vinel_Diff_%'] = (results_df['Vinel_Est'] - results_df['True_LKH_Cost']) / results_df['True_LKH_Cost'] * 100
    results_df['Cavdar_Diff_%'] = (results_df['Cavdar_Est'] - results_df['True_LKH_Cost']) / results_df['True_LKH_Cost'] * 100
    results_df['Composite_Diff_%'] = (results_df['Composite_Est'] - results_df['True_LKH_Cost']) / results_df['True_LKH_Cost'] * 100
    results_df['ML_Alpha_Diff_%'] = (results_df['ML_Alpha_Est'] - results_df['True_LKH_Cost']) / results_df['True_LKH_Cost'] * 100
    results_df['ML_Alpha_Beta_Diff_%'] = (results_df['ML_Alpha_Beta_Est'] - results_df['True_LKH_Cost']) / results_df['True_LKH_Cost'] * 100
    results_df['Feldman_Diff_%'] = (results_df['Feldman_Est'] - results_df['True_LKH_Cost']) / results_df['True_LKH_Cost'] * 100

    xlsx_path = ANALYSIS_OUTPUT_DIR / "benchmark_results.xlsx"
    results_df.to_excel(xlsx_path, index=False)
    print(f"\nAll benchmark results saved to {xlsx_path}")
    
    # --- Generate Plots ---
    plot_analysis(results_df)
    
    print("\nBenchmark and analysis complete.")

def plot_analysis(df: pd.DataFrame):
    """Generates and saves a series of plots for a high-level analysis."""
    print("Generating analysis plots...")
    
    df = df[df['N'] > 10].copy()  # Use .copy() to prevent SettingWithCopyWarning
    
    error_cols = ['Vinel_Diff_%', 'Cavdar_Diff_%', 'Composite_Diff_%', 'ML_Alpha_Diff_%', 'ML_Alpha_Beta_Diff_%']
    
    # Melt the dataframe for plotting
    plot_df = df.melt(id_vars=['Instance', 'N', 'Distribution'], value_vars=error_cols, var_name='Method', value_name='Error')
    
    # Clean up method names for plotting
    plot_df['Method'] = plot_df['Method'].str.replace('_Diff_%', '')

    # Overall Performance: Box Plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Method', y='Error', data=plot_df)
    plt.title("Overall Estimation Error by Method")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig(ANALYSIS_OUTPUT_DIR / "overall_error_boxplot.png")
    plt.close()

    # Performance vs. Instance Size (Bar Plot)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='N', y='Error', hue='Method', data=plot_df, estimator=np.mean)
    plt.title("Mean Estimation Error vs. Number of Nodes (N)")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=45)
    plt.savefig(ANALYSIS_OUTPUT_DIR / "error_by_size_barplot.png")
    plt.close()

    # Performance by Distribution Type
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='Distribution', y='Error', hue='Method', data=plot_df)
    plt.title("Estimation Error by Distribution Type")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(ANALYSIS_OUTPUT_DIR / "error_by_distribution.png")
    plt.close()
    
    print(f"Analysis plots saved to {ANALYSIS_OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    run_benchmarks()