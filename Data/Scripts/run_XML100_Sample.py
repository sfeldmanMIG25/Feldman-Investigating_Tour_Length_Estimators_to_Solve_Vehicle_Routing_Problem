# run_XML100_Sample.py
import os
import sys
import time
import re
import glob
import random
import collections
import traceback
from pathlib import Path
import pandas as pd
import joblib
from tqdm import tqdm
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
# Set the mode: 'RUN' to execute the heuristic, 'AGGREGATE' to analyze results
MODE = 'RUN'
SOLVER_CHOICE = 'tabu2' # Your custom heuristic identifier

# --- Dynamic Import of Solver Class ---
if SOLVER_CHOICE == 'custom':
    from heuristic_strategy import VRPInstanceSolver as SolverClass
    print("✅ Using 'custom' heuristic: VRPInstanceSolver (SP-STS)")
elif SOLVER_CHOICE == 'tabu':
    from heuristic_strategy_tabu import VRPInstanceSolverTabu as SolverClass
    print("✅ Using 'tabu' heuristic: VRPInstanceSolverTabu (DD-kE v1)")
elif SOLVER_CHOICE == 'tabu2':
    from heuristic_strategy_tabu_2 import VRPInstanceSolverTabu2 as SolverClass
    print(f"✅ Using '{SOLVER_CHOICE}' heuristic: VRPInstanceSolverTabu2")
else:
    raise ValueError(f"Invalid SOLVER_CHOICE: '{SOLVER_CHOICE}'.")

# --- Project Paths ---
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))
from vrp_utils import parse_vrp 

DATA_DIR = SCRIPT_DIR.parent
INSTANCE_DIR = DATA_DIR / "instances"
SOLUTION_DIR = DATA_DIR / "solutions"
ML_MODEL_DIR = SCRIPT_DIR / "ML_model"

# Define paths for both sets of results
HEURISTIC_OUTPUT_DIR = SCRIPT_DIR / f"Batch_Run_Results_{SOLVER_CHOICE}"
ORTOOLS_OUTPUT_DIR = SCRIPT_DIR / "Batch_Run_Results" 
CONTROL_CSV_PATH = ORTOOLS_OUTPUT_DIR / "benchmark_instances.csv" 
HEURISTIC_TIMEOUT = 2 * 3600  # 2 hours

# --- Cleanup Function ---
def cleanup_previous_run_files(output_dir: Path):
    """Deletes old heuristic-specific results to ensure a clean run."""
    if not output_dir.exists():
        return

    print("\n--- Cleaning up previous run results ---")
    files_to_delete = []
    
    patterns_to_delete = ["*_solution.txt", "*.png", "*.xlsx"]
    protected_files = ["benchmark_instances.csv", "benchmark_instances_xml100.csv"]

    for pattern in patterns_to_delete:
        for f_path in output_dir.glob(pattern):
            if "ortools" in f_path.name.lower():
                continue
            if f_path.name in protected_files:
                continue
            files_to_delete.append(f_path)

    if not files_to_delete:
        print(" > No old files to delete.")
        return

    for f_path in files_to_delete:
        try: f_path.unlink()
        except OSError as e: print(f" > Error deleting {f_path.name}: {e}")
    
    print(f"Successfully deleted {len(files_to_delete)} old result files.")

# --- Worker Function for Parallel Processing ---
def run_single_solver(instance_filename, solver_class, common_params):
    """Encapsulates the logic to solve a single VRP instance."""
    basename = instance_filename.replace('.vrp', '')
    print(f"  [Worker] Starting to process: {instance_filename}")
    
    try:
        solver_params = {**common_params, 'instance_path': str(INSTANCE_DIR / instance_filename), 'solution_path': str(SOLUTION_DIR / f"{basename}.sol")}
        solver = solver_class(**solver_params)
        solver.solve()
        return f"Successfully processed {instance_filename}"
    except Exception:
        return f"FATAL ERROR while processing {basename}:\n{traceback.format_exc()}"

# --- Sampling Function ---
def select_benchmark_instances(instance_dir: Path, control_csv_path: Path):
    if control_csv_path.exists():
        print(f"Found control file. Loading instances from '{control_csv_path}'.")
        return pd.read_csv(control_csv_path)['instance_filename'].tolist()
    else:
        print(f"Control file not found at '{control_csv_path}'. Please run 'run_ortools_batch.py' first to generate it.")
        sys.exit(1)

def aggregate_and_compare_results(heuristic_dir: Path, ortools_dir: Path):
    """Parses all results into one long-format dataframe, then pivots for comparison."""
    print("\n" + "="*60)
    print("--- Starting Comparative Aggregation (Pivot Method) ---")
    print(f"Heuristic results from: '{heuristic_dir.name}'")
    print(f"OR-Tools results from:  '{ortools_dir.name}'")
    print("="*60)

    def parse_report_file_for_pivot(filepath):
        """Parses a single report file and identifies the solver type."""
        filename = os.path.basename(filepath)
        if '_ortools_solution.txt' in filename:
            solver = 'OR_Tools'
            instance = filename.replace('_ortools_solution.txt', '')
        else:
            solver = SOLVER_CHOICE
            instance = filename.replace('_solution.txt', '')

        with open(filepath, 'r') as f: content = f.read()
        
        patterns = {
            'True_Gap': r"Percentage Difference \(True vs Optimal\)\s*:\s*([+-]?\d+\.\d+)",
            'Estimator_Gap': r"Heuristic Final Gap \(Est\. vs Optimal Est\.\)\s*:\s*([+-]?\d+\.\d+)",
            'Heuristic_Cost': r"Heuristic True Cost \(LKH\)\s*:\s*([\d,]+\.\d+)",
            'Optimal_Cost': r"Optimal VRP Cost \(LKH Recalculated\)\s*:\s*([\d,]+\.\d+)",
            'Total_Time': r"Total Time\s*:\s*([\d,]+\.\d+)"
        }
        
        result = {'instance': instance, 'solver': solver}
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            result[key] = float(match.group(1).replace(',', '')) if match else np.nan

        customer_counts = re.findall(r"Cluster \d+ \((\d+) customers\):", content)
        result['Max_Customers_Per_Vehicle'] = max(int(c) for c in customer_counts) if customer_counts else np.nan
        return result

    # 1. Collect and parse all available solution files
    heuristic_files = glob.glob(str(heuristic_dir / "*_solution.txt"))
    ortools_files = glob.glob(str(ortools_dir / "*_ortools_solution.txt"))
    all_files = heuristic_files + ortools_files

    if not all_files:
        print(f"Could not find any solution files in '{heuristic_dir.name}' or '{ortools_dir.name}'. Cannot aggregate.")
        return

    # 2. Create a single long-format DataFrame
    all_results = [parse_report_file_for_pivot(f) for f in all_files]
    df_long = pd.DataFrame(all_results)
    
    # 3. Pivot the DataFrame to create a wide-format comparison table
    df_merged = df_long.pivot_table(
        index='instance', 
        columns='solver', 
        values=['True_Gap', 'Estimator_Gap', 'Heuristic_Cost', 'Total_Time', 'Optimal_Cost', 'Max_Customers_Per_Vehicle']
    )
    
    # Flatten the multi-level column index
    df_merged.columns = [f'{val}_{col}' for val, col in df_merged.columns]
    
    # Consolidate instance-specific properties (Optimal_Cost, Max_Customers)
    df_merged['Optimal_Cost'] = df_merged[f'Optimal_Cost_{SOLVER_CHOICE}'].fillna(df_merged.get(f'Optimal_Cost_OR_Tools', pd.Series(index=df_merged.index)))
    df_merged['Max_Customers_Per_Vehicle'] = df_merged[f'Max_Customers_Per_Vehicle_{SOLVER_CHOICE}'].fillna(df_merged.get(f'Max_Customers_Per_Vehicle_OR_Tools', pd.Series(index=df_merged.index)))
    
    # Drop redundant columns and reset index
    cols_to_drop = [c for c in df_merged.columns if 'Optimal_Cost_' in str(c) or 'Max_Customers_Per_Vehicle_' in str(c)]
    df_merged = df_merged.drop(columns=cols_to_drop).reset_index()

    # 4. Save extended comparative Excel file
    excel_path = HEURISTIC_OUTPUT_DIR / "summary_results_comparison.xlsx"
    df_merged.to_excel(excel_path, index=False, float_format="%.2f")
    print(f"\Extended comparative results for {len(df_merged)} instances saved to '{excel_path}'")

    # 5. Report Summary Statistics
    print("\n" + "="*60)
    print(f"--- Performance Statistics ({SOLVER_CHOICE} vs. OR-Tools) ---")
    print("="*60)
    
    cols_to_describe = []
    if f'True_Gap_{SOLVER_CHOICE}' in df_merged.columns:
        cols_to_describe.append(f'True_Gap_{SOLVER_CHOICE}')
    if f'Estimator_Gap_{SOLVER_CHOICE}' in df_merged.columns:
        cols_to_describe.append(f'Estimator_Gap_{SOLVER_CHOICE}')
    if f'True_Gap_OR_Tools' in df_merged.columns:
        cols_to_describe.append(f'True_Gap_OR_Tools')
    
    if cols_to_describe:
        # Rename columns for a clearer statistics table
        rename_dict = {
            f'True_Gap_{SOLVER_CHOICE}': f'{SOLVER_CHOICE}_True_Gap',
            f'Estimator_Gap_{SOLVER_CHOICE}': f'{SOLVER_CHOICE}_Estimator_Gap',
            'True_Gap_OR_Tools': 'OR_Tools_True_Gap'
        }
        stats_df = df_merged[cols_to_describe].rename(columns=rename_dict)
        print(stats_df.describe().round(2))
    else:
        print("No gap columns found to describe.")
    print("="*60)

    # 6. Generate Plots
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Comparative Gap Box Plot
    if cols_to_describe:
        plot_cols_gap = [c for c in rename_dict.values() if c in stats_df.columns]
        plt.figure(figsize=(10, 7))
        sns.boxplot(data=stats_df[plot_cols_gap], palette='deep')
        plt.title(f'Solution Gap Comparison: {SOLVER_CHOICE} vs. OR-Tools', fontsize=16)
        plt.ylabel('Percentage Gap from Optimal (%)')
        box_plot_path = HEURISTIC_OUTPUT_DIR / "summary_gap_boxplot.png"
        plt.savefig(box_plot_path, dpi=600); plt.close()
        print(f"Comparative gap box plot saved to '{box_plot_path}'")
    
    # Plot 2: Comparative Absolute Error Box Plot
    if f'Heuristic_Cost_OR_Tools' in df_merged.columns and 'Optimal_Cost' in df_merged.columns:
        df_merged[f'Abs_Error_{SOLVER_CHOICE}'] = df_merged[f'Heuristic_Cost_{SOLVER_CHOICE}'] - df_merged['Optimal_Cost']
        df_merged[f'Abs_Error_OR_Tools'] = df_merged[f'Heuristic_Cost_OR_Tools'] - df_merged['Optimal_Cost']
        
        plot_data = df_merged[[f'Abs_Error_{SOLVER_CHOICE}', 'Abs_Error_OR_Tools']].rename(columns={f'Abs_Error_{SOLVER_CHOICE}': SOLVER_CHOICE, 'Abs_Error_OR_Tools': 'OR-Tools'})
        
        plt.figure(figsize=(10, 7))
        sns.boxplot(data=plot_data, palette='pastel')
        plt.title('Absolute Cost Error Comparison', fontsize=16)
        plt.ylabel('Cost Above Optimal')
        error_plot_path = HEURISTIC_OUTPUT_DIR / "summary_error_boxplot.png"
        plt.savefig(error_plot_path, dpi=600); plt.close()
        print(f"Comparative error box plot saved to '{error_plot_path}'")

    # Plot 3: Scatter Plot by Max Route Size
    if f'Estimator_Gap_{SOLVER_CHOICE}' in df_merged.columns:
        conditions = [
            df_merged['Max_Customers_Per_Vehicle'] <= 10,
            (df_merged['Max_Customers_Per_Vehicle'] > 10) & (df_merged['Max_Customers_Per_Vehicle'] <= 15),
            df_merged['Max_Customers_Per_Vehicle'] > 15
        ]
        choices = ['n <= 10 (Small)', '10 < n <= 15 (Medium)', 'n > 15 (Large)']
        df_merged['Route_Size_Category'] = np.select(conditions, choices, default='Other')
        
        plt.figure(figsize=(12, 9))
        sns.scatterplot(
            data=df_merged.dropna(subset=[f'Estimator_Gap_{SOLVER_CHOICE}', f'True_Gap_{SOLVER_CHOICE}', 'Route_Size_Category']), 
            x=f'Estimator_Gap_{SOLVER_CHOICE}', 
            y=f'True_Gap_{SOLVER_CHOICE}', 
            hue='Route_Size_Category',
            style='Route_Size_Category',
            s=120,
            palette='Set1',
            hue_order=choices
        )
        plt.title(f'Estimator vs. True Gap by Max Route Size ({SOLVER_CHOICE})', fontsize=16)
        plt.xlabel('Final Estimator Gap (%)')
        plt.ylabel('Final True Solution Gap (%)')
        plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='y=x (Perfect Estimator)')
        plt.legend(title='Max Customers per Vehicle')
        plt.grid(True, which='both', linestyle='--')
        scatter_size_plot_path = HEURISTIC_OUTPUT_DIR / "summary_scatter_plot_by_size.png"
        plt.savefig(scatter_size_plot_path, dpi=600); plt.close()
        print(f"✅ Scatter plot by route size saved to '{scatter_size_plot_path}'")

# --- Main Execution Block ---
def main():
    if MODE == 'RUN':
        HEURISTIC_OUTPUT_DIR.mkdir(exist_ok=True)
        cleanup_previous_run_files(HEURISTIC_OUTPUT_DIR)
        common_solver_params = {'output_dir': str(HEURISTIC_OUTPUT_DIR), 'heuristic_timeout': HEURISTIC_TIMEOUT}
        model_path = ML_MODEL_DIR / "alpha_predictor_model.joblib"
        common_solver_params['ml_model'] = joblib.load(model_path)
        instances_to_run = select_benchmark_instances(INSTANCE_DIR, CONTROL_CSV_PATH)
        print(f"\nSelected {len(instances_to_run)} instances for the benchmark run.")
        num_workers = os.cpu_count()
        print(f"Starting parallel batch run on {num_workers} cores with '{SOLVER_CHOICE}' solver.")
        
        task_function = partial(run_single_solver, solver_class=SolverClass, common_params=common_solver_params)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(task_function, instances_to_run), total=len(instances_to_run), desc="Running Benchmark"))
        
        print("\n--- Batch processing complete. Now aggregating... ---")
        aggregate_and_compare_results(HEURISTIC_OUTPUT_DIR, ORTOOLS_OUTPUT_DIR)

    elif MODE == 'AGGREGATE':
        aggregate_and_compare_results(HEURISTIC_OUTPUT_DIR, ORTOOLS_OUTPUT_DIR)
        
    else:
        print(f"Invalid MODE: '{MODE}'. Choose 'RUN' or 'AGGREGATE'.")

if __name__ == "__main__":
    main()