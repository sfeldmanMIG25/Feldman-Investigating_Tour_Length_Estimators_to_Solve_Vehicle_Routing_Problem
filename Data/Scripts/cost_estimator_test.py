# run_single_test.py
"""
A simple diagnostic script to test the VRP cost estimator on a single,
known problematic instance (XML100_1111_01.vrp).

This script generates an initial solution and calculates its estimated cost.
It prints a PASS/FAIL verdict based on whether the cost is non-zero,
allowing for quick verification of fixes to the vrp_utils.py estimator logic.
"""
import sys
from pathlib import Path
import joblib

# --- Setup: Add project directory to path to find other scripts ---
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

try:
    # --- Imports from your project files ---
    from vrp_utils import parse_vrp, parse_sol
    from initial_solution_generator import generate_feasible_solution_regret
    from heuristic_engine import PartitionState
except ImportError as e:
    print(f"❌ FATAL: Could not import a required module: {e}")
    print("Ensure this script is in the same directory as your other .py files.")
    sys.exit(1)

# --- Configuration: Hardcode the problematic instance and paths ---
INSTANCE_FILENAME = "XML100_1111_04.vrp"
DATA_DIR = SCRIPT_DIR.parent
INSTANCE_DIR = DATA_DIR / "instances"
SOLUTION_DIR = DATA_DIR / "solutions"
ML_MODEL_DIR = SCRIPT_DIR / "ML_model"

def run_test():
    """Main function to run the diagnostic test."""
    print("--- VRP Estimator Diagnostic Test ---")
    print(f"🎯 Testing Instance: {INSTANCE_FILENAME}\n")

    # --- 1. Load the ML Model ---
    model_path = ML_MODEL_DIR / "alpha_beta_predictor_model.joblib"
    try:
        loaded_ml_model = joblib.load(model_path)
        print(f"✅ ML Model loaded successfully.")
    except FileNotFoundError:
        print(f"❌ FATAL: ML model not found at '{model_path}'.")
        return

    # --- 2. Load Instance and Optimal Solution Data ---
    try:
        instance_path = INSTANCE_DIR / INSTANCE_FILENAME
        solution_path = SOLUTION_DIR / f"{INSTANCE_FILENAME.replace('.vrp', '.sol')}"
        capacity, depot_id, coords, demands = parse_vrp(instance_path)
        optimal_routes, _ = parse_sol(solution_path)
        num_vehicles = len(optimal_routes)
        print(f"✅ Instance data loaded. Optimal solution has {num_vehicles} vehicles.")
    except FileNotFoundError as e:
        print(f"❌ FATAL: Could not find a required file: {e}")
        return

    # --- 3. Prepare Problem Data Dictionary ---
    customer_data = {nid: {'coords': c, 'demand': d} for nid, c, d in zip(coords.keys(), coords.values(), demands.values()) if nid != depot_id}
    problem_data = {
        'coords': coords, 'demands': demands, 'capacity': capacity,
        'depot_coord': coords[depot_id], 'depot_id': depot_id,
        'customer_data': customer_data, 'optimal_routes': optimal_routes,
        'ml_model': loaded_ml_model
    }

    # --- 4. Generate Initial Solution ---
    print("🛠️  Generating initial solution using Regret heuristic...")
    initial_partition = generate_feasible_solution_regret(
        customer_data, problem_data['depot_coord'], num_vehicles, capacity
    )
    print("✅ Initial solution generated.")

    # --- 5. Calculate Estimated Cost (The Core Test) ---
    print("⚙️  Calculating estimated cost of the initial solution...")
    # The PartitionState constructor automatically calculates the costs.
    state = PartitionState(initial_partition, problem_data)
    final_est_cost = state.get_total_cost()

    # --- 6. Report Verdict ---
    print("\n--- TEST RESULT ---")
    print(f"  Calculated Estimated Cost: {final_est_cost:.4f}")

    if final_est_cost > 0:
        print("\n  ✅ PASS: The estimator returned a valid, non-zero cost.")
        print("     This indicates your fix in vrp_utils.py is working correctly.")
    else:
        print("\n  ❌ FAIL: The estimator returned 0. The 'zero cost' bug is still present.")
        print("     Please review the proposed fixes for the `calculate_features_and_mst_length` function.")


if __name__ == "__main__":
    run_test()