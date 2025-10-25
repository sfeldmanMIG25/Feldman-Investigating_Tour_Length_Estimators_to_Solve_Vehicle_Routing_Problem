# single_instance_runner.py
import os
import sys
from pathlib import Path
import joblib

# --- Configuration ---
# Choose the solver: 'custom', 'tabu', 'tabu2', or 'memetic'
SOLVER_CHOICE = 'tabu2'
INSTANCE_NAME = "XML100_1111_01.vrp"

# --- Dynamic Import of Solver Class ---
if SOLVER_CHOICE == 'custom':
    from heuristic_strategy import VRPInstanceSolver as SolverClass
    print("Using custom heuristic: VRPInstanceSolver")
elif SOLVER_CHOICE == 'tabu':
    from mealpy_strategy import VRPInstanceSolverTabu as SolverClass
    print("Using Mealpy-powered heuristic: TabuVRPSolver")
elif SOLVER_CHOICE == 'tabu2':
    from heuristic_strategy_tabu_2 import VRPInstanceSolverTabu2 as SolverClass
    print("Using Tabu Search heuristic: VRPInstanceSolverTabu")
else:
    raise ValueError(f"Invalid SOLVER_CHOICE: '{SOLVER_CHOICE}'. Must be 'custom', 'tabu', 'tabu2'.")

# --- Project Paths ---
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))
DATA_DIR = SCRIPT_DIR.parent
INSTANCE_DIR = DATA_DIR / "instances"
SOLUTION_DIR = DATA_DIR / "solutions"
OUTPUT_DIR = SCRIPT_DIR / f"run_results_{SOLVER_CHOICE}"
ML_MODEL_DIR = SCRIPT_DIR / "ML_model"

# === MAIN EXECUTION BLOCK ===
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    basename = INSTANCE_NAME.replace('.vrp', '')
    instance_file_path = INSTANCE_DIR / INSTANCE_NAME
    solution_file_path = SOLUTION_DIR / f"{basename}.sol"
    
    if not instance_file_path.exists() or not solution_file_path.exists():
        print(f"FATAL: Instance or solution file not found for '{INSTANCE_NAME}'")
        sys.exit(1)
        
    print(f"Selected instance: {INSTANCE_NAME}")
    
    # --- Prepare Solver Parameters ---
    solver_params = {
        'instance_path': str(instance_file_path),
        'solution_path': str(solution_file_path),
        'output_dir': str(OUTPUT_DIR),
        'heuristic_timeout': 300  # 5 minutes for a single run
    }
    
    # The runner must load the model and pass it to the solver.
    # This is compatible with all four solver classes.
    model_path = ML_MODEL_DIR / "alpha_predictor_model.joblib"
    loaded_ml_model = joblib.load(model_path)
    
    solver_params['ml_model'] = loaded_ml_model

    # --- Instantiate and Run the Solver ---
    solver = SolverClass(**solver_params)
    solver.solve()
    
    print(f"\nAll outputs for '{SOLVER_CHOICE}' solver saved to the '{OUTPUT_DIR.name}' directory.")