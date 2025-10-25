import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import sys

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "ML_model"
DATASET_PATH = DATA_DIR / "vrp_tour_dataset.xlsx"

# Redirect stdout to a file
def tee_output(filepath):
    """(Unused) Helper to tee stdout to a file and console.

    Left in place for backwards compatibility but not used by the
    updated GART-only analysis flow.
    """
    class Tee(object):
        def __init__(self, name, mode):
            self.file = open(name, mode)
            self.stdout = sys.stdout

        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    sys.stdout = Tee(filepath, 'w')

# --- Analysis Functions ---

def load_data(sheet_name):
    """Load and split data from the specified Excel sheet."""
    try:
        df = pd.read_excel(DATASET_PATH, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"FATAL: Dataset not found at {DATASET_PATH}.")
        return None, None, None
    
    test_df = df[df['partition'] == 'test']
    feature_cols = [c for c in df.columns if c not in ['instance', 'partition', 'alpha', 'true_tsp_cost', 'mst_length', 'target_alpha', 'target_beta']]
    
    return test_df, feature_cols

def analyze_model(model_path, sheet_name):
    """Performs full analysis for the alpha-only predictor and writes GART-named outputs.

    This function assumes the model predicts the single-column target 'alpha'.
    All saved reports and plot files are named to reference "GART".
    """
    model_name = Path(model_path).stem
    analysis_dir = DATA_DIR / f"{model_name}_GART_analysis"
    analysis_dir.mkdir(exist_ok=True)

    output_file_path = analysis_dir / 'gart_model_analysis_report.txt'
    # Redirect print statements to the output file
    original_stdout = sys.stdout
    sys.stdout = open(output_file_path, 'w')

    print("--- Model Analysis Report: GART Model ---")

    # 1. Load Model and Data
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"FATAL: Model not found at {model_path}.")
        sys.stdout = original_stdout
        return

    test_df, feature_cols = load_data(sheet_name)
    if test_df is None:
        sys.stdout = original_stdout
        return

    # Target is the single 'alpha' column for GART
    target_col = 'alpha'
    y_test = test_df[target_col]
    X_test = test_df[feature_cols]

    # 2. Save Hyperparameters
    print("\n[Section 1: Model Hyperparameters]\n")
    print("All Model Parameters:")

    # Access parameters of the base estimator if it's a wrapped estimator
    if hasattr(model, 'estimators_'):
        params = model.estimators_[0].get_params()
    else:
        params = model.get_params()

    for k, v in params.items():
        print(f"- {k}: {v}")

    # 3. Predict and Evaluate
    y_pred = model.predict(X_test)

    print("\n[Section 2: Test Set Performance]\n")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("Performance Metrics:")
    print(f"  GART Prediction -> R²: {r2:.4f}, MAE: {mae:.4f}")

    # Calculate residuals for plots
    residuals = y_test - y_pred

    # 4. Feature Importance
    print("\n[Section 3: Feature Importance Analysis]\n")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'estimators_'):
        # For wrapped estimators, average importances
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    else:
        print("Warning: Could not get feature importances.")
        importances = np.zeros(len(feature_cols))

    feat_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)

    # Print and save all feature importances to the file
    print("Full Feature Ranking by Importance:")
    print(feat_df.to_string(index=False))

    # Plot all features (dynamic size)
    plt.figure(figsize=(12, max(8, len(feature_cols) * 0.3)))
    sns.barplot(x='importance', y='feature', data=feat_df)
    plt.title('Full Feature Importance Ranking for GART Model')
    plt.tight_layout()
    plt.savefig(analysis_dir / 'gart_full_feature_importance.png')
    plt.close()

    # 5. Residual and Regression Plots for GART
    print("\n[Section 4: Residual and Regression Plots]\n")
    print("Plots saved to the analysis directory for visual inspection:")

    # Predicted vs. Actual
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=y_test)
    plt.xlabel('Predicted GART')
    plt.ylabel('Actual GART')
    plt.title('Predicted vs. Actual Values for GART Model')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.savefig(analysis_dir / 'gart_predicted_vs_actual.png')
    plt.close()

    # Residuals vs. Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted GART')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot for GART Model')
    plt.tight_layout()
    plt.savefig(analysis_dir / 'gart_residuals_plot.png')
    plt.close()

    # Residuals Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals for GART Model')
    plt.tight_layout()
    plt.savefig(analysis_dir / 'gart_residuals_histogram.png')
    plt.close()

    print("\nAnalysis plots generated successfully.")
    print("\n--- Analysis Complete for GART Model ---")

    # Restore original stdout
    sys.stdout.close()
    sys.stdout = original_stdout

def main():
    """Main function to orchestrate model analysis."""
    print("--- VRP Model Analysis Script (Verbose Mode) ---")

    # Define path to the trained alpha (GART) model
    alpha_model_path = DATA_DIR / "alpha_predictor_model.joblib"

    # Run analysis for the GART (alpha-only) model
    print("\nStarting analysis for the GART (Alpha-Only) Model...")
    analyze_model(alpha_model_path, 'alpha_only_data')

    print("\nGART analysis complete. Check the output folder for the report and plots.")

if __name__ == "__main__":
    main()