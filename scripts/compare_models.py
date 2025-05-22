import mlflow
import pandas as pd

def compare_model_runs(experiment_name="ChurnPredictionExperiment"):
    """
    Fetches runs from an MLflow experiment, displays key metrics in a table,
    and identifies the best model based on F1-score.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Error: Experiment '{experiment_name}' not found.")
            return
        
        experiment_id = experiment.experiment_id
        print(f"Fetching runs for experiment: '{experiment_name}' (ID: {experiment_id})...")

        # Search for all runs in the experiment
        # We can specify search_all_experiments=True if we want to search by name across all experiments
        # but get_experiment_by_name followed by experiment_id is more precise.
        runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

        if runs_df.empty:
            print(f"No runs found in experiment '{experiment_name}'.")
            return

        # Define columns of interest: parameters, metrics, and run info
        # Parameters are prefixed with 'params.', metrics with 'metrics.'
        # Tags can also be included, prefixed with 'tags.'
        # Run info like 'run_id', 'start_time', 'end_time'
        
        # Let's select some key columns. Adjust based on what's logged.
        # Common parameters logged by our script: params.model_name, params.C, params.n_estimators, etc.
        # Common metrics: metrics.test_f1_score, metrics.test_roc_auc, metrics.test_accuracy, etc.
        # Common tags: tags.mlflow.runName, tags.best_overall_model_name (on the last run)
        
        cols_to_display = []
        
        # Try to find common parameter columns (model name is a good one)
        if 'params.model_name' in runs_df.columns:
            cols_to_display.append('params.model_name')
        elif 'tags.mlflow.runName' in runs_df.columns: # Fallback to runName if model_name param isn't there
             cols_to_display.append('tags.mlflow.runName')


        # Add key metrics
        metric_cols = [
            'metrics.test_f1_score', 'metrics.test_roc_auc', 
            'metrics.test_accuracy', 'metrics.test_precision', 'metrics.test_recall',
            'metrics.best_cv_f1_score' # CV score from GridSearchCV
        ]
        for col in metric_cols:
            if col in runs_df.columns:
                cols_to_display.append(col)
        
        # Add run ID for reference
        if 'run_id' in runs_df.columns:
            cols_to_display.append('run_id')

        if not cols_to_display:
            print("Could not find any of the expected parameter or metric columns in the runs.")
            print("Available columns:", runs_df.columns.tolist())
            return

        # Filter out rows where the essential metrics might be NaN (e.g., if a run failed early)
        # Particularly, we want to sort by F1 score.
        if 'metrics.test_f1_score' in runs_df.columns:
            runs_df = runs_df.dropna(subset=['metrics.test_f1_score'])
            if runs_df.empty:
                print(f"No runs with valid 'metrics.test_f1_score' found in experiment '{experiment_name}'.")
                return
            # Sort by F1 score in descending order
            sorted_runs_df = runs_df.sort_values(by='metrics.test_f1_score', ascending=False)
        else:
            print("Metric 'metrics.test_f1_score' not found for sorting. Displaying unsorted results.")
            sorted_runs_df = runs_df
            
        # Display the selected columns
        print("\n--- Model Comparison Table (Sorted by Test F1-Score) ---")
        # Set pandas options to display more content if necessary
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000) 
        print(sorted_runs_df[cols_to_display].to_string())

        # Identify and print the best model based on test F1-score
        if 'metrics.test_f1_score' in sorted_runs_df.columns and not sorted_runs_df.empty:
            best_run = sorted_runs_df.iloc[0]
            best_model_name = best_run.get('params.model_name', best_run.get('tags.mlflow.runName', 'Unknown Model'))
            best_f1_score = best_run.get('metrics.test_f1_score', float('nan'))
            best_roc_auc = best_run.get('metrics.test_roc_auc', float('nan'))
            best_run_id = best_run.get('run_id', 'N/A')

            print("\n--- Summary ---")
            print(f"Best performing model based on Test F1-Score: {best_model_name}")
            print(f"  Run ID: {best_run_id}")
            print(f"  Test F1-Score: {best_f1_score:.4f}")
            print(f"  Test ROC AUC: {best_roc_auc:.4f}")
            
            # You can also fetch the best_overall_model_name tag from the *last* run of the experiment
            # if your training script sets it on the experiment or a specific run.
            # The current train_model.py sets it on the last run, which might not be the best model's run.
            # It's better to rely on sorting the metrics from all runs.
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Ensure MLflow tracking URI is set if not using local file storage
    # For example, if you have a remote MLflow server:
    # mlflow.set_tracking_uri("http://localhost:5000")
    
    # The train_model.py script uses the default local 'mlruns' directory.
    # This comparison script will also use that by default.
    
    compare_model_runs()
