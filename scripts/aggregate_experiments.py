import os
from typing import List
import pandas as pd
import numpy as np

TRIAL_PATHS = ["trial_1.csv", "trial_2.csv", "trial_3.csv"] # Add your paths here
EXPERIMENT_FILE = "results/experiment_results.csv"

EXPERIMENT_SCHEMA = [
    "experiment_id",
    "bit_width",
    "group_size",
    "mean_perplexity",
    "mean_rmse_key",
    "mean_rmse_value",
    "fqa_accuracy",
    "oosqa_accuracy",
    "crqa_accuracy",
    "zero_shot_accuracy",
]

def main(exp_number: int):
    try:
        # 1. Load and Concatenate Trials
        trials_chunks = [pd.read_csv(tp) for tp in TRIAL_PATHS if tp]

        if not trials_chunks:
            print("No trial files found.")
            return

        # ignore_index=True prevents duplicate row numbers (e.g., 0,1,2,0,1,2)
        concatenated_trials = pd.concat(trials_chunks, ignore_index=True)

        # index=False prevents writing the row numbers as a new column in the CSV
        concatenated_trials.to_csv(f"results/trials_results_exp_{exp_number}.csv", index=False)

        # 2. Filter data by category (Correct Pandas syntax)
        fqa_trials = concatenated_trials[concatenated_trials["question_type"] == "factual"]
        oosqa_trials = concatenated_trials[concatenated_trials["question_type"] == "out-of-scope"]
        crqa_trials = concatenated_trials[concatenated_trials["question_type"] == "cross-reference"]

        # 3. Calculate category accuracies (Targeting the 'evaluation' column)
        # Using a fallback of 0.0 in case a category happens to be completely empty
        fqa_accuracy = float(fqa_trials["evaluation"].mean()) if not fqa_trials.empty else 0.0
        oosqa_accuracy = float(oosqa_trials["evaluation"].mean()) if not oosqa_trials.empty else 0.0
        crqa_accuracy = float(crqa_trials["evaluation"].mean()) if not crqa_trials.empty else 0.0

        # Mathematically perfect Zero-Shot Accuracy across all 33 rows
        zero_shot_acc = float(concatenated_trials["evaluation"].mean())

        # 4. Calculate Hardware Metrics
        mean_perplexity = float(concatenated_trials["perplexity"].mean())
        mean_rmse_key = float(concatenated_trials["rmse_key"].mean())
        mean_rmse_value = float(concatenated_trials["rmse_value"].mean())

        # 5. Build the row (Fixed 'bitwidth' typo to 'bit_width')
        row = {
            "experiment_id": exp_number,
            "bit_width": concatenated_trials["bit_width"].iloc[0],
            "group_size": concatenated_trials["group_size"].iloc[0],
            "mean_perplexity": mean_perplexity,
            "mean_rmse_key": mean_rmse_key,
            "mean_rmse_value": mean_rmse_value,
            "fqa_accuracy": fqa_accuracy,
            "oosqa_accuracy": oosqa_accuracy,
            "crqa_accuracy": crqa_accuracy,
            "zero_shot_accuracy": zero_shot_acc
        }

        # 6. Safely append to the Experiment CSV
        row_df = pd.DataFrame([row])

        if os.path.exists(EXPERIMENT_FILE):
            experiments = pd.read_csv(EXPERIMENT_FILE)
            # Concat is the standard way to append rows
            experiments = pd.concat([experiments, row_df], ignore_index=True)
        else:
            # If the file doesn't exist yet, this row becomes the new DataFrame
            experiments = row_df

        experiments.to_csv(EXPERIMENT_FILE, index=False)
        print(f"Experiment {exp_number} successfully aggregated and saved!")

    except Exception as e:
        raise RuntimeError(f"An error occurred. Reason: {e}")

if __name__ == '__main__':
    # Ensure the results directory exists before saving
    os.makedirs("results", exist_ok=True)
    print("Aggregating data...")
    main(0)
