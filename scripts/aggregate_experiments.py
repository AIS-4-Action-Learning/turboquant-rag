import pandas as pd
import numpy as np

EXPERIMENT_PATHS = [""]
TRIAL_PATHS = [""]

def main():
    try:
        experiment_chunks = []
        trials_chunks = []

        for i, ep in enumerate(EXPERIMENT_PATHS):
            experiment_chunks.insert(i, pd.read_csv(ep))

        for i, tp in enumerate(TRIAL_PATHS):
            trials_chunks.insert(i, pd.read_csv(tp))

    except Exception as e:
        raise RuntimeError(f"An error occured. Reason: {e}")

if __name__ == '__main__':
    print("Aggregating data...")


