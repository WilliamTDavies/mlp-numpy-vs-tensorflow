import numpy as np
import tensorflow as tf
import pandas as pd
import numpy_mlp as np_mlp
import tensorflow_mlp as tf_mlp
import time
import re
from pathlib import Path

def setUp():
    #Ensure only the CPU is used to ensure greater similarities
    tf.config.set_visible_devices([], 'GPU')

    #Set random seeds for reproducible results
    tf.random.set_seed(0)
    np.random.seed(0)

def paths():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = ROOT_DIR / "results"
    EXPDATA_DIR = RESULTS_DIR / "expData"
    PLOTS_DIR = RESULTS_DIR / "plots"
    return EXPDATA_DIR, PLOTS_DIR

def experimentParams():
    trials = 5
    batch_sizes = [32, 64, 128, 256, 512]
    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1]
    epochNum = 10
    return trials, batch_sizes, learning_rates, epochNum

def runExperiment(trials, epochNum, batch_sizes, learning_rates, EXPDATA_DIR):
    records = []
    X_train, y_train = np_mlp.getData() #Use same training data for both models
    X_test, y_test = np_mlp.getTest()
    #Generate experimental data
    for batch in batch_sizes:
        for lr in learning_rates:
            for trial in range(1, trials + 1):
                #TensorFlow
                start = time.perf_counter()
                tf_model, tf_history = tf_mlp.modelCreation(X_train, y_train, epochs=epochNum, batchSize=batch, lr=lr)
                tf_time = time.perf_counter() - start
                tf_acc = tf_mlp.accuracy(tf_model, X_test, y_test)
                #Save tf model data
                TFDATA_DIR = EXPDATA_DIR / "rawData" / f"tf_{batch}_{lr}_{trial}.csv"
                pd.DataFrame(tf_history).to_csv(TFDATA_DIR, index=False)
                records.append({
                    "trial": trial,
                    "model": "tensorflow",
                    "batch_size": batch,
                    "learning_rate": lr,
                    "training_time_s": tf_time,
                    "test_accuracy_pct": tf_acc
                })
                #Numpy
                start = time.perf_counter()
                np_model, np_history = np_mlp.modelCreation(X_train, y_train, epochs=epochNum, batchSize=batch, lr=lr)
                np_time = time.perf_counter() - start
                np_acc = np_mlp.accuracy(np_model, X_test, y_test)
                #Save np model data
                NPDATA_DIR = EXPDATA_DIR / "rawData" / f"np_{batch}_{lr}_{trial}.csv"
                pd.DataFrame(np_history).to_csv(NPDATA_DIR, index=False)
                records.append({
                    "trial": trial,
                    "model": "numpy",
                    "batch_size": batch,
                    "learning_rate": lr,
                    "training_time_s": np_time,
                    "test_accuracy_pct": np_acc
                })
    #Create main CSV
    df = pd.DataFrame(records)
    CSV_DIR = EXPDATA_DIR / "processedData" / "tf_vs_np.csv"
    df.to_csv(CSV_DIR, index=False)

def loadAvgHistory(batch, lr, EXPDATA_DIR):
    INPUT_DIR = EXPDATA_DIR / "rawData"
    OUTPUT_DIR = EXPDATA_DIR / "processedData"
    pattern = re.compile(
    r"(?P<model>[^_]+)_(?P<batch>\d+)_(?P<lr>[\d\.e-]+)_(?P<trial>\d+)\.csv"
    )

    groups = {}

    for file in INPUT_DIR.glob("*.csv"):
        name = file.name
        match = pattern.match(name)
        if not match:
            continue
        model = match.group("model")
        batch = match.group("batch")
        lr = match.group("lr")

        key = (model, batch, lr)
        groups.setdefault(key, []).append(file)

    # Process each group
    for (model, batch, lr), file_list in groups.items():
        dfs = [pd.read_csv(f) for f in file_list]
        min_len = min(len(df) for df in dfs)
        dfs = [df.iloc[:min_len] for df in dfs]
        avg_df = sum(dfs) / len(dfs)
        avg_df = avg_df.round(6)
        # Save
        out_name = f"{model}_{batch}_{lr}_averaged.csv"
        avg_df.to_csv((OUTPUT_DIR / out_name), index=False)

def main():
    setUp()
    EXPDATA, _ = paths()
    trials, batch_sizes, learning_rates, epochs = experimentParams()
    runExperiment(trials, epochs, batch_sizes, learning_rates, EXPDATA)
    loadAvgHistory(batch_sizes, learning_rates, EXPDATA)
    print("\nAll experiments complete.\n")

#To ensure the program is not accidentally run
if __name__ == "__main__":
    main()