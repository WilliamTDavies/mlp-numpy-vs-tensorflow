from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def paths():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = ROOT_DIR / "results"
    EXPDATA_DIR = RESULTS_DIR / "expData"
    PLOTS_DIR = RESULTS_DIR / "plots"
    return EXPDATA_DIR, PLOTS_DIR

def plotLossCurves(EXPDATA_DIR, PLOTS_DIR):
    DATA_DIR = EXPDATA_DIR / "processedData"
    plt.figure(figsize=(8, 5))
    batches = set()
    lrs = set()

    for csv in DATA_DIR.glob("*_averaged.csv"):
        stem = csv.stem.split("_")
        batches.add(int(stem[1]))
        lrs.add(float(stem[2]))

    batches = sorted(list(batches))
    lrs = sorted(list(lrs))
    #Pick representative small / mid / large
    rep_batches = [batches[0], batches[len(batches)//2], batches[-1]]
    rep_lrs = [lrs[0], lrs[len(lrs)//2], lrs[-1]]

    #Plot only the representative curves ---
    for csv in DATA_DIR.glob("*_averaged.csv"):
        hist = pd.read_csv(csv)
        stem = csv.stem.split("_")
        model = stem[0] #tf or np
        batch = int(stem[1])
        lr = float(stem[2])
        #Skip anything not representative
        if batch not in rep_batches or lr not in rep_lrs:
            continue

        linestyle = "-" if model == "tf" else "--"
        label = f"{model.upper()} | batch={batch} | lr={lr}"
        plt.plot(hist["loss"], linestyle=linestyle, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curves (Representative Hyperparameters)")
    plt.legend(
        fontsize=5,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )
    plt.tight_layout(rect=[0, 0, 0.8, 1]) 
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curves.svg")
    plt.close()

def plotTimeAccBatch(EXPDATA_DIR, PLOTS_DIR):
    csv_path = EXPDATA_DIR / "processedData" / "tf_vs_np.csv"
    df = pd.read_csv(csv_path)
    #Determine learning rate subset
    all_lrs = sorted(df["learning_rate"].unique())
    lr_subset = [
        all_lrs[0],
        all_lrs[len(all_lrs)//2],
        all_lrs[-1]
    ]
    df = df[df["learning_rate"].isin(lr_subset)]
    markers = {
        "tensorflow": "o",
        "numpy": "s"
    }

    for metric, label in [
        ("training_time_s", "Training Time (s)"),
        ("test_accuracy_pct", "Test Accuracy (%)")
    ]:
        plt.figure(figsize=(8,6))

        for model in ["tensorflow", "numpy"]:
            subset = df[df["model"] == model]
            for lr in lr_subset:
                data = (
                    subset[subset["learning_rate"] == lr]
                    .groupby("batch_size")[metric]
                    .mean()
                )
                if len(data) == 0:
                    continue

                plt.plot(
                    data.index,
                    data.values,
                    marker=markers[model],
                    linestyle="-",
                    label=f"{model} | lr={lr}"
                )
        plt.xlabel("Batch Size")
        plt.ylabel(label)
        plt.title(f"{label} vs Batch Size")
        plt.grid(True)
        plt.legend(title="Model / LR", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        out = PLOTS_DIR / f"{metric}_batch_comparison.svg"
        plt.savefig(out)
        plt.close()

def plotTimeAccLr(EXPDATA_DIR, PLOTS_DIR):
    csv_path = EXPDATA_DIR / "processedData" / "tf_vs_np.csv"
    df = pd.read_csv(csv_path)
    #Determine batch size subset
    all_batch = sorted(df["batch_size"].unique())
    batch_subset = [
        all_batch[0],
        all_batch[len(all_batch)//2],
        all_batch[-1]
    ]
    df = df[df["batch_size"].isin(batch_subset)]
    markers = {
        "tensorflow": "o",
        "numpy": "s"
    }

    for metric, label in [
        ("training_time_s", "Training Time (s)"),
        ("test_accuracy_pct", "Test Accuracy (%)")
    ]:
        plt.figure(figsize=(8,6))

        for model in ["tensorflow", "numpy"]:
            subset = df[df["model"] == model]
            for batchSize in batch_subset:
                data = (
                    subset[subset["batch_size"] == batchSize]
                    .groupby("learning_rate")[metric]
                    .mean()
                )
                if len(data) == 0:
                    continue

                plt.plot(
                    data.index,
                    data.values,
                    marker=markers[model],
                    linestyle="-",
                    label=f"{model} | bs={batchSize}"
                )
        plt.xlabel("Learning Rate")
        plt.ylabel(label)
        plt.title(f"{label} vs Learning Rate")
        plt.grid(True)
        plt.legend(title="Model / Batch Size", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        out = PLOTS_DIR / f"{metric}_lr_comparison.svg"
        plt.savefig(out)
        plt.close()

def plotAccHeatmaps (EXPDATA_DIR, PLOTS_DIR):
    df = pd.read_csv(EXPDATA_DIR / "processedData" / "tf_vs_np.csv")

    #Ensure seaborn style
    sns.set_theme(context="notebook")

    for model in ["numpy", "tensorflow"]:
        sub = df[df["model"] == model]
        heat = sub.pivot_table(
            index="batch_size",
            columns="learning_rate",
            values="test_accuracy_pct",
            aggfunc="mean"
        )

        plt.figure(figsize=(7,5))
        sns.heatmap(
            heat,
            annot=True,
            fmt=".1f",
            cmap="viridis"
        )

        plt.title(f"Test Accuracy (%) Heatmap - {model.capitalize()}")
        plt.xlabel("Learning Rate")
        plt.ylabel("Batch Size")
        plt.tight_layout()

        out_path = PLOTS_DIR / f"acc_heatmap_{model}.svg"
        plt.savefig(out_path)
        plt.close()

def plotTimeHeatmaps (EXPDATA_DIR, PLOTS_DIR):
    df = pd.read_csv(EXPDATA_DIR / "processedData" / "tf_vs_np.csv")

    #Ensure seaborn style
    sns.set_theme(context="notebook")

    for model in ["numpy", "tensorflow"]:
        sub = df[df["model"] == model]
        heat = sub.pivot_table(
            index="batch_size",
            columns="learning_rate",
            values="training_time_s",
            aggfunc="mean"
        )

        plt.figure(figsize=(7,5))
        sns.heatmap(
            heat,
            annot=True,
            fmt=".1f",
            cmap="viridis"
        )

        plt.title(f"Training Time (s) Heatmap - {model.capitalize()}")
        plt.xlabel("Learning Rate")
        plt.ylabel("Batch Size")
        plt.tight_layout()

        out_path = PLOTS_DIR / f"time_heatmap_{model}.svg"
        plt.savefig(out_path)
        plt.close()

def plotSpeedAccuracyScatter(EXPDATA_DIR, PLOTS_DIR):
    df = pd.read_csv(EXPDATA_DIR / "processedData" / "tf_vs_np.csv")
    grouped = (
        df.groupby(["model", "batch_size", "learning_rate"])
          .agg({"training_time_s": "mean", "test_accuracy_pct": "mean"})
          .reset_index()
    )
    unique_lrs = sorted(grouped.learning_rate.unique())
    batch_markers = {b: m for b, m in zip(unique_lrs, ["o", "s", "D", "^", "x"])}
    cmap = "viridis"

    for model in ["tensorflow", "numpy"]:
        sub = grouped[grouped.model == model]
        plt.figure(figsize=(7,6))
        for lr in unique_lrs:
            sb = sub[sub.learning_rate == lr]
            plt.scatter(
                sb["training_time_s"],
                sb["test_accuracy_pct"],
                s=90,
                marker=batch_markers[lr],
                c=sb["batch_size"],
                cmap=cmap,
                alpha=0.8,
                label=f"Rate: {lr}"
            )

        plt.xlabel("Training Time (s)")
        plt.ylabel("Test Accuracy (%)")
        plt.title(f"Speed-Accuracy Tradeoff ({model})")
        plt.grid(True)
        cbar = plt.colorbar()
        cbar.set_label("Batch Size")
        plt.legend(title="Batch Size", loc="best")
        out = PLOTS_DIR / f"speed_accuracy_scatter_{model}.svg"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

def main():
    EXPDATA, PLOTS = paths()
    plotLossCurves(EXPDATA, PLOTS)
    plotTimeAccBatch(EXPDATA, PLOTS)
    plotTimeAccLr(EXPDATA, PLOTS)
    plotAccHeatmaps(EXPDATA, PLOTS)
    plotTimeHeatmaps(EXPDATA, PLOTS)
    plotSpeedAccuracyScatter(EXPDATA, PLOTS)

#Prevents file from accidentally being run
if __name__ == "__main__":
    main()