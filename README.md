# MLP Comparison: NumPy vs TensorFlow on MNIST

This repository compares two implementations of an identical multilayer perceptron (MLP) trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/):

- **NumPy model:** Manual implementation with explicitly coded forward and backward passes  
- **TensorFlow/Keras model:** Uses automatic differentiation and optimised tensor kernels

The goal is to isolate how implementation strategy — rather than model architecture — affects optimisation behaviour, numerical stability, and computational performance. A full academic report is available in `report/report.pdf`.

## Project Structure

```
├── src/
│   ├── numpy_mlp.py                    # Manual MLP (forward + backward)
│   ├── tensorflow_mlp.py               # TensorFlow/Keras implementation
│   ├── experiments.py                  # Full experimental pipeline
│   └── plot_gen.py                     # Plot generation utilities
│
├── visualiser/
│   └── functionality_visualiser.py     # Optional MNIST visualisation tool
│
├── data/                               # MNIST training/testing CSV files
│
├── results/
│   ├── expData/ 
│   │   ├── processedData/              # Averaged histories, master CSVs
│   │   └── rawData/                    # Per-run loss/accuracy logs
│   └── plots/                          # Generated figures
│
├── report/
│   ├── report.pdf
│   └── tex_report/
│       └── report.tex
│
└── environment.yml
```

## Installation

Conda (recommended):
```bash
conda env create -f environment.yml
conda activate mnist-env
```

Pip:
```pip
pip install numpy pandas matplotlib seaborn scipy pillow tensorflow
```
## Running Experiments

Run the full training grid:
`python src/experiments.py`

This script:

- trains NumPy and TensorFlow MLPs across all hyperparameters
- logs loss and accuracy histories
- measures training time and test accuracy
- saves results into results/expData/

Generate all figures:
`python src/plot_gen.py`

Generated plots are stored in `results/plots/`.

## Visualiser (Optional)

A standalone MNIST visualisation tool is included but not used in the experiments:

`python visualiser/functionality_visualiser.py`

## Reproducibility

The full pipeline is scripted and deterministic where possible. The environment is fully specified in environment.yml, raw logs are stored in `results/expData/rawData/`, and averaged histories and consolidated CSVs are provided in `results/expData/processedData/`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Additional Resources

See `report/report.pdf` for the full study.

- [NumPy](https://numpy.org/) – Fundamental Python library for numerical computing
- [TensorFlow](https://www.tensorflow.org/) – Framework for machine learning and neural networks
- [Keras](https://keras.io/) – High-level API for TensorFlow
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) – Standard dataset of handwritten digits