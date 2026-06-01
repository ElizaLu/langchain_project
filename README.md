# TranAD-lumh

A research-oriented fork of **TranAD** for **multivariate gas anomaly detection** in industrial sensor streams.

This repository keeps the core TranAD architecture while adapting the pipeline for gas monitoring scenarios, including custom data preprocessing, score calibration, and POT-based thresholding for anomaly decision making.

## Why this fork

Compared with the original TranAD repository, this version is tailored to a gas-monitoring workflow:

- Custom preprocessing for raw gas sensor text data
- Window-based time-series modeling
- Transformer-based reconstruction and anomaly scoring
- Score transformation for more stable thresholding
- POT/SPOT-based threshold estimation
- Per-dimension inspection and visualization
- Training / testing / plotting integrated in one pipeline

## Main idea

The workflow is:

1. Load raw gas sensor data
2. Preprocess and split it into train / test / labels
3. Convert sequences into sliding windows
4. Train TranAD on normal patterns
5. Compute reconstruction error as anomaly score
6. Calibrate scores and estimate thresholds with POT
7. Visualize predictions, scores, and labels

## Repository structure

```text
TranAD-lumh/
├── data/
│   └── raw_txt/              # raw gas sensor text files
├── src/
│   ├── models.py             # TranAD model definition
│   ├── pot.py                # POT/SPOT thresholding utilities
│   ├── plotting.py           # visualization helpers
│   ├── diagnosis.py          # detection / diagnosis metrics
│   ├── utils.py              # helper functions
│   └── parser.py             # command-line arguments
├── preprocess.py             # preprocessing entry point
├── main.py                   # training / testing / evaluation
├── main.sh                   # example run script
├── requirements.txt
└── README.md
```

## Features

- Multivariate time-series anomaly detection
- Transformer-based reconstruction model
- Windowed sequence modeling
- Train / test separation
- Per-sensor anomaly score analysis
- Threshold calibration using POT/SPOT
- Saved checkpoints for reuse
- Plot generation for debugging and reporting

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- pandas
- tqdm
- scikit-learn
- matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

If your environment needs a specific PyTorch build, install PyTorch first and then install the remaining requirements.

## Data preparation

The pipeline expects preprocessed NumPy arrays such as:

```text
train.npy
test.npy
labels.npy
```

A typical preprocessing workflow is:

```bash
python preprocess.py
```

If you are using a custom gas dataset, make sure the preprocessing script writes the final arrays to the data directory expected by `src/parser.py`.

## Training and evaluation

Train the model:

```bash
python main.py --model TranAD --retrain
```

Run testing / evaluation:

```bash
python main.py --model TranAD --test
```

If you want to train on a smaller subset for quick debugging, use the corresponding reduced-data option in your argument configuration.

## Output

During execution, the script will typically produce:

- training loss logs
- saved checkpoints
- anomaly scores
- POT thresholds
- evaluation metrics
- plots for each sensor dimension

## Score calibration

For this fork, anomaly scores can be post-processed before thresholding to make the tail behavior more stable.

A typical calibration pipeline is:

```text
raw reconstruction loss
→ score scaling
→ log1p transform
→ POT thresholding
```

This is useful when the raw loss values are very small and their distribution is highly concentrated.

## Model overview

TranAD is a Transformer-based anomaly detection model for multivariate time series. It learns to reconstruct the most recent observation in a sliding window and uses reconstruction error as an anomaly signal.

In this repository, the model is used as the core detector for gas sensor streams.

## Customization

You may want to adjust the following parts depending on your dataset:

- `model.n_window` in `src/models.py`
- learning rate and batch size
- preprocessing rules for gas sensor files
- score scaling factor before `log1p`
- POT `level` and `q` parameters
- plotting and diagnosis logic

## Notes

- The model is most stable when the input data are properly normalized.
- If the dataset contains only normal samples, thresholding results should be interpreted carefully.
- The final threshold depends on both the score distribution and the calibration strategy.

## Citation

If you use this repository, please cite the original TranAD paper:

```bibtex
@article{tuli2022tranad,
  title={{TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data}},
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R},
  journal={Proceedings of VLDB},
  volume={15},
  number={6},
  pages={1201-1214},
  year={2022}
}
```

## License

This project retains the original BSD-3-Clause license unless otherwise stated. See `LICENSE` for details.
