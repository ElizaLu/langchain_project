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
## Feature List

| Index | Attribute Code  | Feature Name | Unit  | availability |
| ----- | --------------- | ------------ | ----- |------------- |
| 0     | 400002_TT101_AI | 进口压力         | Kpa   | 未使用 |
| 1     | 400003_PT101_AI | 出口压力         | Kpa   | 
| 2     | 400007_BKSS_FL1 | 1#流量计标况瞬时流量  | Nm3/h | 未使用 |
| 3     | 400009_WD_FL1   | 1#流量计温度      | ℃     | 坏了 |
| 4     | 400011_YL_FL1   | 1#流量计压力      | Kpa   | 坏了 |
| 5     | 400013_BKLJ_FL1 | 1#流量计标况累计流量  | Nm3   | 未使用 |
| 6     | 400015_BKSS_FL2 | 2#流量计标况瞬时流量  | Nm3/h | 未使用 |
| 7     | 400017_BKLJ_FL2 | 2#流量计标况累计流量  | Nm3   | 未使用 |
| 8     | 400019_WD_FL2   | 2#流量计温度      | ℃     |
| 9     | 400021_YL_FL2   | 2#流量计压力      | Kpa   |
| 10    | 400023_BKSS_FL3 | 3#流量计标况瞬时流量  | Nm3/h | 未使用 |
| 11    | 400025_BKLJ_FL3 | 3#流量计标况累计流量  | Nm3   | 未使用 |
| 12    | 400027_WD_FL3   | 3#流量计温度      | ℃     |
| 13    | 400029_YL_FL3   | 3#流量计压力      | Kpa   |
| 14    | 400037_XQYL_AI  | 新撬压力         | Kpa   |
| 15    | 400039_XQXL_AI  | 新撬泄露         | LEL%  |
| 16    | 400041_XQFW_AI  | 新撬阀位         | %     |
| 17    | 400043_YL_FL4   | 4#流量计压力      | Kpa   |
| 18    | 400045_WD_FL4   | 4#流量计温度      | ℃     |
| 19    | 400047_BKSS_FL4 | 4#流量计标况瞬时流量  | Nm3/h |
| 20    | 400049_BKLJ_FL4 | 4#流量计标况累计流量  | Nm3   |
|       |                 |              |       |

The data of sensors that are not available will be deleted.

| Index | Attribute Code  | Feature Name | Unit  | availability |
| ----- | --------------- | ------------ | ----- | ------------ |
| 0     | 400003_PT101_AI | 出口压力         | Kpa   |              |
| 1     | 400019_WD_FL2   | 2#流量计温度      | ℃     |              |
| 2     | 400021_YL_FL2   | 2#流量计压力      | Kpa   |              |
| 3     | 400027_WD_FL3   | 3#流量计温度      | ℃     |              |
| 4     | 400029_YL_FL3   | 3#流量计压力      | Kpa   |              |
| 5     | 400037_XQYL_AI  | 新撬压力         | Kpa   |              |
| 6     | 400039_XQXL_AI  | 新撬泄露         | LEL%  |              |
| 7     | 400041_XQFW_AI  | 新撬阀位         | %     |              |
| 8     | 400043_YL_FL4   | 4#流量计压力      | Kpa   |              |
| 9     | 400045_WD_FL4   | 4#流量计温度      | ℃     |              |
| 10    | 400047_BKSS_FL4 | 4#流量计标况瞬时流量  | Nm3/h |              |
| 11    | 400049_BKLJ_FL4 | 4#流量计标况累计流量  | Nm3   |              |

The state columns are used to indicate whether a data point is an original sensor measurement or a generated value. If sensor data is interrupted and missing timestamps are filled through interpolation, the corresponding state columns are set to **0**. Likewise, if the original sensor value is **NaN**, the corresponding state column is also set to **0**. All valid original sensor measurements have their state columns set to **1**.

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
