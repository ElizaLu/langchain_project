from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json

TRANAD_ROOT = Path("/home/sente/langchain_project/TranAD-lumh")

sys.path.insert(0, str(TRANAD_ROOT))

from src.models import TranAD


class TranADService:
    def __init__(self, feature_names: List[str], scale: float = 1e5):
        self.scale = scale
        self.feature_names = feature_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.optimizer, self.scheduler = self._load_model(len(feature_names))
        self.window_size = self.model.n_window
        self.feature_dim = self.model.n_feats
        self.overall_threshold, self.sensor_thresholds = self._load_thresholds()

    def _load_model(self, dims: int):
        model = TranAD(dims).double().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

        fname = f"/home/sente/langchain_project/TranAD-lumh/checkpoints/TranAD_gas/model.ckpt"
        if os.path.exists(fname):
            checkpoint = torch.load(fname, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        model.eval()
        return model, optimizer, scheduler

    def _load_thresholds(self):
        fname = f"/home/sente/langchain_project/TranAD-lumh/checkpoints/TranAD_gas/thresholds.json"
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"Missing threshold file: {fname}. "
                f"Please run offline calibration first."
            )

        with open(fname, "r", encoding="utf-8") as f:
            obj = json.load(f)

        overall_threshold = float(obj["overall_threshold"])
        sensor_thresholds = np.asarray(obj["sensor_thresholds"], dtype=np.float64)

        if len(sensor_thresholds) != len(self.feature_names):
            raise ValueError(
                f"threshold dim mismatch: got {len(sensor_thresholds)}, "
                f"expected {len(self.feature_names)}"
            )

        return overall_threshold, sensor_thresholds


    def predict_window(self, window_np: np.ndarray) -> Dict[str, Any]:
        """
        window_np: [T, D]
        返回:
            {
              overall_score,
              overall_threshold,
              overall_anomaly,
              channel_scores,
              abnormal_sensors,
              prediction
            }
        """
        x = torch.as_tensor(window_np, dtype=torch.double, device=self.device)  # [T, D]
        window = x.unsqueeze(1)      # [T, 1, D]
        elem = window[-1:, :, :]     # [1, 1, D]

        with torch.no_grad():
            out = self.model(window, elem)
            if isinstance(out, tuple):
                out = out[1]

            pred = out.squeeze(0)       # [1, D]
            target = elem.squeeze(0)    # [1, D]

            # 每个传感器一个 loss
            channel_loss = (pred - target).pow(2).squeeze(0)  # [D]
            channel_scores = torch.log1p(channel_loss * self.scale)  # [D]

            overall_score = channel_scores.mean().item()
            channel_scores_np = channel_scores.detach().cpu().numpy()

            overall_anomaly = bool(overall_score > self.overall_threshold)

            abnormal_sensors = []
            if overall_anomaly:
                idxs = np.where(channel_scores_np > self.sensor_thresholds)[0]
                abnormal_sensors = [self.feature_names[i] for i in idxs]

        return {
            "overall_score": float(overall_score),
            "overall_threshold": float(self.overall_threshold),
            "overall_anomaly": overall_anomaly,
            "channel_scores": channel_scores_np.tolist(),
            "abnormal_sensors": abnormal_sensors,
            "prediction": pred.detach().cpu().numpy().tolist(),
        }