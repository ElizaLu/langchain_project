from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


class DailyJsonlStorage:
    def __init__(self, raw_dir: Path, processed_dir: Path, pred_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.pred_dir = Path(pred_dir)
        self.latest_pred_file = self.pred_dir / "_latest_prediction.json"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir.mkdir(parents=True, exist_ok=True)


    def save_latest_prediction(self, row, result):
        payload = {
            "time": str(row.get("time")),
            "overall_anomaly": result.get("overall_anomaly"),
            "overall_score": result.get("overall_score"),
            "abnormal_sensors": result.get("abnormal_sensors", []),
        }

        tmp_file = self.latest_pred_file.with_suffix(".json.tmp")
        tmp_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_file.replace(self.latest_pred_file)


    def _day(self, t) -> str:
        ts = pd.to_datetime(t, errors="coerce")
        if pd.isna(ts):
            return "unknown_date"
        return ts.strftime("%Y-%m-%d")

    def save_raw_payload(self, payload: Dict[str, Any]) -> None:
        day = self._day(payload.get("time"))
        path = self.raw_dir / f"{day}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_processed_row(self, row: Dict[str, Any]) -> None:
        day = self._day(row.get("time"))
        path = self.processed_dir / f"{day}.jsonl"
        record = dict(row)
        if "time" in record:
            record["time"] = str(pd.to_datetime(record["time"], errors="coerce"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_prediction(self, row: Dict[str, Any], result: Dict[str, Any]) -> None:
        day = self._day(row.get("time"))
        path = self.pred_dir / f"{day}.jsonl"
        record = {
            "time": str(pd.to_datetime(row.get("time"), errors="coerce")),
            "result": result,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")