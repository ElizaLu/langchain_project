from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd


class OnlinePreprocessor:
    def __init__(
        self,
        base_feature_cols: List[str],
        gap_threshold_minutes: int = 2,
        step_minutes: int = 1,
    ):
        self.base_feature_cols = list(base_feature_cols)
        self.state_cols = [f"{c}_is_alive" for c in self.base_feature_cols]

        self.gap_threshold = pd.Timedelta(minutes=gap_threshold_minutes)
        self.step = pd.Timedelta(minutes=step_minutes)

        self.last_raw: dict[str, Any] | None = None

    def _build_state_row(self, row: Dict[str, Any], synthetic: bool) -> Dict[str, Any]:
        out = dict(row)
        for col in self.base_feature_cols:
            if synthetic:
                out[f"{col}_is_alive"] = 0.0
            else:
                out[f"{col}_is_alive"] = 0.0 if pd.isna(out.get(col)) else 1.0
        return out

    def _payload_to_raw_row(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        row = {"time": pd.to_datetime(payload.get("time"), errors="coerce")}
        for col in self.base_feature_cols:
            row[col] = np.nan

        msg = payload.get("msg", [])
        if isinstance(msg, list):
            for item in msg:
                if not isinstance(item, dict):
                    continue
                code = item.get("attributeCode")
                if code in self.base_feature_cols:
                    try:
                        row[code] = float(item.get("attributeData", np.nan))
                    except Exception:
                        row[code] = np.nan

        return row

    def _generate_rows_between(self, prev: Dict[str, Any], curr: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根据 prev 和 curr 生成：
        - 若间隔 > 2 分钟，插入中间点
        - 最后追加 curr
        """
        prev_t = prev["time"]
        curr_t = curr["time"]

        if pd.isna(prev_t) or pd.isna(curr_t):
            return [self._build_state_row(curr, synthetic=False)]

        delta = curr_t - prev_t
        if delta <= pd.Timedelta(0):
            print("警告：当前记录时间不晚于上一条记录，跳过插值")
            return [self._build_state_row(curr, synthetic=False)]

        rows: List[Dict[str, Any]] = []

        if delta > self.gap_threshold:
            missing_rows = max(1, int(round(delta / self.step)) - 1)

            for k in range(1, missing_rows + 1):
                new_time = prev_t + self.step * k
                ratio = (new_time - prev_t) / delta

                new_row = {"time": new_time}
                for col in self.base_feature_cols:
                    v1 = prev.get(col, np.nan)
                    v2 = curr.get(col, np.nan)

                    if pd.notna(v1) and pd.notna(v2):
                        new_row[col] = float(v1 + (v2 - v1) * ratio)
                    elif pd.notna(v1):
                        new_row[col] = float(v1)
                    elif pd.notna(v2):
                        new_row[col] = float(v2)
                    else:
                        new_row[col] = np.nan

                rows.append(self._build_state_row(new_row, synthetic=True))

        rows.append(self._build_state_row(curr, synthetic=False))
        return rows

    def process_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        输入一条 Kafka 消息，输出 1 条或多条“处理后记录”
        """
        curr_raw = self._payload_to_raw_row(payload)
        prev_raw = self.last_raw
        self.last_raw = curr_raw

        if prev_raw is None:
            rows = [self._build_state_row(curr_raw, synthetic=False)]
        else:
            rows = self._generate_rows_between(prev_raw, curr_raw)

        df = pd.DataFrame(rows)
        if df.empty:
            return []

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        df = df.set_index("time")
        df[self.base_feature_cols] = (
            df[self.base_feature_cols]
            .interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )
        df = df.reset_index()

        return df.to_dict("records")

    def row_to_vector(self, row: Dict[str, Any], normalizer) -> np.ndarray:
        cols = self.base_feature_cols + self.state_cols
        vec = []

        for c in cols:
            try:
                vec.append(float(row.get(c, np.nan)))
            except Exception:
                vec.append(np.nan)

        vec = np.asarray(vec, dtype=np.float64)
        print(f"原始向量: {vec}")
        vec = normalizer.transform(vec)
        return vec

    @property
    def model_cols(self) -> List[str]:
        return self.base_feature_cols + self.state_cols