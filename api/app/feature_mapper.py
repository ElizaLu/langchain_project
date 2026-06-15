import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


class FeatureMapper:
    def __init__(self, columns_file: str | Path):
        """
        直接从 json 文件读取 base feature code。
        这个文件里只存 base columns，不包含 *_is_alive。
        """
        with open(columns_file, "r", encoding="utf-8") as f:
            self.base_feature_cols = json.load(f)

        self.base_feature_set = set(self.base_feature_cols)

    def payload_to_raw_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        输出一行原始记录：
        {
            "time": timestamp,
            "feature1": value,
            ...
        }
        """
        row = {"time": pd.to_datetime(payload.get("time"), errors="coerce")}

        for col in self.base_feature_cols:
            row[col] = np.nan

        msg = payload.get("msg", [])
        if isinstance(msg, list):
            for item in msg:
                if not isinstance(item, dict):
                    continue
                code = item.get("attributeCode")
                if code in self.base_feature_set:
                    val = item.get("attributeData", np.nan)
                    try:
                        row[code] = float(val)
                    except Exception:
                        row[code] = np.nan

        return row