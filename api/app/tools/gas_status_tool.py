from pathlib import Path
import json

from langchain_core.tools import tool

LATEST_FILE = Path("/home/sente/langchain_project/api/data/predictions/_latest_prediction.json")


@tool
def get_latest_gas_status() -> str:
    """
    查询当前最新一条燃气在线推理结果，返回是否异常。
    """
    if not LATEST_FILE.exists():
        return "当前还没有最新推理结果，请先等待在线推理程序产生数据。"

    try:
        data = json.loads(LATEST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return "最新推理结果文件损坏，无法读取。"

    time_str = data.get("time", "未知时间")
    overall_anomaly = data.get("overall_anomaly")
    overall_score = data.get("overall_score", "未知分数")
    abnormal_sensors = data.get("abnormal_sensors", [])

    if overall_anomaly:
        status = "异常"
    else:
        status = "正常"

    return (
        f"最新推理时间：{time_str}\n"
        f"状态：{status}\n"
        f"overall_score：{overall_score}\n"
        f"abnormal_sensors：{abnormal_sensors}"
    )