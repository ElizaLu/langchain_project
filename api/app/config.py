from pathlib import Path

KAFKA_TOPIC = "hst-gas-station"
KAFKA_BOOTSTRAP_SERVERS = ["47.96.156.60:19092"]
KAFKA_GROUP_ID = "sente_test"

BASE_DIR = Path("/home/sente/langchain_project/api")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PRED_DIR = DATA_DIR / "predictions"
NORM_PARAMS_FILE = Path("/home/sente/langchain_project/TranAD-lumh/data/processed/norm_params.npz")

# 这里放你离线预处理后保存的列顺序文件
# 推荐用 columns.json，内容应当是：
# base_feature_cols + [f"{col}_is_alive" for col in base_feature_cols]
COLUMNS_FILE = BASE_DIR / "columns.json"

MODEL_CKPT = BASE_DIR / "checkpoints" / "your_model_gas" / "model.ckpt"

GAP_THRESHOLD_MINUTES = 2
INSERT_STEP_MINUTES = 1
SCORE_SCALE = 1e5