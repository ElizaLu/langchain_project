from fastapi import FastAPI

from app.feature_mapper import FeatureMapper
from app.model_service import TranADService
from app.config import FEATURE_CODES_FILE


mapper = FeatureMapper.from_file(FEATURE_CODES_FILE, fill_value=0.0)
service = TranADService(feature_dim=len(mapper.feature_codes), scale=1e2)

app = FastAPI(title="Gas Detection API", version="2.0.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "window_size": service.window_size,
        "feature_dim": service.feature_dim,
    }