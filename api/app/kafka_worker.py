from __future__ import annotations

import json

from kafka import KafkaConsumer

from app.config import (
    KAFKA_TOPIC,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    RAW_DIR,
    PROCESSED_DIR,
    PRED_DIR,
    COLUMNS_FILE,
    SCORE_SCALE,
    NORM_PARAMS_FILE,
)
from app.feature_mapper import FeatureMapper
from app.preprocess import OnlinePreprocessor
from app.window_buffer import ProcessedWindowBuffer
from app.storage import DailyJsonlStorage
from app.model_service import TranADService
from app.normalizer import MinMaxNormalizer


def main():
    # 1) 读取训练时列顺序
    mapper = FeatureMapper(COLUMNS_FILE)

    # 2) 预处理器：只保留全局上一条记录
    preprocessor = OnlinePreprocessor(
        base_feature_cols=mapper.base_feature_cols,
        gap_threshold_minutes=2,
        step_minutes=1,
    )

    normalizer = MinMaxNormalizer.from_file(NORM_PARAMS_FILE)

    # 3) 模型
    model_service = TranADService(
        feature_names=preprocessor.model_cols,
        scale=SCORE_SCALE,
    )

    # 4) 全局滑动窗口
    buffer = ProcessedWindowBuffer(window_size=model_service.window_size)

    # 5) 按天落盘
    storage = DailyJsonlStorage(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        pred_dir=PRED_DIR,
    )

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda x: x.decode("utf-8", errors="ignore"),
        key_deserializer=lambda x: x.decode("utf-8", errors="ignore") if x else None,
    )

    def handle_message(payload: dict):
        # 1. 原始消息存盘
        storage.save_raw_payload(payload)

        # 2. 预处理：若gap大于两分钟则填充
        processed_rows = preprocessor.process_payload(payload)

        for row in processed_rows:
            # 3. 处理后数据存盘
            storage.save_processed_row(row)

            # 4. 转成模型向量，筛选特征并保持和训练时一致的列顺序
            vec = preprocessor.row_to_vector(row, normalizer)

            # 5. 进入全局窗口
            buffer.push(vec)

            # 6. 窗口不够就跳过
            if not buffer.ready():
                continue

            # 7. 推理
            window_np = buffer.get()
            result = model_service.predict_window(window_np) # pred为[1, D]

            storage.save_prediction(row, result)
            storage.save_latest_prediction(row, result)

            if result["overall_anomaly"]:
                print(
                    f"time={row.get('time')} "
                    f"overall_score={result['overall_score']:.6f} "
                    f"ANOMALY sensors={result['abnormal_sensors']}"
                )
            else:
                print(
                    f"time={row.get('time')} "
                    f"overall_score={result['overall_score']:.6f} "
                    f"normal"
                )

    print("开始接收 Kafka 消息...")

    try:
        for msg in consumer:
            raw = msg.value
            try:
                payload = json.loads(raw)
            except Exception:
                print("收到的消息不是 JSON，跳过")
                continue

            # 真正调用处理函数
            handle_message(payload)

    except KeyboardInterrupt:
        print("手动停止")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()