
import json
import unicodedata
from pathlib import Path

import cv2
import pandas as pd


BASE = Path("WatchOut_dataset")
ROOT = Path("WatchOut_MacStudio")

INPUT_PATH = ROOT / "data/manifests/vlm_inputs_400_portable.json"
OUTPUT_PATH = ROOT / "data/manifests/vlm_inputs_400_with_duration.json"


def get_duration(video_path):
    # macOS 한글 경로의 유니코드 표기 차이 처리
    path = unicodedata.normalize("NFD", str(video_path))

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or frame_count <= 0:
            return None

        return round(frame_count / fps, 3)
    finally:
        cap.release()


abnormal = pd.read_csv(BASE / "metadata.csv")
normal = pd.read_csv(
    BASE / "inference_ready_400/normal_metadata_100.csv"
)

metadata = pd.concat(
    [
        abnormal[["video_id", "video_path"]],
        normal[["video_id", "video_path"]],
    ],
    ignore_index=True,
)

if metadata["video_id"].duplicated().any():
    raise ValueError("메타데이터에 중복 video_id가 있습니다.")

video_paths = dict(zip(metadata["video_id"], metadata["video_path"]))

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

failed = []

for item in data:
    video_id = item["video_id"]
    video_path = video_paths.get(video_id)

    duration = get_duration(video_path) if video_path else None

    if duration is None:
        failed.append(video_id)

    item["duration_sec"] = duration

print("전체 영상:", len(data))
print("영상 길이 측정 성공:", len(data) - len(failed))
print("측정 실패:", len(failed))
print("첫 번째 영상 길이:", data[0]["duration_sec"])

if failed:
    print("실패 영상 예시:", failed[:5])
    print("측정 실패가 있어 JSON을 저장하지 않았습니다.")
else:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("저장 완료:", OUTPUT_PATH)
