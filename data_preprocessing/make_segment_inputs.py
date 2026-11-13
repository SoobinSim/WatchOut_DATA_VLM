
import json
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


BASE = Path("WatchOut_dataset")
ROOT = Path("WatchOut_MacStudio")
MANIFEST_DIR = ROOT / "data/manifests"
FRAME_DIR = ROOT / "data/frames_segments"

NUM_FRAMES = 16

segments = pd.read_csv(
    BASE / "inference_ready_400/segments_300.csv"
)

metadata = pd.read_csv(BASE / "metadata.csv")

video_paths = dict(
    zip(metadata["video_id"], metadata["video_path"])
)

with open(
    MANIFEST_DIR / "vlm_inputs_400_with_duration.json",
    "r",
    encoding="utf-8"
) as f:
    duration_data = json.load(f)

duration_map = {
    item["video_id"]: item["duration_sec"]
    for item in duration_data
}

if segments["segment_id"].duplicated().any():
    raise ValueError("중복 segment_id가 있습니다.")

inputs = []
failed = []

for idx, row in segments.iterrows():
    video_id = str(row["video_id"])
    segment_id = str(row["segment_id"])

    start_sec = float(row["start_sec"])
    end_sec = float(row["end_sec"])

    video_path = video_paths.get(video_id)
    duration = duration_map.get(video_id)

    if video_path is None or duration is None:
        failed.append((segment_id, "원본 영상 정보 없음"))
        continue

    video_path = unicodedata.normalize("NFD", video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        failed.append((segment_id, "영상 열기 실패"))
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        cap.release()
        failed.append((segment_id, "FPS 또는 프레임 수 오류"))
        continue

    timestamps = np.linspace(
        start_sec, end_sec, NUM_FRAMES, endpoint=False
    )

    output_dir = FRAME_DIR / segment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    error = None

    try:
        for frame_idx, timestamp in enumerate(timestamps):
            # 실제 비디오 프레임 인덱스로 변환
            source_idx = min(
                max(int(round(float(timestamp) * fps)), 0),
                frame_count - 1
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, source_idx)
            success, image = cap.read()

            if not success:
                error = f"프레임 읽기 실패: {timestamp:.3f}초"
                break

            actual_sec = round(
                cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3
            )

            # 일부 코덱에서 POS_MSEC가 0으로 반환되는 경우 대비
            if actual_sec <= 0 and source_idx > 0:
                actual_sec = round(source_idx / fps, 3)

            frame_id = f"f_{frame_idx:04d}"
            filename = f"frame_{frame_idx:04d}.jpg"
            output_path = output_dir / filename

            if not cv2.imwrite(str(output_path), image):
                error = f"이미지 저장 실패: {filename}"
                break

            relative_path = output_path.relative_to(ROOT)

            frames.append({
                "frame_id": frame_id,
                "timestamp_sec": actual_sec,
                "frame_path": relative_path.as_posix()
            })
    finally:
        cap.release()

    if error or len(frames) != NUM_FRAMES:
        failed.append((segment_id, error or "프레임 수 부족"))
        continue

    # 모델 입력에는 정답 유형이나 XML 라벨을 포함하지 않음
    inputs.append({
        "video_id": video_id,
        "segment_id": segment_id,
        "duration_sec": float(duration),
        "frames": frames
    })

    if (idx + 1) % 50 == 0:
        print(f"진행: {idx + 1}/{len(segments)}")


print("\n===== 생성 결과 =====")
print("전체 후보:", len(segments))
print("성공 구간:", len(inputs))
print("실패 구간:", len(failed))
print("등록된 프레임:", sum(len(x["frames"]) for x in inputs))

if failed:
    print("실패 예시:", failed[:5])
    print("실패 구간이 있어 최종 JSON은 저장하지 않습니다.")
else:
    # B2·B3는 동일 입력을 사용하고 프롬프트만 다르게 적용
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("B2", "B3"):
        output_json = MANIFEST_DIR / f"inputs_{name}.json"

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(inputs, f, ensure_ascii=False, indent=2)

        print(f"{name} JSON 저장 완료:", output_json)

    print("B2·B3 입력 생성 완료!")
