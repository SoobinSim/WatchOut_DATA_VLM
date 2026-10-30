import os
from pathlib import Path
import json
import csv
import re

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")

FRAME_DIR = BASE / "normal/processed/frames"
OUTPUT_DIR = BASE / "normal/processed"
META_PATH = BASE / "normal/normal_metadata_150.csv"
OUTPUT_PATH = OUTPUT_DIR / "vlm_inputs_normal_100.json"

NUM_FRAMES = 16

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 기존에 선별한 정상 구매 영상 목록
with META_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    metadata = list(csv.DictReader(f))

if len(metadata) != 100:
    raise RuntimeError(
        f"정상 구매 메타데이터가 100개여야 합니다. 현재 {len(metadata)}개"
    )

vlm_inputs = []
errors = []

for row in metadata:
    video_id = row["video_id"]
    video_frame_dir = FRAME_DIR / video_id

    # frame_0000.jpg 형태의 프레임 검색
    frame_paths = sorted(
        video_frame_dir.glob("frame_*.jpg"),
        key=lambda p: int(re.search(r"frame_(\d+)", p.stem).group(1))
    )

    if not frame_paths:
        errors.append(f"{video_id}: 추출된 프레임 없음")
        continue

    # 전체 구간에서 16개를 균등 선택
    if len(frame_paths) >= NUM_FRAMES:
        indices = [
            round(i * (len(frame_paths) - 1) / (NUM_FRAMES - 1))
            for i in range(NUM_FRAMES)
        ]
    else:
        # 짧은 영상은 중복 프레임 없이 모두 사용
        indices = list(range(len(frame_paths)))

    selected_frames = []

    for idx in indices:
        frame_path = frame_paths[idx]
        timestamp_sec = int(frame_path.stem.split("_")[-1])

        selected_frames.append({
            "frame_id": f"f_{timestamp_sec:04d}",
            "timestamp_sec": timestamp_sec,
            "frame_path": str(frame_path)
        })

    vlm_inputs.append({
        "video_id": video_id,
        "source_class": "normal_purchase",
        "source_dataset": "AI-Hub 238-1",
        "frames": selected_frames
    })

    print(
        f"[{len(vlm_inputs)}/100] {video_id}: "
        f"{len(selected_frames)}프레임",
        flush=True
    )

if errors:
    print("\n오류 목록:")
    for error in errors:
        print(" -", error)

    raise RuntimeError(
        f"{len(errors)}개 영상의 프레임을 찾지 못했습니다. JSON을 저장하지 않습니다."
    )

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(vlm_inputs, f, ensure_ascii=False, indent=2)

total_frames = sum(len(item["frames"]) for item in vlm_inputs)

print("\n✅ STEP 5 완료!")
print(f"정상 구매 영상: {len(vlm_inputs)}개")
print(f"VLM 입력 프레임: {total_frames}개")
print(f"저장 위치: {OUTPUT_PATH}")