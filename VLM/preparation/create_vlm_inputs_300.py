import os
from pathlib import Path
from collections import Counter
import csv
import json

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")

INPUT_CSV = BASE / "processed" / "frames_metadata_300.csv"
OUTPUT_JSON = BASE / "processed" / "vlm_inputs_300.json"

NUM_FRAMES = 16

# 1. 전처리된 프레임 정보 불러오기
with INPUT_CSV.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

videos = {}

for row in rows:
    key = (row["class"], row["video_id"])
    videos.setdefault(key, []).append(row)

# 2. 영상별 16프레임 균등 선정
vlm_inputs = []

for (category, video_id), frames in sorted(videos.items()):
    frames.sort(key=lambda x: int(x["frame_index"]))

    if len(frames) < NUM_FRAMES:
        raise RuntimeError(f"프레임 부족: {video_id}")

    indices = [
        round(i * (len(frames) - 1) / (NUM_FRAMES - 1))
        for i in range(NUM_FRAMES)
    ]

    selected = []

    for idx in indices:
        frame = frames[idx]
        frame_path = Path(frame["frame_path"])

        if not frame_path.is_file() or frame_path.stat().st_size == 0:
            raise RuntimeError(f"이미지 누락: {frame_path}")

        selected.append({
            "frame_id": f"f_{int(frame['frame_index']):04d}",
            "timestamp_sec": float(frame["timestamp_sec"]),
            "frame_path": str(frame_path),
        })

    vlm_inputs.append({
        "video_id": video_id,
        "class": category,
        "num_frames": NUM_FRAMES,
        "frames": selected,
    })

# 3. JSON 저장
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(vlm_inputs, f, ensure_ascii=False, indent=2)

# 4. 결과 확인
counts = Counter(item["class"] for item in vlm_inputs)

print("전체 영상:", len(vlm_inputs))
print("유형별:", dict(counts))
print("총 프레임 참조:", sum(v["num_frames"] for v in vlm_inputs))
print("저장 경로:", OUTPUT_JSON)

assert len(vlm_inputs) == 300
assert all(
    counts[c] == 75
    for c in ["theft", "damage", "smoking", "dumping"]
)

print("✅ STEP 6 완료!")