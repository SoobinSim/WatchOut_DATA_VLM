import os
from pathlib import Path
import csv
import random
import shutil
import unicodedata

BASE = Path(os.environ["WATCHOUT_BASE"])

DATASET = BASE / "WatchOut_dataset"
NORMAL_DIR = DATASET / "normal"

VIDEO_OUT = NORMAL_DIR / "videos"
LABEL_OUT = NORMAL_DIR / "labels"

SEED = 42
TARGET_COUNT = 150


def normalize(text):
    return unicodedata.normalize("NFC", text)


# 정상 구매 원천 영상·라벨 검색
videos = []
labels = []

for folder in BASE.iterdir():
    if not folder.is_dir() or "238-1." not in folder.name:
        continue

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        name = normalize(path.name)

        if path.suffix.lower() == ".mp4":
            videos.append(path)
        elif path.suffix.lower() == ".xml":
            labels.append(path)


# 파일명 기준으로 매칭
video_map = {normalize(p.stem): p for p in videos}
label_map = {normalize(p.stem): p for p in labels}

matched_ids = sorted(set(video_map) & set(label_map))

print(f"MP4: {len(videos)}개")
print(f"XML: {len(labels)}개")
print(f"매칭 성공: {len(matched_ids)}개")

if not matched_ids:
    raise RuntimeError("매칭된 정상 구매 영상이 없습니다.")

# 재현 가능한 랜덤 선별
rng = random.Random(SEED)
selected_ids = rng.sample(
    matched_ids,
    min(TARGET_COUNT, len(matched_ids))
)

VIDEO_OUT.mkdir(parents=True, exist_ok=True)
LABEL_OUT.mkdir(parents=True, exist_ok=True)

metadata = []

for i, video_id in enumerate(selected_ids, 1):
    video_src = video_map[video_id]
    label_src = label_map[video_id]

    video_dst = VIDEO_OUT / video_src.name
    label_dst = LABEL_OUT / label_src.name

    if not video_dst.exists():
        shutil.copy2(video_src, video_dst)

    if not label_dst.exists():
        shutil.copy2(label_src, label_dst)

    metadata.append({
        "video_id": video_id,
        "source_class": "normal_purchase",
        "video_path": str(video_dst),
        "label_path": str(label_dst),
        "source_dataset": "AI-Hub 238-1",
    })

    print(f"[{i}/{len(selected_ids)}] {video_id}", flush=True)


metadata_path = NORMAL_DIR / "normal_metadata_150.csv"

with metadata_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
    writer.writeheader()
    writer.writerows(metadata)

print("\n✅ 정상 구매 데이터 선별 완료!")
print(f"선별 영상: {len(selected_ids)}개")
print(f"영상 위치: {VIDEO_OUT}")
print(f"라벨 위치: {LABEL_OUT}")
print(f"메타데이터: {metadata_path}")