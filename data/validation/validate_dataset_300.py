import os
from pathlib import Path
from collections import Counter
import csv
import unicodedata

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")

CATEGORIES = ["theft", "damage", "smoking", "dumping"]

def normalize(name):
    return unicodedata.normalize("NFC", name)

all_ids = []
errors = []

for category in CATEGORIES:
    video_dir = BASE / "videos" / category
    label_dir = BASE / "labels" / category

    videos = {
        normalize(p.stem): p
        for p in video_dir.glob("*.mp4")
    }

    labels = {
        normalize(p.stem): p
        for p in label_dir.glob("*.xml")
    }

    missing_labels = set(videos) - set(labels)
    missing_videos = set(labels) - set(videos)

    matched = set(videos) & set(labels)

    print(f"\n[{category}]")
    print(f"영상: {len(videos)}개")
    print(f"XML: {len(labels)}개")
    print(f"매칭: {len(matched)}개")

    if len(matched) != 75:
        errors.append(f"{category}: 75쌍이 아님")

    if missing_labels:
        errors.append(f"{category}: XML 누락 {len(missing_labels)}개")

    if missing_videos:
        errors.append(f"{category}: 영상 누락 {len(missing_videos)}개")

    for key in matched:
        all_ids.append((category, key))

metadata_path = BASE / "metadata.csv"

if not metadata_path.exists():
    errors.append("metadata.csv 없음")
else:
    with metadata_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"\nmetadata.csv: {len(rows)}행")

    if len(rows) != 300:
        errors.append("metadata.csv 행 수가 300개가 아님")

    metadata_ids = [
        (row["class"], normalize(row["video_id"]))
        for row in rows
    ]

    if set(metadata_ids) != set(all_ids):
        errors.append("metadata.csv와 실제 파일 목록 불일치")

    if len(metadata_ids) != len(set(metadata_ids)):
        errors.append("metadata.csv에 중복 ID 존재")

print("\n" + "=" * 35)

if errors:
    print("검사 실패")
    for error in errors:
        print("❌", error)
else:
    print("✅ 검사 통과!")
    print("300개 영상과 300개 XML 매칭 완료")
    print("metadata.csv 일치 확인")