import os
from pathlib import Path
import csv
import shutil
import unicodedata

BASE = Path(os.environ["WATCHOUT_BASE"])
DATASET = BASE / "WatchOut_dataset"

CATEGORIES = {
    "theft": "12.절도",
    "damage": "08.파손",
    "smoking": "10.흡연",
    "dumping": "11.유기",
}

TARGET = 75


def normalized(text):
    return unicodedata.normalize("NFC", text)


def find_source(prefix, keyword):
    matches = [
        p for p in BASE.iterdir()
        if p.is_dir()
        and p.name.startswith(prefix)
        and keyword in normalized(p.name)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{prefix} / {keyword} 폴더 확인 필요: {matches}"
        )

    return matches[0]


def collect_files(folder, extension):
    result = {}

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() == extension:
            key = normalized(path.stem)

            if key in result:
                raise RuntimeError(f"중복 파일명: {key}")

            result[key] = path

    return result


rows = []

for category, keyword in CATEGORIES.items():
    video_source = find_source("VS_", keyword)
    label_source = find_source("VL_", keyword)

    videos = collect_files(video_source, ".mp4")
    labels = collect_files(label_source, ".xml")

    paired = sorted(set(videos) & set(labels))

    if len(paired) < TARGET:
        raise RuntimeError(
            f"{category}: 매칭된 영상이 {len(paired)}개뿐입니다."
        )

    video_dest = DATASET / "videos" / category
    label_dest = DATASET / "labels" / category

    video_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)

    existing = {
        normalized(p.stem)
        for p in video_dest.glob("*.mp4")
    }

    if len(existing) > TARGET:
        raise RuntimeError(
            f"{category}: 기존 영상이 {TARGET}개보다 많습니다."
        )

    # 기존 40개에 포함된 파일을 우선 보존
    selected = sorted(existing)

    for key in paired:
        if len(selected) >= TARGET:
            break

        if key not in existing:
            selected.append(key)

    if len(selected) != TARGET:
        raise RuntimeError(f"{category}: 75개 선별 실패")

    for key in selected:
        if key not in videos or key not in labels:
            raise RuntimeError(f"원본 매칭 실패: {key}")

        dst_video = video_dest / videos[key].name
        dst_label = label_dest / labels[key].name

        if not dst_video.exists():
            shutil.copy2(videos[key], dst_video)

        if not dst_label.exists():
            shutil.copy2(labels[key], dst_label)

        rows.append({
            "video_id": key,
            "class": category,
            "split": "validation",
            "group": "",
            "video_path": str(dst_video),
            "label_path": str(dst_label),
        })

    print(f"{category}: {len(selected)}개 완료")

# 기존 metadata.csv 백업
metadata = DATASET / "metadata.csv"

if metadata.exists():
    shutil.copy2(metadata, DATASET / "metadata_backup_40.csv")

with metadata.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video_id", "class", "split",
            "group", "video_path", "label_path"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n전체 선별 완료:", len(rows), "개")
print("메타데이터:", metadata)