
import argparse
import csv
import os
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from config import (
    SOURCE_ROOT,
    WORKER_ID,
    TARGET_VIDEOS,
    TOTAL_NEW_VIDEOS,
    TOTAL_SELECTION_PER_CATEGORY,
    EXISTING_IDS_CSV,
    ASSIGNMENT_CSV,
    SOURCE_CATEGORIES,
    VIDEO_DIR,
    LABEL_DIR,
    DATASET_DIR,
)


# ============================================================
# 1. 기본 설정
# ============================================================

SOURCE_PREFIXES = {
    "validation": ("VS_", "VL_"),
    "training": ("TS_", "TL_"),
}

# 기존 프로젝트와 동일한 Validation을 먼저 사용
SOURCE_PRIORITY = ("validation", "training")

ASSIGNMENT_FIELDS = [
    "video_id",
    "class",
    "assigned_to",
    "source_split",
    "video_source",
    "label_source",
]

METADATA_FIELDS = [
    "video_id",
    "class",
    "split",
    "group",
    "video_path",
    "label_path",
]


def normalize(text):
    return unicodedata.normalize("NFC", str(text))


# ============================================================
# 2. 기존 400개 영상 ID 로딩
# ============================================================

def read_existing_ids():
    if not EXISTING_IDS_CSV.is_file():
        raise FileNotFoundError(
            f"기존 영상 ID 파일 없음: {EXISTING_IDS_CSV}"
        )

    with EXISTING_IDS_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if "video_id" not in (reader.fieldnames or []):
            raise ValueError(
                "기존 ID CSV에 video_id 열이 없습니다."
            )

        ids = [
            normalize(row["video_id"])
            for row in reader
            if row.get("video_id")
        ]

    if len(ids) != 400:
        raise ValueError(
            f"기존 영상 목록은 400행이어야 합니다. "
            f"현재 {len(ids)}행"
        )

    if len(set(ids)) != 400:
        raise ValueError(
            "기존 영상 목록에 중복 video_id가 있습니다."
        )

    return set(ids)


# ============================================================
# 3. AI-Hub 원본 폴더 검색
# ============================================================

def discover_source_dirs():
    """
    SOURCE_ROOT 아래에서 VS_, VL_, TS_, TL_ 폴더를 검색한다.

    Validation/Training 하위 폴더 구조를 지원한다.
    매칭된 원본 폴더 내부는 별도로 파일을 검색한다.
    """

    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(
            f"AI-Hub 원본 폴더 없음: {SOURCE_ROOT}"
        )

    prefixes = ("VS_", "VL_", "TS_", "TL_")
    source_dirs = []

    for current, dirnames, _ in os.walk(SOURCE_ROOT):
        current_path = Path(current)

        if any(
            normalize(current_path.name).startswith(prefix)
            for prefix in prefixes
        ):
            source_dirs.append(current_path)

            # 매칭된 원본 폴더 안은 여기서 재탐색하지 않음
            dirnames[:] = []

    return sorted(source_dirs)


def find_source_dirs(source_dirs, prefix, keyword):
    matches = [
        path
        for path in source_dirs
        if normalize(path.name).startswith(prefix)
        and keyword in normalize(path.name)
    ]

    return matches


def collect_files(folders, extension):
    """
    여러 원본 폴더에서 동일한 유형의 파일을 모은다.
    동일 유형·데이터 분할 내부의 중복 파일명은 오류 처리한다.
    """

    files = {}

    for folder in folders:
        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() != extension:
                continue

            video_id = normalize(path.stem)

            if video_id in files:
                raise RuntimeError(
                    f"원본 파일명 중복: {video_id}\n"
                    f"첫 번째: {files[video_id]}\n"
                    f"두 번째: {path}"
                )

            files[video_id] = path

    return files


# ============================================================
# 4. 유형별 원본 MP4/XML 인덱스 생성
# ============================================================

def build_source_index():
    """
    key: (category, source_split)

    value:
        {
            video_id: {
                "video_path": Path,
                "label_path": Path,
            }
        }
    """

    source_dirs = discover_source_dirs()
    source_index = {}

    for category, keyword in SOURCE_CATEGORIES.items():
        for split in SOURCE_PRIORITY:
            video_prefix, label_prefix = (
                SOURCE_PREFIXES[split]
            )

            video_dirs = find_source_dirs(
                source_dirs, video_prefix, keyword
            )

            label_dirs = find_source_dirs(
                source_dirs, label_prefix, keyword
            )

            if not video_dirs and not label_dirs:
                source_index[(category, split)] = {}
                continue

            if not video_dirs or not label_dirs:
                raise RuntimeError(
                    f"{category}/{split}: "
                    "MP4 또는 XML 폴더가 누락되었습니다.\n"
                    f"MP4 폴더: {video_dirs}\n"
                    f"XML 폴더: {label_dirs}"
                )

            videos = collect_files(video_dirs, ".mp4")
            labels = collect_files(label_dirs, ".xml")

            paired_ids = sorted(set(videos) & set(labels))

            source_index[(category, split)] = {
                video_id: {
                    "video_path": videos[video_id],
                    "label_path": labels[video_id],
                }
                for video_id in paired_ids
            }

            print(
                f"[{category}/{split}] "
                f"MP4: {len(videos)} | "
                f"XML: {len(labels)} | "
                f"매칭: {len(paired_ids)}"
            )

    return source_index


# ============================================================
# 5. 신규 영상 선별
# ============================================================

def select_new_videos(source_index, existing_ids):
    selected_rows = []
    globally_selected = set()

    for category in SOURCE_CATEGORIES:
        target = TOTAL_SELECTION_PER_CATEGORY[category]

        candidates = {}

        for split in SOURCE_PRIORITY:
            source_files = source_index[(category, split)]

            for video_id, paths in source_files.items():
                if video_id in existing_ids:
                    continue

                # 같은 영상이 Validation과 Training에
                # 모두 있으면 우선순위가 높은 쪽을 사용
                if video_id in candidates:
                    continue

                candidates[video_id] = {
                    "video_id": video_id,
                    "class": category,
                    "source_split": split,
                    "video_source": str(
                        paths["video_path"].resolve()
                    ),
                    "label_source": str(
                        paths["label_path"].resolve()
                    ),
                }

        # Validation 먼저, 그다음 Training 사용
        ordered_candidates = sorted(
            candidates.values(),
            key=lambda row: (
                SOURCE_PRIORITY.index(row["source_split"]),
                row["video_id"],
            ),
        )

        print(
            f"\n[{category}] "
            f"신규 후보 {len(ordered_candidates)}개 / "
            f"목표 {target}개"
        )

        if len(ordered_candidates) < target:
            raise RuntimeError(
                f"{category}: 신규 영상 부족\n"
                f"목표: {target}개\n"
                f"사용 가능: {len(ordered_candidates)}개\n"
                "추가 MP4/XML 확보 또는 유형별 "
                "목표 수량 조정이 필요합니다."
            )

        chosen = ordered_candidates[:target]

        for row in chosen:
            video_id = row["video_id"]

            if video_id in globally_selected:
                raise ValueError(
                    "서로 다른 유형에 동일한 "
                    f"video_id가 있습니다: {video_id}"
                )

            globally_selected.add(video_id)

        selected_rows.extend(chosen)

        counts = Counter(
            row["source_split"] for row in chosen
        )

        print(
            f"선별 완료: {len(chosen)}개 "
            f"| 데이터 출처: {dict(counts)}"
        )

    if len(selected_rows) != TOTAL_NEW_VIDEOS:
        raise ValueError(
            f"신규 영상 수 오류: "
            f"{len(selected_rows)}/{TOTAL_NEW_VIDEOS}"
        )

    return selected_rows


# ============================================================
# 6. Worker A/B 균형 배정
# ============================================================

def assign_workers(selected_rows):
    """
    각 유형에서 번갈아 A/B에 배정한다.

    홀수 수량이 포함된 유형도 전체 배정 수를
    고려하여 균형을 맞춘다.
    """

    assigned = []
    worker_counts = Counter()

    for category in SOURCE_CATEGORIES:
        category_rows = sorted(
            [
                row.copy()
                for row in selected_rows
                if row["class"] == category
            ],
            key=lambda row: (
                row["source_split"],
                row["video_id"],
            ),
        )

        for row in category_rows:
            # 현재 배정 수가 적은 Worker에게 우선 배정
            if worker_counts["A"] <= worker_counts["B"]:
                worker = "A"
            else:
                worker = "B"

            row["assigned_to"] = worker
            worker_counts[worker] += 1
            assigned.append(row)

    if worker_counts["A"] != TARGET_VIDEOS:
        raise ValueError(
            f"Worker A 영상 수 오류: "
            f"{worker_counts['A']}/{TARGET_VIDEOS}"
        )

    if worker_counts["B"] != TARGET_VIDEOS:
        raise ValueError(
            f"Worker B 영상 수 오류: "
            f"{worker_counts['B']}/{TARGET_VIDEOS}"
        )

    print("\n===== Worker 배정 결과 =====")
    print("Worker별:", dict(worker_counts))

    for category in SOURCE_CATEGORIES:
        counts = Counter(
            row["assigned_to"]
            for row in assigned
            if row["class"] == category
        )

        print(category, dict(counts))

    return assigned


# ============================================================
# 7. 배정표 생성 (--prepare)
# ============================================================

def build_assignments():
    if ASSIGNMENT_CSV.exists():
        raise FileExistsError(
            f"배정표가 이미 존재합니다: {ASSIGNMENT_CSV}\n"
            "기존 파일을 덮어쓰지 않습니다."
        )

    existing_ids = read_existing_ids()
    source_index = build_source_index()

    selected = select_new_videos(
        source_index, existing_ids
    )

    assigned = assign_workers(selected)

    selected_ids = {
        row["video_id"] for row in assigned
    }

    if selected_ids & existing_ids:
        raise ValueError(
            "기존 400개 영상과 중복이 발견되었습니다."
        )

    if len(selected_ids) != TOTAL_NEW_VIDEOS:
        raise ValueError(
            "신규 영상 ID 개수가 목표 수량과 다릅니다."
        )

    ASSIGNMENT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ASSIGNMENT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=ASSIGNMENT_FIELDS,
        )

        writer.writeheader()
        writer.writerows(assigned)

    print("\n========================================")
    print("신규 영상 배정 완료")
    print("전체:", len(assigned))
    print("Worker A:", TARGET_VIDEOS)
    print("Worker B:", TARGET_VIDEOS)
    print("배정표:", ASSIGNMENT_CSV)
    print("========================================")


# ============================================================
# 8. 배정표 불러오기
# ============================================================

def load_assignments():
    if not ASSIGNMENT_CSV.is_file():
        raise FileNotFoundError(
            f"배정표가 없습니다: {ASSIGNMENT_CSV}\n"
            "먼저 --prepare를 실행하고 "
            "동일한 CSV를 두 팀원에게 공유하세요."
        )

    with ASSIGNMENT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "video_id",
            "class",
            "assigned_to",
            "source_split",
        }

        if not required.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "assignments.csv의 필수 열이 누락되었습니다."
            )

        rows = list(reader)

    if len(rows) != TOTAL_NEW_VIDEOS:
        raise ValueError(
            f"배정표 영상 수 오류: "
            f"{len(rows)}/{TOTAL_NEW_VIDEOS}"
        )

    normalized_ids = [
        normalize(row["video_id"])
        for row in rows
    ]

    if len(set(normalized_ids)) != len(rows):
        raise ValueError(
            "배정표에 중복 video_id가 있습니다."
        )

    existing_ids = read_existing_ids()

    if set(normalized_ids) & existing_ids:
        raise ValueError(
            "기존 400개와 중복되는 영상이 있습니다."
        )

    counts = Counter(
        row["assigned_to"] for row in rows
    )

    if (
        counts["A"] != TARGET_VIDEOS
        or counts["B"] != TARGET_VIDEOS
    ):
        raise ValueError(
            f"A/B 배정 수량 오류: {dict(counts)}"
        )

    for row in rows:
        if row["class"] not in SOURCE_CATEGORIES:
            raise ValueError(
                f"지원하지 않는 유형: {row['class']}"
            )

        if row["source_split"] not in SOURCE_PREFIXES:
            raise ValueError(
                f"지원하지 않는 데이터 출처: "
                f"{row['source_split']}"
            )

    return rows


# ============================================================
# 9. 담당자 데이터 복사 (--copy)
# ============================================================

def copy_worker_data():
    rows = load_assignments()

    worker_rows = [
        row
        for row in rows
        if row["assigned_to"] == WORKER_ID
    ]

    if len(worker_rows) != TARGET_VIDEOS:
        raise ValueError(
            f"Worker {WORKER_ID}: "
            f"담당 영상 수 오류"
        )

    metadata_path = DATASET_DIR / "metadata.csv"

    if metadata_path.exists():
        raise FileExistsError(
            f"기존 metadata.csv가 있습니다: {metadata_path}\n"
            "덮어쓰기를 방지하기 위해 중단합니다."
        )

    # 담당자 원본 파일 목록을 유형/출처별로 한 번만 검색
    source_index = build_source_index()

    # 복사 전에 필요한 파일이 전부 있는지 확인
    copy_plan = []

    for row in worker_rows:
        category = row["class"]
        split = row["source_split"]
        video_id = normalize(row["video_id"])

        source_files = source_index.get(
            (category, split), {}
        )

        paths = source_files.get(video_id)

        if paths is None:
            raise FileNotFoundError(
                f"원본 MP4/XML 없음: "
                f"{category}/{split}/{video_id}\n"
                "해당 Worker의 원본 데이터가 "
                "컴퓨터에 다운로드되어 있는지 확인하세요."
            )

        source_video = paths["video_path"]
        source_label = paths["label_path"]

        target_video = (
            VIDEO_DIR / category / source_video.name
        )

        target_label = (
            LABEL_DIR / category / source_label.name
        )

        copy_plan.append({
            "video_id": video_id,
            "category": category,
            "split": split,
            "source_video": source_video,
            "source_label": source_label,
            "target_video": target_video,
            "target_label": target_label,
        })

    metadata_rows = []

    for index, item in enumerate(copy_plan, start=1):
        target_video = item["target_video"]
        target_label = item["target_label"]

        target_video.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target_label.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not target_video.exists():
            shutil.copy2(
                item["source_video"],
                target_video,
            )

        if not target_label.exists():
            shutil.copy2(
                item["source_label"],
                target_label,
            )

        metadata_rows.append({
            "video_id": item["video_id"],
            "class": item["category"],
            "split": item["split"],
            "group": "",
            "video_path": str(target_video.resolve()),
            "label_path": str(target_label.resolve()),
        })

        if index % 25 == 0:
            print(
                f"[Worker {WORKER_ID}] "
                f"{index}/{len(copy_plan)}개 복사 완료",
                flush=True,
            )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=METADATA_FIELDS,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n========================================")
    print("Worker 데이터 복사 완료")
    print("담당자:", WORKER_ID)
    print("영상 수:", len(metadata_rows))
    print(
        "데이터 출처:",
        dict(Counter(r["split"] for r in metadata_rows))
    )
    print("metadata.csv:", metadata_path)
    print("========================================")


# ============================================================
# 10. 메인 실행
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "WatchOut! Validation/Training 신규 영상 "
            "선별 및 Worker A/B 배정"
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--prepare",
        action="store_true",
        help="신규 영상 선별 및 A/B 배정표 생성",
    )

    group.add_argument(
        "--copy",
        action="store_true",
        help="담당자의 MP4/XML 복사",
    )

    args = parser.parse_args()

    if args.prepare:
        build_assignments()
    else:
        copy_worker_data()


if __name__ == "__main__":
    main()
