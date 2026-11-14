
import csv
import json
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from config import (
    DATASET_DIR,
    VIDEO_DIR,
    LABEL_DIR,
    PROCESSED_DIR,
    EVENT_LABELS,
    CATEGORIES,
    TARGET_VIDEOS,
    WORKER_ID,
)


OUTPUT_CSV = PROCESSED_DIR / "event_times.csv"


def normalize(text):
    return unicodedata.normalize("NFC", str(text))


def find_file(folder, video_id, extension):
    if not folder.is_dir():
        raise FileNotFoundError(
            f"폴더가 없습니다: {folder}"
        )

    matches = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() == extension
        and normalize(path.stem) == normalize(video_id)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"파일 매칭 오류: {folder}/{video_id} "
            f"(일치 파일 {len(matches)}개)"
        )

    return matches[0]


def extract_events(xml_path, category):
    root = ET.parse(xml_path).getroot()

    start_label, end_label = EVENT_LABELS[category]

    starts = []
    ends = []

    for track in root.iter("track"):
        label = normalize(track.get("label", "")).strip()

        if label not in (start_label, end_label):
            continue

        for box in track.findall("box"):
            # outside=1인 종료 표시 박스는 제외
            if box.get("outside", "0") != "0":
                continue

            frame = box.get("frame")

            if frame is None:
                continue

            frame = int(frame)

            if label == start_label:
                starts.append(frame)
            elif label == end_label:
                ends.append(frame)

    starts = sorted(set(starts))
    ends = sorted(set(ends))

    # 시작 프레임 이후 가장 가까운 종료 프레임과 매칭
    pairs = []
    unused_ends = ends.copy()

    for start in starts:
        possible = [
            end
            for end in unused_ends
            if end > start
        ]

        if possible:
            end = possible[0]
            pairs.append((start, end))
            unused_ends.remove(end)

    needs_review = (
        len(starts) != len(ends)
        or len(pairs) != len(starts)
        or len(pairs) != 1
    )

    return starts, ends, pairs, needs_review


def load_metadata():
    metadata_path = DATASET_DIR / "metadata.csv"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"metadata.csv가 없습니다: {metadata_path}"
        )

    with metadata_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        required = {"video_id", "class"}

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"필수 메타데이터 열 누락: {required}"
            )

        rows = list(reader)

    if len(rows) != TARGET_VIDEOS:
        raise ValueError(
            f"팀원 {WORKER_ID}: "
            f"{TARGET_VIDEOS}개 예상, 실제 {len(rows)}개"
        )

    ids = [normalize(row["video_id"]) for row in rows]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "metadata.csv에 중복 video_id가 있습니다."
        )

    for row in rows:
        if row["class"] not in CATEGORIES:
            raise ValueError(
                f"지원하지 않는 유형: {row['class']}"
            )

    return rows


def main():
    metadata = load_metadata()

    results = []
    errors = []

    print("\n========================================")
    print("WatchOut! XML Event Time Extraction")
    print("========================================")
    print("담당자:", WORKER_ID)
    print("입력 영상:", len(metadata))
    print("출력 파일:", OUTPUT_CSV)

    for index, item in enumerate(metadata, start=1):
        category = item["class"]
        video_id = normalize(item["video_id"])

        try:
            video_path = find_file(
                VIDEO_DIR / category, video_id, ".mp4"
            )

            xml_path = find_file(
                LABEL_DIR / category, video_id, ".xml"
            )

            cap = cv2.VideoCapture(
                unicodedata.normalize("NFD", str(video_path))
            )

            if not cap.isOpened():
                raise RuntimeError(
                    f"영상 열기 실패: {video_path}"
                )

            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )
            finally:
                cap.release()

            if fps <= 0 or frame_count <= 0:
                raise RuntimeError(
                    f"영상 FPS/프레임 수 오류: {video_id}"
                )

            starts, ends, pairs, needs_review = extract_events(
                xml_path, category
            )

            # XML 이벤트 프레임 범위 확인
            if any(
                start < 0
                or end >= frame_count
                or start >= end
                for start, end in pairs
            ):
                raise ValueError(
                    f"XML 이벤트 프레임 범위 오류: {video_id}"
                )

            results.append({
                "video_id": video_id,
                "class": category,
                "fps": round(fps, 5),
                "start_frames": json.dumps(starts),
                "end_frames": json.dumps(ends),
                "event_pairs": json.dumps(pairs),
                "event_times_sec": json.dumps([
                    [
                        round(start / fps, 3),
                        round(end / fps, 3),
                    ]
                    for start, end in pairs
                ]),
                "needs_review": needs_review,
            })

            if index % 25 == 0:
                print(
                    f"[팀원 {WORKER_ID}] "
                    f"{index}/{len(metadata)}개 XML 분석",
                    flush=True,
                )

        except Exception as e:
            errors.append((video_id, str(e)))
            print(
                f"[ERROR] {video_id}: {e}",
                flush=True,
            )

    if errors:
        print("\nXML 분석 실패:", len(errors))

        for video_id, error in errors[:20]:
            print(f"- {video_id}: {error}")

        raise RuntimeError(
            "XML 이벤트 시간 추출에 실패했습니다. "
            "오류를 해결하고 재실행하세요."
        )

    if len(results) != TARGET_VIDEOS:
        raise ValueError(
            f"분석 영상 수 불일치: "
            f"{len(results)}/{TARGET_VIDEOS}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "class",
                "fps",
                "start_frames",
                "end_frames",
                "event_pairs",
                "event_times_sec",
                "needs_review",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    candidate_count = sum(
        len(json.loads(row["event_pairs"]))
        for row in results
    )

    review_count = sum(
        row["needs_review"]
        for row in results
    )

    print("\n========================================")
    print("XML 이벤트 시간 추출 완료")
    print("영상 수:", len(results))
    print("이벤트 후보 구간:", candidate_count)
    print("검토 대상 영상:", review_count)
    print("CSV 저장:", OUTPUT_CSV)
    print("========================================")


if __name__ == "__main__":
    main()
