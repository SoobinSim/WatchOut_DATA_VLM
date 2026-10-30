import os
from pathlib import Path
from collections import Counter
import csv
import json
import unicodedata
import xml.etree.ElementTree as ET

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
OUTPUT = BASE / "processed" / "event_times_300.csv"

EVENT_LABELS = {
    "theft": ("theft_start", "theft_end"),
    "damage": ("broken_start", "broken_end"),
    "smoking": ("smoke_start", "smoke_end"),
    "dumping": ("abandon_start", "abandon_end"),
}

CATEGORIES = list(EVENT_LABELS)


def normalize(text):
    return unicodedata.normalize("NFC", str(text))


def find_xml(video_id, category):
    folder = BASE / "labels" / category

    matches = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".xml"
        and normalize(p.stem) == normalize(video_id)
    ]

    if len(matches) != 1:
        raise RuntimeError(f"XML 매칭 오류: {category}/{video_id}")

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
            # outside=1인 종료 표시용 박스는 제외
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

    # 시작 이후 가장 가까운 종료 프레임과 매칭
    pairs = []
    unused_ends = ends.copy()

    for start in starts:
        possible = [end for end in unused_ends if end > start]

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

def main():
    metadata_path = BASE / "metadata.csv"
    frames_path = BASE / "processed" / "frames_metadata_300.csv"

    with metadata_path.open(encoding="utf-8-sig", newline="") as f:
        metadata = list(csv.DictReader(f))

    # 전처리 결과의 source_frame과 timestamp를 이용해 FPS 추정
    # 실제 시간 변환에는 영상별 원본 FPS가 필요함
    import cv2

    rows = []
    errors = []

    for i, item in enumerate(metadata, 1):
        category = item["class"]
        video_id = normalize(item["video_id"])

        try:
            xml_path = find_xml(video_id, category)

            video_folder = BASE / "videos" / category
            videos = [
                p for p in video_folder.iterdir()
                if p.suffix.lower() == ".mp4"
                and normalize(p.stem) == video_id
            ]

            if len(videos) != 1:
                raise RuntimeError("MP4 매칭 실패")

            cap = cv2.VideoCapture(str(videos[0]))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if fps <= 0:
                raise RuntimeError("FPS 확인 실패")

            starts, ends, pairs, review = extract_events(
                xml_path, category
            )

            rows.append({
                "video_id": video_id,
                "class": category,
                "fps": fps,
                "start_frames": json.dumps(starts),
                "end_frames": json.dumps(ends),
                "event_pairs": json.dumps(pairs),
                "event_times_sec": json.dumps([
                    [round(s / fps, 3), round(e / fps, 3)]
                    for s, e in pairs
                ]),
                "needs_review": review,
            })

            print(
                f"[{i}/{len(metadata)}] {category} | "
                f"구간 {len(pairs)}개 | 검토 {review}"
            )

        except Exception as e:
            errors.append((video_id, str(e)))
            print(f"❌ {video_id}: {e}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id", "class", "fps",
                "start_frames", "end_frames",
                "event_pairs", "event_times_sec",
                "needs_review"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 40)
    print(f"분석 완료 영상: {len(rows)}개")
    print(f"추출된 이벤트 구간 후보: {sum(len(json.loads(r['event_pairs'])) for r in rows)}개")
    print(f"검토 대상 영상: {sum(str(r['needs_review']) == 'True' for r in rows)}개")
    print(f"처리 오류: {len(errors)}개")
    print(f"결과 저장: {OUTPUT}")

    if errors:
        print("\n오류 목록:")
        for video_id, error in errors:
            print(video_id, error)


if __name__ == "__main__":
    main()