
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from config import (
    DATASET_DIR,
    VIDEO_DIR,
    PROCESSED_DIR,
    MANIFEST_DIR,
    SEGMENT_FRAME_DIR,
    VLM_ROOT,
    NUM_FRAMES,
    TARGET_VIDEOS,
    WORKER_ID,
)

SEGMENTS_CSV = PROCESSED_DIR / "segments.csv"
METADATA_CSV = DATASET_DIR / "metadata.csv"
DURATION_JSON = MANIFEST_DIR / "vlm_inputs_with_duration.json"


def normalize(text):
    return unicodedata.normalize("NFC", str(text))


def load_metadata():
    if not METADATA_CSV.is_file():
        raise FileNotFoundError(
            f"메타데이터 없음: {METADATA_CSV}"
        )

    with METADATA_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != TARGET_VIDEOS:
        raise ValueError(
            f"팀원 {WORKER_ID}: "
            f"영상 {TARGET_VIDEOS}개 예상, {len(rows)}개 확인"
        )

    video_paths = {}

    for row in rows:
        video_id = normalize(row["video_id"])
        category = row["class"]

        path = Path(row["video_path"])

        if not path.is_file():
            path = VIDEO_DIR / category / f"{video_id}.mp4"

        if not path.is_file():
            raise FileNotFoundError(
                f"원본 영상 없음: {video_id}"
            )

        if video_id in video_paths:
            raise ValueError(
                f"중복 video_id: {video_id}"
            )

        video_paths[video_id] = path

    return video_paths


def load_durations():
    if not DURATION_JSON.is_file():
        raise FileNotFoundError(
            f"영상 길이 JSON 없음: {DURATION_JSON}\n"
            "먼저 add_duration.py를 실행하세요."
        )

    with DURATION_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        normalize(item["video_id"]): float(item["duration_sec"])
        for item in data
    }


def load_segments():
    if not SEGMENTS_CSV.is_file():
        raise FileNotFoundError(
            f"후보 구간 CSV 없음: {SEGMENTS_CSV}\n"
            "먼저 create_segments.py를 실행하세요."
        )

    with SEGMENTS_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    segment_ids = [row["segment_id"] for row in rows]

    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError(
            "segments.csv에 중복 segment_id가 있습니다."
        )

    return rows


def process_segment(row, video_paths, durations):
    video_id = normalize(row["video_id"])
    segment_id = normalize(row["segment_id"])

    video_path = video_paths.get(video_id)
    duration = durations.get(video_id)

    if video_path is None or duration is None:
        raise ValueError(
            f"영상 정보 또는 duration_sec 누락: {segment_id}"
        )

    start_sec = float(row["start_sec"])
    end_sec = float(row["end_sec"])

    if not (0 <= start_sec < end_sec <= duration + 0.1):
        raise ValueError(
            f"구간 시간 오류: {segment_id} "
            f"({start_sec}~{end_sec}, duration={duration})"
        )

    path = unicodedata.normalize("NFD", str(video_path))
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(
            f"영상 열기 실패: {video_path}"
        )

    output_dir = SEGMENT_FRAME_DIR / segment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(
                f"FPS 또는 프레임 수 오류: {segment_id}"
            )

        timestamps = np.linspace(
            start_sec,
            end_sec,
            NUM_FRAMES,
            endpoint=False,
        )

        for frame_index, timestamp in enumerate(timestamps):
            source_index = min(
                max(int(round(float(timestamp) * fps)), 0),
                frame_count - 1,
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            success, image = cap.read()

            if not success:
                raise RuntimeError(
                    f"프레임 읽기 실패: "
                    f"{segment_id}, {timestamp:.3f}초"
                )

            actual_sec = round(
                cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                3,
            )

            # 일부 코덱에서 타임스탬프가 0으로 반환되는 경우
            if actual_sec <= 0 and source_index > 0:
                actual_sec = round(source_index / fps, 3)

            frame_id = f"f_{frame_index:04d}"
            filename = f"frame_{frame_index:04d}.jpg"
            output_path = output_dir / filename

            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(
                    f"이미지 저장 실패: {output_path}"
                )

            relative_path = output_path.relative_to(VLM_ROOT)

            frames.append({
                "frame_id": frame_id,
                "timestamp_sec": actual_sec,
                "frame_path": relative_path.as_posix(),
            })

    finally:
        cap.release()

    if len(frames) != NUM_FRAMES:
        raise RuntimeError(
            f"프레임 수 부족: {segment_id}"
        )

    return {
        "video_id": video_id,
        "segment_id": segment_id,
        "duration_sec": duration,
        "frames": frames,
    }


def main():
    video_paths = load_metadata()
    durations = load_durations()
    segments = load_segments()

    inputs = []
    failures = []

    print("\n========================================")
    print("WatchOut! Segment Input Generation")
    print("========================================")
    print("담당자:", WORKER_ID)
    print("담당 영상:", TARGET_VIDEOS)
    print("후보 구간:", len(segments))
    print("출력 프레임 폴더:", SEGMENT_FRAME_DIR)

    for index, row in enumerate(segments, start=1):
        segment_id = row["segment_id"]

        try:
            result = process_segment(
                row, video_paths, durations
            )
            inputs.append(result)

            if index % 25 == 0:
                print(
                    f"[팀원 {WORKER_ID}] "
                    f"{index}/{len(segments)}개 구간 완료",
                    flush=True,
                )

        except Exception as e:
            failures.append((segment_id, str(e)))
            print(
                f"[ERROR] {segment_id}: {e}",
                flush=True,
            )

    if failures:
        print("\n처리 실패 구간:", len(failures))

        for segment_id, error in failures[:20]:
            print(f"- {segment_id}: {error}")

        raise RuntimeError(
            "실패한 후보 구간이 있어 JSON을 저장하지 않습니다. "
            "오류 해결 후 다시 실행하세요."
        )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # B2와 B3는 같은 프레임을 사용하고 프롬프트만 다르게 적용
    for experiment in ("B2", "B3"):
        output_path = MANIFEST_DIR / f"inputs_{experiment}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                inputs,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("입력 JSON 저장:", output_path)

    counts = Counter(
        item["video_id"] for item in inputs
    )

    print("\n========================================")
    print("B2/B3 입력 생성 완료")
    print("담당자:", WORKER_ID)
    print("전체 후보 구간:", len(segments))
    print("성공 구간:", len(inputs))
    print("등록 프레임:", sum(len(item["frames"]) for item in inputs))
    print("후보 구간이 있는 영상:", len(counts))
    print("========================================")


if __name__ == "__main__":
    main()
