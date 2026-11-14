
import csv
import json
import unicodedata
from pathlib import Path

import cv2

from config import (
    DATASET_DIR,
    PROCESSED_DIR,
    VIDEO_DIR,
    MANIFEST_DIR,
    TARGET_VIDEOS,
    NUM_FRAMES,
    WORKER_ID,
)


INPUT_JSON = PROCESSED_DIR / "vlm_inputs.json"
OUTPUT_JSON = MANIFEST_DIR / "vlm_inputs_with_duration.json"


def normalize(text):
    return unicodedata.normalize("NFC", str(text))


def get_duration(video_path):
    # macOS 한글 경로의 Unicode 표기 차이 처리
    path = unicodedata.normalize("NFD", str(video_path))

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or frame_count <= 0:
            return None

        return round(frame_count / fps, 3)

    finally:
        cap.release()


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

        required = {"video_id", "class", "video_path"}

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"metadata.csv에 필수 열이 없습니다: {required}"
            )

        rows = list(reader)

    video_paths = {}

    for row in rows:
        video_id = normalize(row["video_id"])
        category = row["class"]

        video_path = Path(row["video_path"])

        # 파일을 다른 컴퓨터로 이동한 경우 경로 재구성
        if not video_path.is_file():
            video_path = VIDEO_DIR / category / f"{video_id}.mp4"

        if video_id in video_paths:
            raise ValueError(
                f"metadata.csv에 중복 video_id: {video_id}"
            )

        video_paths[video_id] = video_path

    return video_paths


def main():
    if not INPUT_JSON.is_file():
        raise FileNotFoundError(
            f"VLM 입력 JSON이 없습니다: {INPUT_JSON}\n"
            "먼저 create_vlm_inputs.py를 실행하세요."
        )

    video_paths = load_metadata()

    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("VLM 입력 JSON은 리스트여야 합니다.")

    if len(data) != TARGET_VIDEOS:
        raise ValueError(
            f"팀원 {WORKER_ID}: "
            f"{TARGET_VIDEOS}개 예상, 실제 {len(data)}개"
        )

    video_ids = [normalize(item["video_id"]) for item in data]

    if len(video_ids) != len(set(video_ids)):
        raise ValueError("VLM 입력 JSON에 중복 video_id가 있습니다.")

    failed = []

    for index, item in enumerate(data, start=1):
        video_id = normalize(item["video_id"])
        video_path = video_paths.get(video_id)

        duration = (
            get_duration(video_path)
            if video_path is not None and video_path.is_file()
            else None
        )

        if duration is None:
            failed.append(video_id)
            continue

        if len(item.get("frames", [])) != NUM_FRAMES:
            failed.append(
                f"{video_id}: 프레임 수 불일치"
            )
            continue

        item["duration_sec"] = duration

        if index % 25 == 0:
            print(
                f"[팀원 {WORKER_ID}] "
                f"{index}/{len(data)}개 영상 길이 확인",
                flush=True,
            )

    print("\n========================================")
    print("WatchOut! Add Video Duration")
    print("========================================")
    print("담당자:", WORKER_ID)
    print("전체 영상:", len(data))
    print("영상 길이 측정 성공:", len(data) - len(failed))
    print("측정 실패:", len(failed))

    if failed:
        print("실패 예시:", failed[:10])

        raise RuntimeError(
            "영상 길이 측정 또는 프레임 검증에 실패했습니다. "
            "JSON을 저장하지 않습니다."
        )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # 영상 길이가 포함된 원본 JSON 저장
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # B0/B1 입력 JSON 갱신
    for experiment in ("B0", "B1"):
        output_path = MANIFEST_DIR / f"inputs_{experiment}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("최종 입력 JSON:", output_path)

    print("\n영상 길이 예시:", data[0]["duration_sec"])
    print("저장 완료:", OUTPUT_JSON)
    print("✅ STEP 5 완료!")


if __name__ == "__main__":
    main()
