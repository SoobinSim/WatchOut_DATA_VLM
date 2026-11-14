
import csv
import json
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from config import (
    DATASET_DIR,
    PROCESSED_DIR,
    FRAME_DIR,
    MANIFEST_DIR,
    NUM_FRAMES,
    TARGET_VIDEOS,
    WORKER_ID,
)

INPUT_CSV = PROCESSED_DIR / "frames_metadata.csv"
OUTPUT_JSON = PROCESSED_DIR / "vlm_inputs.json"

# VLM 추론에서 사용할 전체 영상 프레임 폴더
# FRAME_DIR = workers/{WORKER_ID}/vlm/data/frames/


def normalize(text):
    return unicodedata.normalize("NFC", str(text))


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"프레임 메타데이터가 없습니다: {INPUT_CSV}\n"
            "먼저 video_preprocessing.py를 실행하세요."
        )

    with INPUT_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "video_id",
            "class",
            "frame_index",
            "timestamp_sec",
            "frame_path",
        }

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"프레임 CSV의 필수 열이 누락되었습니다: {required}"
            )

        rows = list(reader)

    videos = defaultdict(list)

    for row in rows:
        video_id = normalize(row["video_id"])
        category = row["class"]
        videos[(category, video_id)].append(row)

    if len(videos) != TARGET_VIDEOS:
        raise ValueError(
            f"팀원 {WORKER_ID}: "
            f"{TARGET_VIDEOS}개 영상 예상, "
            f"{len(videos)}개 확인"
        )

    vlm_inputs = []
    errors = []
    copied_count = 0

    for index, ((category, video_id), frames) in enumerate(
        sorted(videos.items()), start=1
    ):
        frames.sort(key=lambda row: int(row["frame_index"]))

        if len(frames) < NUM_FRAMES:
            errors.append(
                f"{video_id}: 프레임 부족 "
                f"({len(frames)}/{NUM_FRAMES})"
            )
            continue

        # 기존 방식과 동일한 균등 샘플링
        indices = [
            round(i * (len(frames) - 1) / (NUM_FRAMES - 1))
            for i in range(NUM_FRAMES)
        ]

        selected = []

        for frame_number, frame_index in enumerate(indices):
            frame = frames[frame_index]

            # STEP 3의 경로는 DATASET_DIR 기준 상대경로
            source_path = DATASET_DIR / frame["frame_path"]

            if not source_path.is_file():
                errors.append(
                    f"{video_id}: 이미지 없음: {source_path}"
                )
                break

            if source_path.stat().st_size == 0:
                errors.append(
                    f"{video_id}: 빈 이미지: {source_path}"
                )
                break

            # 추론용 프레임 16장만 별도 보관
            output_dir = FRAME_DIR / video_id
            output_dir.mkdir(parents=True, exist_ok=True)

            frame_id = f"f_{frame_number:04d}"
            output_path = output_dir / f"frame_{frame_number:04d}.jpg"

            if not output_path.exists():
                shutil.copy2(source_path, output_path)

            # VLM_ROOT는 FRAME_DIR의 상위 data 폴더를 포함한 루트
            # FRAME_DIR = VLM_ROOT/data/frames
            vlm_root = FRAME_DIR.parent.parent
            relative_path = output_path.relative_to(vlm_root)

            selected.append({
                "frame_id": frame_id,
                "timestamp_sec": float(frame["timestamp_sec"]),
                "frame_path": relative_path.as_posix(),
            })

            copied_count += 1

        if len(selected) != NUM_FRAMES:
            continue

        vlm_inputs.append({
            "video_id": video_id,
            "segment_id": f"{video_id}_full",
            "class": category,
            "num_frames": NUM_FRAMES,
            "frames": selected,
        })

        if index % 25 == 0:
            print(
                f"[팀원 {WORKER_ID}] "
                f"{index}/{len(videos)}개 영상 처리",
                flush=True,
            )

    if errors:
        print("\n프레임 생성 오류:")

        for error in errors[:20]:
            print("-", error)

        raise RuntimeError(
            f"{len(errors)}개 오류가 발생했습니다. "
            "JSON은 저장하지 않습니다."
        )

    if len(vlm_inputs) != TARGET_VIDEOS:
        raise ValueError(
            f"VLM 입력 수 불일치: "
            f"{len(vlm_inputs)}/{TARGET_VIDEOS}"
        )

    frame_count = sum(
        len(item["frames"])
        for item in vlm_inputs
    )

    expected_frames = TARGET_VIDEOS * NUM_FRAMES

    if frame_count != expected_frames:
        raise ValueError(
            f"프레임 수 불일치: "
            f"{frame_count}/{expected_frames}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # 전처리 중간 결과
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            vlm_inputs, f,
            ensure_ascii=False, indent=2
        )

    # B0/B1 공통 전체 영상 입력
    for experiment in ("B0", "B1"):
        output_path = MANIFEST_DIR / f"inputs_{experiment}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                vlm_inputs, f,
                ensure_ascii=False, indent=2
            )

        print("입력 JSON 저장:", output_path)

    counts = Counter(item["class"] for item in vlm_inputs)

    print("\n========================================")
    print("WatchOut! VLM Full-Video Inputs")
    print("========================================")
    print("담당자:", WORKER_ID)
    print("영상 수:", len(vlm_inputs))
    print("유형별:", dict(counts))
    print("영상당 프레임:", NUM_FRAMES)
    print("총 프레임 참조:", frame_count)
    print("추론용 JPG 복사:", copied_count)
    print("중간 JSON:", OUTPUT_JSON)
    print("========================================")


if __name__ == "__main__":
    main()
