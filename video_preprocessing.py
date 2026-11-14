
import csv
import unicodedata
from pathlib import Path

import cv2

from config import (
    DATASET_DIR,
    PROCESSED_DIR,
    VIDEO_DIR,
    CATEGORIES,
    TARGET_VIDEOS,
    OUTPUT_SIZE,
    JPEG_QUALITY,
    WORKER_ID,
)


OUTPUT_DIR = PROCESSED_DIR / "frames"
OUTPUT_CSV = PROCESSED_DIR / "frames_metadata.csv"


def normalize(text):
    return unicodedata.normalize("NFC", str(text))


def load_metadata():
    metadata_path = DATASET_DIR / "metadata.csv"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"메타데이터 파일이 없습니다: {metadata_path}\n"
            "먼저 select_dataset.py --copy를 실행하세요."
        )

    with metadata_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)

        required = {"video_id", "class", "video_path"}

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"metadata.csv에 필수 열이 없습니다: {required}"
            )

        rows = list(reader)

    if len(rows) != TARGET_VIDEOS:
        raise ValueError(
            f"팀원 {WORKER_ID}: "
            f"{TARGET_VIDEOS}개 영상이 필요하지만 "
            f"metadata.csv에는 {len(rows)}개가 있습니다."
        )

    video_ids = [normalize(row["video_id"]) for row in rows]

    if len(video_ids) != len(set(video_ids)):
        raise ValueError("metadata.csv에 중복 video_id가 있습니다.")

    for row in rows:
        if row["class"] not in CATEGORIES:
            raise ValueError(
                f"지원하지 않는 유형: {row['class']}"
            )

    return rows


def process_video(video_path, category, video_id):
    save_dir = OUTPUT_DIR / category / video_id
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            raise RuntimeError(
                f"영상 FPS 또는 프레임 수 오류: {video_path}"
            )

        records = []
        second = 0

        while True:
            source_frame = round(second * fps)

            if source_frame >= total_frames:
                break

            filename = f"frame_{second:04d}.jpg"
            save_path = save_dir / filename

            # 이전 실행 결과가 정상적으로 존재하면 재사용
            if not save_path.is_file() or save_path.stat().st_size == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                success, frame = cap.read()

                if not success:
                    raise RuntimeError(
                        f"프레임 읽기 실패: {video_path} "
                        f"({second}초)"
                    )

                frame = cv2.resize(frame, OUTPUT_SIZE)

                success = cv2.imwrite(
                    str(save_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )

                if not success:
                    raise RuntimeError(
                        f"이미지 저장 실패: {save_path}"
                    )

            # 프로젝트 폴더 기준 상대경로로 기록
            relative_path = save_path.relative_to(DATASET_DIR)

            records.append({
                "video_id": video_id,
                "class": category,
                "frame_index": second,
                "source_frame": source_frame,
                "timestamp_sec": round(source_frame / fps, 3),
                "frame_path": relative_path.as_posix(),
            })

            second += 1

        return records

    finally:
        cap.release()


def main():
    metadata = load_metadata()

    all_records = []
    success_count = 0
    errors = []

    print("\n========================================")
    print("WatchOut! Video Preprocessing")
    print("========================================")
    print("담당자:", WORKER_ID)
    print("목표 영상:", TARGET_VIDEOS)
    print("입력 영상 경로:", VIDEO_DIR)
    print("출력 프레임 경로:", OUTPUT_DIR)
    print("========================================")

    for index, row in enumerate(metadata, start=1):
        category = row["class"]
        video_id = normalize(row["video_id"])

        # metadata.csv에 기록된 경로를 우선 사용
        video_path = Path(row["video_path"])

        if not video_path.is_file():
            # 다른 컴퓨터로 프로젝트를 옮긴 경우
            # 현재 DATASET_DIR 기준으로 경로 재구성
            video_path = VIDEO_DIR / category / f"{video_id}.mp4"

        try:
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"원본 MP4 없음: {video_path}"
                )

            records = process_video(
                video_path=video_path,
                category=category,
                video_id=video_id,
            )

            all_records.extend(records)
            success_count += 1

            print(
                f"[{index}/{len(metadata)}] "
                f"{category}/{video_id}: "
                f"{len(records)}프레임 완료",
                flush=True,
            )

        except Exception as e:
            errors.append((video_id, str(e)))

            print(
                f"[ERROR] {video_id}: {e}",
                flush=True,
            )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 처리 실패가 발생하면 불완전한 CSV를 저장하지 않음
    if errors:
        print("\n전처리 실패 영상:", len(errors))

        for video_id, error in errors:
            print(f"- {video_id}: {error}")

        raise RuntimeError(
            "전처리에 실패한 영상이 있습니다. "
            "오류를 해결하고 다시 실행하세요."
        )

    if success_count != TARGET_VIDEOS:
        raise RuntimeError(
            f"처리 영상 수 불일치: "
            f"{success_count}/{TARGET_VIDEOS}"
        )

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "class",
                "frame_index",
                "source_frame",
                "timestamp_sec",
                "frame_path",
            ],
        )

        writer.writeheader()
        writer.writerows(all_records)

    print("\n========================================")
    print("전처리 완료")
    print("처리 영상:", success_count)
    print("총 프레임:", len(all_records))
    print("CSV 저장:", OUTPUT_CSV)
    print("========================================")


if __name__ == "__main__":
    main()
