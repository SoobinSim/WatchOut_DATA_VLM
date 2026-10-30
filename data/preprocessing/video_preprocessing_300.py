import os
from pathlib import Path
import csv
import cv2
import unicodedata

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
VIDEO_DIR = BASE / "videos"
OUTPUT_DIR = BASE / "processed" / "frames"

CATEGORIES = ["theft", "damage", "smoking", "dumping"]

OUTPUT_SIZE = (960, 540)
JPEG_QUALITY = 85


def normalize(text):
    return unicodedata.normalize("NFC", text)


def process_video(video_path, category):
    video_id = normalize(video_path.stem)
    save_dir = OUTPUT_DIR / category / video_id
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"영상 정보 오류: {video_path}")

    records = []
    second = 0

    while True:
        source_frame = round(second * fps)

        if source_frame >= total_frames:
            break

        filename = f"frame_{second:04d}.jpg"
        save_path = save_dir / filename

        # 기존 이미지가 없을 때만 새로 추출
        if not save_path.exists() or save_path.stat().st_size == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            success, frame = cap.read()

            if not success:
                cap.release()
                raise RuntimeError(f"프레임 읽기 실패: {video_path}")

            frame = cv2.resize(frame, OUTPUT_SIZE)

            success = cv2.imwrite(
                str(save_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if not success:
                cap.release()
                raise RuntimeError(f"이미지 저장 실패: {save_path}")

        records.append({
            "video_id": video_id,
            "class": category,
            "frame_index": second,
            "source_frame": source_frame,
            "timestamp_sec": round(source_frame / fps, 3),
            "frame_path": str(save_path)
        })

        second += 1

    cap.release()
    return records


def main():
    all_records = []
    success_count = 0
    errors = []

    for category in CATEGORIES:
        video_dir = VIDEO_DIR / category
        videos = sorted(video_dir.glob("*.mp4"))

        print(f"\n[{category}] 영상 {len(videos)}개")

        for i, video_path in enumerate(videos, 1):
            try:
                records = process_video(video_path, category)
                all_records.extend(records)
                success_count += 1
                print(f"  [{i}/{len(videos)}] 완료: {len(records)}프레임")

            except Exception as e:
                errors.append((str(video_path), str(e)))
                print(f"  ❌ 오류: {video_path.name} - {e}")

    output_csv = BASE / "processed" / "frames_metadata_300.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id", "class", "frame_index",
                "source_frame", "timestamp_sec", "frame_path"
            ]
        )
        writer.writeheader()
        writer.writerows(all_records)

    print("\n" + "=" * 40)
    print(f"처리 완료 영상: {success_count}개")
    print(f"전체 프레임 기록: {len(all_records)}개")
    print(f"처리 오류: {len(errors)}개")
    print(f"메타데이터 저장: {output_csv}")

    for path, error in errors:
        print(f"오류 파일: {path}\n원인: {error}")


if __name__ == "__main__":
    main()