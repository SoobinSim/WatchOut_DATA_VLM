import os
from pathlib import Path
import cv2
import csv

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
VIDEO_DIR = BASE / "normal/videos"
FRAME_DIR = BASE / "normal/processed/frames"
META_PATH = BASE / "normal/processed/frames_metadata_100.csv"

FRAME_DIR.mkdir(parents=True, exist_ok=True)

rows = []
videos = sorted(VIDEO_DIR.glob("*.mp4"))

for i, video_path in enumerate(videos, 1):
    video_id = video_path.stem
    out_dir = FRAME_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"열기 실패: {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print(f"FPS 오류: {video_path.name}")
        cap.release()
        continue

    duration = total_frames / fps
    saved = 0

    for sec in range(int(duration) + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))

        frame_path = out_dir / f"frame_{sec:04d}.jpg"
        cv2.imwrite(
            str(frame_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        saved += 1

    cap.release()

    rows.append({
        "video_id": video_id,
        "fps_original": fps,
        "duration_sec": round(duration, 2),
        "frames_extracted": saved,
        "frame_dir": str(out_dir)
    })

    print(f"[{i}/{len(videos)}] {video_id}: {saved}프레임", flush=True)

with META_PATH.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n✅ 정상 구매 100개 프레임 추출 완료!")
print(f"메타데이터: {META_PATH}")