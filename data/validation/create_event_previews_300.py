import os
from pathlib import Path
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
INPUT = BASE / "processed" / "segments_review_300.csv"
OUTPUT = BASE / "processed" / "event_previews"

OUTPUT.mkdir(parents=True, exist_ok=True)

with INPUT.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

labels = ["Before", "Start", "Middle", "End", "After"]
positions = [-3, 0, None, 0, 3]

success = 0
errors = []

for i, row in enumerate(rows, 1):
    try:
        video_id = row["video_id"]
        category = row["source_class"]
        start = float(row["start_sec"])
        end = float(row["end_sec"])

        timestamps = [
            max(0, start - 3),
            start,
            (start + end) / 2,
            end,
            end + 3,
        ]

        video_files = list((BASE / "videos" / category).glob("*.mp4"))
        matches = [
            p for p in video_files
            if p.stem == video_id
        ]

        if len(matches) != 1:
            raise RuntimeError("MP4 파일 매칭 실패")

        cap = cv2.VideoCapture(str(matches[0]))

        if not cap.isOpened():
            raise RuntimeError("영상 열기 실패")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total / fps

        frames = []

        for label, sec in zip(labels, timestamps):
            sec = min(max(sec, 0), max(0, duration - 1 / fps))
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)

            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(f"{label} 프레임 추출 실패")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (384, 216))
            frames.append(Image.fromarray(frame))

        cap.release()

        canvas = Image.new("RGB", (384 * 5, 270), "white")
        draw = ImageDraw.Draw(canvas)

        for j, (label, frame) in enumerate(zip(labels, frames)):
            x = 384 * j
            canvas.paste(frame, (x, 30))
            draw.text(
                (x + 10, 8),
                f"{label} ({timestamps[j]:.1f}s)",
                fill="black",
            )

        output_path = OUTPUT / f"{row['segment_id']}.jpg"
        canvas.save(output_path, quality=85)

        success += 1
        print(f"[{i}/{len(rows)}] 완료: {row['segment_id']}")

    except Exception as e:
        errors.append((row["segment_id"], str(e)))
        print(f"❌ {row['segment_id']}: {e}")

print("\n========== 결과 ==========")
print("미리보기 생성:", success)
print("오류:", len(errors))
print("저장 폴더:", OUTPUT)

for segment_id, error in errors[:10]:
    print(segment_id, error)