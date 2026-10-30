import os
from pathlib import Path
import csv
from collections import defaultdict

BASE = Path(
    str(Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
)

INPUT = BASE / "processed" / "segments_300.csv"
OUTPUT = BASE / "processed" / "segments_review_300.csv"

with INPUT.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

review_rows = [
    row for row in rows
    if row["needs_review"].lower() == "true"
]

review_rows.sort(
    key=lambda r: (
        r["source_class"],
        r["video_id"],
        float(r["start_sec"])
    )
)

video_groups = defaultdict(list)

for row in review_rows:
    video_groups[row["video_id"]].append(row)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "segment_id",
            "video_id",
            "source_class",
            "event_type_candidate",
            "start_frame",
            "end_frame",
            "start_sec",
            "end_sec",
            "needs_review",
            "review_status",
            "review_note"
        ]
    )

    writer.writeheader()

    for row in review_rows:
        writer.writerow({
            **row,
            "review_status": "pending",
            "review_note": ""
        })

print("검토 대상 영상:", len(video_groups))
print("검토 대상 구간:", len(review_rows))
print("저장 경로:", OUTPUT)
print("✅ STEP 9 완료!")