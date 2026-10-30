import os
from pathlib import Path
import csv
import json

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
INPUT = BASE / "processed" / "event_times_300.csv"
OUTPUT = BASE / "processed" / "segments_300.csv"

# 팀에서 정한 분류값과 AI-Hub 라벨 연결
# 주의: theft는 unpaid_exit과 완전히 같은 개념이 아니므로 별도 보존
EVENT_TYPES = {
    "theft": "unpaid_exit",
    "damage": "vandalism",
    "smoking": "smoking",
    "dumping": "abandonment",
}

segments = []
review_count = 0

with INPUT.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    video_id = row["video_id"]
    category = row["class"]

    pairs = json.loads(row["event_pairs"])
    times = json.loads(row["event_times_sec"])

    needs_review = row["needs_review"].lower() == "true"

    for index, ((start_frame, end_frame), (start_sec, end_sec)) in enumerate(
        zip(pairs, times), start=1
    ):
        segment_id = f"{video_id}_seg{index:02d}"

        segments.append({
            "segment_id": segment_id,
            "video_id": video_id,
            "source_class": category,
            "event_type_candidate": EVENT_TYPES[category],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "needs_review": needs_review,
        })

        if needs_review:
            review_count += 1

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

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
        ],
    )
    writer.writeheader()
    writer.writerows(segments)

print("이벤트 구간:", len(segments))
print("검토 대상 구간:", review_count)
print("결과 저장:", OUTPUT)
print("✅ STEP 7 완료")