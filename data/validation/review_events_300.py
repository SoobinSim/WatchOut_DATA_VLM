import os
from pathlib import Path
import csv
import subprocess
import sys

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
PROCESSED = BASE / "processed"

INPUT = PROCESSED / "segments_review_300.csv"
OUTPUT = PROCESSED / "segments_review_checked_300.csv"
PREVIEW_DIR = PROCESSED / "event_previews"

FIELDS = [
    "segment_id", "video_id", "source_class",
    "event_type_candidate", "start_frame", "end_frame",
    "start_sec", "end_sec", "needs_review",
    "review_status", "review_note"
]

if OUTPUT.exists():
    source = OUTPUT
    print("기존 검토 기록을 이어서 진행합니다.")
else:
    source = INPUT

with source.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))


def save():
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


print(f"전체 검토 대상: {len(rows)}개")

for i, row in enumerate(rows, 1):
    if row["review_status"] != "pending":
        continue

    segment_id = row["segment_id"]
    preview = PREVIEW_DIR / f"{segment_id}.jpg"

    print(f"\n[{i}/{len(rows)}] {segment_id}")
    print("유형:", row["source_class"])
    print("구간:", row["start_sec"], "~", row["end_sec"], "초")

    if preview.is_file():
        subprocess.run(["open", str(preview)], check=False)
    else:
        print("⚠️ 미리보기 이미지 없음:", preview)

    print("\n1 = 적절한 구간")
    print("2 = 시작·종료 시간 수정 필요")
    print("3 = 이벤트 판단 어려움")
    print("s = 건너뛰기")
    print("q = 저장하고 종료")

    choice = input("선택: ").strip().lower()

    if choice == "q":
        save()
        print("검토 진행 상황 저장 완료")
        sys.exit()

    if choice == "s":
        continue

    status_map = {
        "1": "approved",
        "2": "needs_correction",
        "3": "unclear"
    }

    if choice not in status_map:
        print("잘못된 입력입니다. 다음 구간으로 넘어갑니다.")
        continue

    row["review_status"] = status_map[choice]

    if choice in ("2", "3"):
        row["review_note"] = input("검토 메모: ").strip()

    save()

save()

print("\n✅ 검토 기록 저장 완료")
print("결과 파일:", OUTPUT)