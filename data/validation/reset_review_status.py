import os
from pathlib import Path
import csv
import shutil

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
FILE = BASE / "processed" / "segments_review_checked_300.csv"

# 기존 파일 백업
BACKUP = FILE.with_name("segments_review_checked_300_backup.csv")
shutil.copy2(FILE, BACKUP)

# CSV 읽기
with FILE.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# 검토 상태 초기화
for row in rows:
    row["review_status"] = "pending"
    row["review_note"] = ""

# 저장
with FILE.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ {len(rows)}개 구간의 검토 상태 초기화 완료")
print("백업 파일:", BACKUP)