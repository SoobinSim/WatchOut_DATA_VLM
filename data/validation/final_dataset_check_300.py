import os
from pathlib import Path
from collections import Counter
import csv
import json

BASE = Path(
    str(Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
)
PROCESSED = BASE / "processed"

VLM_FILE = PROCESSED / "vlm_inputs_300.json"
SEGMENTS_FILE = PROCESSED / "segments_300.csv"
REVIEW_FILE = PROCESSED / "segments_review_checked_300.csv"
REVIEW_ORIGINAL = PROCESSED / "segments_review_300.csv"
REPORT_FILE = PROCESSED / "final_dataset_report_300.json"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# 1. VLM 입력 데이터 불러오기
with VLM_FILE.open(encoding="utf-8") as f:
    vlm_inputs = json.load(f)

segments = read_csv(SEGMENTS_FILE)

# 2. 수동 검토 결과 불러오기
if REVIEW_FILE.exists():
    reviews = read_csv(REVIEW_FILE)
    review_source = "manual_review"
elif REVIEW_ORIGINAL.exists():
    reviews = read_csv(REVIEW_ORIGINAL)
    review_source = "pending_only"
else:
    reviews = []
    review_source = "not_found"

# 3. 검토 상태 집계
review_counts = Counter(
    row.get("review_status", "pending").strip().lower()
    for row in reviews
)

# 4. 데이터셋 기본 검사
errors = []

class_counts = Counter(
    video["class"] for video in vlm_inputs
)

expected_classes = ["theft", "damage", "smoking", "dumping"]

if len(vlm_inputs) != 300:
    errors.append("영상 수가 300개가 아닙니다.")

for category in expected_classes:
    if class_counts[category] != 75:
        errors.append(f"{category}: 75개가 아닙니다.")

video_keys = [
    (video["class"], video["video_id"])
    for video in vlm_inputs
]

if len(video_keys) != len(set(video_keys)):
    errors.append("중복 영상 ID가 있습니다.")

total_frame_refs = 0

for video in vlm_inputs:
    frames = video["frames"]

    if len(frames) != 16:
        errors.append(
            f"{video['video_id']}: 입력 프레임이 16개가 아닙니다."
        )

    frame_ids = [frame["frame_id"] for frame in frames]

    if len(frame_ids) != len(set(frame_ids)):
        errors.append(
            f"{video['video_id']}: frame_id 중복"
        )

    for frame in frames:
        image_path = Path(frame["frame_path"])

        if not image_path.is_file():
            errors.append(f"이미지 누락: {image_path}")
        elif image_path.stat().st_size == 0:
            errors.append(f"빈 이미지: {image_path}")

        total_frame_refs += 1

if total_frame_refs != 4800:
    errors.append("전체 프레임 참조가 4,800개가 아닙니다.")

# 5. 이벤트 및 검토 목록 검사
segment_ids = [row["segment_id"] for row in segments]
segment_id_set = set(segment_ids)

if len(segment_ids) != len(segment_id_set):
    errors.append("중복 segment_id가 있습니다.")

review_ids = [row["segment_id"] for row in reviews]

if len(review_ids) != len(set(review_ids)):
    errors.append("검토 목록에 중복 segment_id가 있습니다.")

unknown_review_ids = set(review_ids) - segment_id_set

if unknown_review_ids:
    errors.append("검토 목록에 존재하지 않는 segment_id가 있습니다.")

for segment in segments:
    if float(segment["start_sec"]) >= float(segment["end_sec"]):
        errors.append(
            f"시간 오류: {segment['segment_id']}"
        )

# 검토 대상 63개 영상에 해당하는 구간
expected_review_ids = {
    row["segment_id"]
    for row in segments
    if row["needs_review"].strip().lower() == "true"
}

if set(review_ids) != expected_review_ids:
    errors.append("검토 대상 구간 목록이 원본과 일치하지 않습니다.")

# 6. 결과 리포트
report = {
    "total_videos": len(vlm_inputs),
    "class_counts": dict(class_counts),
    "total_vlm_frame_references": total_frame_refs,
    "total_event_candidates": len(segments),
    "review_target_segments": len(expected_review_ids),
    "review_source": review_source,
    "review_status": {
        "approved": review_counts["approved"],
        "needs_correction": review_counts["needs_correction"],
        "unclear": review_counts["unclear"],
        "pending": review_counts["pending"],
    },
    "validation_errors": errors,
    "basic_validation_passed": len(errors) == 0,
}

with REPORT_FILE.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# 7. 출력
print("\n========== STEP 12 결과 ==========")
print("전체 영상:", len(vlm_inputs))
print("유형별:", dict(class_counts))
print("VLM 입력 프레임 참조:", total_frame_refs)
print("이벤트 구간 후보:", len(segments))

print("\n[수동 검토 현황]")
print("승인:", review_counts["approved"])
print("수정 필요:", review_counts["needs_correction"])
print("판단 어려움:", review_counts["unclear"])
print("미검토:", review_counts["pending"])

print("\n기본 검증 오류:", len(errors))
print("리포트:", REPORT_FILE)

if errors:
    for error in errors[:10]:
        print("❌", error)
else:
    print("✅ 데이터셋 기본 무결성 검사 통과!")

if review_counts["pending"] > 0:
    print("⚠️ 수동 검토가 완료되지 않은 구간이 있습니다.")

print("✅ STEP 12 실행 완료")