import os
from pathlib import Path
from collections import Counter
import csv
import json

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")

INPUT_PATH = BASE / "combined/vlm_inputs_400.json"
SEGMENT_PATH = BASE / "processed/segments_300.csv"
REVIEW_PATH = BASE / "processed/segments_review_checked_300.csv"
OUTPUT_PATH = BASE / "combined/final_validation_report_400.json"

EXPECTED_TOTAL = 400
EXPECTED_ABNORMAL = 300
EXPECTED_NORMAL = 100

errors = []
warnings = []

# 1. 통합 VLM 입력 읽기
with INPUT_PATH.open("r", encoding="utf-8") as f:
    inputs = json.load(f)

if not isinstance(inputs, list):
    raise ValueError("통합 VLM 입력은 리스트 형식이어야 합니다.")

group_counts = Counter(
    item.get("dataset_group") for item in inputs
)

class_counts = Counter(
    item.get("source_class", "unknown") for item in inputs
)

if len(inputs) != EXPECTED_TOTAL:
    errors.append(f"전체 영상 개수 오류: {len(inputs)}개")

if group_counts["abnormal"] != EXPECTED_ABNORMAL:
    errors.append("이상행동 영상 개수 오류")

if group_counts["normal"] != EXPECTED_NORMAL:
    errors.append("정상 구매 영상 개수 오류")


# 2. 영상 ID 및 프레임 검증
video_ids = []
total_frames = 0
missing_frames = []
invalid_frames = []
frame_counts = []

for item in inputs:
    video_id = item.get("video_id")
    video_ids.append(video_id)

    frames = item.get("frames", [])
    frame_counts.append(len(frames))
    total_frames += len(frames)

    if not frames:
        errors.append(f"{video_id}: 프레임 없음")
        continue

    if len(frames) != 16:
        warnings.append(
            f"{video_id}: 프레임 {len(frames)}개 (기준 16개)"
        )

    frame_ids = []

    for frame in frames:
        frame_id = frame.get("frame_id")
        frame_ids.append(frame_id)

        frame_path = frame.get("frame_path")

        if not frame_path:
            invalid_frames.append(f"{video_id}: 경로 없음")
            continue

        if not Path(frame_path).is_file():
            missing_frames.append(frame_path)

        if frame.get("timestamp_sec") is None:
            invalid_frames.append(
                f"{video_id}: timestamp_sec 없음"
            )

    if len(frame_ids) != len(set(frame_ids)):
        errors.append(f"{video_id}: frame_id 중복")

if len(video_ids) != len(set(video_ids)):
    errors.append("중복 video_id 발견")

if missing_frames:
    errors.append(
        f"존재하지 않는 이미지 파일 {len(missing_frames)}개"
    )

if invalid_frames:
    errors.append(
        f"프레임 정보 오류 {len(invalid_frames)}개"
    )


# 3. 이상행동 이벤트 구간 검증
with SEGMENT_PATH.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    segments = list(csv.DictReader(f))

segment_ids = [row["segment_id"] for row in segments]

if len(segment_ids) != len(set(segment_ids)):
    errors.append("중복 segment_id 발견")

abnormal_ids = {
    item["video_id"]
    for item in inputs
    if item["dataset_group"] == "abnormal"
}

for row in segments:
    segment_id = row["segment_id"]

    if row["video_id"] not in abnormal_ids:
        errors.append(
            f"{segment_id}: 없는 이상행동 영상 참조"
        )

    try:
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])

        if start_sec < 0 or end_sec < start_sec:
            errors.append(
                f"{segment_id}: 이벤트 시간 구간 오류"
            )

    except (ValueError, TypeError):
        errors.append(
            f"{segment_id}: 시간 값 변환 오류"
        )


# 4. 수동 검토 현황 확인
review_counts = {}
review_total = 0

if REVIEW_PATH.exists():
    with REVIEW_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reviews = list(csv.DictReader(f))

    review_total = len(reviews)

    status_column = next(
        (
            name for name in (
                "review_status",
                "status"
            )
            if reviews and name in reviews[0]
        ),
        None
    )

    if status_column:
        review_counts = dict(
            Counter(
                (row.get(status_column) or "empty").strip()
                for row in reviews
            )
        )
    else:
        warnings.append(
            "수동 검토 파일의 상태 컬럼을 찾지 못했습니다."
        )
else:
    warnings.append("수동 검토 CSV 파일 없음")


# 5. 결과 보고서 생성
report = {
    "dataset_name": "WatchOut_400",
    "total_videos": len(inputs),
    "group_counts": dict(group_counts),
    "class_counts": dict(class_counts),
    "total_vlm_frames": total_frames,
    "frame_count_min": min(frame_counts, default=0),
    "frame_count_max": max(frame_counts, default=0),
    "missing_frame_count": len(missing_frames),
    "abnormal_event_segments": len(segments),
    "manual_review_entries": review_total,
    "manual_review_status_counts": review_counts,
    "validation_passed": len(errors) == 0,
    "errors": errors,
    "warnings": warnings
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)


# 6. 결과 출력
print("\n========== STEP 7 검증 결과 ==========")
print(f"전체 영상: {len(inputs)}개")
print(f"이상행동: {group_counts['abnormal']}개")
print(f"정상 구매: {group_counts['normal']}개")
print(f"VLM 입력 프레임: {total_frames}개")
print(f"프레임 누락: {len(missing_frames)}개")
print(f"이상행동 이벤트 후보 구간: {len(segments)}개")
print(f"수동 검토 대상: {review_total}개")
print(f"수동 검토 상태: {review_counts}")
print(f"오류: {len(errors)}개")
print(f"경고: {len(warnings)}개")

if errors:
    print("\n오류 목록:")
    for error in errors[:20]:
        print(" -", error)

if warnings:
    print("\n경고 목록 (최대 10개):")
    for warning in warnings[:10]:
        print(" -", warning)

print(f"\n보고서: {OUTPUT_PATH}")

if errors:
    print("❌ STEP 7 검증 실패")
else:
    print("✅ STEP 7 검증 완료!")