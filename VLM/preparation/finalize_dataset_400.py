import os
from pathlib import Path
import shutil
import json
import csv

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
OUTPUT = BASE / "inference_ready_400"

# 원본 파일은 유지하고 메타데이터만 복사
FILES = {
    "vlm_inputs_400.json": BASE / "combined/vlm_inputs_400.json",
    "segments_300.csv": BASE / "processed/segments_300.csv",
    "segments_review_checked_300.csv": (
        BASE / "processed/segments_review_checked_300.csv"
    ),
    "final_validation_report_400.json": (
        BASE / "combined/final_validation_report_400.json"
    ),
    "normal_metadata_100.csv": (
        BASE / "normal/normal_metadata_150.csv"
    ),
}

# 1. 필수 파일 확인
missing = [
    str(src)
    for src in FILES.values()
    if not src.is_file()
]

if missing:
    raise FileNotFoundError(
        "필수 파일이 없습니다:\n" + "\n".join(missing)
    )

# 2. STEP 7 검증 결과 확인
with FILES["final_validation_report_400.json"].open(
    "r", encoding="utf-8"
) as f:
    report = json.load(f)

if not report.get("validation_passed", False):
    raise RuntimeError(
        "STEP 7 검증을 통과하지 못했습니다. "
        "오류를 해결한 뒤 다시 실행하세요."
    )

# 3. 통합 VLM 입력 확인
with FILES["vlm_inputs_400.json"].open(
    "r", encoding="utf-8"
) as f:
    inputs = json.load(f)

if len(inputs) != 400:
    raise ValueError(f"영상 개수 오류: {len(inputs)}개")

# 정상 영상은 이벤트 후보가 없으며,
# 이상행동 이벤트는 아직 최종 정답으로 확정되지 않음
dataset_info = {
    "dataset_name": "WatchOut_400",
    "total_videos": len(inputs),
    "abnormal_videos": 300,
    "normal_purchase_videos": 100,
    "total_vlm_frame_references": sum(
        len(item["frames"]) for item in inputs
    ),
    "abnormal_event_candidate_segments": report.get(
        "abnormal_event_segments", 0
    ),
    "manual_review_entries": report.get(
        "manual_review_entries", 0
    ),
    "manual_review_status_counts": report.get(
        "manual_review_status_counts", {}
    ),
    "normal_label_status": "purchase_behavior_not_manually_verified",
    "segment_label_status": "xml_derived_candidates",
    "validation_passed": True,
    "actual_vlm_inference_completed": False,
}

# 4. 출력 폴더 생성 및 복사
OUTPUT.mkdir(parents=True, exist_ok=True)

for name, src in FILES.items():
    destination = OUTPUT / name
    shutil.copy2(src, destination)
    print(f"복사 완료: {name}")

# 5. 데이터셋 요약 저장
with (OUTPUT / "dataset_info.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(
        dataset_info,
        f,
        ensure_ascii=False,
        indent=2
    )

# 6. README 작성
readme = """# WatchOut! VLM Inference Dataset

## 데이터 구성
- 이상행동 영상: 300개 (AI-Hub 238-2)
- 정상 구매행동 영상: 100개 (AI-Hub 238-1)
- 총 영상: 400개
- 영상별 최대 16개 프레임을 VLM 입력으로 사용

## 주요 파일
- vlm_inputs_400.json: VLM 입력 프레임 경로 및 시간 정보
- segments_300.csv: 이상행동 XML 기반 이벤트 후보 구간
- segments_review_checked_300.csv: 수동 검토 현황
- normal_metadata_100.csv: 정상 구매행동 영상 메타데이터
- final_validation_report_400.json: 데이터 무결성 검사
- dataset_info.json: 데이터셋 구성 요약

## 주의사항
1. 이 폴더에는 원본 MP4, XML, JPG가 포함되어 있지 않습니다.
2. JSON에 기록된 frame_path는 로컬 외장 SSD의 절대경로입니다.
3. 다른 서버에서 추론할 경우 이미지 경로 변경 또는 데이터 복사가 필요합니다.
4. XML 기반 이벤트 구간은 후보이며 최종 검증된 정답이 아닙니다.
5. 구매행동 데이터는 이상행동이 없는지 별도 검토가 필요합니다.
6. 현재 실제 VLM 추론 및 모델 성능 평가는 수행하지 않았습니다.
7. AI-Hub 데이터 및 관련 메타데이터 공유 시 이용 조건을 확인하세요.
"""

(OUTPUT / "README.md").write_text(
    readme,
    encoding="utf-8"
)

# 7. 최종 출력
print("\n========== STEP 8 최종 결과 ==========")
print(f"전체 영상: {dataset_info['total_videos']}개")
print(f"이상행동: {dataset_info['abnormal_videos']}개")
print(f"정상 구매: {dataset_info['normal_purchase_videos']}개")
print(
    "VLM 프레임 참조:",
    dataset_info["total_vlm_frame_references"],
    "개"
)
print(
    "이상행동 이벤트 후보:",
    dataset_info["abnormal_event_candidate_segments"],
    "개"
)
print(f"저장 위치: {OUTPUT}")
print("\n✅ STEP 8 완료! VLM 입력 데이터 준비 완료!")