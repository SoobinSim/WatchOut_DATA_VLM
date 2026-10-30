import os
from pathlib import Path
import csv
import json
import shutil

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")
PROCESSED = BASE / "processed"
OUTPUT = BASE / "inference_ready"

OUTPUT.mkdir(parents=True, exist_ok=True)

files = {
    "vlm_inputs_300.json": "vlm_inputs_300.json",
    "segments_300.csv": "segments_300.csv",
    "final_dataset_report_300.json": "final_dataset_report_300.json",
}

for source_name, target_name in files.items():
    source = PROCESSED / source_name
    target = OUTPUT / target_name

    if not source.exists():
        raise FileNotFoundError(f"필수 파일 누락: {source}")

    shutil.copy2(source, target)
    print("복사 완료:", target_name)

# 검토 결과도 함께 저장
review_file = PROCESSED / "segments_review_checked_300.csv"

if review_file.exists():
    shutil.copy2(
        review_file,
        OUTPUT / "segments_review_checked_300.csv"
    )

# VLM 입력 데이터 최종 확인
with (OUTPUT / "vlm_inputs_300.json").open(encoding="utf-8") as f:
    inputs = json.load(f)

assert len(inputs) == 300
assert all(len(item["frames"]) == 16 for item in inputs)

for item in inputs:
    for frame in item["frames"]:
        if not Path(frame["frame_path"]).is_file():
            raise FileNotFoundError(frame["frame_path"])

# README 생성
readme = """# WatchOut! Anomaly Dataset — Inference Ready

## Dataset
- 300 abnormal-behavior CCTV videos
- 75 videos per class: theft, damage, smoking, dumping
- 16 sampled frames per video
- 4,800 VLM frame references

## Files
- vlm_inputs_300.json: VLM input metadata
- segments_300.csv: XML-derived event interval candidates
- final_dataset_report_300.json: basic validation results
- segments_review_checked_300.csv: manual review status, if available

## Important
- Video frames are referenced by absolute local file paths.
- Original MP4, XML, and JPG files remain in the local dataset folders.
- Event intervals are automatically extracted candidates, not fully verified ground truth.
- Actual Qwen3-VL inference has not yet been performed.
- Do not upload the underlying AI-Hub data to a public repository.
"""

(OUTPUT / "README.md").write_text(readme, encoding="utf-8")

print("\n========== 최종 정리 ==========")
print("입력 영상:", len(inputs))
print("입력 프레임:", sum(len(x["frames"]) for x in inputs))
print("저장 폴더:", OUTPUT)
print("✅ STEP 13 완료!")