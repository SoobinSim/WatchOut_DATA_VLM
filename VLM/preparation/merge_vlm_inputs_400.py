import os
from pathlib import Path
import json
from collections import Counter

BASE = (Path(os.environ["WATCHOUT_BASE"]) / "WatchOut_dataset")

ABNORMAL_PATH = BASE / "processed/vlm_inputs_300.json"
NORMAL_PATH = BASE / "normal/processed/vlm_inputs_normal_100.json"

OUTPUT_DIR = BASE / "combined"
OUTPUT_PATH = OUTPUT_DIR / "vlm_inputs_400.json"

EXPECTED_ABNORMAL = 300
EXPECTED_NORMAL = 100

# 1. JSON 불러오기
with ABNORMAL_PATH.open("r", encoding="utf-8") as f:
    abnormal = json.load(f)

with NORMAL_PATH.open("r", encoding="utf-8") as f:
    normal = json.load(f)


# 2. JSON 구조 확인
def extract_items(data, name):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # JSON이 {"inputs": [...]} 형태인 경우 지원
        for key in ("inputs", "videos", "vlm_inputs", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"{name}: 지원하지 않는 JSON 구조")


abnormal = extract_items(abnormal, "abnormal")
normal = extract_items(normal, "normal")

print(f"이상행동 데이터: {len(abnormal)}개")
print(f"정상 구매 데이터: {len(normal)}개")

if len(abnormal) != EXPECTED_ABNORMAL:
    raise ValueError("이상행동 데이터 개수가 300개가 아닙니다.")

if len(normal) != EXPECTED_NORMAL:
    raise ValueError("정상 구매 데이터 개수가 100개가 아닙니다.")


# 3. 통합 데이터 생성
combined = []

for item in abnormal:
    new_item = dict(item)
    new_item["dataset_group"] = "abnormal"
    combined.append(new_item)

for item in normal:
    new_item = dict(item)
    new_item["dataset_group"] = "normal"
    combined.append(new_item)


# 4. 중복 및 경로 검사
ids = [item["video_id"] for item in combined]

if len(ids) != len(set(ids)):
    duplicates = [
        video_id
        for video_id, count in Counter(ids).items()
        if count > 1
    ]
    raise ValueError(f"중복 video_id 발견: {duplicates[:10]}")

missing_frames = []
invalid_items = []
frame_counts = []

for item in combined:
    video_id = item["video_id"]
    frames = item.get("frames", [])

    if not frames:
        invalid_items.append(video_id)
        continue

    frame_counts.append(len(frames))

    frame_ids = [frame.get("frame_id") for frame in frames]

    if len(frame_ids) != len(set(frame_ids)):
        invalid_items.append(f"{video_id}: 중복 frame_id")

    for frame in frames:
        frame_path = Path(frame["frame_path"])

        if not frame_path.is_file():
            missing_frames.append(str(frame_path))

if invalid_items or missing_frames:
    print("\n검증 오류")
    print("잘못된 영상:", invalid_items[:10])
    print("존재하지 않는 프레임:", missing_frames[:10])
    raise RuntimeError("검증 실패: 통합 JSON을 저장하지 않습니다.")


# 5. 저장
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)


# 6. 결과 출력
group_counts = Counter(item["dataset_group"] for item in combined)

print("\n========== STEP 6 결과 ==========")
print(f"전체 영상: {len(combined)}개")
print(f"이상행동: {group_counts['abnormal']}개")
print(f"정상 구매: {group_counts['normal']}개")
print(f"전체 VLM 프레임: {sum(frame_counts)}개")
print(f"프레임 누락: {len(missing_frames)}개")
print(f"저장 위치: {OUTPUT_PATH}")

print("\n✅ STEP 6 완료!")