
import json
from pathlib import Path

ROOT = Path("WatchOut_MacStudio")
MANIFESTS = ROOT / "data/manifests"

input_path = MANIFESTS / "vlm_inputs_400_with_duration.json"

with open(input_path, "r", encoding="utf-8") as f:
    source_data = json.load(f)

# 정답 정보 제외
inference_data = [
    {
        "video_id": item["video_id"],
        "segment_id": item["video_id"] + "_window01",
        "duration_sec": item["duration_sec"],
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "timestamp_sec": frame["timestamp_sec"],
                "frame_path": frame["frame_path"],
            }
            for frame in item["frames"]
        ],
    }
    for item in source_data
]

# 필수 검사
assert len(inference_data) == 400
assert all(len(item["frames"]) == 16 for item in inference_data)
assert all(item["duration_sec"] is not None for item in inference_data)

# B0, B1은 동일한 프레임 입력을 사용
# 실제 차이는 실행 시 적용하는 프롬프트에서 결정
for name in ("B0", "B1"):
    output_path = MANIFESTS / f"inputs_{name}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inference_data, f, ensure_ascii=False, indent=2)

    print(f"{name} 저장 완료:", output_path)

print("영상 수:", len(inference_data))
print("프레임 수:", sum(len(x["frames"]) for x in inference_data))
print("첫 번째 입력 키:", list(inference_data[0].keys()))
