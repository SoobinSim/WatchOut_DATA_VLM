
from datetime import datetime
from pathlib import Path
import json


EVENT_TYPES = {
    "unpaid_exit",
    "vandalism",
    "smoking",
    "abandonment",
    "normal",
    "unclear",
}

CONFIDENCE_LEVELS = {"high", "medium", "low"}

REQUIRED_FIELDS = {
    "segment_id",
    "video_id",
    "start_time",
    "end_time",
    "event_type",
    "summary",
    "claims",
    "people_count",
    "confidence",
    "uncertain_points",
}


def validate_vlm_output(output, input_data):
    """VLM 출력과 실제 입력 프레임의 형식 일치 여부 검사."""
    errors = []

    if not isinstance(output, dict):
        return ["출력 결과가 JSON 객체가 아닙니다."]

    missing = REQUIRED_FIELDS - output.keys()

    if missing:
        errors.append(f"필수 필드 누락: {sorted(missing)}")

    if output.get("video_id") != input_data.get("video_id"):
        errors.append("video_id가 입력과 일치하지 않습니다.")

    if output.get("segment_id") != input_data.get("segment_id"):
        errors.append("segment_id가 입력과 일치하지 않습니다.")

    if output.get("event_type") not in EVENT_TYPES:
        errors.append("허용되지 않은 event_type입니다.")

    if output.get("confidence") not in CONFIDENCE_LEVELS:
        errors.append("허용되지 않은 confidence입니다.")

    if not isinstance(output.get("summary"), str):
        errors.append("summary는 문자열이어야 합니다.")

    count = output.get("people_count")

    if type(count) is not int or count < 0:
        errors.append("people_count는 0 이상의 정수여야 합니다.")

    uncertain = output.get("uncertain_points")

    if not isinstance(uncertain, list) or not all(
        isinstance(x, str) for x in uncertain
    ):
        errors.append("uncertain_points는 문자열 리스트여야 합니다.")

    # 시간 형식 검사
    # 상대 시간(초): 숫자
    # 절대 시간: ISO 8601 문자열
    times = []

    for field in ("start_time", "end_time"):
        value = output.get(field)

        if type(value) in (int, float) and value >= 0:
            times.append(("relative", float(value)))
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
                times.append(("absolute", parsed))
            except ValueError:
                errors.append(f"{field}의 시간 형식이 잘못되었습니다.")
        else:
            errors.append(f"{field}의 시간 형식이 잘못되었습니다.")

    if len(times) == 2:
        if times[0][0] != times[1][0]:
            errors.append("시작·종료 시간 형식이 서로 다릅니다.")
        else:
            try:
                if times[0][1] > times[1][1]:
                    errors.append("시작 시간이 종료 시간보다 늦습니다.")
            except TypeError:
                errors.append("시작·종료 시간대를 비교할 수 없습니다.")

    # 입력으로 제공된 Key Frame ID만 허용
    frames = input_data.get("frames", [])

    valid_frame_ids = {
        frame["frame_id"]
        for frame in frames
        if isinstance(frame, dict) and "frame_id" in frame
    }

    claims = output.get("claims")

    if not isinstance(claims, list):
        errors.append("claims는 리스트여야 합니다.")
        return errors

    claim_ids = set()

    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            errors.append(f"claims[{idx}]는 객체여야 합니다.")
            continue

        claim_id = claim.get("claim_id")
        text = claim.get("text")
        frame_ids = claim.get("frame_ids")

        if not isinstance(claim_id, str) or not claim_id:
            errors.append(f"claims[{idx}] claim_id 오류")
        elif claim_id in claim_ids:
            errors.append(f"중복 claim_id: {claim_id}")
        else:
            claim_ids.add(claim_id)

        if not isinstance(text, str) or not text.strip():
            errors.append(f"claims[{idx}] text 오류")

        if not isinstance(frame_ids, list):
            errors.append(f"claims[{idx}] frame_ids는 리스트여야 합니다.")
            continue

        for frame_id in frame_ids:
            if not isinstance(frame_id, str) or frame_id not in valid_frame_ids:
                errors.append(
                    f"claims[{idx}] 존재하지 않는 Frame ID: {frame_id}"
                )

    return errors


if __name__ == "__main__":
    # 실제 VLM 추론 전 테스트용 가상 출력
    sample_input = {
        "video_id": "vid01",
        "segment_id": "vid01_window01",
        "frames": [
            {"frame_id": "f_0000", "timestamp_sec": 0.0},
            {"frame_id": "f_0004", "timestamp_sec": 4.0},
        ],
    }

    sample_output = {
        "segment_id": "vid01_window01",
        "video_id": "vid01",
        "start_time": 0.0,
        "end_time": 4.0,
        "event_type": "smoking",
        "summary": "한 사람이 흡연으로 의심되는 동작을 보인다.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "손에 든 물체를 입 근처로 가져간다.",
                "frame_ids": ["f_0004"],
            }
        ],
        "people_count": 1,
        "confidence": "medium",
        "uncertain_points": ["연기는 확인되지 않는다."],
    }

    errors = validate_vlm_output(sample_output, sample_input)

    if errors:
        print("검증 실패")
        for error in errors:
            print("-", error)
    else:
        print("검증 성공: 모든 필드와 Frame ID가 유효합니다.")

    # 의도적으로 잘못된 Frame ID 테스트
    sample_output["claims"][0]["frame_ids"] = ["f_9999"]

    errors = validate_vlm_output(sample_output, sample_input)

    print("\n잘못된 Frame ID 테스트:")
    for error in errors:
        print("-", error)
