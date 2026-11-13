
import math


ALLOWED_EVENTS = {
    "theft",
    "damage",
    "smoking",
    "abandonment",
    "unclear",
}

ALLOWED_CONFIDENCE = {
    "low",
    "medium",
    "high",
}


def valid_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_vlm_output(event, input_data):
    """
    단일 VLM 이벤트 검증.

    시간 필드:
        start_time, end_time

    시간 기준:
        원본 영상 시작 시점 = 0초

    반환:
        []            검증 통과
        [오류 메시지]  검증 실패
    """
    errors = []

    if not isinstance(event, dict):
        return ["이벤트는 JSON 객체여야 합니다."]

    required = {
        "event_type",
        "start_time",
        "end_time",
        "summary",
        "claims",
        "people_count",
        "confidence",
        "uncertain_points",
    }

    missing = required - set(event)

    if missing:
        errors.append(f"필수 필드 누락: {sorted(missing)}")
        return errors

    # 1. 사건 유형
    if event["event_type"] not in ALLOWED_EVENTS:
        errors.append("허용되지 않은 event_type")

    # 2. 시간 범위
    start = event["start_time"]
    end = event["end_time"]

    if not valid_number(start) or not valid_number(end):
        errors.append("start_time/end_time은 유한한 숫자여야 합니다.")
    elif start < 0 or end <= start:
        errors.append("사건 시간 범위가 올바르지 않습니다.")
    else:
        duration = input_data.get("duration_sec")

        if valid_number(duration) and end > duration + 0.1:
            errors.append("사건 종료 시간이 영상 길이를 초과합니다.")

    # 3. 사건 설명
    if (
        not isinstance(event["summary"], str)
        or not event["summary"].strip()
    ):
        errors.append("summary는 비어 있지 않은 문자열이어야 합니다.")

    # 4. Claim과 근거 프레임
    valid_frame_ids = {
        frame["frame_id"]
        for frame in input_data["frames"]
    }

    claims = event["claims"]

    if not isinstance(claims, list):
        errors.append("claims는 리스트여야 합니다.")
    else:
        if not claims:
            errors.append("최소 하나의 Claim이 필요합니다.")

        for idx, claim in enumerate(claims):
            if not isinstance(claim, dict):
                errors.append(f"claims[{idx}]는 객체여야 합니다.")
                continue

            if (
                not isinstance(claim.get("claim_id"), str)
                or not claim["claim_id"].strip()
            ):
                errors.append(f"claims[{idx}] claim_id 오류")

            if (
                not isinstance(claim.get("text"), str)
                or not claim["text"].strip()
            ):
                errors.append(f"claims[{idx}] text 오류")

            frame_ids = claim.get("frame_ids")

            if not isinstance(frame_ids, list) or not frame_ids:
                errors.append(f"claims[{idx}] 근거 프레임 누락")
                continue

            for frame_id in frame_ids:
                if frame_id not in valid_frame_ids:
                    errors.append(
                        f"claims[{idx}] 존재하지 않는 frame_id: {frame_id}"
                    )

    # 5. 인원 수
    people_count = event["people_count"]

    if (
        not isinstance(people_count, int)
        or isinstance(people_count, bool)
        or people_count < 0
    ):
        errors.append("people_count는 0 이상의 정수여야 합니다.")

    # 6. 신뢰도
    if event["confidence"] not in ALLOWED_CONFIDENCE:
        errors.append("confidence 값 오류")

    # 7. 불확실성
    uncertain = event["uncertain_points"]

    if not isinstance(uncertain, list):
        errors.append("uncertain_points는 리스트여야 합니다.")
    elif not all(isinstance(x, str) for x in uncertain):
        errors.append("uncertain_points는 문자열 리스트여야 합니다.")

    return errors
