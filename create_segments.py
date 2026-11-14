
import csv
import json
from collections import Counter

from config import (
    PROCESSED_DIR,
    SOURCE_EVENT_TYPES,
    TARGET_VIDEOS,
    WORKER_ID,
)


INPUT_CSV = PROCESSED_DIR / "event_times.csv"
OUTPUT_CSV = PROCESSED_DIR / "segments.csv"


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"이벤트 시간 파일이 없습니다: {INPUT_CSV}\n"
            "먼저 extract_event_times.py를 실행하세요."
        )

    with INPUT_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "video_id",
            "class",
            "event_pairs",
            "event_times_sec",
            "needs_review",
        }

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"event_times.csv 필수 열 누락: "
                f"{required - set(reader.fieldnames or [])}"
            )

        rows = list(reader)

    if len(rows) != TARGET_VIDEOS:
        raise ValueError(
            f"팀원 {WORKER_ID}: "
            f"{TARGET_VIDEOS}개 영상 예상, "
            f"실제 {len(rows)}개"
        )

    segments = []
    seen_segment_ids = set()
    review_count = 0
    zero_event_videos = []

    for row in rows:
        video_id = row["video_id"]
        category = row["class"]

        if category not in SOURCE_EVENT_TYPES:
            raise ValueError(
                f"지원하지 않는 유형: {category}"
            )

        pairs = json.loads(row["event_pairs"])
        times = json.loads(row["event_times_sec"])

        # zip()에 의해 구간이 조용히 누락되는 문제 방지
        if len(pairs) != len(times):
            raise ValueError(
                f"이벤트 구간 수 불일치: {video_id} "
                f"(프레임={len(pairs)}, 시간={len(times)})"
            )

        needs_review = (
            str(row["needs_review"]).strip().lower() == "true"
        )

        if not pairs:
            zero_event_videos.append(video_id)

        for index, (frame_pair, time_pair) in enumerate(
            zip(pairs, times), start=1
        ):
            if len(frame_pair) != 2 or len(time_pair) != 2:
                raise ValueError(
                    f"이벤트 구간 형식 오류: {video_id}"
                )

            start_frame, end_frame = map(int, frame_pair)
            start_sec, end_sec = map(float, time_pair)

            if start_frame >= end_frame:
                raise ValueError(
                    f"시작/종료 프레임 오류: {video_id}"
                )

            if start_sec < 0 or start_sec >= end_sec:
                raise ValueError(
                    f"시작/종료 시간 오류: {video_id}"
                )

            segment_id = f"{video_id}_seg{index:02d}"

            if segment_id in seen_segment_ids:
                raise ValueError(
                    f"중복 segment_id: {segment_id}"
                )

            seen_segment_ids.add(segment_id)

            segments.append({
                "segment_id": segment_id,
                "video_id": video_id,
                "source_class": category,
                "event_type_candidate": SOURCE_EVENT_TYPES[category],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "needs_review": needs_review,
            })

            if needs_review:
                review_count += 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "segment_id",
                "video_id",
                "source_class",
                "event_type_candidate",
                "start_frame",
                "end_frame",
                "start_sec",
                "end_sec",
                "needs_review",
            ],
        )

        writer.writeheader()
        writer.writerows(segments)

    counts = Counter(
        row["source_class"] for row in segments
    )

    print("\n========================================")
    print("WatchOut! Event Segment Generation")
    print("========================================")
    print("담당자:", WORKER_ID)
    print("분석 영상:", len(rows))
    print("생성 구간:", len(segments))
    print("유형별 구간:", dict(counts))
    print("검토 대상 구간:", review_count)
    print("이벤트 0개 영상:", len(zero_event_videos))
    print("저장 경로:", OUTPUT_CSV)
    print("========================================")

    if zero_event_videos:
        print("\n[주의] 이벤트가 추출되지 않은 영상:")
        for video_id in zero_event_videos[:10]:
            print("-", video_id)
        print(
            "이 영상들은 B2/B3 후보 구간 입력에서 제외됩니다. "
            "XML 및 추출 로직을 확인하세요."
        )

    print("✅ STEP 7 완료!")


if __name__ == "__main__":
    main()
