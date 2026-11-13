
import json
from pathlib import Path

import pandas as pd


BASE = Path("WatchOut_dataset/inference_ready_400")
REVIEW_PATH = BASE / "initial_eval_15_review.csv"
GT_PATH = BASE / "initial_eval_15_events.csv"
PRED_PATH = BASE / "initial_predictions_15.json"

EVENT_TYPES = {
    "vandalism", "abandonment", "smoking", "unpaid_exit"
}
IOU_THRESHOLD = 0.3


def temporal_iou(start1, end1, start2, end2):
    """두 시간 구간의 Intersection over Union."""
    overlap = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)

    if union <= 0:
        return 0.0

    return overlap / union


def evaluate():
    review = pd.read_csv(REVIEW_PATH)
    gt = pd.read_csv(GT_PATH)

    with open(PRED_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        raise ValueError("예측 결과는 JSON 리스트여야 합니다.")

    # 입력 JSON에 등장한 영상만 평가
    valid_video_ids = set(review["video_id"])
    predicted_video_ids = set()

    predicted_events = []
    unclear_videos = set()

    for pred in predictions:
        video_id = pred["video_id"]
        event_type = pred["event_type"]

        if video_id not in valid_video_ids:
            raise ValueError(f"평가셋에 없는 영상: {video_id}")

        predicted_video_ids.add(video_id)

        if event_type == "unclear":
            unclear_videos.add(video_id)
            continue

        if event_type == "normal":
            continue

        if event_type not in EVENT_TYPES:
            raise ValueError(f"알 수 없는 이벤트 유형: {event_type}")

        start = pred["start_time"]
        end = pred["end_time"]

        if not all(type(x) in (int, float) for x in (start, end)):
            raise ValueError(
                "평가 시 start_time/end_time은 영상 기준 초 단위여야 합니다."
            )

        if start < 0 or end <= start:
            raise ValueError(f"잘못된 시간 구간: {video_id}")

        predicted_events.append({
            "video_id": video_id,
            "event_type": event_type,
            "start": float(start),
            "end": float(end),
        })

    missing_videos = valid_video_ids - predicted_video_ids

    if missing_videos:
        raise ValueError(
            f"예측 결과가 없는 영상 {len(missing_videos)}개: "
            f"{sorted(missing_videos)[:5]}"
        )

    # 정답 이벤트 정리
    gt_events = [
        {
            "video_id": row["video_id"],
            "event_type": row["event_type_gt"],
            "start": float(row["start_sec_gt"]),
            "end": float(row["end_sec_gt"]),
        }
        for _, row in gt.iterrows()
    ]

    # 사건별 유형과 시간 IoU를 동시에 만족해야 TP
    # 후보들을 IoU 높은 순으로 정렬한 뒤 일대일 매칭
    candidates = []

    for gi, actual in enumerate(gt_events):
        for pi, predicted in enumerate(predicted_events):
            if actual["video_id"] != predicted["video_id"]:
                continue

            if actual["event_type"] != predicted["event_type"]:
                continue

            iou = temporal_iou(
                actual["start"], actual["end"],
                predicted["start"], predicted["end"]
            )

            if iou >= IOU_THRESHOLD:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matched_ious = []

    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue

        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_ious.append(iou)

    tp = len(matched_gt)
    fp = len(predicted_events) - len(matched_pred)
    fn = len(gt_events) - len(matched_gt)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    # 정상 영상에 이상행동을 예측하면 오탐
    normal_ids = set(
        review.loc[
            review["event_type_gt"] == "normal", "video_id"
        ]
    )

    false_alert_ids = {
        p["video_id"] for p in predicted_events
        if p["video_id"] in normal_ids
    }

    normal_fpr = (
        len(false_alert_ids) / len(normal_ids)
        if normal_ids else 0.0
    )

    print("\n========== VLM 초기 성능 평가 ==========")
    print(f"평가 영상: {len(valid_video_ids)}개")
    print(f"정답 이벤트: {len(gt_events)}개")
    print(f"예측 이벤트: {len(predicted_events)}개")
    print(f"TP: {tp} / FP: {fp} / FN: {fn}")
    print(f"Event Precision: {precision:.4f}")
    print(f"Event Recall:    {recall:.4f}")
    print(f"Event F1:        {f1:.4f}")
    print(f"Matched 평균 tIoU: "
          f"{sum(matched_ious) / len(matched_ious):.4f}"
          if matched_ious else "Matched 평균 tIoU: N/A")
    print(f"정상 영상 오탐률: {normal_fpr:.4f}")
    print(f"Unclear 응답 영상 수: {len(unclear_videos)}개")
    print("=========================================\n")


if __name__ == "__main__":
    if not PRED_PATH.exists():
        print(f"VLM 예측 파일이 없습니다: {PRED_PATH}")
        print("실제 VLM 추론 후 해당 파일을 생성해 주세요.")
    else:
        evaluate()
