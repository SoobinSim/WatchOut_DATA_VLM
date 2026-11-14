
import argparse
import json
import time

from collections import defaultdict
from pathlib import Path

from config import (
    PROJECT_ROOT,
    VLM_ROOT,
    MANIFEST_DIR,
    RESULT_DIR,
    TARGET_VIDEOS,
    WORKER_ID,
)

from validate_vlm_output import validate_vlm_output


PROMPT_DIR = PROJECT_ROOT / "prompts"

DEFAULT_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"

PROMPTS = {
    "A": "promptA.txt",
    "B": "promptB.txt",
    "C": "promptC.txt",
}

MODEL_STATE = {}


# ============================================================
# 1. 입력 데이터
# ============================================================

def load_inputs(experiment):
    path = MANIFEST_DIR / f"inputs_{experiment}.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"입력 JSON 없음: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("입력 JSON은 리스트여야 합니다.")

    required = {
        "video_id",
        "segment_id",
        "duration_sec",
        "frames",
    }

    for item in data:
        missing = required - set(item)

        if missing:
            raise ValueError(
                f"입력 필드 누락: {missing}"
            )

        if len(item["frames"]) != 16:
            raise ValueError(
                f"16프레임이 아닙니다: {item['segment_id']}"
            )

    unique_videos = {item["video_id"] for item in data}

    if len(unique_videos) != TARGET_VIDEOS and experiment in ("B0", "B1"):
        raise ValueError(
            f"전체 영상 수 불일치: "
            f"{len(unique_videos)}/{TARGET_VIDEOS}"
        )

    if len(unique_videos) > TARGET_VIDEOS:
        raise ValueError(
            "입력에 담당 영상 수보다 많은 video_id가 있습니다."
        )

    return data


def load_prompt(prompt_id):
    path = PROMPT_DIR / PROMPTS[prompt_id]

    if not path.is_file():
        raise FileNotFoundError(
            f"프롬프트 파일 없음: {path}"
        )

    prompt = path.read_text(encoding="utf-8").strip()

    if not prompt:
        raise ValueError("프롬프트가 비어 있습니다.")

    return prompt


def check_frames(item):
    frames = item["frames"]

    for frame in frames:
        path = VLM_ROOT / frame["frame_path"]

        if not path.is_file():
            raise FileNotFoundError(
                f"이미지 파일 없음: {path}"
            )

        if path.stat().st_size == 0:
            raise ValueError(
                f"빈 이미지 파일: {path}"
            )


# ============================================================
# 2. JSON 파싱 및 검증
# ============================================================

def parse_and_validate(raw_output, input_data):
    try:
        text = raw_output.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()

        parsed = json.loads(text)

    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        return None, [f"JSON 파싱 실패: {e}"]

    if isinstance(parsed, dict) and "events" in parsed:
        events = parsed["events"]
    elif isinstance(parsed, list):
        events = parsed
    elif isinstance(parsed, dict):
        events = [parsed]
    else:
        return parsed, ["지원하지 않는 JSON 구조"]

    if not isinstance(events, list):
        return parsed, ["events는 리스트여야 합니다."]

    errors = []

    allowed_events = {
        "theft",
        "damage",
        "smoking",
        "abandonment",
        "unclear",
    }

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"events[{index}]: 객체가 아닙니다.")
            continue

        if event.get("event_type") not in allowed_events:
            errors.append(
                f"events[{index}]: 잘못된 event_type "
                f"({event.get('event_type')})"
            )

        try:
            event_errors = validate_vlm_output(
                event,
                input_data,
            )

            for error in event_errors:
                errors.append(
                    f"events[{index}]: {error}"
                )

        except Exception as e:
            errors.append(
                f"events[{index}] 검증 오류: {e}"
            )

    return parsed, errors


# ============================================================
# 3. MLX-VLM 모델
# ============================================================

def initialize_model(model_name):
    if model_name not in MODEL_STATE:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(
            f"[MODEL] 로딩 시작: {model_name}",
            flush=True,
        )

        model, processor = load(model_name)
        config = load_config(model_name)

        MODEL_STATE[model_name] = (
            model,
            processor,
            config,
        )

        print("[MODEL] 로딩 완료", flush=True)

    return MODEL_STATE[model_name]


def run_model(item, prompt, model_name, max_tokens):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = initialize_model(model_name)

    frames = sorted(
        item["frames"],
        key=lambda frame: frame["timestamp_sec"],
    )

    image_paths = [
        str(VLM_ROOT / frame["frame_path"])
        for frame in frames
    ]

    frame_info = "\n".join(
        f"{index + 1}. "
        f"frame_id={frame['frame_id']}, "
        f"timestamp_sec={frame['timestamp_sec']}"
        for index, frame in enumerate(frames)
    )

    full_prompt = (
        prompt
        + "\n\n[입력 영상 정보]\n"
        + f"video_id: {item['video_id']}\n"
        + f"duration_sec: {item['duration_sec']}\n"
        + "모든 시간은 원본 영상 기준 초 단위입니다.\n"
        + "아래 순서대로 Key Frame 이미지가 제공됩니다.\n"
        + frame_info
        + "\n\n[출력 지침]\n"
        + "입력 이미지만 근거로 사건을 분석하세요.\n"
        + "event_type은 theft, damage, smoking, "
          "abandonment, unclear 중 하나를 사용하세요.\n"
        + "사건 시간은 start_time, end_time으로 출력하세요.\n"
        + "서로 다른 사건은 events 배열에 기록하세요.\n"
        + '이상행동이 없으면 {"events": []}를 반환하세요.\n'
        + "반드시 지정된 JSON 구조만 출력하세요."
    )

    formatted_prompt = apply_chat_template(
        processor,
        config,
        full_prompt,
        num_images=len(image_paths),
    )

    output = generate(
        model,
        processor,
        formatted_prompt,
        image_paths,
        max_tokens=max_tokens,
    )

    if isinstance(output, str):
        return output

    if hasattr(output, "text"):
        return output.text

    raise TypeError(
        f"지원하지 않는 모델 출력 타입: {type(output).__name__}"
    )


# ============================================================
# 4. 메인 실행
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="WatchOut! Worker VLM Inference"
    )

    parser.add_argument(
        "--experiment",
        choices=["B0", "B1", "B2", "B3"],
        required=True,
    )

    parser.add_argument(
        "--prompt",
        choices=["A", "B", "C"],
        required=True,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")

    if args.max_tokens < 1:
        parser.error("--max-tokens는 1 이상이어야 합니다.")

    inputs = load_inputs(args.experiment)
    prompt = load_prompt(args.prompt)

    grouped = defaultdict(list)

    for item in inputs:
        grouped[item["video_id"]].append(item)

    video_ids = sorted(grouped)

    if args.limit is not None:
        video_ids = video_ids[:args.limit]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    mode = "dryrun" if args.dry_run else "inference"

    output_path = RESULT_DIR / (
        f"{args.experiment}_{args.prompt}_{mode}.jsonl"
    )

    # 기존 추론 결과 보호
    if output_path.exists() and not args.dry_run:
        raise FileExistsError(
            f"기존 결과 파일이 존재합니다: {output_path}\n"
            "백업하거나 파일명을 변경한 뒤 실행하세요."
        )

    print("\n========================================")
    print("WatchOut! VLM Inference")
    print("========================================")
    print("담당자:", WORKER_ID)
    print("실험:", args.experiment)
    print("프롬프트:", args.prompt)
    print("처리 영상:", len(video_ids))
    print("입력 구간:", sum(len(grouped[v]) for v in video_ids))
    print("Dry-run:", args.dry_run)
    print("결과 저장:", output_path)
    print("========================================")

    with output_path.open("w", encoding="utf-8") as out:
        for index, video_id in enumerate(video_ids, start=1):
            video_start = time.perf_counter()
            segment_results = []

            for item in grouped[video_id]:
                segment_start = time.perf_counter()

                result = {
                    "segment_id": item["segment_id"],
                    "status": None,
                    "raw_output": None,
                    "parsed_output": None,
                    "validation_errors": [],
                    "inference_time_sec": None,
                    "error": None,
                }

                try:
                    check_frames(item)

                    if args.dry_run:
                        result["status"] = "dry_run"

                    else:
                        raw_output = run_model(
                            item,
                            prompt,
                            args.model,
                            args.max_tokens,
                        )

                        result["raw_output"] = raw_output

                        parsed, errors = parse_and_validate(
                            raw_output,
                            item,
                        )

                        result["parsed_output"] = parsed
                        result["validation_errors"] = errors

                        result["status"] = (
                            "completed"
                            if not errors
                            else "validation_failed"
                        )

                except Exception as e:
                    result["status"] = "error"
                    result["error"] = (
                        f"{type(e).__name__}: {e}"
                    )

                    print(
                        f"[ERROR] {video_id}: {result['error']}",
                        flush=True,
                    )

                finally:
                    if not args.dry_run:
                        result["inference_time_sec"] = round(
                            time.perf_counter() - segment_start,
                            3,
                        )

                segment_results.append(result)

            statuses = [
                item["status"]
                for item in segment_results
            ]

            if all(status == "dry_run" for status in statuses):
                video_status = "dry_run"
            elif all(status == "completed" for status in statuses):
                video_status = "completed"
            elif "error" in statuses:
                video_status = "error"
            else:
                video_status = "validation_failed"

            elapsed = time.perf_counter() - video_start

            record = {
                "video_id": video_id,
                "worker_id": WORKER_ID,
                "experiment_id": args.experiment,
                "prompt_id": args.prompt,
                "model_name": (
                    None if args.dry_run else args.model
                ),
                "status": video_status,
                "segment_count": len(segment_results),
                "segment_results": segment_results,
                "inference_time_sec": (
                    None if args.dry_run
                    else round(elapsed, 3)
                ),
            }

            out.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            out.flush()

            print(
                f"[{index}/{len(video_ids)}] "
                f"{video_id} | {video_status} "
                f"| {elapsed:.2f}초",
                flush=True,
            )

    print("\n추론 작업 종료")
    print("결과 파일:", output_path)


if __name__ == "__main__":
    main()
