
import argparse
import csv
import json
import time

from collections import defaultdict
from pathlib import Path

from validate_vlm_output import validate_vlm_output


# ============================================================
# 1. 경로 및 기본 설정
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_DIR = ROOT / "data" / "manifests"
PROMPT_DIR = ROOT / "prompts"
RESULT_DIR = ROOT / "results"

DEFAULT_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"

PROMPTS = {
    "A": "promptA.txt",
    "B": "promptB.txt",
    "C": "promptC.txt",
}

# 모델은 최초 1회만 로딩하고 재사용
MODEL_STATE = {}


# ============================================================
# 2. 입력 데이터 로딩
# ============================================================

def load_inputs(experiment):
    path = MANIFEST_DIR / f"inputs_{experiment}.json"

    if not path.is_file():
        raise FileNotFoundError(f"입력 JSON 없음: {path}")

    with open(path, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    if not isinstance(inputs, list):
        raise ValueError("입력 JSON은 리스트여야 합니다.")

    return inputs


def load_prompt(prompt_id):
    path = PROMPT_DIR / PROMPTS[prompt_id]

    if not path.is_file():
        raise FileNotFoundError(f"프롬프트 파일 없음: {path}")

    prompt = path.read_text(encoding="utf-8").strip()

    if not prompt:
        raise ValueError(f"프롬프트가 비어 있습니다: {path}")

    return prompt


def filter_by_split(inputs, split):
    split_path = MANIFEST_DIR / "split_dev_test.csv"

    if not split_path.is_file():
        raise FileNotFoundError(
            f"dev/test 분할 파일 없음: {split_path}"
        )

    with open(split_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    split_map = {
        row["video_id"]: row["split"]
        for row in rows
    }

    missing = [
        item["video_id"]
        for item in inputs
        if item["video_id"] not in split_map
    ]

    if missing:
        raise ValueError(
            f"분할 정보 누락: {missing[:5]}"
        )

    return [
        item
        for item in inputs
        if split_map[item["video_id"]] == split
    ]


def check_frames(item):
    frames = item.get("frames", [])

    if not frames:
        raise ValueError(
            f"프레임이 없는 입력: {item['segment_id']}"
        )

    for frame in frames:
        path = ROOT / frame["frame_path"]

        if not path.is_file():
            raise FileNotFoundError(
                f"프레임 파일 없음: {path}"
            )


# ============================================================
# 3. VLM 응답 JSON 파싱 및 검증
# ============================================================

def parse_and_validate(raw_output, input_data):
    """
    VLM 원본 응답을 JSON으로 변환하고,
    각 이벤트를 validate_vlm_output으로 검증한다.

    반환:
        parsed_output, validation_errors
    """

    try:
        text = raw_output.strip()

        # 모델이 마크다운 코드 블록으로 반환한 경우
        if text.startswith("```"):
            lines = text.splitlines()

            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()

        parsed = json.loads(text)

    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        return None, [f"JSON 파싱 실패: {e}"]

    # 여러 출력 형태 지원
    if isinstance(parsed, dict) and "events" in parsed:
        events = parsed["events"]

    elif isinstance(parsed, list):
        events = parsed

    elif isinstance(parsed, dict):
        events = [parsed]

    else:
        return parsed, ["JSON 출력 구조 오류"]

    if not isinstance(events, list):
        return parsed, ["events는 리스트여야 합니다."]

    errors = []

    for idx, event in enumerate(events):
        try:
            event_errors = validate_vlm_output(
                event,
                input_data
            )

            for error in event_errors:
                errors.append(
                    f"events[{idx}]: {error}"
                )

        except Exception as e:
            errors.append(
                f"events[{idx}] 검증 중 오류: "
                f"{type(e).__name__}: {e}"
            )

    return parsed, errors


# ============================================================
# 4. Qwen3-VL 모델 로딩
# ============================================================

def initialize_model(model_name):
    """
    MLX-VLM 모델을 최초 한 번만 로딩한다.
    이후 영상에서는 동일 모델을 재사용한다.
    """

    if model_name not in MODEL_STATE:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        print(
            f"\n[MODEL] 모델 로딩 시작: {model_name}",
            flush=True
        )

        model, processor = load(model_name)
        config = load_config(model_name)

        MODEL_STATE[model_name] = (
            model,
            processor,
            config,
        )

        print(
            "[MODEL] 모델 로딩 완료",
            flush=True
        )

    return MODEL_STATE[model_name]


# ============================================================
# 5. 실제 VLM 추론
# ============================================================

def run_model(item, prompt):
    """
    Key Frame 16개와 프롬프트를 Qwen3-VL에 전달한다.

    모든 timestamp는 원본 영상 기준 초 단위이다.
    """

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model_name = MODEL_STATE["selected_model"]
    max_tokens = MODEL_STATE["max_tokens"]

    model, processor, config = initialize_model(model_name)

    # 프레임 시간순 정렬
    frames = sorted(
        item["frames"],
        key=lambda f: f["timestamp_sec"]
    )

    # 실제 이미지 경로
    image_paths = [
        str(ROOT / frame["frame_path"])
        for frame in frames
    ]

    # 프레임 ID와 timestamp 정보를 텍스트로 구성
    frame_info = "\n".join(
        (
            f"{i + 1}. "
            f"frame_id={frame['frame_id']}, "
            f"timestamp_sec={frame['timestamp_sec']}"
        )
        for i, frame in enumerate(frames)
    )

    # 모델에 전달할 최종 프롬프트
    full_prompt = (
        prompt
        + "\n\n[입력 영상 정보]\n"
        + f"video_id: {item['video_id']}\n"
        + f"duration_sec: {item['duration_sec']}\n"
        + "모든 시간은 원본 영상 시작 기준 초 단위입니다.\n"
        + "아래 순서대로 Key Frame 이미지가 제공됩니다.\n"
        + frame_info
        + "\n\n[출력 지침]\n"
        + "입력된 이미지만 근거로 사건을 분석하세요.\n"
        + "사건 시간은 start_time, end_time으로 출력하세요.\n"
        + "시간 단위는 초이며 원본 영상 기준입니다.\n"
        + "서로 다른 사건은 events 배열에 기록하세요.\n"
        + "이상행동이 없으면 {\"events\": []}를 반환하세요.\n"
        + "반드시 지정된 JSON 구조만 출력하세요."
    )

    # Qwen3-VL 채팅 템플릿 적용
    formatted_prompt = apply_chat_template(
        processor,
        config,
        full_prompt,
        num_images=len(image_paths),
    )

    # 실제 모델 추론
    output = generate(
        model,
        processor,
        formatted_prompt,
        image_paths,
        max_tokens=max_tokens,
    )

    # 출력 문자열 추출
    if isinstance(output, str):
        return output

    if hasattr(output, "text"):
        return output.text

    raise TypeError(
        "지원하지 않는 모델 출력 타입: "
        f"{type(output).__name__}"
    )


# ============================================================
# 6. 메인 실행
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="WatchOut! Qwen3-VL inference"
    )

    parser.add_argument(
        "--experiment",
        choices=["B0", "B1", "B2", "B3"],
        required=True
    )

    parser.add_argument(
        "--prompt",
        choices=["A", "B", "C"],
        required=True
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        required=True
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 영상 수. 생략하면 전체 처리"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="기존에 수동 검수한 15개 영상만 실행"
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상의 정수여야 합니다.")

    if args.max_tokens < 1:
        parser.error("--max-tokens는 1 이상이어야 합니다.")

    MODEL_STATE["selected_model"] = args.model
    MODEL_STATE["max_tokens"] = args.max_tokens

    # --------------------------------------------------------
    # 입력 및 프롬프트 준비
    # --------------------------------------------------------

    inputs = load_inputs(args.experiment)
    inputs = filter_by_split(inputs, args.split)

    prompt = load_prompt(args.prompt)

    print("\n========================================")
    print("WatchOut! VLM Inference")
    print("========================================")
    print("실험:", args.experiment)
    print("프롬프트:", args.prompt)
    print("데이터 분할:", args.split)
    print("입력 구간 수:", len(inputs))
    print("Dry-run:", args.dry_run)

    # --------------------------------------------------------
    # 영상 단위로 그룹화
    # --------------------------------------------------------

    grouped = defaultdict(list)

    for item in inputs:
        grouped[item["video_id"]].append(item)

        video_ids = list(grouped)

    if args.reviewed_only:
        review_path = (
            ROOT / "ground_truth" / "initial_eval_15_review.csv"
        )

        with open(review_path, "r", encoding="utf-8-sig") as f:
            reviewed_ids = {
                row["video_id"]
                for row in csv.DictReader(f)
            }

        video_ids = [
            video_id
            for video_id in video_ids
            if video_id in reviewed_ids
        ]

        if len(video_ids) != 15:
            raise ValueError(
                f"검수 영상 15개를 기대했으나 "
                f"{len(video_ids)}개가 선택됨"
            )

    if args.limit is not None:
        video_ids = video_ids[:args.limit]

    print("처리 예정 영상 수:", len(video_ids))

    # --------------------------------------------------------
    # 출력 파일 경로
    # --------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    mode_name = (
        "dryrun" if args.dry_run else "inference"
    )

    output_path = RESULT_DIR / (
        f"{args.experiment}_"
        f"{args.prompt}_"
        f"{args.split}_"
        f"{mode_name}.jsonl"
    )

    # 실제 추론 결과 덮어쓰기 방지
    if output_path.exists() and not args.dry_run:
        raise FileExistsError(
            "\n기존 추론 결과 파일이 존재합니다.\n"
            f"경로: {output_path}\n"
            "기존 파일을 백업하거나 이름을 변경한 뒤 "
            "다시 실행하세요."
        )

    print("결과 저장 경로:", output_path)
    print("========================================\n")

    # --------------------------------------------------------
    # 영상별 추론 및 JSONL 저장
    # --------------------------------------------------------

    with open(output_path, "w", encoding="utf-8") as out:

        for index, video_id in enumerate(video_ids, start=1):

            video_start = time.perf_counter()

            segments = grouped[video_id]
            responses = []

            print(
                f"\n[{index}/{len(video_ids)}] "
                f"영상: {video_id}",
                flush=True
            )

            for item in segments:

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
                    # 이미지 파일 존재 확인
                    check_frames(item)

                    if args.dry_run:
                        result["status"] = "dry_run"

                    else:
                        # 실제 Qwen3-VL 실행
                        raw_output = run_model(
                            item,
                            prompt
                        )

                        result["raw_output"] = raw_output

                        # JSON 파싱 및 이벤트 검증
                        parsed, errors = parse_and_validate(
                            raw_output,
                            item
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
                        f"[ERROR] {item['segment_id']}: "
                        f"{result['error']}",
                        flush=True
                    )

                finally:

                    if not args.dry_run:
                        result["inference_time_sec"] = round(
                            time.perf_counter() - segment_start,
                            3
                        )

                responses.append(result)

            # ------------------------------------------------
            # 영상 단위 결과 구성
            # ------------------------------------------------

            elapsed = time.perf_counter() - video_start

            statuses = [
                response["status"]
                for response in responses
            ]

            if args.dry_run:

                video_status = (
                    "dry_run"
                    if all(s == "dry_run" for s in statuses)
                    else "error"
                )

            elif all(s == "completed" for s in statuses):

                video_status = "completed"

            elif "error" in statuses:

                video_status = "error"

            else:

                video_status = "validation_failed"

            record = {
                "video_id": video_id,
                "experiment_id": args.experiment,
                "prompt_id": args.prompt,
                "split": args.split,
                "model_name": (
                    None if args.dry_run else args.model
                ),
                "status": video_status,
                "segment_count": len(segments),
                "segment_results": responses,
                "inference_time_sec": (
                    None
                    if args.dry_run
                    else round(elapsed, 3)
                ),
            }

            # 영상당 JSONL 한 줄 저장
            out.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            # 매 영상 처리 후 즉시 저장
            out.flush()

            print(
                f"완료 | 상태: {video_status} "
                f"| 구간: {len(segments)}개 "
                f"| 시간: {elapsed:.2f}초",
                flush=True
            )

    print("\n========================================")
    print("전체 처리 완료")
    print("처리 영상 수:", len(video_ids))
    print("결과 파일:", output_path)
    print("========================================")


if __name__ == "__main__":
    main()
