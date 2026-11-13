# Video VLM Inference

Qwen3-VL 기반 CCTV 이상행동 분석을 수행하는 Python 코드와 프롬프트를 관리합니다.

## File Descriptions

| File | Description |
|---|---|
| `run_vlm.py` | 전체 영상 기반 VLM 추론, JSON 검증, 결과 저장 |
| `run_vlm_b2.py` | 사건 후보 구간 기반 VLM 추론 코드 |
| `validate_vlm_output.py` | VLM 출력 JSON 구조, 시간 범위 및 근거 Frame ID 검증 |
| `prompts/promptA.txt` | 기본 행동 분석 |
| `prompts/promptB.txt` | 시간적 추론 강화 |
| `prompts/promptC.txt` | 근거 검증 및 환각 억제 |

## Model

- Base Model: Qwen3-VL-8B-Instruct
- Apple Silicon 실행 방식: MLX-VLM
- MLX 4-bit 모델 ID: `mlx-community/Qwen3-VL-8B-Instruct-4bit`

모델 다운로드 및 실행 가능 여부는 운영체제, 메모리, 설치된 라이브러리 버전에 따라 달라질 수 있습니다.

## Inference Flow

```text
VLM Input JSON
     |
     v
Key Frame Loading
     |
     v
Prompt A / B / C
     |
     v
Qwen3-VL Inference
     |
     v
JSON Parsing
     |
     v
Schema & Evidence Validation
     |
     v
JSONL Result Saving
```

## Experiment Inputs

| Experiment | Input |
|---|---|
| B0 | 전체 영상에서 선택한 16개 프레임 |
| B1 | 전체 영상에서 선택한 16개 프레임 |
| B2 | XML 후보 구간 내부에서 선택한 16개 프레임 |
| B3 | XML 후보 구간 내부에서 선택한 16개 프레임 |

B0/B1과 B2/B3는 입력 구간의 조건이 다르며, 같은 종류의 성능 평가로 직접 비교하면 안 됩니다.

B0와 B1의 입력 프레임은 동일하게 생성되어 있으므로, 실험별 프롬프트 및 실행 설정을 명확히 기록해야 합니다.

## Output Schema

```json
{
  "events": [
    {
      "event_type": "smoking",
      "start_time": 10.0,
      "end_time": 25.0,
      "summary": "Example summary",
      "claims": [
        {
          "claim_id": "c1",
          "text": "Example claim",
          "frame_ids": ["f_0003"]
        }
      ],
      "people_count": 1,
      "confidence": "medium",
      "uncertain_points": []
    }
  ]
}
```

위 JSON은 실제 모델 예측이 아닌 구조 예시입니다.

- `start_time` / `end_time`: 원본 영상 시작 기준 초 단위
- `event_type`: theft, damage, smoking, dumping, unclear
- `claims`: 관찰 사실 및 근거 프레임
- `frame_ids`: 입력 JSON에 존재하는 Frame ID만 허용
- `uncertain_points`: 근거가 부족하거나 확인되지 않은 정보
- `events: []`: 보고할 사건이 없는 경우

## Environment Setup

Apple Silicon 기반 macOS에서 다음과 같이 설치할 수 있습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install mlx-vlm
```

해당 설치 명령은 예시이며, 실제 실행 환경과 모델 호환성을 확인해야 합니다.

## Running Inference

**현재 `run_vlm.py`는 개발 당시의 `WatchOut_MacStudio` 폴더 구조를 기준으로 작성되었습니다.**

이 저장소의 `vlm/` 폴더로 파일을 이동한 경우, 실행 전 다음 경로를 반드시 사용자 환경에 맞게 수정해야 합니다.

- `ROOT`
- `MANIFEST_DIR`
- `PROMPT_DIR`
- `RESULT_DIR`

`validate_vlm_output.py`가 import 가능한 위치에 있는지도 확인해야 합니다.

원래 프로젝트 구조에서 사용한 실행 명령 예시는 다음과 같습니다.

```bash
python WatchOut_MacStudio/scripts/run_vlm.py \
  --experiment B0 \
  --prompt A \
  --split dev \
  --dry-run \
  --limit 1
```

위 명령어는 기존 폴더 구조에서만 그대로 사용할 수 있습니다.

실제 모델 추론에서는 `--dry-run`을 제외합니다.

## Important: User-Specific Configuration

**데이터 경로, 이미지 위치, 모델 설정은 각자의 로컬 환경에 맞게 직접 수정해야 합니다.**

특히 다음 항목을 확인하세요.

1. VLM 입력 JSON 위치
2. JPG 프레임 이미지 폴더
3. 프롬프트 파일 위치
4. 모델 ID 또는 로컬 가중치 경로
5. JSONL 결과 저장 위치
6. Python 및 MLX-VLM 버전

프레임의 `frame_path`는 프로젝트 루트 기준 상대경로로 구성되어 있으므로, 실행 코드의 `ROOT`와 실제 데이터 폴더 구조가 일치해야 합니다.

## Limitations

- Key Frame 기반 추론은 연속 영상의 일부 시간 정보를 잃을 수 있습니다.
- 근거 Frame ID가 유효하다고 해서 Claim의 내용이 사실이라는 의미는 아닙니다.
- JSON 구조 검증과 실제 이상행동 판별 성능 평가는 별개입니다.
- Apple MLX 실행 코드는 NVIDIA CUDA 환경에 그대로 사용할 수 없습니다.
