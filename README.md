# WatchOut! — CCTV Abnormal Behavior Understanding with Video VLM

**Video VLM 기반 CCTV 이상행동 이해 및 대응 AI Agent 프로젝트**

WatchOut!은 무인점포 CCTV 영상을 분석하여 이상행동을 식별하고, 사건 발생 시점과 판단 근거를 구조화하여 후속 AI Agent의 상황 판단 및 대응을 지원하는 프로젝트입니다.

이 저장소는 전체 WatchOut! 시스템 중 **데이터 구축·영상 전처리·Video VLM 추론·성능 평가** 관련 코드를 정리한 저장소입니다.

## 1. Project Overview

기존 CCTV 관제는 다수의 영상을 사람이 직접 확인해야 하며, 단순 이상행동 탐지만으로는 사건의 맥락과 판단 근거를 파악하기 어렵다는 한계가 있습니다.

본 프로젝트는 다음 과정을 목표로 합니다.

1. CCTV 영상에서 이상행동 후보 탐색
2. Key Frame 기반 영상 이해
3. 사건 유형 및 발생 시간 추정
4. 시각적 근거 프레임 연결
5. 구조화된 JSON 분석 결과 생성
6. 후속 AI Agent 및 백엔드에 결과 전달

**분석 대상 행동**
- Theft: 절도 의심 행동
- Damage: 기물 파손
- Smoking: 흡연
- Dumping: 물건 유기·투기
- Normal: 정상 이용 행동

영상에서 직접 확인할 수 없는 의도나 결제 여부는 확정하지 않고, 불확실성을 별도로 기록하도록 설계했습니다.

## 2. My Contributions

프로젝트에서 직접 수행한 주요 작업입니다.

- AI-Hub CCTV 영상 데이터 선별 및 전처리
- XML 어노테이션 분석 및 사건 후보 구간 추출
- 전체 영상 및 사건 구간 기반 Key Frame 생성
- Video VLM 입력 JSON 및 메타데이터 구축
- 촬영 그룹 기반 dev/test 데이터 분할
- Qwen3-VL 추론 파이프라인 구현
- 기본 분석·시간적 추론·근거 검증 프롬프트 설계
- JSON 출력 형식 및 근거 프레임 유효성 검증
- 이벤트 탐지 및 시간 정확도 평가 체계 설계

## 3. Dataset Construction

AI-Hub 실내 이상행동 데이터를 기반으로 실험용 데이터셋을 구축했습니다.

| Dataset | Videos |
|---|---:|
| Theft | 75 |
| Damage | 75 |
| Smoking | 75 |
| Dumping | 75 |
| Normal purchase | 100 |
| **Total** | **400** |

### Key Frame Extraction

- 전체 영상 기반: 400개 영상 × 16프레임 = 6,400개
- XML 기반 이상행동 후보: 363개 구간
- 구간 기반: 363개 × 16프레임 = 5,808개
- 영상별 프레임 ID, 타임스탬프, 상대경로를 JSON으로 관리

### Dataset Split

| Split | Videos |
|---|---:|
| Development | 322 |
| Test | 78 |

촬영 그룹이 서로 다른 데이터 분할에 포함되지 않도록 그룹 단위로 분리했습니다. 촬영 그룹은 영상 ID의 명명 규칙으로 추정한 값이므로 실제 촬영 세션 정보와의 추가 대조가 필요합니다.

## 4. Repository Structure

```text
WatchOut_DATA_VLM/
├── data_preprocessing/
├── vlm/
│   └── prompts/
├── evaluation/
├── validation/
└── README.md
```

각 폴더의 역할은 다음과 같습니다.

| Directory | Description |
|---|---|
| `data_preprocessing/` | 데이터 선별, XML 분석, 프레임 추출, VLM 입력 생성 |
| `vlm/` | Qwen3-VL 추론 코드, 출력 검증 및 프롬프트 |
| `evaluation/` | 수동 검수 지원 및 모델 성능 평가 |
| `validation/` | 데이터 및 모델 출력의 유효성 검증 |

자세한 파일 설명은 각 폴더의 README를 참고하세요.

## 5. VLM Inference

사용 모델: **Qwen3-VL-8B-Instruct**

Apple Silicon 환경에서는 MLX-VLM 기반 4-bit 모델을 사용하도록 추론 코드를 구성했습니다.

### Prompt Strategies

| Prompt | Strategy |
|---|---|
| A | 기본 행동 분석 |
| B | 시간적 추론 강화 |
| C | 시각적 근거 검증 및 환각 억제 |

세 프롬프트는 동일한 JSON 출력 형식을 사용하며, 분석 전략만 다르게 설계했습니다.

### Output Schema

```json
{
  "events": [
    {
      "event_type": "damage",
      "start_time": 26.0,
      "end_time": 52.0,
      "summary": "Example event summary",
      "claims": [
        {
          "claim_id": "c1",
          "text": "Example visual evidence",
          "frame_ids": ["f_0005"]
        }
      ],
      "people_count": 1,
      "confidence": "medium",
      "uncertain_points": []
    }
  ]
}
```

위 JSON은 실제 추론 결과가 아닌 형식 설명용 예시입니다.

시간 단위는 초(seconds)이며, `start_time`과 `end_time`은 원본 영상 시작 시점(0초)을 기준으로 합니다.

## 6. Evaluation Design

주요 평가 지표는 다음과 같습니다.

- Event Precision / Recall / F1
- Temporal Intersection over Union (tIoU)
- False Alert Rate
- Evidence Accuracy
- Hallucination Rate
- JSON Validation Success Rate
- Inference Time

이벤트 단위의 탐지 정확도뿐 아니라 사건 발생 시간, 근거 프레임의 적절성, 근거 없는 설명 생성 여부를 함께 평가하도록 설계했습니다.

실제 모델 성능 수치는 평가가 완료된 후 추가할 예정입니다.

## 7. Environment & Path Configuration

**실행 전 각 Python 파일의 경로 설정을 본인의 환경에 맞게 수정해야 합니다.**

본 프로젝트는 macOS 및 외장 SSD 환경에서 개발되었으며, 일부 스크립트에는 개발 당시 사용한 로컬 경로 또는 디렉터리 구조가 반영되어 있습니다.

또한 GitHub에 코드를 정리하면서 기존 프로젝트의 폴더 구조가 변경되었으므로, 실행 전 다음 항목을 확인해야 합니다.

| 설정 항목 | 설명 |
|---|---|
| Dataset Path | 원본 CCTV 영상 및 XML 데이터 경로 |
| Frame Directory | 추출된 JPG 프레임 이미지 폴더 |
| Manifest Path | VLM 입력 JSON 및 메타데이터 파일 경로 |
| Prompt Directory | 프롬프트 A·B·C 파일 위치 |
| Model Path | Hugging Face 모델 ID 또는 로컬 모델 경로 |
| Output Directory | 추론 및 평가 결과 저장 경로 |

예를 들어 `vlm/run_vlm.py`에서는 다음 경로 변수들을 실제 디렉터리 구조에 맞게 설정해야 합니다.

- `ROOT`
- `MANIFEST_DIR`
- `PROMPT_DIR`
- `RESULT_DIR`

다른 전처리 및 평가 스크립트 역시 파일 상단의 경로 관련 변수 또는 입력 파일 경로를 확인한 후 실행해야 합니다.

**원본 AI-Hub 데이터셋과 추출 프레임은 본 저장소에 포함되어 있지 않으며, 데이터 이용 권한과 파일은 사용자가 별도로 확보해야 합니다.**

따라서 본 저장소의 코드는 모든 환경에서 수정 없이 즉시 실행되는 형태가 아니며, 사용자별 환경 설정이 필요합니다.
## 8. Limitations

- 제한된 Key Frame만으로 분석하기 때문에 프레임 사이의 행동을 정확히 복원하기 어렵습니다.
- 영상만으로 결제 여부나 사람의 의도를 확정하기 어렵습니다.
- XML 후보 구간을 사용하는 실험은 전체 영상 탐지 실험과 평가 조건이 다릅니다.
- 데이터 분할의 촬영 그룹은 파일명 기반으로 추정했습니다.
- 구조화된 JSON 출력의 유효성과 실제 사건 판단의 정확성은 별도로 검증해야 합니다.

## 9. Data Availability

본 저장소는 데이터 처리 및 모델 실험 코드를 공유하기 위한 저장소입니다.

원본 AI-Hub CCTV 데이터, 영상에서 추출한 프레임, 실제 어노테이션 파일 등은 데이터 이용 조건과 개인정보 보호를 고려하여 공개하지 않습니다.

실행하려는 사용자는 관련 데이터의 이용 권한을 별도로 확보해야 합니다.

---

**Tech Stack:** Python, Pandas, OpenCV, XML, JSON/JSONL, Qwen3-VL, MLX-VLM
