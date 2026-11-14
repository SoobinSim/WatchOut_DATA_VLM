# WatchOut! — Video Preprocessing & VLM Inference Guide

## 1. 프로젝트 개요

본 프로젝트는 CCTV 영상의 이상행동 분석을 위해 AI-Hub 데이터를 전처리하고, Qwen3-VL 기반 Video VLM 추론을 수행하는 파이프라인입니다.

기존 400개 영상과 중복되지 않는 신규 500개를 선별하여 팀원 2명이 각각 250개씩 담당합니다.

| 구분 | 영상 수 | 작업 |
|---|---:|---|
| 기존 데이터셋 | 400개 | 전처리 완료 |
| Worker A | 250개 | 신규 전처리 및 VLM 추론 |
| Worker B | 250개 | 신규 전처리 및 VLM 추론 |
| **합계** | **900개** | 기존 + 신규 |

### 전체 작업 흐름

1. AI-Hub Validation 및 필요한 Training 원본 MP4·XML 다운로드
2. 기존 400개와 중복되지 않는 신규 500개 선별
3. 동일한 `assignments.csv`로 A/B 각각 250개 배정
4. 두 팀원이 각자 컴퓨터에서 전처리 수행
5. 전처리된 JPG와 JSON을 외장 SSD 하나에 통합
6. Mac Studio 1에서 Worker A 추론
7. Mac Studio 2에서 Worker B 추론
8. 두 컴퓨터의 JSONL 추론 결과를 외장 SSD에 통합

**전처리는 Mac Studio 방문 전에 완료하는 것을 원칙으로 하며, Mac Studio에서는 VLM 추론을 수행합니다.**

---

## 2. 분석 대상 행동

| 원본 데이터 분류 | 최종 VLM 이벤트 유형 | 설명 |
|---|---|---|
| `theft` | `theft` | 절도 의심 행동 |
| `damage` | `damage` | 기물 파손 |
| `smoking` | `smoking` | 흡연 |
| `dumping` | `abandonment` | 물건 유기·투기 |

`dumping`은 AI-Hub 원본 데이터 분류명 및 폴더명으로 유지합니다.

최종 VLM 출력에서는 `abandonment`를 사용합니다. 이상행동이 관측되지 않은 경우 `{"events": []}`를 반환하도록 프롬프트를 구성합니다.

---

## 3. AI-Hub 원본 데이터 다운로드

### 3.1 데이터셋 정보

- **데이터셋명:** 실내(편의점, 매장) 사람 이상행동 데이터
- **제공 플랫폼:** AI-Hub
- **데이터셋 ID:** 71550
- **공식 페이지:** https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=71550

기존 프로젝트에서는 해당 데이터셋의 Validation(검증) 데이터를 사용했습니다.

신규 500개를 확보할 때는 기존 400개와의 중복을 제외하고, Validation 데이터가 부족한 경우 같은 데이터셋의 Training(학습) 데이터를 추가로 사용합니다.

### 3.2 다운로드 절차

1. AI-Hub 데이터셋 페이지에 접속합니다.
2. 계정 로그인 후 데이터 이용 신청 및 승인 상태를 확인합니다.
3. 다운로드 화면으로 이동합니다.
4. Validation 데이터의 원천데이터와 라벨링데이터를 확인합니다.
5. 아래 네 가지 행동 유형에 해당하는 MP4 및 XML ZIP 파일을 선택합니다.
6. 다운로드 및 압축 해제 후 기존 영상과 중복되지 않는 수량을 확인합니다.
7. Validation 데이터만으로 신규 500개 확보가 어렵다면 Training의 동일 행동 유형 데이터를 추가로 다운로드합니다.

**반드시 원천 영상(MP4)과 해당 라벨(XML)을 함께 다운로드해야 합니다.**

### 3.3 Validation 다운로드 대상

| 유형 | 원천데이터 | 라벨링데이터 |
|---|---|---|
| 파손 | `VS_03.이상행동_08.파손.zip` | `VL_03.이상행동_08.파손.zip` |
| 흡연 | `VS_03.이상행동_10.흡연.zip` | `VL_03.이상행동_10.흡연.zip` |
| 유기 | `VS_03.이상행동_11.유기.zip` | `VL_03.이상행동_11.유기.zip` |
| 절도 | `VS_03.이상행동_12.절도.zip` | `VL_03.이상행동_12.절도.zip` |

위 파일명은 기존에 사용한 Validation 데이터 기준입니다. 실제 다운로드 화면에서 파일명과 제공 구성을 다시 확인하세요.

### 3.4 Training 다운로드 대상

Validation에서 신규 영상이 부족하면 같은 유형의 Training 데이터를 확보합니다.

| 유형 | 원천데이터 | 라벨링데이터 |
|---|---|---|
| 파손 | `TS_03.이상행동_08.파손.zip` | `TL_03.이상행동_08.파손.zip` |
| 흡연 | `TS_03.이상행동_10.흡연.zip` | `TL_03.이상행동_10.흡연.zip` |
| 유기 | `TS_03.이상행동_11.유기.zip` | `TL_03.이상행동_11.유기.zip` |
| 절도 | `TS_03.이상행동_12.절도.zip` | `TL_03.이상행동_12.절도.zip` |

Training의 파일명은 Validation과 같은 명명 규칙을 가정한 예시이므로, **다운로드 화면에 표시된 실제 파일명과 폴더 구조를 확인한 뒤 사용해야 합니다.**

데이터 유형별 접두어는 다음과 같습니다.

| 접두어 | 의미 |
|---|---|
| `VS_` | Validation 원천 영상 |
| `VL_` | Validation 라벨 |
| `TS_` | Training 원천 영상 |
| `TL_` | Training 라벨 |

최신 `select_dataset.py`는 **Validation과 Training 모두 지원**합니다.

### 3.5 분할 ZIP 처리

AI-Hub 파일이 여러 개의 `.part` 파일로 다운로드되면 전체 파일을 확보한 후 병합해야 합니다.

1. 동일 ZIP에 해당하는 분할 파일을 모두 다운로드합니다.
2. 파일 번호가 누락되지 않았는지 확인합니다.
3. 분할 파일을 번호순으로 병합합니다.
4. 병합된 ZIP의 정상 여부를 확인합니다.
5. 압축을 해제합니다.
6. MP4 및 XML 파일의 존재 여부를 확인합니다.

일반 ZIP 형태로 다운로드되었다면 별도 병합 없이 압축을 해제하면 됩니다.

### 3.6 원본 데이터 저장 구조

다음과 같은 구조를 권장합니다.

```text
AIHub_Download/
├── Validation/
│   ├── VS_03.이상행동_08.파손/
│   ├── VL_03.이상행동_08.파손/
│   ├── VS_03.이상행동_10.흡연/
│   ├── VL_03.이상행동_10.흡연/
│   ├── VS_03.이상행동_11.유기/
│   ├── VL_03.이상행동_11.유기/
│   ├── VS_03.이상행동_12.절도/
│   └── VL_03.이상행동_12.절도/
└── Training/
    ├── TS_03.이상행동_08.파손/
    ├── TL_03.이상행동_08.파손/
    └── ...
```

폴더 이름은 압축 해제 결과에 따라 다를 수 있습니다.

최신 `select_dataset.py`는 `SOURCE_ROOT`의 하위 디렉터리를 탐색하여 `VS_`, `VL_`, `TS_`, `TL_`로 시작하는 원본 폴더를 검색합니다.

ZIP 파일이 아닌 **압축 해제된 폴더**를 찾는 방식이므로 반드시 압축을 먼저 해제해야 합니다.

---

## 4. 프로젝트 파일 구성

```text
WatchOut_Preprocessing/
├── config.py
├── select_dataset.py
├── video_preprocessing.py
├── create_vlm_inputs.py
├── add_duration.py
├── extract_event_times.py
├── create_segments.py
├── make_segment_inputs.py
├── run_vlm.py
├── validate_vlm_output.py
├── prompts/
│   ├── promptA.txt
│   ├── promptB.txt
│   └── promptC.txt
├── existing_video_ids_400.csv
└── README.md
```

`assignments.csv`는 신규 영상 선별 단계에서 생성됩니다.

각 Python 파일은 공통 `config.py`의 경로 및 수량 설정을 사용합니다.

---

## 5. 공통 환경 설정

### 5.1 전처리 패키지 설치

```bash
python -m pip install opencv-python numpy pandas
```

Mac Studio의 MLX-VLM 패키지는 실제 실행 환경의 Python 버전 및 Apple Silicon 지원 여부를 확인한 뒤 설치합니다.

### 5.2 config.py 설정

프로젝트 경로 및 담당자 설정은 다음과 같이 사용합니다.

```python
from pathlib import Path
import os

SOURCE_ROOT = Path("/본인의/AIHub_다운로드_폴더")

PROJECT_ROOT = Path(__file__).resolve().parent

WORKER_ID = os.environ.get("WATCHOUT_WORKER_ID", "A")

TARGET_VIDEOS = 250
```

`SOURCE_ROOT`에는 압축을 해제한 Validation 및 Training 원본 폴더들이 포함된 상위 경로를 지정합니다.

`PROJECT_ROOT`는 실행 중인 `config.py`의 위치를 자동으로 인식합니다. 따라서 외장 SSD로 프로젝트를 옮겨도 프로젝트 내부 상대경로 구조를 유지할 수 있습니다.

### 5.3 영상 수 설정

```python
TOTAL_SELECTION_PER_CATEGORY = {
    "theft": 125,
    "damage": 125,
    "smoking": 125,
    "dumping": 125,
}

TARGET_VIDEOS = 250
TOTAL_NEW_VIDEOS = sum(
    TOTAL_SELECTION_PER_CATEGORY.values()
)
```

유형별 125개는 현재 기본 설정이며, 데이터 확보 가능 수량에 따라 변경할 수 있습니다.

전체 500개 및 Worker별 250개 조건은 유지해야 합니다.

### 5.4 Worker 설정

Worker A:

```bash
export WATCHOUT_WORKER_ID=A
```

Worker B:

```bash
export WATCHOUT_WORKER_ID=B
```

이후 같은 터미널에서 Python 스크립트를 실행하면 각 Worker의 전용 폴더를 사용합니다.

---

## 6. STEP 1 — 신규 영상 선별 및 A/B 배정

실행 파일: `select_dataset.py`

### 6.1 기존 400개 영상 ID 준비

`existing_video_ids_400.csv`를 프로젝트 루트에 저장합니다.

이 CSV에는 기존에 사용한 400개의 고유 `video_id`가 포함되어 있습니다.

새 데이터 선별 시 해당 ID와 중복되는 영상은 제외됩니다.

### 6.2 신규 500개 선별

**한 명만 실행합니다.**

```bash
python select_dataset.py --prepare
```

최신 `select_dataset.py`는 다음 순서로 처리합니다.

1. Validation(`VS/VL`)과 Training(`TS/TL`) 폴더 탐색
2. 각 유형별 MP4와 XML의 `video_id` 매칭
3. 기존 400개 영상 ID 제외
4. 같은 `video_id`가 Validation과 Training에 모두 있는 경우 Validation 우선
5. 유형별 목표 수량에 맞게 신규 영상 선택
6. Worker A/B 각각 250개 배정
7. 공통 `assignments.csv` 저장

생성되는 파일:

```text
WatchOut_Preprocessing/
└── assignments.csv
```

배정표의 주요 필드:

| 필드 | 의미 |
|---|---|
| `video_id` | 영상 식별자 |
| `class` | 원본 행동 유형 |
| `assigned_to` | A 또는 B |
| `source_split` | validation 또는 training |
| `video_source` | 배정표 생성 당시 MP4 경로 |
| `label_source` | 배정표 생성 당시 XML 경로 |

**선별 기준:** Validation 우선, 부족한 수량은 Training에서 확보합니다.

단, 동일 촬영 세션의 영상이나 이름이 바뀐 복제 영상은 `video_id` 비교만으로 검출되지 않을 수 있으므로 별도로 검토해야 합니다.

### 6.3 공통 assignments.csv 공유

배정표는 한 명이 생성한 후 팀원 A와 B에게 동일한 파일을 전달해야 합니다.

**각자 `--prepare`를 실행하지 마세요.**

다른 컴퓨터에서 서로 다른 영상이 선별되어 A/B가 중복될 수 있습니다.

### 6.4 담당 영상 MP4·XML 복사

Worker A:

```bash
WATCHOUT_WORKER_ID=A python select_dataset.py --copy
```

Worker B:

```bash
WATCHOUT_WORKER_ID=B python select_dataset.py --copy
```

최신 코드는 배정표에 들어 있는 `video_id`, `class`, `source_split`을 기준으로 **현재 컴퓨터의 `SOURCE_ROOT`에서 원본 MP4·XML을 다시 검색**합니다.

따라서 `assignments.csv`에 저장된 다른 사람 컴퓨터의 절대경로를 직접 변경할 필요는 없습니다.

단, 본인에게 배정된 MP4와 XML 파일이 현재 컴퓨터에 다운로드되어 있어야 합니다.

출력:

```text
workers/
├── A/
│   └── dataset/
│       ├── metadata.csv
│       ├── videos/
│       └── labels/
└── B/
    └── dataset/
        ├── metadata.csv
        ├── videos/
        └── labels/
```

각 `metadata.csv`에는 담당 영상 250개와 원본 출처(`training` 또는 `validation`)가 기록됩니다.

---

## 7. STEP 2 — 1FPS 프레임 전처리

실행 파일: `video_preprocessing.py`

```bash
python video_preprocessing.py
```

처리 내용:

- 원본 MP4에서 약 1초 간격으로 프레임 추출
- 이미지 크기: 960×540
- JPEG 품질: 85
- 프레임별 시간 정보 저장

출력:

```text
dataset/processed/
├── frames/
└── frames_metadata.csv
```

이 단계는 전체 1FPS 이미지를 생성하는 과정입니다.

최종 VLM에 입력할 16개 프레임은 다음 단계에서 선택합니다.

---

## 8. STEP 3 — 전체 영상 16프레임 생성 (B0/B1)

실행 파일: `create_vlm_inputs.py`

```bash
python create_vlm_inputs.py
```

처리 내용:

1. `frames_metadata.csv` 로딩
2. 영상별 1FPS 이미지 정렬
3. 전체 영상에서 16개 프레임 균등 선택
4. 최종 16개 JPG를 별도 추론용 폴더에 복사
5. VLM 입력 JSON 생성

출력:

```text
vlm/data/
├── frames/
└── manifests/
    ├── inputs_B0.json
    └── inputs_B1.json
```

Worker별 250개 영상 × 16개 프레임 = **4,000장 JPG**가 생성됩니다.

B0와 B1은 동일한 JPG를 공유합니다.

---

## 9. STEP 4 — 영상 길이 추가

실행 파일: `add_duration.py`

```bash
python add_duration.py
```

원본 MP4의 FPS 및 프레임 수로 `duration_sec`를 계산합니다.

출력:

```text
vlm/data/manifests/
├── vlm_inputs_with_duration.json
├── inputs_B0.json
└── inputs_B1.json
```

`inputs_B0.json`과 `inputs_B1.json`은 영상 길이 정보가 포함된 최종 입력 파일로 갱신됩니다.

---

## 10. STEP 5 — XML 이벤트 시간 추출

실행 파일: `extract_event_times.py`

```bash
python extract_event_times.py
```

이벤트 시작·종료 프레임을 읽고 원본 영상 FPS를 이용해 초 단위로 변환합니다.

| 원본 유형 | 시작 라벨 | 종료 라벨 |
|---|---|---|
| theft | `theft_start` | `theft_end` |
| damage | `broken_start` | `broken_end` |
| smoking | `smoke_start` | `smoke_end` |
| dumping | `abandon_start` | `abandon_end` |

출력:

```text
dataset/processed/event_times.csv
```

Training XML을 새로 사용하는 경우에도 위 라벨 구조와 시간축이 동일한지 소량 샘플로 확인해야 합니다.

---

## 11. STEP 6 — 이벤트 후보 구간 생성

실행 파일: `create_segments.py`

```bash
python create_segments.py
```

`event_times.csv`의 이벤트 시작·종료 시간을 구간별 행으로 변환합니다.

출력:

```text
dataset/processed/segments.csv
```

주요 필드:

- `segment_id`
- `video_id`
- `source_class`
- `event_type_candidate`
- `start_sec`
- `end_sec`
- `needs_review`

이벤트 후보 구간은 자동 생성 결과이며, 검수가 완료된 Ground Truth를 의미하지 않습니다.

---

## 12. STEP 7 — B2/B3 후보 구간 프레임 생성

실행 파일: `make_segment_inputs.py`

```bash
python make_segment_inputs.py
```

각 후보 구간에 대해 원본 MP4에서 16개 프레임을 균등 추출합니다.

출력:

```text
vlm/data/
├── frames_segments/
└── manifests/
    ├── inputs_B2.json
    └── inputs_B3.json
```

B2와 B3는 동일한 후보 구간 JPG를 공유합니다.

**B2/B3 입력 개수는 250개로 고정되지 않습니다.** 한 영상에 여러 후보 구간이 존재하거나 후보 구간이 없는 경우도 있기 때문입니다.

---

## 13. 외장 SSD에 두 팀원의 전처리 결과 통합

### 13.1 작업 원칙

각 팀원이 본인 컴퓨터에서 전처리를 끝낸 뒤, 최종 추론에 필요한 파일을 **하나의 외장 SSD**에 모읍니다.

전체 1FPS 이미지, 원본 MP4, 원본 XML을 모두 Mac Studio로 옮길 필요는 없습니다.

다만 원본 데이터와 중간 메타데이터는 재전처리를 위해 별도로 보존합니다.

### 13.2 외장 SSD 구조

```text
외장SSD/
└── WatchOut_Preprocessing/
    ├── config.py
    ├── run_vlm.py
    ├── validate_vlm_output.py
    ├── prompts/
    │   ├── promptA.txt
    │   ├── promptB.txt
    │   └── promptC.txt
    └── workers/
        ├── A/
        │   └── vlm/
        │       ├── data/
        │       │   ├── frames/
        │       │   ├── frames_segments/
        │       │   └── manifests/
        │       │       ├── inputs_B0.json
        │       │       ├── inputs_B1.json
        │       │       ├── inputs_B2.json
        │       │       └── inputs_B3.json
        │       └── results/
        └── B/
            └── vlm/
                ├── data/
                │   ├── frames/
                │   ├── frames_segments/
                │   └── manifests/
                │       ├── inputs_B0.json
                │       ├── inputs_B1.json
                │       ├── inputs_B2.json
                │       └── inputs_B3.json
                └── results/
```

### 13.3 팀원별 전달 폴더

Worker A:

`workers/A/vlm/`

Worker B:

`workers/B/vlm/`

각 폴더를 그대로 복사합니다.

**A/B의 `frames/`, `frames_segments/`, `manifests/` 폴더를 서로 합치지 마세요.**

JSON의 `frame_path`가 각 `vlm/` 폴더를 기준으로 작성되어 있기 때문입니다.

### 13.4 외장 SSD 복사 완료 검증

| 항목 | Worker A | Worker B |
|---|---:|---:|
| B0 영상 | 250개 | 250개 |
| B1 영상 | 250개 | 250개 |
| B0/B1 JPG | 4,000장 | 4,000장 |
| B2/B3 JPG | 후보 구간 수 × 16장 | 후보 구간 수 × 16장 |
| JSON 경로 누락 | 0개 | 0개 |

외장 SSD에 복사한 뒤에는 **외장 SSD의 실제 파일을 기준으로** 모든 JSON이 참조하는 JPG가 존재하는지 확인해야 합니다.

---

## 14. Mac Studio 2대에서 VLM 추론

### 14.1 Mac Studio 사용 방식

| 구분 | Mac Studio 1 | Mac Studio 2 |
|---|---|---|
| 담당 | Worker A | Worker B |
| 영상 수 | 250개 | 250개 |
| 데이터 | `workers/A/vlm/` | `workers/B/vlm/` |
| 결과 | A 전용 JSONL | B 전용 JSONL |

외장 SSD 하나를 두 대에 동시에 직접 연결할 수 없으므로, 다음 방식으로 진행합니다.

1. 외장 SSD를 Mac Studio 1에 연결합니다.
2. Mac Studio 1에서 Worker A의 데이터로 추론합니다.
3. Worker B의 추론용 데이터와 공통 코드를 Mac Studio 2의 로컬 저장공간으로 복사합니다.
4. Mac Studio 2에서 Worker B의 데이터로 추론합니다.
5. Worker B의 추론이 완료되면 결과 JSONL을 외장 SSD의 B 전용 결과 폴더로 복사합니다.

Mac Studio 2에 필요한 저장공간이 충분한지 미리 확인합니다.

### 14.2 Mac Studio 1 — Worker A

프로젝트 폴더에서 실행합니다.

```bash
export WATCHOUT_WORKER_ID=A
```

먼저 Dry-run:

```bash
python run_vlm.py --experiment B0 --prompt A --dry-run
```

실제 영상 1개 테스트:

```bash
python run_vlm.py --experiment B0 --prompt A --limit 1
```

정상 작동 확인 후 전체 실행:

```bash
python run_vlm.py --experiment B0 --prompt A
```

### 14.3 Mac Studio 2 — Worker B

프로젝트와 B 전용 데이터가 복사된 폴더에서 실행합니다.

```bash
export WATCHOUT_WORKER_ID=B
```

Dry-run:

```bash
python run_vlm.py --experiment B0 --prompt A --dry-run
```

실제 영상 1개 테스트:

```bash
python run_vlm.py --experiment B0 --prompt A --limit 1
```

전체 실행:

```bash
python run_vlm.py --experiment B0 --prompt A
```

### 14.4 추론 파일 덮어쓰기 주의

현재 `run_vlm.py`는 기존 추론 결과 파일이 있을 경우 덮어쓰지 않도록 설계되어 있습니다.

따라서 `--limit 1`로 테스트한 경우 해당 JSONL의 파일명을 변경하거나 백업한 후 전체 추론을 실행합니다.

오류로 추론이 중단되더라도 일부 결과가 저장되었을 수 있으므로 재실행 시 확인해야 합니다.

---

## 15. B0~B3 실험

| 실험 | 입력 방식 |
|---|---|
| B0 | 전체 영상의 16개 균등 프레임 |
| B1 | 전체 영상의 16개 균등 프레임 |
| B2 | XML 후보 구간의 16개 프레임 |
| B3 | XML 후보 구간의 16개 프레임 |

프롬프트 A·B·C 조합은 팀에서 정한 실험 정의를 따릅니다.

실행 예시:

```bash
python run_vlm.py --experiment B0 --prompt A
python run_vlm.py --experiment B1 --prompt B
python run_vlm.py --experiment B2 --prompt B
python run_vlm.py --experiment B3 --prompt C
```

위 명령은 예시이며, 실험별 프롬프트 배정이 최종 실험 계획과 일치하는지 확인해야 합니다.

### 모델

`mlx-community/Qwen3-VL-8B-Instruct-4bit`

실행 환경은 Apple Silicon Mac Studio와 MLX-VLM을 기준으로 합니다.

**실제 모델 호출은 Mac Studio의 설치 버전에 따라 확인해야 합니다.** Dry-run 성공만으로 모델 추론 성공을 보장하지 않습니다.

---

## 16. 추론 결과 백업 및 통합

각 Worker의 결과는 별도로 저장합니다.

```text
workers/
├── A/
│   └── vlm/
│       └── results/
│           └── B0_A_inference.jsonl
└── B/
    └── vlm/
        └── results/
            └── B0_A_inference.jsonl
```

두 파일의 이름이 같더라도 Worker별 폴더가 다르므로 구분됩니다.

추론 결과 통합 시 `worker_id`, `video_id`, `experiment_id`, `prompt_id`를 확인하여 중복 데이터를 방지합니다.

원본 JSONL을 보존한 상태에서 별도의 통합 결과를 생성하는 것을 권장합니다.

---

## 17. 최종 체크리스트

### 데이터 다운로드

- [ ] AI-Hub 데이터 이용 권한 확인
- [ ] Validation 원천 MP4·XML 확보
- [ ] 필요 시 Training 원천 MP4·XML 추가 확보
- [ ] 분할 ZIP 병합 및 압축 해제 완료
- [ ] 실제 원본 폴더명 확인
- [ ] MP4·XML 매칭 확인

### 신규 영상 선별

- [ ] 기존 400개 ID 목록 준비
- [ ] `select_dataset.py --prepare` 한 번만 실행
- [ ] 신규 500개 선별
- [ ] A/B 각각 250개 배정
- [ ] 동일한 `assignments.csv` 공유
- [ ] 각자 `select_dataset.py --copy` 실행
- [ ] `metadata.csv`에 원본 데이터 출처 기록 확인

### 사전 전처리

- [ ] `video_preprocessing.py` 완료
- [ ] `create_vlm_inputs.py` 완료
- [ ] `add_duration.py` 완료
- [ ] `extract_event_times.py` 완료
- [ ] `create_segments.py` 완료
- [ ] `make_segment_inputs.py` 완료
- [ ] 각 Worker의 B0~B3 JSON 생성
- [ ] 실제 JPG 경로 누락 없음

### 외장 SSD 준비

- [ ] A의 `vlm/` 폴더 복사
- [ ] B의 `vlm/` 폴더 복사
- [ ] 공통 Python 코드 복사
- [ ] 프롬프트 A/B/C 파일 복사
- [ ] `validate_vlm_output.py`에 `abandonment` 허용 확인
- [ ] Mac Studio 2에 B 데이터 복사할 저장공간 확인
- [ ] 외장 SSD 복사 후 프레임 경로 검증

### Mac Studio 추론

- [ ] Mac Studio 1에서 Worker A 설정
- [ ] Mac Studio 2에서 Worker B 설정
- [ ] 두 컴퓨터에서 MLX-VLM 패키지 확인
- [ ] B0 Dry-run 성공
- [ ] 영상 1개 실제 추론 성공
- [ ] 추론 결과 JSON 검증
- [ ] 담당 영상 추론 실행
- [ ] 결과 JSONL 백업
- [ ] A/B 결과 통합 및 중복 검사

---

## 18. 주의사항

- Validation과 Training을 함께 사용할 수 있지만, 두 출처를 구분하여 기록해야 합니다.
- 실제 XML 이벤트 라벨과 시간축이 동일한지 확인해야 합니다.
- `dumping`은 원본 데이터 분류명이며, 최종 VLM 이벤트 유형은 `abandonment`입니다.
- XML 후보 구간은 수동 검수가 완료된 Ground Truth와 구분해야 합니다.
- 기존 400개 영상과 중복되는 `video_id`는 신규 데이터에 포함하지 않습니다.
- 동일 촬영 세션에서 나온 연관 영상은 추후 평가 데이터 분할 시 함께 관리하는 것을 권장합니다.
- JSON과 JPG를 서로 다른 위치에 옮기는 경우 상대경로 기준을 확인해야 합니다.
- 외장 SSD의 원본 파일을 삭제하지 말고, Mac Studio 2의 결과도 외장 SSD에 백업해야 합니다.
- AI-Hub 데이터의 이용·공유·재배포는 해당 데이터셋의 이용 조건을 준수해야 합니다.

---
