# Data Preprocessing

CCTV 영상 데이터를 선별하고, XML 어노테이션을 분석하여 Video VLM이 사용할 수 있는 입력 데이터로 변환하는 코드입니다.

## Processing Pipeline

```text
Raw CCTV Videos & XML
        |
        v
Dataset Selection
        |
        v
XML Event Analysis
        |
        v
Key Frame Extraction
        |
        v
VLM Input JSON Generation
        |
        v
Development / Test Split
```

## File Descriptions

### 1. Data Collection & Selection

| File | Description |
|---|---|
| `merge_zip.py` | 분할 압축 데이터 병합 |
| `select_dataset_300.py` | 4개 이상행동 유형에서 총 300개 영상 선별 |
| `select_normal_100.py` | 정상 이용 영상 100개 선별 |
| `finalize_anomaly_300.py` | 이상행동 데이터셋 구성 및 정리 |
| `finalize_dataset_400.py` | 이상행동·정상행동 데이터셋 통합 |

### 2. XML & Event Processing

| File | Description |
|---|---|
| `xml_analysis.py` | XML 어노테이션 구조 분석 |
| `extract_event_times_300.py` | XML 기반 사건 발생 시간 추출 |
| `create_segments_300.py` | 이벤트 후보 구간 생성 |

XML에서 추출한 시간은 모델 입력 후보 구간을 구성하기 위한 것입니다. XML 후보가 곧 수동 검수로 확정된 정답을 의미하지는 않습니다.

### 3. Frame Extraction

| File | Description |
|---|---|
| `video_preprocessing_300.py` | 이상행동 영상 프레임 전처리 |
| `extract_normal_purchase.py` | 정상 구매 영상 추출 |
| `preprocess_normal_100.py` | 정상 영상 전처리 |

프레임 추출 및 선택 과정을 통해 영상당 16개 Key Frame을 사용하는 실험 입력을 구성했습니다.

### 4. VLM Input Generation

| File | Description |
|---|---|
| `create_vlm_inputs_300.py` | 이상행동 영상 VLM 입력 JSON 생성 |
| `create_normal_vlm_inputs.py` | 정상 영상 VLM 입력 JSON 생성 |
| `merge_vlm_inputs_400.py` | 이상행동 및 정상행동 입력 병합 |
| `add_duration.py` | 영상 길이 메타데이터 추가 |
| `make_baseline_inputs.py` | B0·B1 전체 영상 기반 입력 생성 |
| `make_segment_inputs.py` | B2·B3 후보 구간 기반 입력 생성 |
| `make_dev_test_split.py` | 촬영 그룹 기반 dev/test 분할 |

## Dataset Summary

| Item | Count |
|---|---:|
| Total videos | 400 |
| Abnormal videos | 300 |
| Normal videos | 100 |
| Full-video Key Frames | 6,400 |
| XML candidate segments | 363 |
| Segment-based Key Frames | 5,808 |

## Input Data Format

VLM 입력 JSON은 영상 식별자, 구간 식별자, 영상 길이 및 프레임 정보를 포함하도록 구성했습니다.

```json
{
  "video_id": "example_video",
  "segment_id": "example_segment",
  "duration_sec": 60.0,
  "frames": [
    {
      "frame_id": "f_0000",
      "timestamp_sec": 0.0,
      "frame_path": "data/frames/example/frame_0000.jpg"
    }
  ]
}
```

위 데이터는 형식 설명을 위한 예시입니다.

## Path Configuration

**실행 전 사용자 환경에 맞게 모든 입력·출력 경로를 확인하고 수정해야 합니다.**

특히 다음 경로를 확인하세요.

- 원본 MP4 데이터가 저장된 위치
- XML 어노테이션 파일 위치
- 추출한 JPG 프레임 저장 위치
- 메타데이터 CSV 저장 위치
- VLM 입력 JSON 저장 위치

일부 코드에는 개발 당시 사용한 로컬 디렉터리 또는 외장 SSD 경로가 포함되어 있을 수 있습니다.

파일을 실행하기 전에 `Path(...)`, `ROOT`, `BASE_DIR`, `DATA_DIR`, `OUTPUT_DIR` 등 경로 관련 설정을 확인하세요. 실제 변수명은 파일마다 다를 수 있습니다.

원본 데이터의 접근 권한과 디렉터리 구조가 동일하지 않으면 코드를 그대로 실행할 수 없습니다.

## Notes

- 데이터셋 원본은 본 저장소에 포함되지 않습니다.
- XML 후보 이벤트는 수동 검수된 Ground Truth와 구분해야 합니다.
- 데이터 분할은 영상 ID에서 추정한 촬영 그룹을 기준으로 구성했습니다.
- 재현 실험에서는 데이터 분할과 랜덤 시드를 고정하는 것을 권장합니다.
