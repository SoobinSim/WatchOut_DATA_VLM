# Validation

데이터 전처리 결과와 Video VLM 출력의 유효성을 점검하기 위한 검증 코드입니다.

## File Descriptions

| File | Description |
|---|---|
| `check_source_data.py` | 원본 데이터의 파일 구성 및 상태 확인 |
| `validate_dataset_300.py` | 이상행동 영상 300개 데이터셋 검증 |
| `validate_combined_400.py` | 이상행동·정상행동 통합 400개 데이터셋 검증 |
| `validate_vlm_output.py` | VLM 출력 구조 및 필수 필드 검증 |

## Validation Targets

### 1. Dataset Integrity

- 데이터 파일 존재 여부
- 영상 식별자 및 데이터 수량
- 메타데이터와 실제 파일의 대응 관계
- 데이터셋 구성 오류 여부

### 2. VLM Input Integrity

- 입력 JSON 구성
- Frame ID
- 이미지 경로
- 타임스탬프 및 영상 길이

### 3. VLM Output Integrity

- 필수 JSON 필드 존재 여부
- 사건 유형 값의 유효성
- `start_time`, `end_time` 숫자형 검사
- 사건 시작·종료 시간의 순서 및 영상 길이 범위
- 근거 Frame ID의 입력 데이터 내 존재 여부
- Claim 및 불확실성 필드 구조

## Validation vs. Evaluation

**Validation**은 데이터 또는 출력 형식이 올바른지 검사합니다.

**Evaluation**은 모델이 실제 사건을 정확히 탐지하고 설명했는지 평가합니다.

예를 들어 존재하는 Frame ID를 반환했다고 해서 해당 이미지가 Claim을 실제로 증명하는 것은 아닙니다.

따라서 구조 검증과 시각적 근거 검증은 별도로 수행해야 합니다.

## Path Configuration

**데이터 검증 스크립트의 입력 경로와 출력 경로는 각자의 실행 환경에 맞게 직접 설정해야 합니다.**

프로젝트 원본 데이터의 디렉터리 구조, 메타데이터 파일명 및 JSON 구조가 달라지면 코드 내 경로와 입력 항목도 수정해야 합니다.

일부 스크립트는 개발 당시 사용한 로컬 경로를 포함할 수 있습니다.

## Notes

본 저장소의 검증 스크립트는 실제 데이터 파일이 별도로 준비되어 있어야 실행할 수 있습니다.

또한 `vlm/validate_vlm_output.py`와 이 폴더의 동일한 이름의 파일은 복사 시점에 따라 내용이 달라질 수 있으므로, 실제 추론에 사용하는 검증기 버전을 기준으로 일관성을 유지해야 합니다.
