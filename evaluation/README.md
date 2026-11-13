# Evaluation

Video VLM의 이상행동 탐지 성능과 사건 시간 정확도를 평가하고, Ground Truth 검수를 지원하기 위한 코드입니다.

## File Descriptions

| File | Description |
|---|---|
| `evaluate_vlm.py` | VLM 예측 결과와 GT를 비교하여 이벤트 기반 평가 지표 계산 |
| `create_review_list_300.py` | 수동 검수 대상 이벤트 후보 목록 생성 |
| `review_events_300.py` | 이벤트 후보 검수 작업 지원 |
| `create_event_previews_300.py` | 검수용 영상 또는 이벤트 미리보기 생성 |

## Ground Truth

XML 어노테이션에서 추출된 사건 후보는 수동 검수를 거친 확정 GT와 구분합니다.

평가 데이터는 실제 영상을 직접 검토하여 사건 유형 및 시작·종료 시간을 확인하는 방식으로 구축합니다.

초기 평가용으로 수동 검수한 15개 영상은 개발 및 검증 용도로 사용합니다.

## Evaluation Metrics

### Event Precision

모델이 예측한 사건 중 실제 정답 사건과 일치하는 비율입니다.

### Event Recall

실제 사건 중 모델이 성공적으로 탐지한 비율입니다.

### Event F1

Precision과 Recall의 조화평균입니다.

### Temporal IoU

예측한 사건 시간 구간과 GT 시간 구간의 겹침 정도를 측정합니다.

기존 평가 코드에서는 tIoU 0.3 이상을 이벤트 매칭 기준으로 사용하는 테스트를 진행했습니다.

### False Alert Rate

정상 영상 중 이상행동을 잘못 보고한 영상의 비율입니다.

### Evidence Accuracy

모델이 제시한 Claim이 실제 근거 프레임에서 확인되는지 평가합니다.

### Hallucination Rate

영상에 존재하지 않거나 근거가 없는 내용을 생성하는 비율을 평가합니다.

Evidence Accuracy와 Hallucination Rate는 별도의 수동 검토 기준이 필요합니다.

## Evaluation Workflow

```text
Manual Ground Truth
        +
VLM Prediction Results
        |
        v
Event Matching
        |
        v
Temporal IoU Calculation
        |
        v
Precision / Recall / F1
        |
        v
False Alert Analysis
        |
        v
Evidence & Hallucination Review
```

## Path Configuration

**실행 전 GT 및 예측 결과 파일 경로를 사용자 환경에 맞게 지정해야 합니다.**

필요한 데이터는 다음과 같습니다.

- 수동 검수 Ground Truth CSV
- VLM 추론 결과 JSON 또는 JSONL
- 원본 영상 또는 근거 이미지
- 영상별 dev/test 분할 정보

실제 파일 구조와 평가 코드가 기대하는 입력 형식이 다를 수 있으므로, 실행 전에 CSV 컬럼 및 JSON 구조를 확인해야 합니다.

특히 기존 `evaluate_vlm.py`는 이벤트 단위 예측 결과를 대상으로 작성되었으므로, 영상당 한 줄로 저장하는 최신 `run_vlm.py`의 JSONL 결과는 별도의 변환 과정이 필요할 수 있습니다.

## Evaluation Policy

- Ground Truth는 모델 결과를 확인하기 전에 확정합니다.
- 이미 프롬프트 개발에 사용한 dev 샘플을 최종 test 성능으로 보고하지 않습니다.
- XML 후보 이벤트를 검수 없이 확정 정답으로 사용하지 않습니다.
- 모델 출력의 구조적 유효성과 사건 판단 정확도를 구분합니다.
- 정량 성능 수치는 실제 평가 실행 결과에 근거하여 기록합니다.
