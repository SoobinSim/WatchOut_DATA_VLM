import os
from pathlib import Path

# ============================================================
# 1. 사용자별 환경 설정
# ============================================================

# AI-Hub VS_ / VL_ 원본 폴더들이 있는 위치
SOURCE_ROOT = Path("/사용자/AIHub_다운로드_폴더").expanduser()

# 전처리 프로젝트 폴더
PROJECT_ROOT = Path(__file__).resolve().parent

WORKER_ID = os.environ.get("WATCHOUT_WORKER_ID", "A")

# ============================================================
# 2. 영상 선별 설정
# ============================================================

# 신규 500개 유형별 목표 수량 (임시 균등 배분안)
TOTAL_SELECTION_PER_CATEGORY = {
    "theft": 125,
    "damage": 125,
    "smoking": 125,
    "dumping": 125,
}

TARGET_VIDEOS = 250
TOTAL_NEW_VIDEOS = sum(TOTAL_SELECTION_PER_CATEGORY.values())

# 기존 400개 영상 ID 목록
EXISTING_IDS_CSV = PROJECT_ROOT / "existing_video_ids_400.csv"

# A/B 공통 배정표 (다음 단계에서 생성)
ASSIGNMENT_CSV = PROJECT_ROOT / "assignments.csv"

# AI-Hub 원본 폴더명 검색 키워드
SOURCE_CATEGORIES = {
    "theft": "12.절도",
    "damage": "08.파손",
    "smoking": "10.흡연",
    "dumping": "11.유기",
}

CATEGORIES = list(SOURCE_CATEGORIES)

# ============================================================
# 3. 팀원별 출력 경로
# ============================================================

WORKER_ROOT = PROJECT_ROOT / "workers" / WORKER_ID

DATASET_DIR = WORKER_ROOT / "dataset"
VIDEO_DIR = DATASET_DIR / "videos"
LABEL_DIR = DATASET_DIR / "labels"
PROCESSED_DIR = DATASET_DIR / "processed"

# VLM 입력 및 결과
VLM_ROOT = WORKER_ROOT / "vlm"
MANIFEST_DIR = VLM_ROOT / "data" / "manifests"
FRAME_DIR = VLM_ROOT / "data" / "frames"
SEGMENT_FRAME_DIR = VLM_ROOT / "data" / "frames_segments"
RESULT_DIR = VLM_ROOT / "results"

# ============================================================
# 4. 프레임 전처리 설정
# ============================================================

NUM_FRAMES = 16
OUTPUT_SIZE = (960, 540)
JPEG_QUALITY = 85

# ============================================================
# 5. XML 이벤트 라벨
# ============================================================

EVENT_LABELS = {
    "theft": ("theft_start", "theft_end"),
    "damage": ("broken_start", "broken_end"),
    "smoking": ("smoke_start", "smoke_end"),
    "dumping": ("abandon_start", "abandon_end"),
}

# 최종 VLM 이벤트 유형
EVENT_TYPES = {
    "theft": "theft",
    "damage": "damage",
    "smoking": "smoking",
    "dumping": "abandonment",
}

# AI-Hub 원본 후보 라벨과 구분
SOURCE_EVENT_TYPES = {
    "theft": "unpaid_exit",
    "damage": "vandalism",
    "smoking": "smoking",
    "dumping": "abandonment",
}

# ============================================================
# 6. 설정 검증
# ============================================================

if WORKER_ID not in ("A", "B"):
    raise ValueError("WORKER_ID는 A 또는 B여야 합니다.")

if TOTAL_NEW_VIDEOS != 2 * TARGET_VIDEOS:
    raise ValueError(
        "신규 전체 수량과 팀원별 목표 수량이 일치하지 않습니다."
    )

if set(TOTAL_SELECTION_PER_CATEGORY) != set(CATEGORIES):
    raise ValueError("유형별 목표 수량의 클래스가 일치하지 않습니다.")

if any(count < 0 for count in TOTAL_SELECTION_PER_CATEGORY.values()):
    raise ValueError("유형별 목표 수량은 음수가 될 수 없습니다.")


if __name__ == "__main__":
    print("===== WatchOut! Configuration =====")
    print("담당자:", WORKER_ID)
    print("담당 영상:", TARGET_VIDEOS)
    print("신규 전체 영상:", TOTAL_NEW_VIDEOS)
    print("유형별 목표:", TOTAL_SELECTION_PER_CATEGORY)
    print("원본 경로:", SOURCE_ROOT)
    print("프로젝트 경로:", PROJECT_ROOT)
    print("담당자 출력:", WORKER_ROOT)
    print("===================================")
