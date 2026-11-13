from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

# ==========================================
# 1. 데이터 경로
# ==========================================

BASE_DIR = Path(
    "/Volumes/SSB 010-3362-8464/"
    "생성형AI/WatchOut_dataset"
)

LABEL_DIR = BASE_DIR / "labels"

CLASSES = [
    "damage",
    "smoking",
    "dumping",
    "theft"
]

# ==========================================
# 2. XML 구조 분석 함수
# ==========================================

def analyze_xml(xml_path):

    print("\n" + "=" * 70)
    print(f"XML 파일: {xml_path.name}")
    print("=" * 70)

    # XML 파일 읽기
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. 최상위 태그 확인
    print("\n[1] 최상위 태그")
    print("Root:", root.tag)
    print("Attributes:", root.attrib)

    # 2. 전체 태그 종류 확인
    print("\n[2] XML 태그 종류 및 개수")

    tags = Counter(
        element.tag for element in root.iter()
    )

    for tag, count in tags.most_common(25):
        print(f"{tag}: {count}개")

    # 3. XML 내부 구조 미리보기
    print("\n[3] XML 구조 미리보기")

    def show_structure(element, depth=0, max_depth=3):

        if depth > max_depth:
            return

        indent = "  " * depth

        print(
            f"{indent}<{element.tag}> "
            f"{element.attrib}"
        )

        # 같은 구조 반복 출력 방지
        seen_tags = set()

        for child in element:

            if child.tag in seen_tags:
                continue

            seen_tags.add(child.tag)

            show_structure(
                child,
                depth + 1,
                max_depth
            )

    show_structure(root)

    # 4. 시간 및 행동 관련 정보 탐색
    print("\n[4] 행동 및 시간 관련 정보")

    keywords = [
        "action",
        "event",
        "behavior",
        "label",
        "start",
        "end",
        "time",
        "frame",
        "duration",
        "bbox"
    ]

    found = 0

    for element in root.iter():

        tag = element.tag.lower()

        if any(keyword in tag for keyword in keywords):

            print(
                f"태그: {element.tag}",
                f"| 속성: {element.attrib}",
                f"| 값: {(element.text or '').strip()[:100]}"
            )

            found += 1

        if found >= 20:
            break

    if found == 0:
        print("키워드에 해당하는 태그가 없습니다.")

    print("\n분석 완료!")


# ==========================================
# 3. 유형별 XML 하나씩 분석
# ==========================================

for class_name in CLASSES:

    class_dir = LABEL_DIR / class_name

    xml_files = sorted(
        class_dir.rglob("*.xml")
    )

    print(f"\n[{class_name}]")
    print(f"XML 파일 수: {len(xml_files)}개")

    if not xml_files:
        print("XML 파일 없음!")
        continue

    # 유형별 XML 1개 분석
    analyze_xml(xml_files[0])


print("\n" + "=" * 70)
print("전체 XML 구조 분석 완료!")
print("=" * 70)