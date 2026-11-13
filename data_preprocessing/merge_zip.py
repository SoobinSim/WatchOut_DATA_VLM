from pathlib import Path
from collections import defaultdict
import unicodedata
import zipfile
import shutil
import re

# ==========================================
# 1. 데이터 폴더 경로
# ==========================================

folder = Path(
    "/Volumes/SSB 010-3362-8464/"
    "생성형AI/"
    "238-2.실내(편의점,_매장)_사람_이상행동_데이터 8/"
    "01-1.정식개방데이터/"
    "Validation/"
    "02.라벨링데이터"
)

if not folder.exists():
    raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder}")

# ==========================================
# 2. 병합할 행동 유형
# ==========================================

TARGET_CLASSES = ["절도", "파손", "흡연", "유기"]

# ==========================================
# 3. 분할 파일 자동 검색
# ==========================================

groups = defaultdict(list)

for file in folder.iterdir():

    if not file.is_file():
        continue

    # Mac 한글 파일명 정규화
    name = unicodedata.normalize("NFC", file.name)

    # 분할 파일만 검색
    match = re.fullmatch(r"(.+\.zip)\.part(\d+)", name)

    if not match:
        continue

    # 원하는 유형인지 확인
    if not any(f".{cls}.zip" in name for cls in TARGET_CLASSES):
        continue

    zip_name = match.group(1)

    groups[zip_name].append(file)

print("=" * 60)
print("분할 ZIP 파일 검색 결과")
print("=" * 60)

for zip_name, parts in groups.items():
    print(f"{zip_name}: {len(parts)}개")

# ==========================================
# 4. ZIP 병합 함수
# ==========================================

def merge_zip(zip_name, parts):

    print(f"\n{'=' * 60}")
    print(f"병합 시작: {zip_name}")
    print("=" * 60)

    # 분할 번호순 정렬
    parts.sort(
        key=lambda p: int(p.name.rsplit(".part", 1)[-1])
    )

    # 분할 파일 누락 여부 확인
    expected_offset = 0

    for part in parts:

        offset = int(part.name.rsplit(".part", 1)[-1])

        if offset != expected_offset:
            raise ValueError(
                f"분할 파일 누락 또는 크기 불일치: {part.name}"
            )

        expected_offset += part.stat().st_size

    # 원본 파일명을 사용해 ZIP 경로 생성
    output_name = parts[0].name.rsplit(".part", 1)[0]
    output_path = folder / output_name
    temp_path = folder / (output_name + ".merging")

    # 이미 병합된 파일은 건너뛰기
    if output_path.exists():
        if not zipfile.is_zipfile(output_path):
            raise ValueError(f"기존 ZIP 파일이 손상되었을 수 있습니다: {output_path}")

        print("이미 병합된 ZIP이 존재합니다. 건너뜁니다.")
        return

    # 저장 공간 확인
    required_space = sum(p.stat().st_size for p in parts)
    free_space = shutil.disk_usage(folder).free

    if free_space < required_space + 256 * 1024**2:
        raise RuntimeError("외장 SSD의 여유 공간이 부족합니다.")

    if temp_path.exists():
        raise FileExistsError(f"이전 임시 병합 파일을 확인해주세요: {temp_path}")

    # 분할 파일 병합
    try:
        with open(temp_path, "wb") as output:

            for i, part in enumerate(parts, 1):

                print(
                    f"[{i}/{len(parts)}] 병합 중: {part.name}"
                )

                with open(part, "rb") as source:

                    shutil.copyfileobj(
                        source,
                        output,
                        length=8 * 1024 * 1024
                    )

        # ZIP 정상 여부 검사
        with zipfile.ZipFile(temp_path, "r") as zf:

            print("ZIP 무결성 검사 중...")

            bad_file = zf.testzip()

            if bad_file is not None:
                raise ValueError(
                    f"ZIP 내부 파일 손상: {bad_file}"
                )

        temp_path.rename(output_path)

        print(f"\n✅ 병합 완료: {output_path.name}")

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


# ==========================================
# 5. 모든 유형 자동 병합
# ==========================================

if not groups:

    print("분할 파일을 찾지 못했습니다.")

else:

    success = []
    failed = []

    for zip_name, parts in sorted(groups.items()):

        try:
            merge_zip(zip_name, parts)
            success.append(zip_name)

        except Exception as e:
            print(f"❌ 오류: {zip_name}")
            print(e)
            failed.append(zip_name)

    print("\n" + "=" * 60)
    print("최종 병합 결과")
    print("=" * 60)

    print(f"성공: {len(success)}개")
    print(f"실패: {len(failed)}개")

    for name in success:
        print(f"✅ {name}")

    for name in failed:
        print(f"❌ {name}")