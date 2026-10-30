import os
from pathlib import Path, PurePosixPath
import zipfile
import unicodedata
import shutil

BASE = Path(os.environ["WATCHOUT_BASE"])


def normalize(text):
    return unicodedata.normalize("NFC", text)


zip_files = [
    p for p in BASE.rglob("*.zip")
    if p.name.startswith(("VS_02.", "VL_02."))
    and "04.구매" in normalize(p.name)
]

if len(zip_files) != 2:
    raise RuntimeError(f"ZIP 파일 2개가 필요합니다. 현재 {len(zip_files)}개 발견")


for zip_path in sorted(zip_files):
    output_dir = zip_path.parent / zip_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n압축 해제 시작: {zip_path.name}", flush=True)

    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            # ZIP 내부의 맨 앞 / 제거
            name = member.filename.lstrip("/\\")
            path = PurePosixPath(name)

            # 상위 폴더 이동 및 비정상 경로 차단
            if not name or ".." in path.parts:
                raise RuntimeError(f"안전하지 않은 경로: {member.filename}")

            target = output_dir.joinpath(*path.parts)

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            # 기존 파일이 정상적으로 있으면 재사용
            if target.is_file() and target.stat().st_size == member.file_size:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)

            with z.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

    mp4_count = len(list(output_dir.rglob("*.mp4")))
    xml_count = len(list(output_dir.rglob("*.xml")))

    print(f"✅ 완료: {zip_path.name}")
    print(f"MP4: {mp4_count}개 / XML: {xml_count}개")


print("\n✅ 정상 구매 데이터 압축 해제 완료!")