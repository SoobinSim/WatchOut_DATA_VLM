from pathlib import Path
from collections import Counter
import unicodedata

BASE = Path("/Volumes/SSB 010-3362-8464/생성형AI")

VIDEO_KEYWORDS = {
    "theft": "절도",
    "damage": "파손",
    "smoking": "흡연",
    "dumping": "유기",
}

print("외장 SSD 연결:", BASE.exists())

if not BASE.exists():
    raise SystemExit("외장 SSD 경로를 확인해 주세요.")

for class_name, keyword in VIDEO_KEYWORDS.items():
    print(f"\n[{class_name} / {keyword}]")

    

    folders = [
        p for p in BASE.iterdir()
        if p.is_dir()
        and keyword in unicodedata.normalize("NFC", p.name)
        and (p.name.startswith("VS_") or p.name.startswith("VL_"))
    ]

    if not folders:
        print("원본 폴더를 찾지 못했습니다.")
        continue

    for folder in sorted(folders):
        extension = ".mp4" if folder.name.startswith("VS_") else ".xml"

        files = list(folder.rglob(f"*{extension}"))

        print(f"폴더: {folder}")
        print(f"파일 수: {len(files)}")
        print(f"파일 예시: {files[0].name if files else '없음'}")