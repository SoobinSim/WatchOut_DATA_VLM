import os
from pathlib import Path
from collections import Counter

BASE = Path(os.environ["WATCHOUT_BASE"])

for folder in BASE.iterdir():
    if not folder.is_dir() or "238-1." not in folder.name:
        continue

    print(f"\n{'=' * 50}")
    print(f"폴더: {folder.name}")

    files = [p for p in folder.rglob("*") if p.is_file()]

    print(f"전체 파일 수: {len(files)}")
    print("확장자별 개수:", dict(Counter(p.suffix.lower() for p in files)))

    print("\n파일 목록 (최대 15개):")
    for path in files[:15]:
        print(" -", path.relative_to(folder))