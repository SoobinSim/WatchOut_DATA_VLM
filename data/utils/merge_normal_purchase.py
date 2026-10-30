import os
from pathlib import Path
import re

BASE = Path(os.environ["WATCHOUT_BASE"])

def merge_parts(parts):
    parts = sorted(
        parts,
        key=lambda p: int(re.search(r"\.part(\d+)$", p.name).group(1))
    )

    first = parts[0]
    output = first.with_name(first.name.split(".part")[0])

    if output.exists():
        print(f"이미 존재하여 건너뜀: {output.name}")
        return

    print(f"\n병합 시작: {output.name}")
    print(f"분할 파일: {len(parts)}개")

    with output.open("wb") as out:
        for i, part in enumerate(parts, 1):
            print(f"  [{i}/{len(parts)}] {part.name}", flush=True)

            with part.open("rb") as src:
                while True:
                    chunk = src.read(8 * 1024 * 1024)

                    if not chunk:
                        break

                    out.write(chunk)

    print(f"✅ 병합 완료: {output}")


for folder in BASE.iterdir():
    if not folder.is_dir() or "238-1." not in folder.name:
        continue

    groups = {}

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        match = re.match(r"(.+\.zip)\.part(\d+)$", path.name)

        if match:
            zip_name = match.group(1)
            groups.setdefault(
                (path.parent, zip_name), []
            ).append(path)

    for (_, _), parts in groups.items():
        merge_parts(parts)