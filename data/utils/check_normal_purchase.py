import os
from pathlib import Path
import unicodedata

BASE = Path(os.environ["WATCHOUT_BASE"])

def normalize(text):
    return unicodedata.normalize("NFC", text)

videos = []
labels = []

for folder in BASE.iterdir():
    if not folder.is_dir() or "238-1." not in folder.name:
        continue

    print(f"\n폴더: {folder.name}")

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".mp4":
            videos.append(path)
        elif path.suffix.lower() == ".xml":
            labels.append(path)

print("\n========== 정상 구매 데이터 ==========")
print("영상 MP4:", len(videos))
print("라벨 XML:", len(labels))

video_ids = {normalize(p.stem) for p in videos}
label_ids = {normalize(p.stem) for p in labels}

print("매칭된 영상:", len(video_ids & label_ids))
print("XML 없는 영상:", len(video_ids - label_ids))
print("영상 없는 XML:", len(label_ids - video_ids))

if videos:
    print("\n영상 예시:", videos[0].name)
if labels:
    print("XML 예시:", labels[0].name)