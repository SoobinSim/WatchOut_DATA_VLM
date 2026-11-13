
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("WatchOut_dataset")
ROOT = Path("WatchOut_MacStudio")
OUTPUT = ROOT / "data/manifests/split_dev_test.csv"

abnormal = pd.read_csv(BASE / "metadata.csv")
normal = pd.read_csv(
    BASE / "inference_ready_400/normal_metadata_100.csv"
)
review = pd.read_csv(
    BASE / "inference_ready_400/initial_eval_15_review.csv"
)

abnormal = abnormal[["video_id", "class"]].rename(
    columns={"class": "source_class"}
)
normal = normal[["video_id", "source_class"]]

all_data = pd.concat([abnormal, normal], ignore_index=True)

if len(all_data) != 400:
    raise ValueError("전체 영상 수가 400개가 아닙니다.")

if all_data["video_id"].duplicated().any():
    raise ValueError("중복 video_id가 있습니다.")

# 카메라 식별자 이후를 제외해 촬영 그룹 후보 구성
all_data["scene_group"] = all_data["video_id"].str.replace(
    r"_C[A-Z]_RGB_.*$", "", regex=True
)

if all_data["scene_group"].isna().any():
    raise ValueError("촬영 그룹 추출 실패")

# 기존 수동 검수한 15개 영상은 dev에 고정
reviewed_ids = set(review["video_id"])

if not reviewed_ids.issubset(set(all_data["video_id"])):
    raise ValueError("기존 검수 영상 중 전체 데이터에 없는 ID가 있습니다.")

fixed_dev_groups = set(
    all_data.loc[
        all_data["video_id"].isin(reviewed_ids), "scene_group"
    ]
)

# 그룹 단위로 dev/test 분리
rng = np.random.default_rng(42)

group_info = (
    all_data.groupby("scene_group")
    .agg(
        source_class=("source_class", "first"),
        n_videos=("video_id", "size"),
    )
    .reset_index()
)

group_splits = {}

for source_class, class_groups in group_info.groupby("source_class"):
    groups = class_groups["scene_group"].tolist()
    remaining = [g for g in groups if g not in fixed_dev_groups]
    rng.shuffle(remaining)

    target_test = max(1, round(len(groups) * 0.2))
    test_groups = set(remaining[:target_test])

    for group in groups:
        group_splits[group] = (
            "test" if group in test_groups else "dev"
        )

all_data["split"] = all_data["scene_group"].map(group_splits)

# 안전성 검증
assert all(
    all_data.loc[
        all_data["video_id"].isin(reviewed_ids), "split"
    ] == "dev"
)

assert (
    all_data.groupby("scene_group")["split"].nunique().max()
    == 1
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
all_data.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("저장 완료:", OUTPUT)
print("\n전체 분할:")
print(all_data["split"].value_counts())

print("\n유형별 분할:")
print(pd.crosstab(all_data["source_class"], all_data["split"]))

print("\n기존 검수 영상 dev 배정:",
      sum(all_data["video_id"].isin(reviewed_ids)
          & (all_data["split"] == "dev")))

print("촬영 그룹 중복 분할:",
      sum(all_data.groupby("scene_group")["split"].nunique() > 1))
