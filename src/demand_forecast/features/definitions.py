"""特征名称和模型特征集的唯一来源。"""

WINDOWS = (7, 14, 30, 60, 90)
FEATURE_VERSION = "v1"

CURRENT_MEAN_FEATURES = tuple(f"current_mean_{window}" for window in WINDOWS)
YOY_MEAN_FEATURES = tuple(f"yoy_mean_{window}" for window in WINDOWS)
OCCURRENCE_RATE_FEATURES = tuple(f"occurrence_rate_{window}" for window in WINDOWS)
CALENDAR_FEATURES = ("dow_sin", "dow_cos")

FEATURE_SETS = {
    "five_period": CURRENT_MEAN_FEATURES,
    "direct10": CURRENT_MEAN_FEATURES + YOY_MEAN_FEATURES,
    "hurdle": CURRENT_MEAN_FEATURES + OCCURRENCE_RATE_FEATURES + CALENDAR_FEATURES,
    "all": CURRENT_MEAN_FEATURES + YOY_MEAN_FEATURES + OCCURRENCE_RATE_FEATURES + CALENDAR_FEATURES,
}


def feature_names_for(feature_set: str) -> tuple[str, ...]:
    """返回固定顺序的特征名，未知集合立即报错，避免训练/推理列错位。"""
    try:
        return FEATURE_SETS[feature_set]
    except KeyError as error:
        raise ValueError(f"未知特征集: {feature_set}") from error
