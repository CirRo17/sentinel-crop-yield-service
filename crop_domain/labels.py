"""作物类别编码、标签和历史类别映射。"""

import numpy as np


AGRIFIELDNET_LABELS = {
    0: "Background",
    1: "Wheat",
    2: "Mustard",
    3: "Lentil",
    4: "Green pea",
    5: "Sugarcane",
    6: "Garlic",
    7: "Maize",
    8: "Gram",
    9: "Coriander",
    10: "Potato",
    11: "Bersem",
    12: "Rice",
}

TARGET_LABELS = {
    0: "Others",
    1: "Rice",
    2: "Wheat",
    3: "Maize",
    4: "Rapeseed",
}

VALID_OUTPUT_CLASSES = frozenset(TARGET_LABELS)


def normalize_output_classes(classes):
    """把历史或技术标签归一化到对外公开的 0-4 类别集合。"""

    normalized = np.asarray(classes).copy()
    normalized[~np.isin(normalized, list(VALID_OUTPUT_CLASSES))] = 0
    return normalized.astype("uint8")


AGRIFIELDNET_TO_TARGET = {
    0: 0,
    1: 2,
    2: 4,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 3,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 1,
}
