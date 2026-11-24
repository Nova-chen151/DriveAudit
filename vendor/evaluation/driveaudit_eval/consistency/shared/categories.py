from __future__ import annotations


DYNAMIC_CLASSES = {"car", "truck", "construction_vehicle", "bus", "trailer",
                   "motorcycle", "bicycle", "pedestrian"}
STATIC_CLASSES = {"barrier", "traffic_cone"}
SEMANTIC_CATEGORY = {
    "car": "vehicle", "truck": "vehicle", "construction_vehicle": "vehicle",
    "bus": "vehicle", "trailer": "vehicle", "pedestrian": "pedestrian",
    "motorcycle": "bicycle", "bicycle": "bicycle",
}


def semantic_category(category: str) -> str:
    return SEMANTIC_CATEGORY.get(category, category)
