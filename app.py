import base64
import json
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import requests
import streamlit as st
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    pipeline,
)

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - optional at runtime until dependency is installed
    st_autorefresh = None


st.set_page_config(page_title="Hong Kong Tunnel Traffic Monitor", page_icon="🚗", layout="wide")

REQUEST_TIMEOUT_SECONDS = 10
IMAGE_CACHE_TTL_SECONDS = 60
DETECTOR_FEED_CACHE_TTL_SECONDS = 60
AUTO_REFRESH_INTERVAL_MS = 120_000
DETECTOR_CONFIDENCE_THRESHOLD = 0.20
DETECTOR_NMS_IOU_THRESHOLD = 0.55
DETECTOR_MODEL_ID = "Gegeishit/hk-traffic-detector-detr"
TREND_WINDOW_SECONDS = 4 * 60 * 60
TREND_CHART_WINDOW_SECONDS = 4 * 60 * 60
TREND_BUCKET_SECONDS = 2 * 60
PERSISTED_HISTORY_PATH = Path(".streamlit/traffic_history.json")
BUSY_OCCUPANCY_THRESHOLD = 0.5
SPIKE_DOMINANCE_THRESHOLD = 0.58
LARGE_VEHICLE_NEAR_CAMERA_RATIO = 0.66
TRAFFIC_DETECTOR_XML_URL = "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml"
TRAFFIC_DETECTOR_HEADERS = {"User-Agent": "Mozilla/5.0"}
SERVICE_CHECK_MODEL_ID = "google/siglip-base-patch16-224"
SERVICE_SCREEN_LABELS = {
    "a yellow no service warning screen": True,
    "a service unavailable placeholder screen": True,
    "a traffic CCTV camera view of a road": False,
}
SERVICE_SCREEN_THRESHOLD = 0.55
LARGE_VEHICLE_LABELS = {"bus", "truck", "tractor", "multiaxle"}
DETECTOR_VEHICLE_LABELS = {
    "bus",
    "car",
    "motorcycle",
    "truck",
}
ANNOTATION_COLORS = {
    "car": (59, 130, 246),
    "bus": (234, 88, 12),
    "truck": (220, 38, 38),
    "motorcycle": (16, 185, 129),
}
ANNOTATION_SHORT_LABELS = {
    "car": "Car",
    "bus": "Bus",
    "truck": "Truck",
    "motorcycle": "Moto",
}

DEFAULT_BASELINE_SPEED_KMH = {
    "Cross Harbour Tunnel": 40.0,
    "Eastern Harbour Crossing": 40.0,
    "Western Harbour Crossing": 40.0,
}
LIVE_BASELINE_SPEED_WEIGHT = 0.35
TUNNEL_LENGTHS_KM = {
    "Cross Harbour Tunnel": 1.86,
    "Eastern Harbour Crossing": 2.2,
    "Western Harbour Crossing": 2.0,
}
TUNNEL_SPEED_LIMITS_KMH = {
    "Cross Harbour Tunnel": 50.0,
    "Eastern Harbour Crossing": 70.0,
    "Western Harbour Crossing": 70.0,
}
BASELINE_DETECTOR_IDS = {
    "Cross Harbour Tunnel": {
        "Hong Kong": ["AID01112"],
        "Kowloon": ["AID01213"],
    },
    "Eastern Harbour Crossing": {
        "Hong Kong": ["AID04222"],
        "Kowloon": ["AID04222"],
    },
    "Western Harbour Crossing": {
        "Hong Kong": ["AID03106"],
        "Kowloon": ["AID03211"],
    },
}

MAX_EXTRA_DELAY_SECONDS = {
    "Cross Harbour Tunnel": 480,
    "Eastern Harbour Crossing": 420,
    "Western Harbour Crossing": 360,
}
RECENT_LOAD_WEIGHT = 0.2
CURRENT_LOAD_WEIGHT = 0.8
LARGE_VEHICLE_COUNT_PENALTY = 0.15
HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")

CAMERA_SOURCE_URLS = {
    "K107F-KL2HK": "https://tdcctv.data.one.gov.hk/K107F.JPG",
    "K107F-HK2KL": "https://tdcctv.data.one.gov.hk/K107F.JPG",
    "K952F-KL2HK": "https://tdcctv.data.one.gov.hk/K952F.JPG",
    "K952F-HK2KL": "https://tdcctv.data.one.gov.hk/K952F.JPG",
    "H702F": "https://tdcctv.data.one.gov.hk/H702F.JPG",
    "K901F": "https://tdcctv.data.one.gov.hk/K901F.JPG",
}
CAMERA_DISPLAY_NAMES = {
    "K107F-KL2HK": "Cross Harbour Tunnel KL → HK [K107F]",
    "K107F-HK2KL": "Cross Harbour Tunnel HK → KL [K107F]",
    "K952F-KL2HK": "Eastern Harbour Crossing KL → HK [K952F]",
    "K952F-HK2KL": "Eastern Harbour Crossing HK → KL [K952F]",
    "H702F": "Western Harbour Crossing HK → KL [H702F]",
    "K901F": "Western Harbour Crossing KL → HK [K901F]",
}
TUNNEL_LOGO_PATHS = {
    "Cross Harbour Tunnel": "images/CHT.avif",
    "Eastern Harbour Crossing": "images/EHC.avif",
    "Western Harbour Crossing": "images/WHC.avif",
}

ROI_CAPACITY_BY_CAMERA = {
    "K107F-KL2HK": 90,
    "K107F-HK2KL": 60,
    "K952F-KL2HK": 150,
    "K952F-HK2KL": 150,
    "H702F": 30,
    "K901F": 30,
}

TUNNELS = {
    "Cross Harbour Tunnel": {
        "Kowloon": ["K107F-KL2HK"],
        "Hong Kong": ["K107F-HK2KL"],
    },
    "Eastern Harbour Crossing": {
        "Kowloon": ["K952F-KL2HK"],
        "Hong Kong": ["K952F-HK2KL"],
    },
    "Western Harbour Crossing": {
        "Hong Kong": ["H702F"],
        "Kowloon": ["K901F"],
    },
}

# Fill each polygon with pixel (x, y) points that cover the drivable road area
# for that specific fixed camera. Empty lists mean the camera is not calibrated yet.
ROAD_ROIS = {
    "K107F-KL2HK": [(97, 223), (94, 199), (107, 168), (124, 143), (134, 135), (174, 132), (183, 120), (147, 121), (161, 103), (180, 83), (183, 71), (169, 62), (180, 55), (199, 60), (213, 74), (213, 92), (193, 118), (164, 150), (148, 178), (149, 199), (163, 220)],
    "K107F-HK2KL": [(5, 219), (53, 218), (173, 94), (179, 79), (181, 61), (173, 54), (163, 54), (146, 69), (173, 69), (169, 76), (136, 78), (134, 85), (103, 104), (67, 124), (141, 121), (136, 126), (60, 132), (44, 146), (18, 170), (3, 175)],
    "K952F-KL2HK": [(153, 21), (141, 64), (140, 64), (137, 91), (139, 132), (151, 145), (154, 203), (143, 220), (136, 199), (142, 185), (144, 154), (136, 145), (134, 197), (141, 221), (79, 222), (76, 165), (83, 155), (94, 147), (96, 119), (100, 103), (114, 65), (131, 42), (139, 20), (143, 26), (113, 87), (107, 131), (113, 144), (117, 102), (131, 60), (142, 25)],
    "K952F-HK2KL": [(166, 218), (165, 152), (163, 104), (160, 62), (154, 19), (163, 17), (178, 46), (192, 81), (192, 87), (168, 89), (184, 146), (185, 94), (193, 93), (200, 221)],
    "H702F": [(113, 158), (86, 163), (104, 172), (106, 187), (116, 187), (142, 177), (235, 147), (239, 151), (301, 128), (314, 117), (313, 102), (286, 91), (250, 85), (211, 85), (239, 91), (252, 99), (245, 110), (215, 125), (164, 143), (127, 149), (144, 156), (135, 160)],
    "K901F": [(7, 90), (206, 60), (260, 61), (260, 99), (183, 222), (3, 219), (5, 97)],
}


def init_session_state() -> None:
    persisted_history = load_persisted_history()
    if "traffic_status_history" not in st.session_state:
        st.session_state.traffic_status_history = persisted_history.get("traffic_status_history", [])
    if "camera_flow_history" not in st.session_state:
        st.session_state.camera_flow_history = persisted_history.get("camera_flow_history", {})


def prune_history_rows(history: list[dict[str, Any]], cutoff: int) -> list[dict[str, Any]]:
    return [row for row in history if int(row.get("timestamp", 0)) >= cutoff]


def prune_camera_history(history_by_camera: dict[str, list[dict[str, Any]]], cutoff: int) -> dict[str, list[dict[str, Any]]]:
    pruned_history = {}
    for camera_url, entries in history_by_camera.items():
        kept_entries = prune_history_rows(entries, cutoff)
        if kept_entries:
            pruned_history[camera_url] = kept_entries
    return pruned_history


def load_persisted_history() -> dict[str, Any]:
    if not PERSISTED_HISTORY_PATH.exists():
        return {}

    try:
        payload = json.loads(PERSISTED_HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    cutoff = bucket_timestamp(datetime.now().timestamp()) - TREND_WINDOW_SECONDS
    traffic_history = payload.get("traffic_status_history", [])
    camera_flow_history = payload.get("camera_flow_history", {})
    return {
        "traffic_status_history": prune_history_rows(traffic_history, cutoff),
        "camera_flow_history": prune_camera_history(camera_flow_history, cutoff),
    }


def persist_history() -> None:
    cutoff = bucket_timestamp(datetime.now().timestamp()) - TREND_WINDOW_SECONDS
    payload = {
        "traffic_status_history": prune_history_rows(
            st.session_state.get("traffic_status_history", []),
            cutoff,
        ),
        "camera_flow_history": prune_camera_history(
            st.session_state.get("camera_flow_history", {}),
            cutoff,
        ),
    }
    try:
        PERSISTED_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        PERSISTED_HISTORY_PATH.write_text(
            json.dumps(payload, ensure_ascii=True),
            encoding="utf-8",
        )
    except OSError:
        pass


@st.cache_resource(show_spinner="Loading service-screen classifier...")
def load_service_screen_classifier() -> tuple[Any | None, str | None]:
    try:
        return (
            pipeline(
                "zero-shot-image-classification",
                model=SERVICE_CHECK_MODEL_ID,
                device=-1,
                model_kwargs={"low_cpu_mem_usage": True},
            ),
            None,
        )
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Loading Conditional DETR detector...")
def load_object_detector() -> tuple[Any | None, str | None]:
    try:
        processor = AutoImageProcessor.from_pretrained(DETECTOR_MODEL_ID, use_fast=False)
        model = AutoModelForObjectDetection.from_pretrained(
            DETECTOR_MODEL_ID,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return (
            {
                "kind": "hf_object_detector",
                "processor": processor,
                "model": model,
            },
            None,
        )
    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl=IMAGE_CACHE_TTL_SECONDS, show_spinner=False)
def download_image_bytes(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": "hk-traffic-monitor/1.0"},
    )
    response.raise_for_status()
    return response.content


@st.cache_data(ttl=DETECTOR_FEED_CACHE_TTL_SECONDS, show_spinner=False)
def download_detector_feed_xml() -> str:
    response = requests.get(
        TRAFFIC_DETECTOR_XML_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers=TRAFFIC_DETECTOR_HEADERS,
    )
    response.raise_for_status()
    return response.text


def fetch_image(url: str) -> Image.Image | None:
    try:
        image_bytes = download_image_bytes(url)
        with Image.open(BytesIO(image_bytes)) as img:
            return img.convert("RGB")
    except (requests.RequestException, UnidentifiedImageError, OSError):
        return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bucket_timestamp(timestamp: float) -> int:
    return int(timestamp // TREND_BUCKET_SECONDS * TREND_BUCKET_SECONDS)


def queue_state_to_band(label: str) -> int | None:
    mapping = {
        "Clear": 0,
        "Busy but moving": 1,
        "Slowing": 2,
        "Congested": 3,
    }
    return mapping.get(label)


def band_to_status_label(status_band: int) -> str:
    return {
        0: "Clear",
        1: "Busy but moving",
        2: "Slowing",
        3: "Congested",
    }[status_band]


def get_camera_flow_history(camera_id: str, current_bucket: int | None = None) -> list[dict[str, Any]]:
    history_by_camera = st.session_state.get("camera_flow_history", {})
    history = list(history_by_camera.get(camera_id, []))
    if current_bucket is None:
        return history
    return [entry for entry in history if entry["timestamp"] < current_bucket]


def is_large_vehicle_spike(
    detections: list[dict[str, Any]],
    image_size: tuple[int, int] | None,
) -> bool:
    if not detections or image_size is None:
        return False

    box_areas = []
    for detection in detections:
        box = detection["box"]
        box_width = max(box["xmax"] - box["xmin"], 1)
        box_height = max(box["ymax"] - box["ymin"], 1)
        box_areas.append((box_width * box_height, detection))

    total_box_area = sum(area for area, _ in box_areas)
    if total_box_area <= 0:
        return False

    dominant_box_area, dominant_detection = max(box_areas, key=lambda item: item[0])
    _, center_y = box_center(dominant_detection["box"])
    vertical_ratio = center_y / max(image_size[1], 1)
    image_area = max(image_size[0] * image_size[1], 1)
    area_share = dominant_box_area / total_box_area
    image_area_ratio = dominant_box_area / image_area
    return (
        dominant_detection["label"] in LARGE_VEHICLE_LABELS
        and vertical_ratio >= LARGE_VEHICLE_NEAR_CAMERA_RATIO
        and area_share >= SPIKE_DOMINANCE_THRESHOLD
        and image_area_ratio >= 0.03
    )


def compute_camera_load(
    camera_capacity: int | None,
    on_road_vehicle_count: int,
    large_vehicle_spike_flag: bool,
    roi_configured: bool,
    detector_available: bool,
) -> float:
    if not roi_configured or not detector_available or not camera_capacity or on_road_vehicle_count == 0:
        return 0.0

    count_score = min(on_road_vehicle_count / camera_capacity, 1.0)
    if large_vehicle_spike_flag and on_road_vehicle_count <= 2:
        count_score -= LARGE_VEHICLE_COUNT_PENALTY
    return round(min(max(count_score, 0.0), 1.0), 3)


def count_persistent_high(history: list[dict[str, Any]], current_score: float, current_bucket: int) -> int:
    if current_score < BUSY_OCCUPANCY_THRESHOLD:
        return 0

    count = 1
    expected_timestamp = current_bucket - TREND_BUCKET_SECONDS
    for entry in reversed(history):
        if entry["timestamp"] != expected_timestamp:
            break
        if entry.get("camera_load", 0.0) < BUSY_OCCUPANCY_THRESHOLD:
            break
        count += 1
        expected_timestamp -= TREND_BUCKET_SECONDS
    return count


def derive_camera_flow_metrics(
    camera_id: str,
    snapshot_time: float,
    on_road_vehicle_count: int,
    camera_load: float,
) -> dict[str, Any]:
    current_bucket = bucket_timestamp(snapshot_time)
    history = get_camera_flow_history(camera_id, current_bucket)
    persistent_high_count = count_persistent_high(history, camera_load, current_bucket)

    if on_road_vehicle_count == 0 or camera_load < BUSY_OCCUPANCY_THRESHOLD:
        camera_flow_state = "Clear"
    elif persistent_high_count >= 3:
        camera_flow_state = "Congested"
    elif persistent_high_count >= 2:
        camera_flow_state = "Slowing"
    else:
        camera_flow_state = "Busy but moving"

    return {
        "persistent_high_count": persistent_high_count,
        "camera_flow_state": camera_flow_state,
    }


def average_detector_speed_kmh(detector_element: ET.Element) -> float | None:
    for lane_element in detector_element.findall("./lanes/lane"):
        lane_name = (lane_element.findtext("lane_id") or "").strip().lower()
        if "slow" not in lane_name:
            continue

        valid_flag = (lane_element.findtext("valid") or "").strip().upper()
        if valid_flag != "Y":
            continue

        speed = parse_float(lane_element.findtext("speed"))
        if speed is None or speed <= 0:
            continue

        return round(speed, 2)

    return None


def load_detector_speed_map() -> tuple[dict[str, float], str | None]:
    try:
        xml_text = download_detector_feed_xml()
        root = ET.fromstring(xml_text)
    except (requests.RequestException, ET.ParseError, OSError) as exc:
        return {}, str(exc)

    periods = root.findall("./periods/period")
    if not periods:
        return {}, "No detector periods found"

    latest_period = periods[-1]
    speed_by_detector: dict[str, float] = {}
    for detector_element in latest_period.findall("./detectors/detector"):
        detector_id = (detector_element.findtext("detector_id") or "").strip()
        if not detector_id:
            continue
        average_speed = average_detector_speed_kmh(detector_element)
        if average_speed is not None:
            speed_by_detector[detector_id] = average_speed

    return speed_by_detector, None


def dynamic_baseline_seconds(tunnel: str, side: str, detector_speed_map: dict[str, float]) -> tuple[int | None, str, float | None]:
    detector_ids = BASELINE_DETECTOR_IDS[tunnel][side]
    tunnel_length_km = TUNNEL_LENGTHS_KM[tunnel]
    speed_limit_kmh = TUNNEL_SPEED_LIMITS_KMH[tunnel]
    default_speed_kmh = DEFAULT_BASELINE_SPEED_KMH[tunnel]
    available_speeds = [
        min(detector_speed_map[detector_id], speed_limit_kmh)
        for detector_id in detector_ids
        if detector_id in detector_speed_map
    ]
    if not available_speeds:
        return None, ",".join(detector_ids), None

    live_speed_kmh = round(sum(available_speeds) / len(available_speeds), 1)
    if live_speed_kmh <= 0:
        return None, ",".join(detector_ids), None

    baseline_speed = round(
        default_speed_kmh + (LIVE_BASELINE_SPEED_WEIGHT * (live_speed_kmh - default_speed_kmh)),
        1,
    )
    baseline_speed = min(max(baseline_speed, 1.0), speed_limit_kmh)
    baseline_seconds = round((tunnel_length_km / baseline_speed) * 3600)
    return baseline_seconds, ",".join(detector_ids), baseline_speed


def run_service_screen_check(img: Image.Image | None, classifier: Any | None) -> list[dict[str, Any]]:
    if img is None:
        return []
    if classifier is None:
        return []

    try:
        result = classifier(
            img,
            candidate_labels=list(SERVICE_SCREEN_LABELS),
        )
    except Exception:
        return []

    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def detect_service_unavailable(img: Image.Image | None, classifier: Any | None) -> tuple[bool, str | None]:
    predictions = run_service_screen_check(img, classifier)
    if not predictions:
        return False, None

    top_prediction = predictions[0]
    top_label = str(top_prediction.get("label", "")).strip()
    top_score = float(top_prediction.get("score", 0.0) or 0.0)
    is_service_screen = (
        SERVICE_SCREEN_LABELS.get(top_label, False)
        and top_score >= SERVICE_SCREEN_THRESHOLD
    )
    return is_service_screen, f"{top_label} ({top_score:.2f})"


def detect_vehicles(img: Image.Image | None, detector: Any | None) -> list[dict[str, Any]]:
    if img is None or detector is None:
        return []

    try:
        if isinstance(detector, dict) and detector.get("kind") == "hf_object_detector":
            processor = detector["processor"]
            model = detector["model"]
            model_device = next(model.parameters()).device
            inputs = processor(images=img, return_tensors="pt")
            inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]], device=model_device)
            processed = processor.post_process_object_detection(
                outputs,
                threshold=DETECTOR_CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes,
            )[0]
            results = []
            id_to_label = model.config.id2label
            for score, label_id, box in zip(
                processed["scores"],
                processed["labels"],
                processed["boxes"],
            ):
                box_values = box.tolist()
                results.append(
                    {
                        "label": id_to_label[int(label_id)].lower().strip(),
                        "score": float(score),
                        "box": {
                            "xmin": float(box_values[0]),
                            "ymin": float(box_values[1]),
                            "xmax": float(box_values[2]),
                            "ymax": float(box_values[3]),
                        },
                    }
                )
        elif isinstance(detector, dict) and detector.get("kind") == "pipeline":
            results = detector["runner"](img, threshold=DETECTOR_CONFIDENCE_THRESHOLD)
        else:
            results = detector(img, threshold=DETECTOR_CONFIDENCE_THRESHOLD)
    except Exception:
        return []

    if not results:
        return []

    detections = []
    for result in results:
        try:
            label = str(result.get("label", "")).lower().strip()
            score = float(result.get("score", 0.0) or 0.0)
            box = result.get("box", {}) or {}
            xmin = int(float(box.get("xmin", 0)))
            ymin = int(float(box.get("ymin", 0)))
            xmax = int(float(box.get("xmax", 0)))
            ymax = int(float(box.get("ymax", 0)))
        except (AttributeError, TypeError, ValueError):
            continue

        if label not in DETECTOR_VEHICLE_LABELS:
            continue

        detections.append(
            {
                "label": label,
                "score": score,
                "box": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                },
            }
        )

    return dedupe_vehicle_detections(detections)

def point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    x, y = point
    inside = False
    point_count = len(polygon)

    for index in range(point_count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % point_count]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1
        )
        if intersects:
            inside = not inside

    return inside


def box_center(box: dict[str, int]) -> tuple[float, float]:
    return ((box["xmin"] + box["xmax"]) / 2.0, (box["ymin"] + box["ymax"]) / 2.0)


def box_iou(box_a: dict[str, int], box_b: dict[str, int]) -> float:
    inter_xmin = max(box_a["xmin"], box_b["xmin"])
    inter_ymin = max(box_a["ymin"], box_b["ymin"])
    inter_xmax = min(box_a["xmax"], box_b["xmax"])
    inter_ymax = min(box_a["ymax"], box_b["ymax"])

    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0

    area_a = max(box_a["xmax"] - box_a["xmin"], 0) * max(box_a["ymax"] - box_a["ymin"], 0)
    area_b = max(box_b["xmax"] - box_b["xmin"], 0) * max(box_b["ymax"] - box_b["ymin"], 0)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def dedupe_vehicle_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(detections) <= 1:
        return detections

    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if any(box_iou(detection["box"], existing["box"]) >= DETECTOR_NMS_IOU_THRESHOLD for existing in kept):
            continue
        kept.append(detection)
    return kept


def roi_for_camera(camera_id: str) -> list[tuple[int, int]]:
    polygon = ROAD_ROIS.get(camera_id, [])
    return polygon if len(polygon) >= 3 else []


def filter_detections_to_road(
    detections: list[dict[str, Any]],
    polygon: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    if not polygon:
        return []

    return [
        detection
        for detection in detections
        if point_in_polygon(box_center(detection["box"]), polygon)
    ]


def annotate_image(
    img: Image.Image | None,
    display_detections: list[dict[str, Any]],
    roi_polygon: list[tuple[int, int]] | None = None,
) -> Image.Image | None:
    if img is None:
        return None

    annotated = img.convert("RGBA")
    if roi_polygon:
        roi_overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        roi_alpha = Image.new("L", annotated.size, 128)
        roi_alpha_draw = ImageDraw.Draw(roi_alpha)
        roi_alpha_draw.polygon(roi_polygon, fill=0)
        roi_overlay.putalpha(roi_alpha)
        annotated = Image.alpha_composite(annotated, roi_overlay)
        roi_outline = ImageDraw.Draw(annotated)
        roi_outline.line(roi_polygon + [roi_polygon[0]], fill=(148, 210, 189, 220), width=3)

    if not display_detections:
        return annotated.convert("RGB")

    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in display_detections:
        color = ANNOTATION_COLORS.get(detection["label"], (220, 38, 38))

        box = detection["box"]
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]
        caption = ANNOTATION_SHORT_LABELS.get(detection["label"], detection["label"].title())

        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=2)
        if hasattr(draw, "textbbox"):
            text_bbox = draw.textbbox((xmin, ymin), caption, font=font)
        else:
            text_width, text_height = draw.textsize(caption, font=font)
            text_bbox = (xmin, ymin, xmin + text_width, ymin + text_height)
        background = (
            text_bbox[0] - 1,
            text_bbox[1] - 1,
            text_bbox[2] + 1,
            text_bbox[3] + 1,
        )
        draw.rectangle(background, fill=color)
        draw.text((xmin, ymin), caption, fill="white", font=font)

    return annotated.convert("RGB")


def icon_for_flow_label(flow_label: str) -> str:
    if flow_label in {"No data", "Uncalibrated", "No road data"}:
        return "❓"
    if flow_label in {"Clear", "Busy but moving"}:
        return "🟢"
    if flow_label == "Slowing":
        return "🟡"
    return "🔴"


def side_direction_label(side: str) -> str:
    if side == "Hong Kong":
        return "HK → KL"
    if side == "Kowloon":
        return "KL → HK"
    return side


def status_color(flow_label: str) -> str:
    if flow_label == "Clear":
        return "#68d391"
    if flow_label == "Busy but moving":
        return "#f6e05e"
    if flow_label == "Slowing":
        return "#f6ad55"
    if flow_label == "Congested":
        return "#fc8181"
    return "#cbd5e0"


def ordered_sides(side_map: dict[str, Any]) -> list[str]:
    side_order = {"Kowloon": 0, "Hong Kong": 1}
    return sorted(side_map.keys(), key=lambda side: (side_order.get(side, 99), side))


def estimate_delay_seconds(
    tunnel: str,
    current_load: float,
    recent_load: float | None,
    total_on_road_vehicle_count: int,
) -> int:
    if total_on_road_vehicle_count == 0 or current_load < BUSY_OCCUPANCY_THRESHOLD:
        return 0

    smoothed_load = (
        (CURRENT_LOAD_WEIGHT * current_load) + (RECENT_LOAD_WEIGHT * recent_load)
        if recent_load is not None
        else current_load
    )
    effective_load = min(max(smoothed_load, 0.0), 1.0)
    normalized_load = min(max((effective_load - BUSY_OCCUPANCY_THRESHOLD) / (1.0 - BUSY_OCCUPANCY_THRESHOLD), 0.0), 1.0)
    delay = (normalized_load ** 1.15) * MAX_EXTRA_DELAY_SECONDS[tunnel]
    return int(round(min(delay, MAX_EXTRA_DELAY_SECONDS[tunnel])))


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "No data"
    minutes, remainder = divmod(max(int(seconds), 0), 60)
    return f"{minutes}m {remainder}s"


def format_vehicle_type_counts(vehicle_counts: dict[str, int]) -> str:
    if not vehicle_counts:
        return "none"
    ordered_counts = sorted(vehicle_counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{label} {count}" for label, count in ordered_counts)


def fixed_baseline_seconds(tunnel: str) -> int:
    default_speed_kmh = DEFAULT_BASELINE_SPEED_KMH[tunnel]
    if default_speed_kmh <= 0:
        return 0
    return round((TUNNEL_LENGTHS_KM[tunnel] / default_speed_kmh) * 3600)


def default_baseline_speed_kmh(tunnel: str) -> float:
    return round(DEFAULT_BASELINE_SPEED_KMH[tunnel], 1)


def baseline_caption(summary: dict[str, Any]) -> str:
    speed_kmh = (
        summary.get("baseline_speed_kmh")
        if summary.get("baseline_source") == "dynamic" and summary.get("baseline_speed_kmh") is not None
        else summary.get("default_baseline_speed_kmh")
    )
    if speed_kmh is None:
        return "Vehicle speed: N/A"
    return f"Vehicle speed: {speed_kmh:.1f}km/h"


def render_tags(tags: list[str]) -> None:
    if not tags:
        return
    chips_html = "".join(
        f"<span class='traffic-tag'>{tag}</span>"
        for tag in tags
    )
    st.markdown(f"<div class='traffic-tag-row'>{chips_html}</div>", unsafe_allow_html=True)


def render_side_badge(direction: str, status_icon: str) -> None:
    safe_direction = (
        direction.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    st.markdown(
        f"""
        <div class="traffic-side-badge">
            <span class="traffic-side-icon">{status_icon}</span>
            <span class="traffic-tag">{safe_direction}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def camera_code_from_source_url(url: str) -> str:
    return url.rsplit("/", maxsplit=1)[-1].replace(".JPG", "")


def camera_modal_id(camera_id: str) -> str:
    return camera_id.replace("/", "_").replace(".", "_")


def image_to_data_uri(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_hover_image(image: Image.Image | None, caption: str, modal_id: str) -> None:
    image_uri = image_to_data_uri(image)
    if image_uri is None:
        st.info("No image available.")
        return

    safe_caption = caption.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""
        <div class="traffic-image-card">
            <input class="traffic-image-toggle" type="checkbox" id="traffic-image-{modal_id}" />
            <label class="traffic-image-frame" for="traffic-image-{modal_id}">
                <img src="{image_uri}" alt="{safe_caption}" />
                <div class="traffic-image-overlay">Click to enlarge</div>
            </label>
            <div class="traffic-image-caption">{safe_caption}</div>
            <label class="traffic-image-modal" for="traffic-image-{modal_id}">
                <div class="traffic-image-modal-backdrop"></div>
                <div class="traffic-image-modal-content">
                    <img src="{image_uri}" alt="{safe_caption}" />
                    <div class="traffic-image-modal-caption">{safe_caption}</div>
                    <div class="traffic-image-modal-close">Close</div>
                </div>
            </label>
        </div>
        """,
        unsafe_allow_html=True,
    )


def service_classifier_status(records_by_tunnel: dict[str, Any]) -> tuple[str, str]:
    unavailable_codes: list[str] = []
    for side_map in records_by_tunnel.values():
        for records in side_map.values():
            for record in records:
                if record.get("service_unavailable_detected"):
                    unavailable_codes.append(camera_code_from_source_url(record["source_url"]))

    unavailable_codes = sorted(set(unavailable_codes))
    if unavailable_codes:
        joined_codes = ", ".join(unavailable_codes)
        verb = "is" if len(unavailable_codes) == 1 else "are"
        return f"Camera {joined_codes} {verb} not available at the moment", "#b7791f"

    return "All cameras are working", "#2f855a"


def summarize_side(
    tunnel: str,
    side: str,
    records: list[dict[str, Any]],
    detector_available: bool,
    detector_speed_map: dict[str, float],
) -> dict[str, Any]:
    available_records = [record for record in records if record["image"] is not None]
    analyzable_records = [record for record in available_records if record["analysis_enabled"]]
    calibrated_records = [record for record in analyzable_records if record["roi_configured"]]
    fallback_baseline = fixed_baseline_seconds(tunnel)
    default_speed_kmh = default_baseline_speed_kmh(tunnel)
    dynamic_baseline, baseline_detector_id, baseline_speed_kmh = dynamic_baseline_seconds(
        tunnel=tunnel,
        side=side,
        detector_speed_map=detector_speed_map,
    )
    baseline = dynamic_baseline if dynamic_baseline is not None else fallback_baseline
    baseline_source = "dynamic" if dynamic_baseline is not None else "fallback"

    if not available_records:
        return {
            "side": side,
            "direction": side_direction_label(side),
            "baseline_seconds": baseline,
            "baseline_source": baseline_source,
            "baseline_detector_id": baseline_detector_id,
            "baseline_speed_kmh": baseline_speed_kmh,
            "default_baseline_speed_kmh": default_speed_kmh,
            "extra_delay_seconds": None,
            "estimated_crossing_seconds": None,
            "status_label": "No data",
            "status_icon": icon_for_flow_label("No data"),
            "flow_label": "No data",
            "camera_count": 0,
            "average_on_road_vehicle_count": 0.0,
            "eta_confidence": "none",
            "provisional_eta": False,
        }

    if analyzable_records and not calibrated_records:
        return {
            "side": side,
            "direction": side_direction_label(side),
            "baseline_seconds": baseline,
            "baseline_source": baseline_source,
            "baseline_detector_id": baseline_detector_id,
            "baseline_speed_kmh": baseline_speed_kmh,
            "default_baseline_speed_kmh": default_speed_kmh,
            "extra_delay_seconds": None,
            "estimated_crossing_seconds": None,
            "status_label": "No calibrated road area",
            "status_icon": icon_for_flow_label("Uncalibrated"),
            "flow_label": "Uncalibrated",
            "camera_count": 0,
            "average_on_road_vehicle_count": 0.0,
            "eta_confidence": "none",
            "provisional_eta": False,
        }

    if available_records and not analyzable_records:
        return {
            "side": side,
            "direction": side_direction_label(side),
            "baseline_seconds": baseline,
            "baseline_source": baseline_source,
            "baseline_detector_id": baseline_detector_id,
            "baseline_speed_kmh": baseline_speed_kmh,
            "default_baseline_speed_kmh": default_speed_kmh,
            "extra_delay_seconds": None,
            "estimated_crossing_seconds": None,
            "status_label": "N/A",
            "status_icon": "❓",
            "flow_label": "N/A",
            "camera_count": 0,
            "average_on_road_vehicle_count": 0.0,
            "eta_confidence": "none",
            "provisional_eta": False,
        }

    if not detector_available:
        return {
            "side": side,
            "direction": side_direction_label(side),
            "baseline_seconds": baseline,
            "baseline_source": baseline_source,
            "baseline_detector_id": baseline_detector_id,
            "baseline_speed_kmh": baseline_speed_kmh,
            "default_baseline_speed_kmh": default_speed_kmh,
            "extra_delay_seconds": None,
            "estimated_crossing_seconds": None,
            "status_label": "Detector unavailable",
            "status_icon": icon_for_flow_label("No road data"),
            "flow_label": "No road data",
            "camera_count": len(calibrated_records),
            "average_on_road_vehicle_count": 0.0,
            "eta_confidence": "none",
            "provisional_eta": False,
        }

    primary_record = calibrated_records[0]
    current_load = float(primary_record["camera_load"])
    recent_load = (
        float(primary_record["recent_camera_load"])
        if primary_record.get("recent_camera_load") is not None
        else None
    )
    total_on_road_vehicle_count = int(primary_record["on_road_vehicle_count"])
    average_on_road_vehicle_count = float(total_on_road_vehicle_count)
    flow_label = str(primary_record["camera_flow_state"])
    eta_confidence = "provisional" if flow_label == "Busy but moving" else "confirmed"
    extra_delay_seconds = estimate_delay_seconds(
        tunnel=tunnel,
        current_load=current_load,
        recent_load=recent_load,
        total_on_road_vehicle_count=total_on_road_vehicle_count,
    )

    return {
        "side": side,
        "direction": side_direction_label(side),
        "baseline_seconds": baseline,
        "baseline_source": baseline_source,
        "baseline_detector_id": baseline_detector_id,
        "baseline_speed_kmh": baseline_speed_kmh,
        "default_baseline_speed_kmh": default_speed_kmh,
        "extra_delay_seconds": extra_delay_seconds,
        "estimated_crossing_seconds": baseline + extra_delay_seconds,
        "status_label": flow_label,
        "status_icon": icon_for_flow_label(flow_label),
        "flow_label": flow_label,
        "camera_count": len(calibrated_records),
        "average_on_road_vehicle_count": round(average_on_road_vehicle_count, 1),
        "eta_confidence": eta_confidence,
        "provisional_eta": flow_label == "Busy but moving",
        "camera_load": round(current_load, 3),
    }


def record_camera_flow_history(snapshot_time: float, records_by_tunnel: dict[str, Any]) -> None:
    bucketed_time = bucket_timestamp(snapshot_time)
    cutoff = bucketed_time - TREND_WINDOW_SECONDS
    history_by_camera = st.session_state.get("camera_flow_history", {}).copy()

    for side_records in records_by_tunnel.values():
        for camera_records in side_records.values():
            for record in camera_records:
                if record["image"] is None or not record["roi_configured"] or not record["analysis_enabled"]:
                    continue

                history = [
                    entry
                    for entry in history_by_camera.get(record["camera_id"], [])
                    if entry["timestamp"] >= cutoff
                ]
                history = [
                    entry
                    for entry in history
                    if entry["timestamp"] != bucketed_time
                ]
                history.append(
                    {
                        "timestamp": bucketed_time,
                        "on_road_vehicle_count": record["on_road_vehicle_count"],
                        "camera_load": record["camera_load"],
                        "camera_flow_state": record["camera_flow_state"],
                    }
                )
                history.sort(key=lambda entry: entry["timestamp"])
                history_by_camera[record["camera_id"]] = history

    st.session_state.camera_flow_history = history_by_camera
    persist_history()


def record_traffic_status_history(snapshot_time: float, tunnel_metrics: dict[str, Any]) -> None:
    bucketed_time = bucket_timestamp(snapshot_time)

    cutoff = bucketed_time - TREND_WINDOW_SECONDS
    history = [
        row for row in st.session_state.traffic_status_history
        if row["timestamp"] >= cutoff
    ]

    for tunnel, metrics in tunnel_metrics.items():
        status_label = metrics.get("trend_status_label")
        status_band = metrics.get("trend_status_band")
        if status_label is None or status_band is None:
            continue

        history = [
            row for row in history
            if not (row["timestamp"] == bucketed_time and row["tunnel"] == tunnel)
        ]

        history.append(
            {
                "timestamp": bucketed_time,
                "tunnel": tunnel,
                "status_band": status_band,
                "status_label": status_label,
            }
        )

    st.session_state.traffic_status_history = history
    persist_history()


def build_trend_dataframe(snapshot_time: float) -> pd.DataFrame:
    history = st.session_state.get("traffic_status_history", [])
    if not history:
        return pd.DataFrame()

    tunnel_order = [
        "Cross Harbour Tunnel",
        "Eastern Harbour Crossing",
        "Western Harbour Crossing",
    ]
    df = pd.DataFrame(history).copy()
    df["timestamp"] = df["timestamp"].apply(bucket_timestamp)
    cutoff = bucket_timestamp(snapshot_time) - TREND_WINDOW_SECONDS
    df = df[df["timestamp"] >= cutoff].copy()

    if "status_band" not in df.columns and "status_label" in df.columns:
        df["status_band"] = df["status_label"].map(queue_state_to_band)

    # Always rebuild labels from the canonical band so legacy values such as
    # "Building" are normalized to the current wording ("Slowing").
    df["status_label"] = df["status_band"].apply(
        lambda band: band_to_status_label(int(band)) if pd.notna(band) else None
    )

    df = (
        df.dropna(subset=["timestamp", "tunnel", "status_band", "status_label"])
        .drop_duplicates(subset=["timestamp", "tunnel"], keep="last")
        .sort_values(["tunnel", "timestamp"])
    )
    window_start = bucket_timestamp(snapshot_time) - TREND_CHART_WINDOW_SECONDS
    window_end = bucket_timestamp(snapshot_time)
    bucket_values = list(range(window_start, window_end + TREND_BUCKET_SECONDS, TREND_BUCKET_SECONDS))

    full_grid = pd.DataFrame(
        [(tunnel, timestamp) for tunnel in tunnel_order for timestamp in bucket_values],
        columns=["tunnel", "timestamp"],
    )
    df = full_grid.merge(
        df[["tunnel", "timestamp", "status_band", "status_label"]],
        on=["tunnel", "timestamp"],
        how="left",
    )
    df["status_band"] = df["status_band"].fillna(-1)
    df["status_label"] = df["status_label"].fillna("No data")
    df["time_label"] = df["timestamp"].apply(
        lambda ts: datetime.fromtimestamp(int(ts), HONG_KONG_TZ).strftime("%H:%M")
    )
    return df


def build_snapshot() -> tuple[float, dict[str, Any], dict[str, Any], dict[str, str]]:
    service_classifier, service_classifier_error = load_service_screen_classifier()
    detector, detector_error = load_object_detector()
    detector_available = detector is not None
    detector_speed_map, detector_feed_error = load_detector_speed_map()

    snapshot_time = datetime.now().timestamp()
    records_by_tunnel: dict[str, Any] = {}
    tunnel_metrics: dict[str, Any] = {}

    for tunnel, sides in TUNNELS.items():
        side_records: dict[str, Any] = {}
        side_summaries: dict[str, Any] = {}
        active_cameras = 0
        tunnel_camera_scores: list[float] = []

        for side, camera_ids in sides.items():
            camera_records = []

            for camera_id in camera_ids:
                source_url = CAMERA_SOURCE_URLS[camera_id]
                image = fetch_image(source_url)
                service_unavailable_detected, service_check_result = detect_service_unavailable(image, service_classifier)
                analysis_enabled = image is not None and not service_unavailable_detected
                all_detections = detect_vehicles(image, detector) if analysis_enabled else []
                polygon = roi_for_camera(camera_id)
                roi_configured = bool(polygon)
                camera_capacity = ROI_CAPACITY_BY_CAMERA.get(camera_id)
                capacity_configured = camera_capacity is not None and camera_capacity > 0
                on_road_detections = filter_detections_to_road(all_detections, polygon)
                on_road_vehicle_types = dict(
                    sorted(Counter(detection["label"] for detection in on_road_detections).items())
                )
                large_vehicle_spike_flag = is_large_vehicle_spike(
                    on_road_detections,
                    image.size if analysis_enabled else None,
                )
                camera_load = (
                    compute_camera_load(
                        camera_capacity=camera_capacity,
                        large_vehicle_spike_flag=large_vehicle_spike_flag,
                        roi_configured=roi_configured and capacity_configured,
                        detector_available=detector_available,
                        on_road_vehicle_count=len(on_road_detections),
                    )
                    if analysis_enabled
                    else 0.0
                )
                history = get_camera_flow_history(camera_id, bucket_timestamp(snapshot_time))
                recent_camera_loads = [
                    float(entry.get("camera_load", 0.0))
                    for entry in history[-2:]
                    if entry.get("camera_load") is not None
                ]
                recent_camera_load = (
                    round(sum(recent_camera_loads) / len(recent_camera_loads), 3)
                    if recent_camera_loads
                    else None
                )
                camera_flow_metrics = derive_camera_flow_metrics(
                    camera_id=camera_id,
                    snapshot_time=snapshot_time,
                    on_road_vehicle_count=len(on_road_detections),
                    camera_load=camera_load,
                ) if analysis_enabled else {
                    "persistent_high_count": 0,
                    "camera_flow_state": "N/A",
                }
                annotated_image = image
                if analysis_enabled and image is not None:
                    annotated_image = annotate_image(
                        image,
                        on_road_detections,
                        polygon,
                    )

                if image is not None:
                    active_cameras += 1

                camera_records.append(
                    {
                        "camera_id": camera_id,
                        "source_url": source_url,
                        "url": source_url,
                        "name": CAMERA_DISPLAY_NAMES.get(camera_id, camera_id),
                        "image": image,
                        "annotated_image": annotated_image if annotated_image is not None else image,
                        "analysis_enabled": analysis_enabled,
                        "service_unavailable_detected": service_unavailable_detected,
                        "service_check_result": service_check_result,
                        "on_road_vehicle_count": len(on_road_detections),
                        "on_road_vehicle_types": on_road_vehicle_types,
                        "camera_load": camera_load,
                        "recent_camera_load": recent_camera_load,
                        "roi_configured": roi_configured and capacity_configured,
                        **camera_flow_metrics,
                    }
                )

            side_records[side] = camera_records
            side_summaries[side] = summarize_side(
                tunnel=tunnel,
                side=side,
                records=camera_records,
                detector_available=detector_available,
                detector_speed_map=detector_speed_map,
            )
            if side_summaries[side].get("camera_load") is not None:
                tunnel_camera_scores.append(side_summaries[side]["camera_load"])

        side_flow_bands = [
            queue_state_to_band(summary["flow_label"])
            for summary in side_summaries.values()
            if queue_state_to_band(summary["flow_label"]) is not None
        ]
        trend_status_band = (
            int(round(sum(side_flow_bands) / len(side_flow_bands)))
            if side_flow_bands
            else None
        )
        trend_status_label = band_to_status_label(trend_status_band) if trend_status_band is not None else None
        tunnel_score = (
            sum(tunnel_camera_scores) / len(tunnel_camera_scores)
            if tunnel_camera_scores
            else None
        )
        records_by_tunnel[tunnel] = side_records
        tunnel_metrics[tunnel] = {
            "status_label": trend_status_label if trend_status_label is not None else "No calibrated road data",
            "status_icon": icon_for_flow_label(trend_status_label or "No road data"),
            "active_cameras": active_cameras,
            "trend_status_label": trend_status_label,
            "trend_status_band": trend_status_band,
            "tunnel_load": round(tunnel_score, 3) if tunnel_score is not None else None,
            "sides": side_summaries,
        }

    return snapshot_time, records_by_tunnel, tunnel_metrics, {
        "service_classifier": service_classifier_error or "",
        "detector": detector_error or "",
        "detector_feed": detector_feed_error or "",
    }


def render_top_bar(snapshot_time: float, model_errors: dict[str, str], records_by_tunnel: dict[str, Any]) -> None:
    with st.container(border=True):
        status_column, action_column = st.columns([4.8, 1.2], vertical_alignment="center")
        with status_column:
            service_message, service_color = service_classifier_status(records_by_tunnel)
            warnings = []
            if model_errors.get("service_classifier"):
                warnings.append("Service-screen check unavailable")
            if model_errors.get("detector"):
                warnings.append("Object detector unavailable")
            if model_errors.get("detector_feed"):
                warnings.append("Speed baseline feed unavailable")
            status_text = " | ".join(warnings) if warnings else service_message
            status_line_color = "#f6ad55" if warnings else service_color
            st.markdown(
                f"<div style='color:{status_line_color};font-size:1.58rem;font-weight:700;display:flex;align-items:center;min-height:3rem;line-height:1.2;margin:0;'>"
                f"Status: {status_text}</div>",
                unsafe_allow_html=True,
            )
            error_details = {
                "Service-screen check": model_errors.get("service_classifier", "").strip(),
                "Object detector": model_errors.get("detector", "").strip(),
                "Speed baseline feed": model_errors.get("detector_feed", "").strip(),
            }
            visible_errors = {
                label: detail
                for label, detail in error_details.items()
                if detail
            }
            if visible_errors:
                with st.expander("Diagnostics", expanded=False):
                    for label, detail in visible_errors.items():
                        st.code(f"{label}: {detail}", language="text")
        with action_column:
            if st.button("Refresh data", use_container_width=True, type="secondary"):
                download_image_bytes.clear()
                download_detector_feed_xml.clear()
                st.rerun()


def render_trend_chart(snapshot_time: float) -> None:
    st.subheader("Flow Timeline (Last 4 Hours)")
    st.caption("Live tunnel-status timeline in 2-minute blocks: Clear, Busy but moving, Slowing, Congested.")

    trend_df = build_trend_dataframe(snapshot_time)
    if trend_df.empty:
        st.info("Trend history will appear after a few refresh cycles.")
        return

    tunnel_order = [
        "Cross Harbour Tunnel",
        "Eastern Harbour Crossing",
        "Western Harbour Crossing",
    ]
    time_order = (
        trend_df.sort_values("timestamp")["time_label"]
        .drop_duplicates()
        .tolist()
    )
    tick_labels = time_order[::5] if len(time_order) > 5 else time_order

    chart = (
        alt.Chart(trend_df)
        .mark_rect(
            cornerRadius=2,
        )
        .encode(
            x=alt.X(
                "time_label:O",
                title="Time",
                axis=alt.Axis(
                    values=tick_labels,
                    labelAngle=0,
                ),
                sort=time_order,
            ),
            y=alt.Y(
                "tunnel:N",
                title=None,
                sort=tunnel_order,
                axis=alt.Axis(labelLimit=240),
            ),
            color=alt.Color(
                "status_label:N",
                title="Status",
                scale=alt.Scale(
                    domain=["No data", "Clear", "Busy but moving", "Slowing", "Congested"],
                    range=["#8c99a5", "#2f855a", "#d4a017", "#dd6b20", "#c53030"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("tunnel:N", title="Tunnel"),
                alt.Tooltip("time_label:N", title="Time"),
                alt.Tooltip("status_label:N", title="Status"),
            ],
        )
        .properties(height=210)
    )
    st.altair_chart(chart, use_container_width=True)


def render_dashboard(snapshot_time: float, records_by_tunnel: dict[str, Any], tunnel_metrics: dict[str, Any]) -> None:
    st.markdown(
        """
        <style>
        .traffic-tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.2rem 0 0.4rem 0;
        }
        .traffic-tag {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: #c5cfd7;
            border: 1px solid #adb9c3;
            color: #23313d;
            font-size: 0.78rem;
            line-height: 1.2;
            white-space: nowrap;
        }
        .traffic-image-card {
            display: block;
            margin: 0.25rem 0 0.65rem 0;
        }
        .traffic-image-toggle {
            position: absolute;
            opacity: 0;
            pointer-events: none;
        }
        .traffic-image-frame {
            position: relative;
            display: block;
            overflow: hidden;
            border-radius: 14px;
            border: 1px solid #d6e1e8;
            background: transparent;
            cursor: zoom-in;
        }
        .traffic-image-frame img {
            display: block;
            width: 100%;
            height: auto;
            transition: transform 180ms ease, filter 180ms ease;
        }
        .traffic-image-overlay {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(20, 33, 42, 0.18);
            color: #ffffff;
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            opacity: 0;
            transition: opacity 180ms ease, background 180ms ease;
        }
        .traffic-image-card:hover .traffic-image-frame img {
            transform: scale(1.02);
            filter: saturate(1.03);
        }
        .traffic-image-card:hover .traffic-image-overlay {
            opacity: 1;
            background: rgba(20, 33, 42, 0.3);
        }
        .traffic-image-caption {
            margin-top: 0.35rem;
            font-size: 0.82rem;
            color: #4a5560;
            line-height: 1.35;
        }
        .traffic-image-modal {
            position: fixed;
            inset: 0;
            z-index: 9999;
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            transition: opacity 220ms ease, visibility 220ms ease;
        }
        .traffic-image-toggle:checked + .traffic-image-frame + .traffic-image-caption + .traffic-image-modal {
            opacity: 1;
            visibility: visible;
            pointer-events: auto;
            cursor: zoom-out;
        }
        .traffic-image-modal-backdrop {
            position: absolute;
            inset: 0;
            background: rgba(10, 16, 24, 0.86);
            backdrop-filter: blur(4px);
            transition: background 220ms ease;
        }
        .traffic-image-modal-content {
            position: relative;
            z-index: 1;
            width: min(98vw, 1600px);
            margin: 2vh auto 0 auto;
            padding: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
            transform: scale(0.94) translateY(10px);
            transform-origin: center top;
            transition: transform 220ms ease;
        }
        .traffic-image-toggle:checked + .traffic-image-frame + .traffic-image-caption + .traffic-image-modal .traffic-image-modal-content {
            transform: scale(1) translateY(0);
        }
        .traffic-image-modal-content img {
            display: block;
            width: min(98vw, 1600px);
            max-width: 100%;
            max-height: 88vh;
            height: auto;
            margin: 0 auto;
            border-radius: 16px;
            box-shadow: 0 22px 80px rgba(0, 0, 0, 0.45);
            object-fit: contain;
        }
        .traffic-image-modal-caption {
            margin-top: 0.7rem;
            text-align: center;
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.88);
            line-height: 1.35;
        }
        .traffic-image-modal-close {
            margin-top: 0.45rem;
            text-align: center;
            font-size: 0.82rem;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.72);
        }
        .tunnel-title-text {
            display: flex;
            align-items: center;
            min-height: 3.1rem;
            margin: 0;
            font-size: 1.62rem;
            font-weight: 700;
            line-height: 1.15;
            color: #ffffff;
        }
        .traffic-side-badge {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0.05rem 0 0.55rem 0;
        }
        .traffic-side-icon {
            font-size: 1rem;
            line-height: 1;
        }
        .tunnel-header-divider {
            height: 1px;
            margin: 0.2rem 0 0.85rem 0;
            background: linear-gradient(90deg, rgba(177, 188, 198, 0.9), rgba(177, 188, 198, 0.35));
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: linear-gradient(180deg, #dbe7f2 0%, #c7d7e7 100%);
            border: 1px solid #a7bfd6;
            color: #1f3448;
            font-weight: 600;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            border-color: #8eaac3;
            color: #15283a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='margin-bottom:0.2rem;'>🚗 Hong Kong Tunnel Traffic Monitor</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Snapshot captured at {datetime.fromtimestamp(snapshot_time, HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S')} HKT "
        f"({'Auto refresh every 2 min' if st_autorefresh is not None else 'Auto refresh unavailable'})"
    )
    render_top_bar(snapshot_time, st.session_state.get("model_errors", {}), records_by_tunnel)
    render_trend_chart(snapshot_time)

    st.caption(
        "Est. crossing time uses official detector speed as the baseline when available; otherwise the fixed tunnel baseline."
    )

    for tunnel, side_map in TUNNELS.items():
        tunnel_status = tunnel_metrics[tunnel]
        side_order = ordered_sides(side_map)
        with st.container(border=True):
            logo_column, title_column = st.columns([0.08, 0.92], vertical_alignment="center")
            with logo_column:
                logo_path = TUNNEL_LOGO_PATHS.get(tunnel)
                if logo_path:
                    st.image(logo_path, width=58)
            with title_column:
                st.markdown(
                    f"<div class='tunnel-title-text'>{tunnel}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div class='tunnel-header-divider'></div>", unsafe_allow_html=True)
            side_columns = st.columns(len(side_order), gap="large")

            for index, side in enumerate(side_order):
                with side_columns[index]:
                    summary = tunnel_metrics[tunnel]["sides"][side]
                    side_records = records_by_tunnel[tunnel][side]
                    has_analyzable_side_record = any(
                        record["image"] is not None
                        and record["roi_configured"]
                        and record["analysis_enabled"]
                        for record in side_records
                    )
                    primary_record = side_records[0] if side_records else None
                    info_column, image_column = st.columns([0.9, 1.1], vertical_alignment="top")

                    with info_column:
                        render_side_badge(summary["direction"], summary["status_icon"])
                        if summary["estimated_crossing_seconds"] is None:
                            st.metric("Est. crossing time", "N/A")
                            st.caption(baseline_caption(summary))
                        else:
                            st.metric(
                                "Est. crossing time",
                                format_duration(summary["estimated_crossing_seconds"]),
                            )
                            st.caption(baseline_caption(summary))

                        if primary_record is None or primary_record["image"] is None:
                            st.write("**Side flow:** N/A  \n**Vehicles in ROI:** N/A")
                        elif not primary_record["roi_configured"]:
                            st.info("ROI not configured; excluded from road-flow calculation.")
                        elif not primary_record["analysis_enabled"]:
                            st.write("Feed: Service unavailable | Traffic analysis: N/A")
                            if primary_record["service_check_result"]:
                                st.caption(f"Feed check: {primary_record['service_check_result']}")
                        else:
                            st.markdown(
                                f"**Side flow:** {primary_record['camera_flow_state']}  \n"
                                f"**Vehicles in ROI:** {format_vehicle_type_counts(primary_record['on_road_vehicle_types'])}"
                            )

                    with image_column:
                        if primary_record is None or primary_record["image"] is None:
                            missing_name = primary_record["name"] if primary_record else summary["direction"]
                            st.warning(f"{missing_name}: camera unavailable")
                        else:
                            render_hover_image(
                                primary_record["annotated_image"],
                                primary_record["name"],
                                camera_modal_id(primary_record["camera_id"]),
                            )


def main() -> None:
    init_session_state()

    if st_autorefresh is not None:
        st_autorefresh(interval=AUTO_REFRESH_INTERVAL_MS, key="live_dashboard_refresh")

    with st.spinner("Refreshing live traffic snapshot..."):
        snapshot_time, records_by_tunnel, tunnel_metrics, model_errors = build_snapshot()
    st.session_state["model_errors"] = model_errors
    record_camera_flow_history(snapshot_time, records_by_tunnel)
    record_traffic_status_history(snapshot_time, tunnel_metrics)
    render_dashboard(snapshot_time, records_by_tunnel, tunnel_metrics)


if __name__ == "__main__":
    main()
