import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, UnidentifiedImageError
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

STREAMLIT_FRAGMENT = getattr(st, "fragment", None)

st.set_page_config(page_title="Hong Kong Tunnel Traffic Monitor", page_icon="🚗", layout="wide")

# Keep requests short so the app does not hang on slow feeds.
REQUEST_TIMEOUT_SECONDS = 10
# Cache live feeds briefly to reduce repeat downloads during reruns.
IMAGE_CACHE_TTL_SECONDS = 60
DETECTOR_FEED_CACHE_TTL_SECONDS = 60
# Refresh the dashboard every 5 minutes.
AUTO_REFRESH_INTERVAL_MS = 300_000
# Detector threshold for keeping a predicted vehicle box.
DETECTOR_CONFIDENCE_THRESHOLD = 0.60
DETECTOR_MODEL_ID = "Gegeishit/yolos-small-hk-traffic-cctv-finetuned"
# The timeline shows the last 4 hours in 5-minute buckets.
TREND_WINDOW_SECONDS = 4 * 60 * 60
TREND_CHART_WINDOW_SECONDS = 4 * 60 * 60
TREND_BUCKET_SECONDS = 5 * 60
PERSISTED_HISTORY_PATH = Path(".streamlit/traffic_history.json")
STYLESHEET_PATH = Path("styles.css")
# Expand each box slightly so occupancy reflects car spacing, not just raw box size.
OCCUPANCY_BOX_PADDING_RATIO = 0.12
OCCUPANCY_BOX_PADDING_MIN_PX = 4
# Count a detection as "in the road" only if at least 40% of its box overlaps the ROI.
ROI_MIN_BOX_OVERLAP_RATIO = 0.40
# These settings soften the impact of large near-camera WHC buses and trucks.
WHC_PERSPECTIVE_CAMERA_IDS = {"H702F", "K901F"}
WHC_FOREGROUND_LARGE_VEHICLE_LABELS = {"bus", "truck"}
WHC_FOREGROUND_MAX_BIG_VEHICLES = 2
WHC_BIG_BOX_MIN_ROI_SHARE = 0.08
WHC_FOREGROUND_CORRECTION_MIN_ROI_COUNT = 30
TRAFFIC_SEGMENT_SPEED_XML_URL = "https://resource.data.one.gov.hk/td/traffic-detectors/irnAvgSpeed-all.xml"
TRAFFIC_SEGMENT_SPEED_HEADERS = {"User-Agent": "hk-traffic-monitor/1.0"}
SERVICE_CHECK_MODEL_ID = "openai/clip-vit-base-patch32"
# The zero-shot classifier decides whether an image is a real road view or a placeholder screen.
SERVICE_SCREEN_LABELS = {
    "a yellow no service warning screen": True,
    "a service unavailable placeholder screen": True,
    "a traffic CCTV camera view of a road": False,
}
SERVICE_SCREEN_THRESHOLD = 0.55
DETECTOR_VEHICLE_LABELS = {
    "bus",
    "car",
    "motorcycle",
    "truck",
}
ANNOTATION_COLORS = {
    "car": (0, 240, 255),
    "bus": (255, 214, 10),
    "truck": (255, 90, 90),
    "motorcycle": (70, 255, 140),
}
ANNOTATION_BOX_ALPHA = 235

# Use this fallback speed only when the live XML speed is unavailable.
DEFAULT_BASELINE_SPEED_KMH = {
    "Cross Harbour Tunnel": 60.0,
    "Eastern Harbour Crossing": 60.0,
    "Western Harbour Crossing": 60.0,
}
# Road occupancy is converted into user-facing traffic states using these cutoffs.
FLOW_STATE_LOAD_THRESHOLDS = {
    "busy_but_moving": 0.45,
    "slowing": 0.70,
    "congested": 0.88,
}
# Camera-derived traffic state applies only a mild correction to the official speed baseline.
FLOW_SPEED_FACTORS = {
    "Clear": 1.0,
    "Busy but moving": 0.95,
    "Slowing": 0.85,
    "Congested": 0.70,
}
# Fixed tunnel lengths are used to convert speed into estimated crossing time.
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
# Each tunnel side maps to one official road-network segment in the XML speed feed.
BASELINE_SEGMENT_IDS = {
    "Cross Harbour Tunnel": {
        "Hong Kong": "2905",
        "Kowloon": "105057",
    },
    "Eastern Harbour Crossing": {
        "Hong Kong": "101734",
        "Kowloon": "101735",
    },
    "Western Harbour Crossing": {
        "Hong Kong": "106784",
        "Kowloon": "106785",
    },
}
HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")

# Camera feed URLs are grouped by logical tunnel side.
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

# This structure controls how the UI groups cameras under tunnels and directions.
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
    "K107F-KL2HK": [(154, 222), (148, 196), (151, 170), (168, 148), (176, 128), (190, 103), (204, 74), (187, 61), (168, 63), (184, 73), (177, 84), (161, 102), (141, 118), (176, 117), (165, 132), (127, 134), (113, 155), (100, 173), (94, 192), (94, 223)],
    "K107F-HK2KL": [(4, 202), (86, 142), (119, 138), (134, 120), (106, 120), (158, 82), (183, 80), (48, 221), (2, 220)],
    "K952F-KL2HK": [(153, 21), (141, 64), (140, 64), (137, 91), (139, 132), (151, 145), (154, 203), (143, 220), (136, 199), (142, 185), (144, 154), (136, 145), (134, 197), (141, 221), (79, 222), (76, 165), (83, 155), (94, 147), (96, 119), (100, 103), (114, 65), (131, 42), (139, 20), (143, 26), (113, 87), (107, 131), (113, 144), (117, 102), (131, 60), (142, 25)],
    "K952F-HK2KL": [(166, 218), (165, 152), (163, 104), (160, 62), (154, 19), (163, 17), (178, 46), (192, 81), (192, 87), (168, 89), (184, 146), (185, 94), (193, 93), (200, 221)],
    "H702F": [(113, 158), (86, 163), (104, 172), (106, 187), (116, 187), (142, 177), (235, 147), (239, 151), (301, 128), (314, 117), (313, 102), (286, 91), (250, 85), (211, 85), (239, 91), (252, 99), (245, 110), (215, 125), (164, 143), (127, 149), (144, 156), (135, 160)],
    "K901F": [(7, 90), (206, 60), (260, 61), (260, 99), (183, 222), (3, 219), (5, 97)],
}


def init_session_state() -> None:
    # Restore saved chart history only once per browser session.
    persisted_history = load_persisted_history()
    if "traffic_status_history" not in st.session_state:
        st.session_state.traffic_status_history = persisted_history.get("traffic_status_history", [])


def prune_history_rows(history: list[dict[str, Any]], cutoff: int) -> list[dict[str, Any]]:
    # The timeline is rolling, so old rows outside the window are dropped.
    return [row for row in history if int(row.get("timestamp", 0)) >= cutoff]


def load_persisted_history() -> dict[str, Any]:
    # Load saved tunnel-status history from disk so the timeline survives app reruns.
    if not PERSISTED_HISTORY_PATH.exists():
        return {}

    try:
        payload = json.loads(PERSISTED_HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    cutoff = bucket_timestamp(datetime.now().timestamp()) - TREND_WINDOW_SECONDS
    traffic_history = payload.get("traffic_status_history", [])
    return {
        "traffic_status_history": prune_history_rows(traffic_history, cutoff),
    }


def persist_history() -> None:
    # Persist only the recent history window that is still relevant to the chart.
    cutoff = bucket_timestamp(datetime.now().timestamp()) - TREND_WINDOW_SECONDS
    payload = {
        "traffic_status_history": prune_history_rows(
            st.session_state.get("traffic_status_history", []),
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


def load_stylesheet() -> str:
    # Keep CSS outside Python so UI styling is easier to maintain.
    try:
        return STYLESHEET_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


@st.cache_resource(show_spinner="Loading service-screen classifier...")
def load_service_screen_classifier() -> tuple[Any | None, str | None]:
    # Load the classifier that screens out "service unavailable" camera frames.
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
    # Load the Hugging Face object detector once and reuse it across reruns.
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
    # Download one raw CCTV snapshot for a camera.
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": "hk-traffic-monitor/1.0"},
    )
    response.raise_for_status()
    return response.content


@st.cache_data(ttl=DETECTOR_FEED_CACHE_TTL_SECONDS, show_spinner=False)
def download_segment_speed_xml() -> str:
    # Download the official XML feed that contains live average traffic speeds.
    response = requests.get(
        TRAFFIC_SEGMENT_SPEED_XML_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers=TRAFFIC_SEGMENT_SPEED_HEADERS,
    )
    response.raise_for_status()
    return response.text


def fetch_image(url: str) -> Image.Image | None:
    # Convert downloaded bytes into a PIL image for classification, detection, and display.
    try:
        image_bytes = download_image_bytes(url)
        with Image.open(BytesIO(image_bytes)) as img:
            return img.convert("RGB")
    except (requests.RequestException, UnidentifiedImageError, OSError):
        return None


def parse_float(value: str | None) -> float | None:
    # XML parsing is safer when invalid values become None instead of throwing errors.
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bucket_timestamp(timestamp: float) -> int:
    # The chart stores each snapshot in a fixed 5-minute bucket.
    return int(timestamp // TREND_BUCKET_SECONDS * TREND_BUCKET_SECONDS)


def queue_state_to_band(label: str) -> int | None:
    # Altair uses numeric bands, so traffic labels are converted to simple integers.
    mapping = {
        "Clear": 0,
        "Busy but moving": 1,
        "Slowing": 2,
        "Congested": 3,
    }
    return mapping.get(label)


def band_to_status_label(status_band: int) -> str:
    # This reverses the numeric chart band back to a human-readable label.
    return {
        0: "Clear",
        1: "Busy but moving",
        2: "Slowing",
        3: "Congested",
    }[status_band]


def build_roi_mask(
    image_size: tuple[int, int],
    polygon: list[tuple[int, int]],
) -> tuple[Image.Image, int]:
    # The ROI mask is the base geometry used for overlap tests and occupancy math.
    roi_mask = Image.new("L", image_size, 0)
    roi_draw = ImageDraw.Draw(roi_mask)
    roi_draw.polygon(polygon, fill=255)
    roi_area = sum(roi_mask.histogram()[1:])
    return roi_mask, roi_area


def box_overlap_ratio_in_roi(
    box: dict[str, float],
    roi_mask: Image.Image,
    image_size: tuple[int, int],
) -> float:
    # This measures how much of one detection box lies inside the road ROI.
    clipped_box = clip_box_to_image(box, image_size)
    if clipped_box is None:
        return 0.0
    box_area = max((clipped_box["xmax"] - clipped_box["xmin"]) * (clipped_box["ymax"] - clipped_box["ymin"]), 1)
    overlap_crop = roi_mask.crop(
        (clipped_box["xmin"], clipped_box["ymin"], clipped_box["xmax"], clipped_box["ymax"])
    )
    overlap_area = sum(overlap_crop.histogram()[1:])
    return overlap_area / box_area


def box_roi_share(
    box: dict[str, float],
    roi_mask: Image.Image,
    image_size: tuple[int, int],
    roi_area: int,
) -> float:
    # This measures how much of the full ROI is covered by one box.
    clipped_box = clip_box_to_image(box, image_size)
    if clipped_box is None or roi_area <= 0:
        return 0.0
    overlap_crop = roi_mask.crop(
        (clipped_box["xmin"], clipped_box["ymin"], clipped_box["xmax"], clipped_box["ymax"])
    )
    overlap_area = sum(overlap_crop.histogram()[1:])
    return overlap_area / roi_area


def whc_foreground_mask(
    camera_id: str,
    image_size: tuple[int, int],
    polygon: list[tuple[int, int]],
) -> Image.Image | None:
    # WHC has perspective distortion, so we define a simple "foreground half" mask per camera.
    if camera_id not in WHC_PERSPECTIVE_CAMERA_IDS or not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    region_mask = Image.new("L", image_size, 0)
    region_draw = ImageDraw.Draw(region_mask)
    if camera_id == "K901F":
        split_y = (min_y + max_y) / 2.0
        region_draw.rectangle((min_x, split_y, max_x, max_y), fill=255)
    else:
        split_x = (min_x + max_x) / 2.0
        region_draw.rectangle((min_x, min_y, split_x, max_y), fill=255)
    return region_mask


def whc_big_foreground_detections(
    camera_id: str,
    detections: list[dict[str, Any]],
    roi_mask: Image.Image,
    image_size: tuple[int, int],
    roi_area: int,
) -> list[int]:
    # Only a few large buses or trucks should trigger the WHC foreground correction.
    if camera_id not in WHC_PERSPECTIVE_CAMERA_IDS:
        return []

    candidate_indices: list[int] = []
    for index, detection in enumerate(detections):
        if detection["label"] not in WHC_FOREGROUND_LARGE_VEHICLE_LABELS:
            continue
        if box_roi_share(detection["box"], roi_mask, image_size, roi_area) < WHC_BIG_BOX_MIN_ROI_SHARE:
            continue
        candidate_indices.append(index)

    return candidate_indices if 1 <= len(candidate_indices) <= WHC_FOREGROUND_MAX_BIG_VEHICLES else []


def compute_road_occupancy(
    camera_id: str,
    image: Image.Image | None,
    polygon: list[tuple[int, int]],
    on_road_detections: list[dict[str, Any]],
    on_road_vehicle_count: int,
    detector_available: bool,
) -> float:
    # Core occupancy formula:
    # road_occupancy = padded vehicle area inside ROI / total ROI area
    if not polygon or not detector_available or on_road_vehicle_count == 0:
        return 0.0

    if image is not None:
        roi_mask, roi_area = build_roi_mask(image.size, polygon)
        if roi_area > 0:
            # Only correct near-camera WHC buses/trucks when the road is already reasonably busy.
            foreground_mask = None
            big_foreground_indices: set[int] = set()
            if on_road_vehicle_count >= WHC_FOREGROUND_CORRECTION_MIN_ROI_COUNT:
                foreground_mask = whc_foreground_mask(camera_id, image.size, polygon)
                big_foreground_indices = set(
                    whc_big_foreground_detections(
                        camera_id,
                        on_road_detections,
                        roi_mask,
                        image.size,
                        roi_area,
                    )
                )
            vehicle_mask = Image.new("L", image.size, 0)
            for index, detection in enumerate(on_road_detections):
                # Expand the box before occupancy is measured so the score reflects practical spacing.
                box = expand_box_for_occupancy(detection["box"], image.size)
                if box is None:
                    continue
                detection_mask = Image.new("L", image.size, 0)
                detection_draw = ImageDraw.Draw(detection_mask)
                detection_draw.rectangle(
                    (box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
                    fill=255,
                )
                if index in big_foreground_indices and foreground_mask is not None:
                    # Limit corrected WHC vehicles to the foreground half only.
                    effective_mask = ImageChops.multiply(detection_mask, foreground_mask)
                else:
                    effective_mask = detection_mask
                vehicle_mask = ImageChops.lighter(vehicle_mask, effective_mask)
            # The final occupancy score is a normalized 0..1 ratio.
            covered_vehicle_area = sum(ImageChops.multiply(vehicle_mask, roi_mask).histogram()[1:])
            bbox_occupancy_ratio = min(max(covered_vehicle_area / roi_area, 0.0), 1.0)
            return round(bbox_occupancy_ratio, 3)
    return 0.0


def derive_camera_flow_state(
    on_road_vehicle_count: int,
    road_occupancy: float,
) -> str:
    # Road occupancy is translated into four user-facing traffic states.
    if on_road_vehicle_count == 0 or road_occupancy < FLOW_STATE_LOAD_THRESHOLDS["busy_but_moving"]:
        return "Clear"
    if road_occupancy >= FLOW_STATE_LOAD_THRESHOLDS["congested"]:
        return "Congested"
    if road_occupancy >= FLOW_STATE_LOAD_THRESHOLDS["slowing"]:
        return "Slowing"
    return "Busy but moving"


def load_segment_speed_map() -> tuple[dict[str, float], str | None]:
    # Turn the XML feed into a simple segment_id -> speed_kmh lookup table.
    try:
        xml_text = download_segment_speed_xml()
        root = ET.fromstring(xml_text)
    except (requests.RequestException, ET.ParseError, OSError, ValueError) as exc:
        return {}, str(exc)

    speed_by_segment: dict[str, float] = {}
    for segment_element in root.findall(".//segment"):
        segment_id = (segment_element.findtext("segment_id") or "").strip()
        if not segment_id:
            continue
        valid_flag = (segment_element.findtext("valid") or "").strip().upper()
        if valid_flag and valid_flag != "Y":
            continue
        speed = parse_float(segment_element.findtext("speed"))
        if speed is None or speed <= 0:
            continue
        speed_by_segment[segment_id] = round(speed, 2)

    if not speed_by_segment:
        return {}, "No valid segment speeds found"

    return speed_by_segment, None


def dynamic_baseline_seconds(
    tunnel: str,
    side: str,
    speed_map: dict[str, float],
) -> tuple[int | None, str, float | None, float | None]:
    # Use the official XML speed as the main ETA baseline for each tunnel side.
    segment_id = BASELINE_SEGMENT_IDS[tunnel][side]
    tunnel_length_km = TUNNEL_LENGTHS_KM[tunnel]
    speed_limit_kmh = TUNNEL_SPEED_LIMITS_KMH[tunnel]
    live_speed_kmh = speed_map.get(segment_id)
    if live_speed_kmh is None or live_speed_kmh <= 0:
        return None, segment_id, None, None
    fetched_speed_kmh = round(live_speed_kmh, 1)
    baseline_speed = min(max(fetched_speed_kmh, 1.0), speed_limit_kmh)
    baseline_seconds = round((tunnel_length_km / baseline_speed) * 3600)
    return baseline_seconds, segment_id, baseline_speed, fetched_speed_kmh


def run_service_screen_check(img: Image.Image | None, classifier: Any | None) -> list[dict[str, Any]]:
    # Run the service-screen classifier and return its raw predictions.
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
    # Mark a feed as unavailable only when the classifier is confident enough.
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
    # Run object detection and keep only vehicle classes that matter to the traffic view.
    if img is None or detector is None:
        return []

    try:
        if isinstance(detector, dict) and detector.get("kind") == "hf_object_detector":
            # For the Hugging Face model, post-process raw logits into image-scale boxes.
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
        # Normalize every detection into one simple shared shape for later processing.
        try:
            label = str(result.get("label", "")).lower().strip()
            score = float(result.get("score", 0.0) or 0.0)
            box = result.get("box", {}) or {}
            xmin = float(box.get("xmin", 0))
            ymin = float(box.get("ymin", 0))
            xmax = float(box.get("xmax", 0))
            ymax = float(box.get("ymax", 0))
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

    return detections


def clip_box_to_image(
    box: dict[str, float],
    image_size: tuple[int, int],
) -> dict[str, int] | None:
    # Clip a box to the image so downstream image math does not go out of bounds.
    image_width, image_height = image_size
    xmin = max(0, min(int(box["xmin"]), image_width - 1))
    ymin = max(0, min(int(box["ymin"]), image_height - 1))
    xmax = max(xmin + 1, min(int(box["xmax"]), image_width))
    ymax = max(ymin + 1, min(int(box["ymax"]), image_height))
    if xmax <= xmin or ymax <= ymin:
        return None
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def expand_box_for_occupancy(
    box: dict[str, float],
    image_size: tuple[int, int],
) -> dict[str, int] | None:
    # Expand the raw box by a small margin before using it for occupancy math.
    clipped_box = clip_box_to_image(box, image_size)
    if clipped_box is None:
        return None
    box_width = clipped_box["xmax"] - clipped_box["xmin"]
    box_height = clipped_box["ymax"] - clipped_box["ymin"]
    margin_x = max(OCCUPANCY_BOX_PADDING_MIN_PX, int(box_width * OCCUPANCY_BOX_PADDING_RATIO))
    margin_y = max(OCCUPANCY_BOX_PADDING_MIN_PX, int(box_height * OCCUPANCY_BOX_PADDING_RATIO))
    return clip_box_to_image(
        {
            "xmin": clipped_box["xmin"] - margin_x,
            "ymin": clipped_box["ymin"] - margin_y,
            "xmax": clipped_box["xmax"] + margin_x,
            "ymax": clipped_box["ymax"] + margin_y,
        },
        image_size,
    )


def roi_for_camera(camera_id: str) -> list[tuple[int, int]]:
    # Empty or tiny polygons are treated as "not calibrated yet".
    polygon = ROAD_ROIS.get(camera_id, [])
    return polygon if len(polygon) >= 3 else []


def filter_detections_to_road(
    detections: list[dict[str, Any]],
    polygon: list[tuple[int, int]],
    image_size: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    # A detection counts only when enough of its box lies inside the road ROI.
    if not polygon or image_size is None:
        return []

    roi_mask, roi_area = build_roi_mask(image_size, polygon)
    if roi_area <= 0:
        return []

    return [
        detection
        for detection in detections
        if box_overlap_ratio_in_roi(detection["box"], roi_mask, image_size) >= ROI_MIN_BOX_OVERLAP_RATIO
    ]


def annotate_image(
    img: Image.Image | None,
    display_detections: list[dict[str, Any]],
    roi_polygon: list[tuple[int, int]] | None = None,
) -> Image.Image | None:
    # The annotation image is for human auditing: ROI outline plus raw detector boxes.
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
        roi_outline.line(roi_polygon + [roi_polygon[0]], fill=(148, 210, 189, 220), width=1)

    if not display_detections:
        return annotated.convert("RGB")

    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for detection in display_detections:
        color = ANNOTATION_COLORS.get(detection["label"], (220, 38, 38))
        box_color = (*color, ANNOTATION_BOX_ALPHA)

        box = detection["box"]
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]
        overlay_draw.rectangle((xmin, ymin, xmax, ymax), outline=box_color, width=1)

    annotated = Image.alpha_composite(annotated, overlay)
    return annotated.convert("RGB")


def icon_for_flow_label(flow_label: str) -> str:
    # Match each flow label to a simple status icon in the card UI.
    if flow_label in {"No data", "Uncalibrated", "No road data"}:
        return "❓"
    if flow_label == "Clear":
        return "🟢"
    if flow_label == "Busy but moving":
        return "🟡"
    if flow_label == "Slowing":
        return "🟠"
    return "🔴"


def side_direction_label(side: str) -> str:
    # Convert the internal tunnel-side key into the user-facing direction text.
    if side == "Hong Kong":
        return "HK → KL"
    if side == "Kowloon":
        return "KL → HK"
    return side


def ordered_sides(side_map: dict[str, Any]) -> list[str]:
    # Keep the tunnel-side layout stable instead of relying on dict insertion order.
    side_order = {"Kowloon": 0, "Hong Kong": 1}
    return sorted(side_map.keys(), key=lambda side: (side_order.get(side, 99), side))


def format_duration(seconds: int | None) -> str:
    # Show ETA in a short format that is easy to scan in the card.
    if seconds is None:
        return "No data"
    minutes, remainder = divmod(max(int(seconds), 0), 60)
    return f"{minutes}m {remainder}s"


def fixed_baseline_seconds(tunnel: str) -> int:
    # This fallback is used only when live XML speed cannot be fetched.
    default_speed_kmh = DEFAULT_BASELINE_SPEED_KMH[tunnel]
    if default_speed_kmh <= 0:
        return 0
    return round((TUNNEL_LENGTHS_KM[tunnel] / default_speed_kmh) * 3600)


def baseline_caption(summary: dict[str, Any]) -> str:
    # Prefer showing the actual fetched speed; fall back only when live speed is unavailable.
    speed_kmh = (
        summary.get("fetched_speed_kmh")
        if summary.get("baseline_source") == "dynamic" and summary.get("fetched_speed_kmh") is not None
        else summary.get("default_baseline_speed_kmh")
    )
    if speed_kmh is None:
        return "Avg. traffic speed: N/A"
    return f"Avg. traffic speed: {speed_kmh:.1f}km/h"


def render_side_badge(direction: str, status_icon: str) -> None:
    # Render the small direction badge and colored state icon for one tunnel side.
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
    # The UI status line shows compact camera codes such as K107F or H702F.
    return url.rsplit("/", maxsplit=1)[-1].replace(".JPG", "")


def camera_modal_id(camera_id: str) -> str:
    # Build a safe HTML id for the click-to-enlarge image modal.
    return camera_id.replace("/", "_").replace(".", "_")


def image_to_data_uri(image: Image.Image | None) -> str | None:
    # Streamlit HTML blocks use base64 image URIs for hover and modal rendering.
    if image is None:
        return None
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_hover_image(image: Image.Image | None, caption: str, modal_id: str) -> None:
    # Render the image card with hover hint and a full-size modal view.
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
    # Summarize camera availability for the top status bar.
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


def make_side_summary(
    side: str,
    baseline: int,
    baseline_source: str,
    baseline_detector_id: str,
    baseline_speed_kmh: float | None,
    fetched_speed_kmh: float | None,
    default_speed_kmh: float,
    *,
    status_label: str,
    flow_label: str,
    estimated_crossing_seconds: int | None = None,
    road_occupancy: float | None = None,
) -> dict[str, Any]:
    # Standardize the shape of one side summary so the UI can render it consistently.
    return {
        "side": side,
        "direction": side_direction_label(side),
        "baseline_seconds": baseline,
        "baseline_source": baseline_source,
        "baseline_detector_id": baseline_detector_id,
        "baseline_speed_kmh": baseline_speed_kmh,
        "fetched_speed_kmh": fetched_speed_kmh,
        "default_baseline_speed_kmh": default_speed_kmh,
        "estimated_crossing_seconds": estimated_crossing_seconds,
        "status_label": status_label,
        "status_icon": icon_for_flow_label(flow_label),
        "flow_label": flow_label,
        "road_occupancy": road_occupancy,
    }


def summarize_side(
    tunnel: str,
    side: str,
    records: list[dict[str, Any]],
    detector_available: bool,
    detector_speed_map: dict[str, float],
) -> dict[str, Any]:
    # This function turns raw camera records into the final side-level card output.
    available_records = [record for record in records if record["image"] is not None]
    analyzable_records = [record for record in available_records if record["analysis_enabled"]]
    calibrated_records = [record for record in analyzable_records if record["roi_configured"]]
    fallback_baseline = fixed_baseline_seconds(tunnel)
    default_speed_kmh = round(DEFAULT_BASELINE_SPEED_KMH[tunnel], 1)
    dynamic_baseline, baseline_detector_id, baseline_speed_kmh, fetched_speed_kmh = dynamic_baseline_seconds(
        tunnel=tunnel,
        side=side,
        speed_map=detector_speed_map,
    )
    baseline = dynamic_baseline if dynamic_baseline is not None else fallback_baseline
    baseline_source = "dynamic" if dynamic_baseline is not None else "fallback"

    summary_base = {
        "side": side,
        "baseline": baseline,
        "baseline_source": baseline_source,
        "baseline_detector_id": baseline_detector_id,
        "baseline_speed_kmh": baseline_speed_kmh,
        "fetched_speed_kmh": fetched_speed_kmh,
        "default_speed_kmh": default_speed_kmh,
    }

    if not available_records:
        return make_side_summary(**summary_base, status_label="No data", flow_label="No data")

    if analyzable_records and not calibrated_records:
        return make_side_summary(
            **summary_base,
            status_label="No calibrated road area",
            flow_label="Uncalibrated",
        )

    if available_records and not analyzable_records:
        return make_side_summary(**summary_base, status_label="N/A", flow_label="N/A")

    if not detector_available:
        return make_side_summary(
            **summary_base,
            status_label="Detector unavailable",
            flow_label="No road data",
        )

    primary_record = calibrated_records[0]
    current_load = float(primary_record["road_occupancy"])
    flow_label = str(primary_record["camera_flow_state"])
    base_speed_kmh = baseline_speed_kmh if baseline_speed_kmh is not None else default_speed_kmh
    # ETA formula:
    # adjusted_speed = baseline_speed * flow_factor
    # eta_seconds = tunnel_length / adjusted_speed * 3600
    adjusted_speed_kmh = (
        round(max(base_speed_kmh * FLOW_SPEED_FACTORS.get(flow_label, 1.0), 5.0), 1)
        if base_speed_kmh is not None and base_speed_kmh > 0
        else None
    )
    estimated_crossing_seconds = (
        round((TUNNEL_LENGTHS_KM[tunnel] / adjusted_speed_kmh) * 3600)
        if adjusted_speed_kmh is not None
        else None
    )
    return make_side_summary(
        **summary_base,
        status_label=flow_label,
        flow_label=flow_label,
        estimated_crossing_seconds=estimated_crossing_seconds,
        road_occupancy=round(current_load, 3),
    )


def record_traffic_status_history(snapshot_time: float, tunnel_metrics: dict[str, Any]) -> None:
    # Save one tunnel-level state per refresh so the 4-hour chart can be rebuilt later.
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
    # Build a complete time grid so the chart can show "No data" gaps explicitly.
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

    # Always rebuild labels from the canonical numeric band so wording stays consistent.
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
    # Missing buckets are shown as "No data" instead of being silently skipped.
    df["status_band"] = df["status_band"].fillna(-1)
    df["status_label"] = df["status_label"].fillna("No data")
    df["time_label"] = df["timestamp"].apply(
        lambda ts: datetime.fromtimestamp(int(ts), HONG_KONG_TZ).strftime("%H:%M")
    )
    return df


def build_snapshot() -> tuple[float, dict[str, Any], dict[str, Any], dict[str, str]]:
    # This is the main live-data pipeline for one refresh cycle.
    service_classifier, service_classifier_error = load_service_screen_classifier()
    detector, detector_error = load_object_detector()
    detector_available = detector is not None

    snapshot_time = datetime.now().timestamp()
    records_by_tunnel: dict[str, Any] = {}
    tunnel_metrics: dict[str, Any] = {}

    for tunnel, sides in TUNNELS.items():
        side_records: dict[str, Any] = {}

        for side, camera_ids in sides.items():
            camera_records = []

            for camera_id in camera_ids:
                # Fetch images first so all camera views belong to the same refresh cycle.
                source_url = CAMERA_SOURCE_URLS[camera_id]
                image = fetch_image(source_url)
                service_unavailable_detected, service_check_result = detect_service_unavailable(image, service_classifier)
                analysis_enabled = image is not None and not service_unavailable_detected
                all_detections = detect_vehicles(image, detector) if analysis_enabled else []
                polygon = roi_for_camera(camera_id)
                roi_configured = bool(polygon)
                on_road_detections = filter_detections_to_road(
                    all_detections,
                    polygon,
                    image.size if image is not None else None,
                )
                on_road_vehicle_count = len(on_road_detections)
                # Road occupancy is the image-derived congestion signal used by side flow.
                road_occupancy = (
                    compute_road_occupancy(
                        camera_id=camera_id,
                        image=image,
                        polygon=polygon,
                        on_road_detections=on_road_detections,
                        on_road_vehicle_count=on_road_vehicle_count,
                        detector_available=detector_available,
                    )
                    if analysis_enabled
                    else 0.0
                )
                camera_flow_state = (
                    derive_camera_flow_state(
                        on_road_vehicle_count=on_road_vehicle_count,
                        road_occupancy=road_occupancy,
                    )
                    if analysis_enabled
                    else "N/A"
                )
                annotated_image = image
                if analysis_enabled and image is not None:
                    # The displayed image uses the already filtered road detections.
                    annotated_image = annotate_image(
                        image,
                        on_road_detections,
                        polygon,
                    )

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
                        "road_occupancy": road_occupancy,
                        "roi_configured": roi_configured,
                        "camera_flow_state": camera_flow_state,
                    }
                )

            side_records[side] = camera_records
        records_by_tunnel[tunnel] = side_records

    # Fetch the official speed feed once after images are ready.
    detector_speed_map, detector_feed_error = load_segment_speed_map()

    for tunnel, side_records in records_by_tunnel.items():
        side_summaries: dict[str, Any] = {}
        tunnel_camera_scores: list[float] = []

        for side, camera_records in side_records.items():
            side_summaries[side] = summarize_side(
                tunnel=tunnel,
                side=side,
                records=camera_records,
                detector_available=detector_available,
                detector_speed_map=detector_speed_map,
            )
            if side_summaries[side].get("road_occupancy") is not None:
                tunnel_camera_scores.append(side_summaries[side]["road_occupancy"])

        side_flow_bands = [
            queue_state_to_band(summary["flow_label"])
            for summary in side_summaries.values()
            if queue_state_to_band(summary["flow_label"]) is not None
        ]
        # The timeline is tunnel-level, so side states are averaged and rounded into one tunnel state.
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
        tunnel_metrics[tunnel] = {
            "status_label": trend_status_label if trend_status_label is not None else "No calibrated road data",
            "status_icon": icon_for_flow_label(trend_status_label or "No road data"),
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
    # Show the global app status, any warnings, and the manual refresh button.
    with st.container(border=True):
        status_column, action_column = st.columns([4.8, 1.2], vertical_alignment="center")
        with status_column:
            service_message, service_color = service_classifier_status(records_by_tunnel)
            warnings = []
            feed_warning = bool(model_errors.get("detector_feed"))
            if model_errors.get("service_classifier"):
                warnings.append("Service-screen check unavailable")
            if model_errors.get("detector"):
                warnings.append("Object detector unavailable")
            if feed_warning:
                warnings.append("Speed baseline feed unavailable")
            status_text = " | ".join(warnings) if warnings else service_message
            status_line_color = "#f6ad55" if warnings else service_color
            st.markdown(
                f"<div style='color:{status_line_color};font-size:1.58rem;font-weight:700;display:flex;align-items:center;min-height:3rem;line-height:1.2;margin:0;'>"
                f"Status: {status_text}</div>",
                unsafe_allow_html=True,
            )
            if feed_warning:
                st.markdown(
                    "<div style='color:#f6ad55;font-size:0.96rem;font-weight:600;line-height:1.25;margin-top:0.15rem;'>"
                    "Live speed data is currently not available.</div>",
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
                # Clear cached live feeds so the next rerun fetches fresh data immediately.
                download_image_bytes.clear()
                download_segment_speed_xml.clear()
                if STREAMLIT_FRAGMENT is not None:
                    st.rerun(scope="fragment")
                else:
                    st.rerun()


def render_trend_chart(snapshot_time: float) -> None:
    # Render the last 4 hours of tunnel-level traffic states as an Altair heatmap.
    st.subheader("Flow Timeline (Last 4 Hours)")
    st.caption("Live tunnel-status timeline in 5-minute blocks: Clear, Busy but moving, Slowing, Congested.")

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
        # Each rectangle is one tunnel status in one 5-minute time bucket.
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
    # Render the whole page: header, status bar, timeline, and tunnel cards.
    stylesheet = load_stylesheet()
    if stylesheet:
        st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='margin-bottom:0.2rem;'>🚗 Hong Kong Tunnel Traffic Monitor</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Snapshot captured at {datetime.fromtimestamp(snapshot_time, HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S')} HKT "
        f"({'Auto refresh every 5 min' if STREAMLIT_FRAGMENT is not None or st_autorefresh is not None else 'Auto refresh unavailable'})"
    )
    render_top_bar(snapshot_time, st.session_state.get("model_errors", {}), records_by_tunnel)
    render_trend_chart(snapshot_time)

    st.caption(
        "Est. crossing time is calculated from tunnel length and vehicle speed, with camera-derived traffic flow reducing speed when the approach looks crowded."
    )

    for tunnel, side_map in TUNNELS.items():
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
                    primary_record = side_records[0] if side_records else None
                    info_column, image_column = st.columns([0.9, 1.1], vertical_alignment="top")

                    with info_column:
                        render_side_badge(summary["direction"], summary["status_icon"])
                        if summary["estimated_crossing_seconds"] is None:
                            st.metric("Est. crossing time (from entrance to exit)", "N/A")
                            st.caption(baseline_caption(summary))
                        else:
                            st.metric(
                                "Est. crossing time (from entrance to exit)",
                                format_duration(summary["estimated_crossing_seconds"]),
                            )
                            st.caption(baseline_caption(summary))

                        if primary_record is None or primary_record["image"] is None:
                            st.write("**Side flow:** N/A  \n**Road occupancy:** N/A")
                        elif not primary_record["roi_configured"]:
                            st.info("ROI not configured; excluded from road-flow calculation.")
                        elif not primary_record["analysis_enabled"]:
                            st.write("Feed: Service unavailable | Traffic analysis: N/A")
                            if primary_record["service_check_result"]:
                                st.caption(f"Feed check: {primary_record['service_check_result']}")
                        else:
                            road_occupancy = summary.get("road_occupancy")
                            road_occupancy_text = (
                                f"{round(float(road_occupancy) * 100)}%"
                                if road_occupancy is not None
                                else "N/A"
                            )
                            st.markdown(
                                f"**Side flow:** {primary_record['camera_flow_state']}  \n"
                                f"**Road occupancy:** {road_occupancy_text}"
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


def render_live_dashboard_cycle() -> None:
    # Run one full refresh, update history, and redraw the page.
    with st.spinner("Refreshing live traffic snapshot..."):
        snapshot_time, records_by_tunnel, tunnel_metrics, model_errors = build_snapshot()
    st.session_state["model_errors"] = model_errors
    record_traffic_status_history(snapshot_time, tunnel_metrics)
    render_dashboard(snapshot_time, records_by_tunnel, tunnel_metrics)


render_live_dashboard_fragment = (
    # Prefer fragment refresh when Streamlit supports it; otherwise fall back to a normal rerun.
    STREAMLIT_FRAGMENT(run_every=AUTO_REFRESH_INTERVAL_MS / 1000)(render_live_dashboard_cycle)
    if STREAMLIT_FRAGMENT is not None
    else render_live_dashboard_cycle
)


def main() -> None:
    # Initialize state and start the live dashboard refresh flow.
    init_session_state()

    if STREAMLIT_FRAGMENT is None and st_autorefresh is not None:
        # This fallback is used only on Streamlit versions that do not support fragments.
        st_autorefresh(interval=AUTO_REFRESH_INTERVAL_MS, key="live_dashboard_refresh")

    render_live_dashboard_fragment()


if __name__ == "__main__":
    main()
