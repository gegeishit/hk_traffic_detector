# Hong Kong Tunnel Traffic Monitor

Streamlit dashboard for monitoring the three main Hong Kong harbour crossings with live CCTV snapshots and official Transport Department traffic-speed data.

The app estimates side-specific tunnel conditions by combining:
- live vehicle detection from tunnel camera images
- official average traffic speed from the Hong Kong Transport Department

It is designed to give a practical view of:
- current traffic condition per tunnel side
- estimated crossing time
- vehicles currently detected inside the configured road ROI

## What The App Covers

- Cross Harbour Tunnel
- Eastern Harbour Crossing
- Western Harbour Crossing

Each tunnel is shown by side:
- `KL -> HK`
- `HK -> KL`

For Cross Harbour Tunnel and Eastern Harbour Crossing, one shared image source is reused for both directions with different ROI polygons. Western Harbour Crossing uses separate source images per direction.

## Live Stack

- UI: `Streamlit`
- Object detector: `Gegeishit/yolos-small-hk-traffic-finetuned`
- Service-screen classifier: `google/siglip-base-patch16-224`
- Image processing: `Pillow`
- Data handling/charting: `pandas`, `altair`

## Data Sources

- CCTV snapshots: Hong Kong Transport Department traffic camera feeds
- Live speed feed: `irnAvgSpeed-all.xml`
  - URL: `https://resource.data.one.gov.hk/td/traffic-detectors/irnAvgSpeed-all.xml`

Tunnel-side speed mapping currently uses these road-network segment IDs:
- WHC `HK -> KL`: `106784`
- WHC `KL -> HK`: `106785`
- CHT `HK -> KL`: `2905`
- CHT `KL -> HK`: `105057`
- EHC `HK -> KL`: `101734`
- EHC `KL -> HK`: `101735`

## How It Works

On each refresh cycle, the app:

1. fetches the latest CCTV snapshot for each logical camera
2. checks whether the image is a service-unavailable screen
3. runs vehicle detection on valid road images
4. filters detections to the configured road ROI
5. converts detected on-road vehicles into a road-occupancy score
6. fetches live average traffic speed for each tunnel side from the TD XML feed
7. derives a traffic-state label from road occupancy
8. adjusts speed using the traffic state
9. calculates estimated crossing time from tunnel length and adjusted speed
10. renders the dashboard

The snapshot build currently fetches all camera images first, then fetches the XML speed feed once, and then calculates the side summaries and ETA metrics in the same refresh cycle.

## Traffic Logic

Core load formula:

```text
road_occupancy = covered_bounding_box_area_inside_roi / roi_area
```

`road_occupancy` is kept internally as a `0..1` float and displayed in the UI as a percentage.

Traffic-state bands:
- `Clear`: load `< 0.20`
- `Busy but moving`: load `0.20` to `< 0.45`
- `Slowing`: load `0.45` to `< 0.70`
- `Congested`: load `>= 0.70`

Speed adjustment factors:
- `Clear`: `100%`
- `Busy but moving`: `95%`
- `Slowing`: `85%`
- `Congested`: `70%`

Fallback behavior:
- if live XML speed is unavailable, the app falls back to `60 km/h`
- if ROI-area occupancy cannot be computed, the app falls back to `on_road_vehicle_count / road_capacity`

Current key constants:
- detector confidence threshold: `0.65`
- auto refresh: `120 seconds`
- tunnel lengths:
  - CHT: `1.86 km`
  - EHC: `2.2 km`
  - WHC: `2.0 km`
- speed caps:
  - CHT: `50 km/h`
  - EHC: `70 km/h`
  - WHC: `70 km/h`

Configured road capacities (fallback/reference):
- CHT `HK -> KL` (`K107F-HK2KL`): `174`
- CHT `KL -> HK` (`K107F-KL2HK`): `171`
- EHC `HK -> KL` (`K952F-HK2KL`): `86`
- EHC `KL -> HK` (`K952F-KL2HK`): `133`
- WHC `HK -> KL` (`H702F`): `86`
- WHC `KL -> HK` (`K901F`): `41`

These capacities are currently based on measured visible road length and lane count, using a `7m` effective vehicle spacing assumption.

## Run Locally

Recommended Python: `3.11` or `3.12`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Streamlit Cloud Notes

- Main entrypoint: `app.py`
- First startup can be slower because Hugging Face models need to download
- The repo is configured to use CPU-friendly PyTorch wheels on Linux via `requirements.txt`
- If the app UI still shows older behavior after a push, reboot the app from Streamlit Cloud

## Repo Notes

- Main app: [app.py](/Users/alan/Desktop/hk_traffic_detector/app.py)

This repository contains some experimental and historical assets as well. The live dashboard behavior is driven by `app.py`.
