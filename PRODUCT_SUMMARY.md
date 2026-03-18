# Hong Kong Tunnel Traffic Monitor: Product Summary

Company name: Transport Department

After the introduction of time-varying tolls, peak traffic at the Western Harbour Crossing rose by about 20%, while the Cross-Harbour Tunnel and Eastern Harbour Crossing saw only modest reductions, resulting in more volatile tunnel demand. As the Department does not yet provide real-time tunnel traffic information, this project applies deep learning to detect vehicles at tunnel entrances/exits and deliver accurate, public-facing travel-time and congestion information. 

Ref:  
- https://www.legco.gov.hk/yr2024/english/panels/tp/papers/tp20240517cb4-584-3-e.pdf 
- https://www.info.gov.hk/gia/general/202501/22/P2025012200478p.htm 
- https://www.i-cable.com/%E6%96%B0%E8%81%9E%E8%B3%87%E8%A8%8A/424132/%E4%B8%80%E7%B7%9A%E6%90%9C%E6%9F%A5-%E4%B8%89%E9%9A%A7%E5%88%86%E6%B5%81%E5%B1%AF%E9%96%80%E7%84%A1%E8%91%97%E6%95%B8-%E6%89%93%E5%B7%A5%E4%BB%94%E9%80%9A%E5%8B%A4%E5%A4%9A45%E5%88%86 

## App Feature

This app monitors the three main Hong Kong harbour tunnels using live Transport Department CCTV images and official speed detector data. Its goal is to give a quick, side-specific view of tunnel conditions, including a simple traffic state label and an estimated crossing time. 

At a high level, the app combines two live signals: 

- what the road looks like right now from CCTV
- how fast traffic is officially moving right now from the Transport Department detector feed

The result is a practical estimate for each tunnel direction rather than a perfect traffic simulation.

## Model Selection Strategy 

We compare three distinct Transformer-based pipelines to find the optimal balance of accuracy and runtime: 

- PekingU/rtdetr_r50vd: A hybrid model optimized for real-time speed. 
- microsoft/conditional-detr-resnet-50: Optimized for faster attention convergence – best results with the lowest loss, meaning the fewest mistakes in understanding images 
- facebook/detr-resnet-50: The foundational baseline for Transformer object detection. 

Experiment: https://colab.research.google.com/drive/14c50cbyXbDUn29HCkwhlJPWFCRMdxZg2?usp=sharing

Experiment for the other pipeline that detects service unavailable screen: 

- openai/clip-vit-base-patch32
- laion/clip-vit-l-14-laion2b-s32b-b82k
- google/siglip-base-patch16-224 – fastest runtime, results similar with the other 2 

Experiment: https://colab.research.google.com/drive/12hHJvKs2EEN2_UZSBBvoHGbh7UKLTlQQ?usp=sharing

## Architecture and Data Flow

The app is built as a Streamlit dashboard. On each refresh cycle, it runs the following flow:

1. Fetch the latest CCTV image for each logical camera.
2. Check whether the image is actually a service-unavailable screen using `google/siglip-base-patch16-224`.
3. Run object detection on valid images using `microsoft/conditional-detr-resnet-50`.
4. Keep only vehicle detections whose bounding-box center falls inside the configured road ROI (Region of Interest).
5. Count those on-road vehicles and turn the count into a load score for that camera.
6. Classify the traffic state using the current load plus recent saved history.
7. Fetch the official XML speed detector feed from the Transport Department.
8. Convert the detector speed into a dynamic baseline crossing time for that tunnel side.
9. Add camera-derived delay on top of that baseline and render the final result in the dashboard.

The camera setup is intentionally simple:

- Cross Harbour Tunnel uses 1 camera angle at Kowloon side entrance & exit to detect traffic from Kownloon & traffic from HK Island, but each side has its own ROI polygon. 
- Eastern Harbour Crossing also uses 1 one camera angle at Kowloon side entrance & exit to detect traffic from Kownloon & traffic from HK Island, but each side has its own ROI polygon. 
- Western Harbour Crossing uses separate images for the entrances at Kowloon side and HK Island side respectively, as the exit on either side is too far away from the camera for detection. Each side currently behaves like one active camera, so side state comes directly from that single camera rather than averaging multiple feeds. 

There is also a visual enhancement layer:

- `facebook/sam2-hiera-tiny` is used only to refine the displayed overlay on the image.
- It does not change the counts, traffic state, or ETA math.
- ROI masking is also visual-only; it helps users see what part of the road is being counted.

## Traffic Logic and ETA Math

The app’s core metric is `camera_load`, which measures how busy a side looks relative to a manually configured capacity:

`camera_load = min(on_road_vehicle_count / camera_capacity, 1.0)`

Example: if a side has capacity `50` and the detector finds `25` vehicles inside the ROI, the load is `0.5`.

There is one correction to avoid false congestion from a single large nearby vehicle. If one very large bus or truck dominates a sparse frame, the app subtracts `0.15` from the load. This prevents one oversized close-up detection from making the road look more congested than it really is.

Traffic state labels are based on both load and persistence:

- `Clear`: no on-road vehicles, or load below `0.5`
- `Busy but moving`: load is at least `0.5` in the current 2-minute bucket
- `Slowing`: high load persists for 2 consecutive buckets
- `Congested`: high load persists for 3 consecutive buckets

Estimated crossing time starts with a dynamic baseline from the official speed detector feed. Lane speeds are averaged using valid lanes and volume weighting, then capped by the tunnel speed limit.

`baseline_seconds = (tunnel_length_km / speed_kmh) * 3600`

If detector data is unavailable, the app falls back to a fixed baseline travel time.

The app then adds extra delay only when the road is meaningfully busy. If load is below `0.5`, extra delay is zero. If it is above `0.5`, the app smooths the current reading with recent history:

`effective_load = 0.8 * current_load + 0.2 * recent_load`

It then normalizes only the busy portion:

`normalized_load = (effective_load - 0.5) / 0.5`

This value is clamped to `0..1`, curved using exponent `1.15`, and scaled by a tunnel-specific max delay:

`extra_delay = (normalized_load ^ 1.15) * tunnel_max_extra_delay`

Final estimate:

`estimated_crossing_seconds = baseline_seconds + extra_delay_seconds`

## Practical Product Notes

Key constants in the current live app:

- Busy threshold: `0.5`
- Smoothing weights: `0.8 current / 0.2 recent`
- Large-vehicle penalty: `0.15`
- Delay exponent: `1.15`
- Tunnel lengths:
  - Cross Harbour Tunnel: `1.86 km`
  - Eastern Harbour Crossing: `2.2 km`
  - Western Harbour Crossing: `2.0 km`
- Speed caps:
  - Cross Harbour Tunnel: `50 km/h`
  - Eastern Harbour Crossing: `70 km/h`
  - Western Harbour Crossing: `70 km/h`

Three product takeaways:

- ETA combines an official live speed baseline with visible queueing from CCTV.
- Flow labels depend on persistence, not just a one-frame spike.
- ROI calibration matters, because only vehicles inside the road area count toward traffic state and ETA.

## Finetuning 

1. Prepare training data: ~2000 snapshots from the used camera from Kaggle, ~3000 snapshots from other angles
2. Auto label training data with PekingU/rtdetr_r50vd
3. Clean up annotation in coco json file, with 5 labels remaining: car, bus, truck, motorcycle, person
4. Finetuning of 25 epochs with microsoft/conditional-detr-resnet-50 and unlabelled snapshot images & annotation file
5. Further finetuneing of 100 epochs with manually labelled & annotated snapshot images (~400) for calibration and enhanced accuracy (AI-assisted manual annotation tool on Roboflow: Segment Anything 3)
