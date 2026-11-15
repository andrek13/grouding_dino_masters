import os
import glob

import torch
import cv2
import numpy as np
from PIL import Image
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
import groundingdino.datasets.transforms as T

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "GroundingDINO", "weights", "groundingdino_swint_ogc.pth")

VIDEO_DIR = os.path.join(BASE_DIR, "videoset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VIDEO_EXTENSIONS = (".mp4", ".ts", ".mov", ".avi", ".mkv")

TEXT_PROMPT = "door"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DETECT_EVERY_N_FRAMES = 5  # bump this up (e.g. 5) to skip frames for speed

RED = (0, 0, 255)  # OpenCV uses BGR, so this is red

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading model on device: {device}")
model = load_model(CONFIG_FILE, CHECKPOINT_FILE).to(device)


def load_image_from_array(image_rgb: np.ndarray):
    """Same preprocessing load_image() applies to a file, but for an in-memory frame."""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_pil = Image.fromarray(image_rgb)
    image_transformed, _ = transform(image_pil, None)
    return image_transformed


def draw_detections(frame_bgr, boxes, logits, phrases):
    h, w, _ = frame_bgr.shape
    boxes_abs = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
    boxes_abs = boxes_abs.numpy()

    for box, phrase, logit in zip(boxes_abs, phrases, logits):
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), RED, thickness=2)

        label = f"{phrase} {logit:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), RED, -1)
        cv2.putText(frame_bgr, label, (x0 + 2, max(0, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return len(boxes_abs)


def process_video(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    last_boxes, last_logits, last_phrases = torch.empty(0, 4), torch.empty(0), []
    total_detections = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_transformed = load_image_from_array(frame_rgb)

            boxes, logits, phrases = predict(
                model=model,
                image=image_transformed,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=device,
            )
            last_boxes, last_logits, last_phrases = boxes, logits, phrases

        n = draw_detections(frame_bgr, last_boxes, last_logits, last_phrases)
        total_detections += n

        out.write(frame_bgr)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"  Done -> {output_path} "
          f"({frame_idx} frames, {total_detections} total detections drawn)")


def find_videos(video_dir: str):
    paths = []
    for ext in VIDEO_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


if __name__ == "__main__":
    videos = find_videos(VIDEO_DIR)

    if not videos:
        print(f"No videos found in {VIDEO_DIR} (looked for {VIDEO_EXTENSIONS})")
    else:
        print(f"Found {len(videos)} video(s) in {VIDEO_DIR}")

        for video_path in videos:
            filename = os.path.basename(video_path)
            name_no_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(OUTPUT_DIR, f"annotated_{name_no_ext}.mp4")

            print(f"\nProcessing: {filename}")
            process_video(video_path, output_path)

        print(f"\nAll done. Annotated videos saved in: {OUTPUT_DIR}")