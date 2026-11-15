"""
Auto-label every video in a folder with GroundingDINO and export a YOLO dataset.

pip install torch torchvision transformers opencv-python pillow
python label_videos.py
"""
import cv2
import torch
from pathlib import Path
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ---------------- Config ----------------
VIDEO_DIR = Path("videoset")           # folder with your videos
OUT = Path("dataset")
CLASSES = ["door"]   # index = YOLO class id
FRAME_STEP = 10                      # keep every Nth frame
BOX_THR, TEXT_THR = 0.35, 0.25
NMS_IOU = 0.6                        # per-class NMS to drop duplicate boxes
VAL_FRACTION = 0.2                   # last 20% of each video -> val
SKIP_EMPTY = True                    # skip frames with no detections
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MODEL_ID = "IDEA-Research/grounding-dino-base"
# ----------------------------------------

for split in ("train", "val"):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()


def detect(frame):
    """Return list of YOLO label lines for one BGR frame."""
    h, w = frame.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=[CLASSES], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THR, text_threshold=TEXT_THR,  # older versions: box_threshold=
        target_sizes=[(h, w)],
    )[0]

    labels = res.get("text_labels", res.get("labels"))
    boxes, scores, cls_ids = [], [], []
    for box, score, label in zip(res["boxes"], res["scores"], labels):
        if label not in CLASSES:  # skip merged/partial phrases
            continue
        boxes.append(box)
        scores.append(score)
        cls_ids.append(CLASSES.index(label))
    if not boxes:
        return []

    boxes = torch.stack(boxes).cpu()
    scores = torch.stack(scores).cpu()
    cls_t = torch.tensor(cls_ids)
    keep = []
    for c in cls_t.unique():
        idx = (cls_t == c).nonzero(as_tuple=True)[0]
        k = nms(boxes[idx], scores[idx], NMS_IOU)
        keep.extend(idx[k].tolist())

    lines = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i].tolist()
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls_ids[i]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


videos = sorted(p for p in VIDEO_DIR.rglob("*") if p.suffix.lower() in VIDEO_EXT)
if not videos:
    raise SystemExit(f"No videos found in {VIDEO_DIR.resolve()}")

totals = {"train": 0, "val": 0}
for vid_idx, video in enumerate(videos, 1):
    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    split_at = int(n_frames * (1 - VAL_FRACTION))  # frame index where val starts
    stem = video.stem.replace(" ", "_")
    print(f"[{vid_idx}/{len(videos)}] {video.name} ({n_frames} frames)")

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % FRAME_STEP == 0:
            lines = detect(frame)
            if lines or not SKIP_EMPTY:
                split = "train" if i < split_at else "val"
                name = f"{stem}_{i:07d}"
                cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), frame)
                (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                totals[split] += 1
        i += 1
    cap.release()

(OUT / "data.yaml").write_text(
    f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
    + "".join(f"  {k}: {c}\n" for k, c in enumerate(CLASSES))
)
print(f"Done. train={totals['train']} val={totals['val']} -> {OUT.resolve()}")