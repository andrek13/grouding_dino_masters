import torch
import cv2
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util import box_ops

CONFIG_FILE = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_FILE = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
IMAGE_PATH = "images/dooors.jpg"
TEXT_PROMPT = "door"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(CONFIG_FILE, CHECKPOINT_FILE).to(device)
image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=device,
)

h, w, _ = image_source.shape
boxes_abs = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([w, h, w, h])
boxes_abs = boxes_abs.numpy()

# image_source is RGB (from load_image); convert to BGR for OpenCV
annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

RED = (0, 0, 255)  # OpenCV uses BGR, so this is red

for box, phrase, logit in zip(boxes_abs, phrases, logits):
    x0, y0, x1, y1 = box.astype(int)
    cv2.rectangle(annotated_frame, (x0, y0), (x1, y1), RED, thickness=2)

    label = f"{phrase} {logit:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(annotated_frame, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), RED, -1)
    cv2.putText(annotated_frame, label, (x0 + 2, max(0, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imwrite("annotated_image.jpg", annotated_frame)
print(f"Done, saved annotated_image.jpg with {len(boxes_abs)} door(s) detected")