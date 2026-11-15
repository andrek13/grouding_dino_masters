import os
import glob
import re
import cv2
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "videoset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_mapped")
VIDEO_EXTENSIONS = (".mp4", ".ts", ".mov", ".avi", ".mkv")

MERGED_CSV = "merged_clusters_doors_gajdos.csv"   # or merged_clusters_doors_gajdos.csv

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(MERGED_CSV)

# Build, per video, a sorted list of (second, cluster, door_detected) we actually
# have labels for. Between labeled seconds we carry the last known value forward,
# since only every Nth frame was clustered/detected, not every single frame.
per_video_labels = {
    video: group.sort_values("second")[["second", "cluster", "door_detected"]].values.tolist()
    for video, group in df.groupby("video")
}


def label_for_second(labels, second):
    """Most recent known (cluster, door_detected) at or before this second."""
    chosen = None
    for s, cluster, door in labels:
        if s <= second:
            chosen = (cluster, door)
        else:
            break
    return chosen if chosen else (None, None)


def find_videos(video_dir: str):
    paths = []
    for ext in VIDEO_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def map_video(video_path: str, labels: list):
    filename = os.path.basename(video_path)
    name_no_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"mapped_{name_no_ext}.mp4")

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
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        second = frame_idx / fps
        cluster, door = label_for_second(labels, second)

        if cluster is not None:
            text = f"cluster {int(cluster)} | door: {'YES' if door else 'no'}"
            color = (0, 0, 255) if door else (0, 200, 0)
        else:
            text = "no label"
            color = (150, 150, 150)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (5, 5), (15 + tw, 15 + th + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"  Done -> {output_path} ({frame_idx} frames)")


if __name__ == "__main__":
    videos = find_videos(VIDEO_DIR)
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
    else:
        for video_path in videos:
            filename = os.path.basename(video_path)
            labels = per_video_labels.get(filename)
            if labels is None:
                print(f"[SKIP] {filename}: no entries in {MERGED_CSV}")
                continue
            print(f"Mapping: {filename}")
            map_video(video_path, labels)

        print(f"\nAll done. Mapped videos saved in: {OUTPUT_DIR}")