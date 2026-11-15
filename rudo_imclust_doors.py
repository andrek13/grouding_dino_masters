# merge_doors_clusters.py
import re
import pandas as pd

def video_and_second(path, is_frame_file=True):
    base = path.split("/")[-1]
    if is_frame_file:
        m = re.match(r"(.+)_(\d+)\.jpg$", base)
        if not m:
            return None, None
        video, n = m.group(1), int(m.group(2))
        return video + ".mp4", n - 1
    return base, None

# --- load imclust's cluster CSV ---
clust = pd.read_csv(
    "rudo-results/frames_flat.km8.densenet201-pca200.csv",   # <-- fix to your real path
    sep=" ", names=["path", "cluster", "dist", "pdist", "bad"], skiprows=1
)
clust["video"], clust["second"] = zip(*clust["path"].map(video_and_second))

# --- load GroundingDINO's door-detection CSV ---
doors = pd.read_csv("output/door_detections.csv")   # <-- fix to your real path
doors["second"] = doors["second"].round().astype(int)

# --- merge ---
merged = clust.merge(doors, on=["video", "second"], how="inner")

print(merged.groupby("cluster")["door_detected"].mean())
print(merged.groupby("video")["door_detected"].agg(["sum", "mean"]))

merged.to_csv("merged_clusters_doors.csv", index=False)