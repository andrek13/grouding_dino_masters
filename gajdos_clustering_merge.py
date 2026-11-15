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

# --- gajdosandrej's cluster CSV ---
clust = pd.read_csv("cluster_assignments_ResNet50_8.csv")   # path, cluster
clust["video"], clust["second"] = zip(*clust["path"].map(video_and_second))

# --- GroundingDINO's door-detection CSV ---
doors = pd.read_csv("output/door_detections.csv")   # <-- fix to your real path
doors["second"] = doors["second"].round().astype(int)

# --- merge ---
merged = clust.merge(doors, on=["video", "second"], how="inner")

print(merged.groupby("cluster")["door_detected"].mean())
print(merged.groupby("video")["door_detected"].agg(["sum", "mean"]))

merged.to_csv("merged_clusters_doors_gajdos.csv", index=False)