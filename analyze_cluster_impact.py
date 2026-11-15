import pandas as pd

MERGED_CSV = "merged_clusters_doors_gajdos.csv"   

df = pd.read_csv(MERGED_CSV)

print(f"Total matched frames: {len(df)}")
print(f"Overall door rate: {df['door_detected'].mean():.2%}\n")

# --- 1. Door rate per cluster ---
by_cluster = df.groupby("cluster")["door_detected"].agg(["count", "sum", "mean"])
by_cluster.columns = ["frames", "door_frames", "door_rate"]
by_cluster = by_cluster.sort_values("door_rate", ascending=False)
print("Door rate by cluster:")
print(by_cluster, "\n")

# --- 2. How concentrated are doors in a few clusters? ---
# If clustering is useful, a small number of clusters should hold most of the doors.
by_cluster_sorted = by_cluster.sort_values("door_frames", ascending=False)
by_cluster_sorted["cum_door_share"] = by_cluster_sorted["door_frames"].cumsum() / by_cluster_sorted["door_frames"].sum()
print("Cumulative share of all door-detections, by cluster rank:")
print(by_cluster_sorted[["door_frames", "cum_door_share"]], "\n")

# --- 3. Compute savings if you only checked 1 frame per (video, cluster) ---
sampled = df.groupby(["video", "cluster"]).first().reset_index()
full_door_frames = df["door_detected"].sum()
sampled_clusters_with_door = sampled["door_detected"].sum()
clusters_found_vs_frames_checked = (
    f"{len(sampled)} cluster-samples checked vs {len(df)} total frames "
    f"({len(sampled)/len(df):.1%} of the work)"
)
print("If you'd only run GroundingDINO once per (video, cluster) instead of every sampled frame:")
print(clusters_found_vs_frames_checked)
print(f"Clusters flagged as door-containing this way: {sampled_clusters_with_door} "
      f"out of {sampled['cluster'].nunique()} distinct clusters checked")
print(f"(compare to: {full_door_frames} individual frames actually had a door out of {len(df)})")