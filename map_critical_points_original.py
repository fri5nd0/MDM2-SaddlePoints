import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

# === Load critical points ===
df = pd.read_csv("critical_points.csv")
param_list = [np.array(ast.literal_eval(p)) for p in df["parameters"]]
param_matrix = np.vstack(param_list)

# === Load and process training trajectory ===
trajectory_df = pd.read_csv("training_trajectory.csv")

# --- Auto-detect parameter columns ---
param_cols = [
    c for c in trajectory_df.columns
    if (
        c.startswith("param_")
        or "_" in c
        or c.isdigit()
    )
    and c not in ("epoch", "loss", "grad_norm")
]

if len(param_cols) == 0:
    raise ValueError("No parameter columns found in training_trajectory.csv.")

print(f"Detected {len(param_cols)} parameter columns in trajectory CSV.")

# --- Extract parameter values ---
trajectory_points = trajectory_df[param_cols].values

if trajectory_points.shape[1] != param_matrix.shape[1]:
    print("Warning: parameter count mismatch between trajectory and critical points!")
    print(f"Trajectory columns: {trajectory_points.shape[1]} | Critical points params: {param_matrix.shape[1]}")

# === Compute covariance and principal axes from trajectory ===
covariance_matrix = np.cov(trajectory_points.T)
variances = np.diag(covariance_matrix)
top3_idx = np.argsort(variances)[-3:][::-1]  # largest → smallest variance

# === Project points using trajectory-based axes ===
projected_trajectory = trajectory_points[:, top3_idx]
projected_points = param_matrix[:, top3_idx]

# === Identify saddle points and compute distances ===
saddle_mask = df["type"].str.lower().str.contains("saddle")
saddle_points = param_matrix[saddle_mask]

if len(saddle_points) == 0:
    raise ValueError("No saddle points detected in critical_points.csv!")

# Compute min distance from each saddle to trajectory
dists = cdist(saddle_points, trajectory_points)
min_dist_idx = np.argmin(dists, axis=1)
min_dist_val = np.min(dists, axis=1)

# Collect summary
saddle_info = pd.DataFrame({
    "saddle_index": np.arange(len(saddle_points)),
    "closest_traj_step": min_dist_idx,
    "min_distance": min_dist_val,
})
closest_saddles = saddle_info.nsmallest(10, "min_distance")  # Top 10 closest

print("\n=== Top 10 Closest Saddle Points to Trajectory ===")
print(closest_saddles)

# === Plot setup (3D view) ===
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# === Critical points (scatter) ===
types = df["type"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
color_map = dict(zip(types, colors))

for t in types:
    mask = df["type"] == t
    ax.scatter(
        projected_points[mask, 0],
        projected_points[mask, 1],
        projected_points[mask, 2],
        label=t,
        color=color_map[t],
        s=70,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.4,
    )

# === Trajectory (line) ===
ax.plot(
    projected_trajectory[:, 0],
    projected_trajectory[:, 1],
    projected_trajectory[:, 2],
    color="orange",
    linewidth=2.5,
    alpha=0.95,
    label="Training Trajectory",
)

# Mark start and end
ax.scatter(*projected_trajectory[0, :3], color="black", s=120, marker="o", label="Start")
ax.scatter(*projected_trajectory[-1, :3], color="lime", s=120, marker="X", label="End")

# === Highlight closest saddle points with red stars ===
projected_saddles = saddle_points[:, top3_idx]
highlighted = projected_saddles[closest_saddles["saddle_index"].values]

ax.scatter(
    highlighted[:, 0],
    highlighted[:, 1],
    highlighted[:, 2],
    color="red",
    s=200,  # Increased size for better visibility
    edgecolors="black",
    linewidths=1.5,
    marker="*",  # Changed to star marker
    label="Top 10 Closest Saddles",
)

# === Labels & aesthetics ===
ax.set_xlabel(f"Axis {top3_idx[0]} (largest variance)", fontsize=12)
ax.set_ylabel(f"Axis {top3_idx[1]} (2nd largest)", fontsize=12)
ax.set_zlabel(f"Axis {top3_idx[2]} (3rd largest)", fontsize=12)
ax.set_title("3D Landscape: Trajectory and Top 10 Closest Saddle Points", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# === Zoom around trajectory region ===
traj_max_range = np.ptp(projected_trajectory, axis=0).max() / 2
traj_center = projected_trajectory.mean(axis=0)
zoom_factor = 1.3
ax.set_xlim(traj_center[0] - zoom_factor * traj_max_range, traj_center[0] + zoom_factor * traj_max_range)
ax.set_ylim(traj_center[1] - zoom_factor * traj_max_range, traj_center[1] + zoom_factor * traj_max_range)
ax.set_zlim(traj_center[2] - zoom_factor * traj_max_range, traj_center[2] + zoom_factor * traj_max_range)

plt.tight_layout()
plt.show(block=True)

# === Additional summary statistics ===
print(f"\n=== Summary Statistics for Top 10 Closest Saddles ===")
print(f"Average distance: {closest_saddles['min_distance'].mean():.2e}")
print(f"Minimum distance: {closest_saddles['min_distance'].min():.2e}")
print(f"Maximum distance: {closest_saddles['min_distance'].max():.2e}")
print(f"Median distance: {closest_saddles['min_distance'].median():.2e}")

# Plot distance distribution
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), closest_saddles['min_distance'].values, 
        color='skyblue', edgecolor='navy', alpha=0.7)
plt.xlabel('Saddle Rank (1 = Closest)')
plt.ylabel('Minimum Distance to Trajectory')
plt.title('Distance Distribution of Top 10 Closest Saddles')
plt.grid(True, alpha=0.3)
for i, v in enumerate(closest_saddles['min_distance'].values):
    plt.text(i+1, v, f'{v:.1e}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()