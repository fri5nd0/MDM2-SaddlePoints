import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq

# === Load critical points ===
df = pd.read_csv("critical_points.csv")

# Parse parameter arrays (stored as stringified lists)
param_list = [np.array(ast.literal_eval(p)) for p in df['parameters']]
param_matrix = np.vstack(param_list)

# === Compute covariance matrix ===
covariance_matrix = np.cov(param_matrix.T)

# === Find top 3 axes with largest covariance ===
variances = np.diag(covariance_matrix)
top3_idx = np.argsort(variances)[-3:][::-1]  # largest 3 variances

print("Top 3 axes (parameter indices):", top3_idx)

# === Project critical points onto top 3 dimensions ===
projected_points = param_matrix[:, top3_idx]

# === Prepare color mapping by critical point type ===
types = df['type'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
color_map = dict(zip(types, colors))

# === Plot in 3D ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for t in types:
    mask = df['type'] == t
    ax.scatter(
        projected_points[mask, 0],
        projected_points[mask, 1],
        projected_points[mask, 2],
        label=t,
        color=color_map[t],
        s=50,
        alpha=0.8
    )

ax.set_xlabel(f"Axis {top3_idx[0]} (largest variance)")
ax.set_ylabel(f"Axis {top3_idx[1]} (2nd largest)")
ax.set_zlabel(f"Axis {top3_idx[2]} (3rd largest)")
ax.set_title("3D Mapping of Critical Points in Loss Landscape")
ax.legend()
plt.tight_layout()
plt.show()
