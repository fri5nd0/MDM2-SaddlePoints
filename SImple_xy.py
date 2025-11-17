import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

# === Load data ===
df = pd.read_csv("critical_points.csv")
param_list = [np.array(ast.literal_eval(p)) for p in df["parameters"]]
param_matrix = np.vstack(param_list)

# Get actual loss values for critical points
critical_losses = df["loss"].values

trajectory_df = pd.read_csv("training_trajectory.csv")
param_cols = [c for c in trajectory_df.columns if c not in ("epoch", "loss", "grad_norm")]
trajectory_points = trajectory_df[param_cols].values
trajectory_losses = trajectory_df["loss"].values

# === Find closest saddles (using entire trajectory) ===
saddle_mask = df["type"].str.lower().str.contains("saddle")
saddle_points = param_matrix[saddle_mask]
saddle_losses = critical_losses[saddle_mask]

# Compute distances over entire trajectory
dists = cdist(saddle_points, trajectory_points)
min_dist_idx = np.argmin(dists, axis=1)
min_dist_val = np.min(dists, axis=1)

saddle_info = pd.DataFrame({
    "saddle_index": np.arange(len(saddle_points)),
    "closest_traj_step": min_dist_idx,
    "min_distance": min_dist_val,
    "saddle_loss": saddle_losses
})
closest_saddles = saddle_info.nsmallest(10, "min_distance")  # Top 10 closest

print("Top 10 closest saddles to trajectory (global search):")
print(closest_saddles)

# === 3D Visualization Functions ===

def create_saddle_3d_trajectory(saddle_params, saddle_loss, trajectory_points, trajectory_losses, closest_step, window=50, saddle_idx=0):
    """
    Create a 3D visualization of the trajectory navigating around a saddle point using actual loss
    
    Parameters:
    -----------
    window : int
        The number of trajectory steps to show before and after the closest point to the saddle.
        This creates a segment of length 2*window + 1 (if within bounds).
        A larger window shows more context but may include irrelevant parts of the trajectory.
        A smaller window focuses on the immediate navigation around the saddle point.
    """
    # Get local trajectory segment
    lo = max(0, closest_step - window)
    hi = min(len(trajectory_points), closest_step + window)
    local_traj = trajectory_points[lo:hi]
    local_losses = trajectory_losses[lo:hi]
    
    # Center trajectory around saddle
    traj_centered = local_traj - saddle_params
    
    # Use PCA on local trajectory to find meaningful directions
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    traj_proj = pca.fit_transform(traj_centered)
    
    # Create separate figure window
    fig = plt.figure(figsize=(10, 8))
    
    # Plot 3D trajectory with actual loss coloring
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by actual loss values
    scatter = ax.scatter(traj_proj[:, 0], traj_proj[:, 1], traj_proj[:, 2], 
                         c=local_losses, cmap='viridis', s=40, alpha=0.8, 
                         label='Trajectory Points')
    
    # Add colorbar for loss values
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Actual Loss Value', rotation=270, labelpad=15)
    
    # Plot lines connecting points
    ax.plot(traj_proj[:, 0], traj_proj[:, 1], traj_proj[:, 2], 
             'gray', alpha=0.3, linewidth=1, label='Trajectory Path')
    
    # Mark key points
    ax.scatter(0, 0, 0, color='red', s=200, marker='*', 
               label=f'Saddle (loss: {saddle_loss:.4f})', edgecolors='black', linewidth=2)
    ax.scatter(traj_proj[0, 0], traj_proj[0, 1], traj_proj[0, 2], 
                color='green', s=120, marker='s', 
                label='Segment Start', edgecolors='black')
    ax.scatter(traj_proj[-1, 0], traj_proj[-1, 1], traj_proj[-1, 2], 
                color='orange', s=120, marker='^', 
                label='Segment End', edgecolors='black')
    
    ax.set_xlabel('PC1 (Maximum Variance Direction)')
    ax.set_ylabel('PC2 (Second Maximum Variance)')
    ax.set_zlabel('PC3 (Third Maximum Variance)')
    ax.set_title(f'Saddle {saddle_idx}: 3D Trajectory Navigation\n'
                f'Saddle Loss: {saddle_loss:.4f}, Min Distance: {np.min(np.linalg.norm(traj_centered, axis=1)):.2e}\n'
                f'Window: {window} steps around closest point', 
                fontsize=12, pad=20)
    
    # Enhanced legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.9)
    
    # Add explanatory text
    explanation = (f"Window Explanation:\n"
                  f"• Shows {window} steps before & after closest point\n"
                  f"• Total segment: {len(local_traj)} steps\n"
                  f"• Colors: Actual loss values\n"
                  f"• Red Star: Saddle point\n" 
                  f"• Green Square: Segment start\n"
                  f"• Orange Triangle: Segment end")
    
    ax.text2D(0.02, 0.02, explanation, transform=ax.transAxes, 
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
              fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_velocity_vectors(saddle_params, saddle_loss, trajectory_points, trajectory_losses, closest_step, window=50, saddle_idx=0):
    """
    Create velocity vector visualization with actual loss coloring
    
    Parameters:
    -----------
    window : int
        The number of trajectory steps to show before and after the closest point to the saddle.
        This creates a focused view of how the trajectory moves around the saddle point.
    """
    # Get local trajectory segment
    lo = max(0, closest_step - window)
    hi = min(len(trajectory_points), closest_step + window)
    local_traj = trajectory_points[lo:hi]
    local_losses = trajectory_losses[lo:hi]
    
    # Center trajectory around saddle
    traj_centered = local_traj - saddle_params
    
    # Use PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    traj_proj = pca.fit_transform(traj_centered)
    
    # Create separate figure window
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate velocity vectors
    velocities = np.diff(traj_proj, axis=0)
    positions = traj_proj[:-1]
    position_losses = local_losses[:-1]  # Loss at start of each velocity vector
    
    # Normalize velocities for consistent arrow sizes
    vel_magnitudes = np.linalg.norm(velocities, axis=1)
    vel_normalized = velocities / vel_magnitudes[:, np.newaxis] * 0.15  # Scale factor
    
    # Plot trajectory with color indicating actual loss
    trajectory_scatter = ax.scatter(traj_proj[:, 0], traj_proj[:, 1], traj_proj[:, 2], 
                                   c=local_losses, cmap='viridis', s=30, alpha=0.7, 
                                   label='Trajectory Points')
    
    # Add colorbar for loss values
    cbar = plt.colorbar(trajectory_scatter, ax=ax, pad=0.1)
    cbar.set_label('Actual Loss Value', rotation=270, labelpad=15)
    
    # Plot velocity vectors (every 2nd point for clarity)
    for i in range(0, len(positions), 2):
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                  vel_normalized[i, 0], vel_normalized[i, 1], vel_normalized[i, 2],
                  color='red', alpha=0.8, linewidth=1.5, 
                  arrow_length_ratio=0.4, label='Velocity Vector' if i == 0 else "")
    
    # Mark key points
    ax.scatter(0, 0, 0, color='darkred', s=250, marker='*', 
               label=f'Saddle (loss: {saddle_loss:.4f})', edgecolors='black')
    ax.scatter(traj_proj[0, 0], traj_proj[0, 1], traj_proj[0, 2], 
                color='darkgreen', s=120, marker='s', 
                label='Start', edgecolors='black')
    ax.scatter(traj_proj[-1, 0], traj_proj[-1, 1], traj_proj[-1, 2], 
                color='darkorange', s=120, marker='^', 
                label='End', edgecolors='black')
    
    ax.set_xlabel('PC1 Direction')
    ax.set_ylabel('PC2 Direction')
    ax.set_zlabel('PC3 Direction')
    ax.set_title(f'Saddle {saddle_idx}: Velocity Vector Field\n'
                f'Saddle Loss: {saddle_loss:.4f}, Window: {window} steps', 
                fontsize=12, pad=20)
    
    # Enhanced legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.9)
    
    # Add speed information
    avg_speed = np.mean(vel_magnitudes)
    max_speed = np.max(vel_magnitudes)
    min_speed = np.min(vel_magnitudes)
    
    speed_info = (f"Window: {window} steps around closest point\n"
                  f"Speed Statistics:\n"
                  f"• Average: {avg_speed:.2e}\n"
                  f"• Maximum: {max_speed:.2e}\n"
                  f"• Minimum: {min_speed:.2e}\n\n"
                  f"Arrow length indicates relative speed\n"
                  f"Colors show actual loss values")
    
    ax.text2D(0.02, 0.02, speed_info, transform=ax.transAxes,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
              fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_distance_plot(saddle_params, saddle_loss, trajectory_points, trajectory_losses, closest_step, window=50, saddle_idx=0):
    """
    Create distance from saddle plot with actual loss overlay
    
    Parameters:
    -----------
    window : int
        The number of trajectory steps to show before and after the closest point to the saddle.
        This defines the analysis window for studying how the trajectory approaches and leaves the saddle.
    """
    # Get local trajectory segment
    lo = max(0, closest_step - window)
    hi = min(len(trajectory_points), closest_step + window)
    local_traj = trajectory_points[lo:hi]
    local_losses = trajectory_losses[lo:hi]
    
    # Center trajectory around saddle
    traj_centered = local_traj - saddle_params
    
    # Calculate distances and get loss values
    distances = np.linalg.norm(traj_centered, axis=1)
    steps = np.arange(len(distances))
    
    # Create separate figure window
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Distance over time
    ax1.plot(steps, distances, 'b-', linewidth=2.5, label='Distance from saddle')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, 
                label='Saddle position (zero)')
    
    # Mark the closest point
    closest_local_idx = np.argmin(distances)
    ax1.scatter(closest_local_idx, distances[closest_local_idx], 
                color='red', s=150, zorder=5, label=f'Closest point (step {closest_local_idx})')
    
    # Mark start and end
    ax1.scatter(0, distances[0], color='green', s=100, marker='s', 
                label='Segment start')
    ax1.scatter(len(distances)-1, distances[-1], color='orange', s=100, marker='^', 
                label='Segment end')
    
    ax1.set_xlabel('Step in Local Trajectory')
    ax1.set_ylabel('Distance from Saddle')
    ax1.set_title(f'Saddle {saddle_idx}: Distance from Saddle Over Time\n'
                 f'Saddle Loss: {saddle_loss:.4f}, Window: {window} steps', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actual loss over time
    ax2.plot(steps, local_losses, 'purple', linewidth=2.5, label='Actual Loss')
    ax2.axhline(y=saddle_loss, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'Saddle loss: {saddle_loss:.4f}')
    
    # Mark key points on loss plot
    ax2.scatter(closest_local_idx, local_losses[closest_local_idx], 
                color='red', s=150, zorder=5, label=f'Closest point loss: {local_losses[closest_local_idx]:.4f}')
    ax2.scatter(0, local_losses[0], color='green', s=100, marker='s', 
                label=f'Start loss: {local_losses[0]:.4f}')
    ax2.scatter(len(local_losses)-1, local_losses[-1], color='orange', s=100, marker='^', 
                label=f'End loss: {local_losses[-1]:.4f}')
    
    ax2.set_xlabel('Step in Local Trajectory')
    ax2.set_ylabel('Actual Loss')
    ax2.set_title('Actual Loss Values Along Trajectory', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Speed (derivative of distance) - LOCAL calculation
    if len(distances) > 1:
        speed = np.abs(np.diff(distances))
        speed_steps = steps[:-1] + 0.5  # Center between distance points
        
        ax3.plot(speed_steps, speed, 'teal', linewidth=2, label='Absolute speed')
        ax3.axhline(y=np.mean(speed), color='teal', linestyle='--', alpha=0.7,
                   label=f'Average speed: {np.mean(speed):.2e}')
        
        # Mark slow regions
        low_speed_threshold = np.mean(speed) * 0.5
        low_speed_regions = speed < low_speed_threshold
        
        start_idx = None
        slow_region_label_added = False
        
        for i in range(1, len(low_speed_regions)):
            if low_speed_regions[i] and not low_speed_regions[i-1]:
                start_idx = i
            elif not low_speed_regions[i] and low_speed_regions[i-1] and start_idx is not None:
                label = 'Slow regions' if not slow_region_label_added else ""
                ax3.axvspan(speed_steps[start_idx], speed_steps[i], 
                           alpha=0.2, color='red', label=label)
                if not slow_region_label_added:
                    slow_region_label_added = True
                start_idx = None
        
        if start_idx is not None and low_speed_regions[-1]:
            label = 'Slow regions' if not slow_region_label_added else ""
            ax3.axvspan(speed_steps[start_idx], speed_steps[-1], 
                       alpha=0.2, color='red', label=label)
    
    ax3.set_xlabel('Step in Local Trajectory')
    ax3.set_ylabel('Speed (|Δdistance|)')
    ax3.set_title('Movement Speed Around Saddle (Local Calculation)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_energy_landscape_3d(saddle_params, saddle_loss, trajectory_points, trajectory_losses, closest_step, window=30, saddle_idx=0):
    """
    Create a 3D energy landscape visualization using actual loss values
    
    Parameters:
    -----------
    window : int
        The number of trajectory steps to show before and after the closest point to the saddle.
        Used to create a focused view of the loss landscape around the saddle region.
    """
    # Get local trajectory
    lo = max(0, closest_step - window)
    hi = min(len(trajectory_points), closest_step + window)
    local_traj = trajectory_points[lo:hi]
    local_losses = trajectory_losses[lo:hi]
    
    # Use PCA to find the two most significant directions
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    traj_centered = local_traj - saddle_params
    traj_2d = pca.fit_transform(traj_centered)
    
    # Create grid for actual loss interpolation
    from scipy.interpolate import griddata
    n_points = 25
    
    # Create grid based on trajectory extent
    x_range = np.linspace(traj_2d[:, 0].min(), traj_2d[:, 0].max(), n_points)
    y_range = np.linspace(traj_2d[:, 1].min(), traj_2d[:, 1].max(), n_points)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Interpolate actual loss values on the grid
    points = traj_2d
    values = local_losses
    Z_grid = griddata(points, values, (X_grid, Y_grid), method='cubic', fill_value=np.nan)
    
    # Create separate figure window
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: 3D energy landscape with actual loss
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot interpolated loss surface
    surf = ax1.plot_surface(X_grid, Y_grid, Z_grid, cmap='coolwarm', 
                           alpha=0.7, linewidth=0, antialiased=True)
    
    # Add colorbar for actual loss
    cbar1 = plt.colorbar(surf, ax=ax1, pad=0.1)
    cbar1.set_label('Actual Loss Value', rotation=270, labelpad=15)
    
    # Plot trajectory points with their actual loss
    scatter = ax1.scatter(traj_2d[:, 0], traj_2d[:, 1], local_losses,
                         c=local_losses, cmap='viridis', s=50, alpha=1.0,
                         edgecolors='black', linewidth=0.5, label='Trajectory Points')
    
    # Mark key points
    ax1.scatter(traj_2d[0, 0], traj_2d[0, 1], local_losses[0], 
                color='lime', s=120, marker='s', edgecolors='black',
                label=f'Start (loss: {local_losses[0]:.4f})')
    ax1.scatter(traj_2d[-1, 0], traj_2d[-1, 1], local_losses[-1], 
                color='yellow', s=120, marker='^', edgecolors='black',
                label=f'End (loss: {local_losses[-1]:.4f})')
    ax1.scatter(0, 0, saddle_loss, color='red', s=200, marker='*', 
                edgecolors='black', linewidth=2, label=f'Saddle (loss: {saddle_loss:.4f})')
    
    ax1.set_xlabel('PC1 Direction')
    ax1.set_ylabel('PC2 Direction')
    ax1.set_zlabel('Actual Loss')
    ax1.set_title(f'Saddle {saddle_idx}: 3D Loss Landscape\n(Window: {window} steps around saddle)', 
                  fontsize=11)
    ax1.legend()
    
    # Plot 2: Contour plot with actual loss
    ax2 = fig.add_subplot(122)
    
    # Create contour plot of interpolated loss
    contour = ax2.contour(X_grid, Y_grid, Z_grid, levels=15, cmap='coolwarm', alpha=0.8)
    cbar2 = plt.colorbar(contour, ax=ax2)
    cbar2.set_label('Actual Loss Value')
    
    # Plot trajectory on contour
    trajectory_line = ax2.plot(traj_2d[:, 0], traj_2d[:, 1], 'o-', color='black', 
             linewidth=2, markersize=5, label='Trajectory')[0]
    
    # Color trajectory points by actual loss
    scatter2 = ax2.scatter(traj_2d[:, 0], traj_2d[:, 1], 
                         c=local_losses, cmap='viridis', 
                         s=60, alpha=1.0, zorder=5, edgecolors='black')
    
    # Mark key points
    ax2.scatter(traj_2d[0, 0], traj_2d[0, 1], color='lime', s=120, 
                marker='s', edgecolors='black', label='Start')
    ax2.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color='yellow', s=120, 
                marker='^', edgecolors='black', label='End')
    ax2.scatter(0, 0, color='red', s=150, marker='*', 
                edgecolors='black', label='Saddle')
    
    ax2.set_xlabel('PC1 Direction')
    ax2.set_ylabel('PC2 Direction')
    ax2.set_title('Contour View with Trajectory\n(Colors show actual loss values)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# === Global distance analysis ===
print("=== Global Distance Analysis ===")
print("Finding closest saddles across entire trajectory...")

# Create global distance plot for all top saddles
plt.figure(figsize=(14, 8))
for i, (idx, row) in enumerate(closest_saddles.iterrows()):
    saddle_idx = int(row["saddle_index"])
    saddle_params = saddle_points[saddle_idx]
    saddle_loss = row["saddle_loss"]
    
    # Compute distance over entire trajectory
    traj_centered_global = trajectory_points - saddle_params
    distances_global = np.linalg.norm(traj_centered_global, axis=1)
    
    plt.plot(distances_global, label=f'Saddle {saddle_idx} (loss: {saddle_loss:.4f})', 
             alpha=0.8, linewidth=2)

plt.xlabel('Training Step (Entire Trajectory)')
plt.ylabel('Distance from Saddle')
plt.title('Global Distance Analysis: All Top 10 Saddles\n(Distance computed over entire trajectory)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Main visualization execution ===
print("Creating 3D visualizations of saddle navigation using ACTUAL LOSS values...")
print("Key Changes:")
print("• Closest approach: Computed over ENTIRE trajectory (global)")
print("• Speed analysis: Computed LOCALLY around closest point")
print("• Window Parameter:")
print("  - 'window' defines local analysis region around closest point")
print("  - Larger window = more context but potentially less focused")
print("  - Smaller window = focused view of immediate saddle navigation")
print("  - Default: 50 steps for trajectory views, 30 steps for landscape views")
print()

# Generate visualizations for each closest saddle
for i, (idx, row) in enumerate(closest_saddles.iterrows()):
    saddle_idx = int(row["saddle_index"])
    closest_step = int(row["closest_traj_step"])  # This is GLOBAL closest step
    saddle_params = saddle_points[saddle_idx]
    saddle_loss = row["saddle_loss"]
    
    print(f"\n=== Creating Visualizations for Saddle {saddle_idx} (Rank {i+1}) ===")
    print(f"Saddle Loss: {saddle_loss:.6f}")
    print(f"Global closest step: {closest_step}")
    print(f"Global minimum distance: {row['min_distance']:.2e}")
    
    # Each visualization in separate window
    print("1. Creating 3D trajectory plot with actual loss...")
    create_saddle_3d_trajectory(saddle_params, saddle_loss, trajectory_points, trajectory_losses, closest_step, 
                               window=50, saddle_idx=saddle_idx)
    
    print("2. Creating velocity vector plot with actual loss...")
    create_velocity_vectors(saddle_params, saddle_loss, trajectory_points, trajectory_losses, closest_step, 
                           window=50, saddle_idx=saddle_idx)
    
    print("3. Creating distance analysis plot with actual loss...")
    create_distance_plot(saddle_params, saddle_loss, trajectory_points, trajectory_losses, closest_step, 
                        window=50, saddle_idx=saddle_idx)
    
    print("4. Creating energy landscape plot with actual loss...")
    create_energy_landscape_3d(saddle_params, saddle_loss, trajectory_points, trajectory_losses, closest_step, 
                              window=30, saddle_idx=saddle_idx)
    
    print(f"✓ Completed all visualizations for Saddle {saddle_idx} (Rank {i+1})\n")

print("=== All visualizations completed ===")
print("Summary:")
print(f"• Analyzed top 10 closest saddles to the training trajectory")
print("• Closest approach computed GLOBALLY across entire trajectory")
print("• Speed analysis computed LOCALLY around closest point")
print("• Each visualization uses ACTUAL LOSS VALUES from critical points and trajectory data")
print("• Different window sizes used for different visualization types (50 for trajectories, 30 for landscapes)")