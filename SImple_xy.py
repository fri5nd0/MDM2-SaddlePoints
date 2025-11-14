import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torch
import os

# =============================================================
# Neural Network (NumPy + PyTorch hybrid for analysis)
# =============================================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights, self.biases = [], []
        
        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        
        # Hidden to hidden layers
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * np.sqrt(2.0 / hidden_sizes[i-1]))
            self.biases.append(np.zeros((1, hidden_sizes[i])))
        
        # Last hidden to output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2.0 / hidden_sizes[-1]))
        self.biases.append(np.zeros((1, output_size)))
        
        self.trajectory_data = []  # Store trajectory during training
    
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    def sigmoid_derivative(self, x): 
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        self.activations, self.z_values = [X], []
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.sigmoid(z))
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a = self.sigmoid(z)
        self.z_values.append(z)
        self.activations.append(a)
        return a
    
    def backward(self, X, y, output):
        m = X.shape[0]
        dZ = output - y
        dW = (1/m) * np.dot(self.activations[-2].T, dZ)
        db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        self.weights[-1] -= self.learning_rate * dW
        self.biases[-1] -= self.learning_rate * db
        for i in range(len(self.weights) - 2, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)  
            dZ = dA * self.sigmoid_derivative(self.z_values[i])
            dW = (1/m) * np.dot(self.activations[i].T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def compute_loss(self, y_true, y_pred):
        """Mean Squared Error (MSE) loss."""
        return np.mean((y_true - y_pred) ** 2)

    def get_parameters(self):
        params = []
        for w in self.weights:
            params.append(w.flatten())
        for b in self.biases:
            params.append(b.flatten())
        return np.concatenate(params)
    
    def set_parameters(self, params):
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            self.weights[i] = params[idx:idx + w_size].reshape(self.weights[i].shape)
            idx += w_size
        for i in range(len(self.biases)):
            b_size = self.biases[i].size
            self.biases[i] = params[idx:idx + b_size].reshape(self.biases[i].shape)
            idx += b_size

    def save_trajectory_snapshot(self, epoch, loss, grad_norm=None):
        """Save current parameters and training info to trajectory data"""
        params = self.get_parameters()
        snapshot = {
            'epoch': epoch,
            'loss': loss,
            'grad_norm': grad_norm if grad_norm is not None else np.nan,
            'parameters': params.tolist()
        }
        self.trajectory_data.append(snapshot)

    def compute_analytical_gradient(self, X, y):
        """Compute gradient using PyTorch autograd consistent with MSE loss."""
        X_t = torch.tensor(X, dtype=torch.float64, requires_grad=False)
        y_t = torch.tensor(y, dtype=torch.float64, requires_grad=False)
        params = self.get_parameters()
        theta = torch.tensor(params, dtype=torch.float64, requires_grad=True)

        # Unpack parameters as views into theta
        idx = 0
        weights, biases = [], []
        for w in self.weights:
            w_shape = w.shape
            w_size = int(np.prod(w_shape))
            weights.append(theta[idx:idx + w_size].view(w_shape))
            idx += w_size
        for b in self.biases:
            b_shape = b.shape
            b_size = int(np.prod(b_shape))
            biases.append(theta[idx:idx + b_size].view(b_shape))
            idx += b_size

        # Forward pass (same network) - using sigmoid for all layers
        a = X_t
        for i in range(len(weights) - 1):
            z = a @ weights[i] + biases[i]
            a = torch.sigmoid(z)
        z = a @ weights[-1] + biases[-1]
        output = torch.sigmoid(z)

        # Use MSE loss to match compute_loss()
        loss = torch.mean((y_t - output) ** 2)

        # Backprop to get gradient wrt theta
        loss.backward()
        grad = theta.grad.detach().cpu().numpy()
        return grad

    def compute_hessian_torch(self, X, y):
        """Compute the exact Hessian matrix using PyTorch autograd with higher precision."""
        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        params = self.get_parameters()
        theta = torch.tensor(params, dtype=torch.float64, requires_grad=True)

        # unpack parameters
        idx = 0
        weights, biases = [], []
        for w in self.weights:
            w_shape = w.shape
            w_size = int(np.prod(w_shape))
            weights.append(theta[idx:idx + w_size].view(w_shape))
            idx += w_size
        for b in self.biases:
            b_shape = b.shape
            b_size = int(np.prod(b_shape))
            biases.append(theta[idx:idx + b_size].view(b_shape))
            idx += b_size

        # forward pass - using sigmoid for all layers
        a = X_t
        for i in range(len(weights) - 1):
            z = a @ weights[i] + biases[i]
            a = torch.sigmoid(z)
        z = a @ weights[-1] + biases[-1]
        output = torch.sigmoid(z)
        loss = torch.mean((y_t - output) ** 2)

        # compute full Hessian
        grad = torch.autograd.grad(loss, theta, create_graph=True)[0]
        n_params = len(theta)
        H = torch.zeros((n_params, n_params), dtype=torch.float64)
        for i in range(n_params):
            grad2 = torch.autograd.grad(grad[i], theta, retain_graph=True)[0]
            H[i] = grad2
        return H.detach().cpu().numpy(), grad.detach().cpu().numpy()

    def find_saddle_points_newton(
        self,
        X, y,
        n_initial_points=50,
        max_iter=50,
        grad_tol=1e-6,
        duplicate_tol=1e-3,
        damping=1e-3
    ):
        """
        Find saddle points specifically using Newton-Raphson method.
        Returns only saddle points.
        """
        print(f"Searching for saddle points using Newton-Raphson ({n_initial_points} initializations)...")
        saddle_points = []
        n_params = len(self.get_parameters())
        original_params = self.get_parameters()

        # Generate random initial points
        init_points = []
        for _ in range(n_initial_points):
            # Vary initialization scale to explore different regions
            scale = np.random.choice([0.1, 0.5, 1.0, 2.0])
            init_points.append(np.random.randn(n_params) * scale)

        # Run Newton-Raphson search
        for i, current_params in enumerate(init_points):
            if i % 10 == 0:
                print(f"  Initialization {i}/{n_initial_points}")

            current_params = current_params.copy()

            for iteration in range(max_iter):
                self.set_parameters(current_params)
                H, grad = self.compute_hessian_torch(X, y)
                grad_norm = np.linalg.norm(grad)

                if grad_norm < grad_tol:
                    # Classify critical point
                    eigenvalues = np.linalg.eigvals(H)
                    
                    # Remove numerical noise near zero
                    max_abs_eigen = np.max(np.abs(eigenvalues))
                    if max_abs_eigen > 0:
                        pos_threshold = 1e-12 * max_abs_eigen
                        neg_threshold = -1e-12 * max_abs_eigen
                    else:
                        pos_threshold = 0
                        neg_threshold = 0
                    
                    pos = np.sum(eigenvalues > pos_threshold)
                    neg = np.sum(eigenvalues < neg_threshold)
                    
                    # Only save saddle points (both positive and negative eigenvalues)
                    if pos > 0 and neg > 0:
                        # Check duplicates
                        duplicate = False
                        for sp in saddle_points:
                            if np.linalg.norm(sp['parameters'] - current_params) < duplicate_tol:
                                duplicate = True
                                break
                        if duplicate:
                            break

                        loss = self.compute_loss_from_params(current_params, X, y)
                        saddle_points.append({
                            'parameters': current_params.copy(),
                            'min_eigen': np.min(eigenvalues).real,
                            'max_eigen': np.max(eigenvalues).real,
                            'positive_eigen': pos,
                            'negative_eigen': neg,
                            'grad_norm': grad_norm,
                            'loss': loss
                        })
                        print(f"  → Found saddle point at iter {iteration} (loss={loss:.6f}, ±eigen: {pos}/{neg})")
                    break

                # Newton-Raphson update (with damping)
                try:
                    step = np.linalg.solve(H + damping * np.eye(n_params), grad)
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(H + damping * np.eye(n_params), grad, rcond=None)[0]

                current_params -= step

        self.set_parameters(original_params)
        print(f"Found {len(saddle_points)} saddle points.")
        return saddle_points

    def find_all_critical_points_newton(
        self,
        X, y,
        n_initial_points=50,
        max_iter=50,
        grad_tol=1e-6,
        duplicate_tol=1e-3,
        damping=1e-3
    ):
        """
        Find all types of critical points using Newton-Raphson method.
        Returns critical points of all types (minima, maxima, saddles).
        """
        print(f"Finding all critical points using Newton-Raphson ({n_initial_points} initializations)...")
        critical_points = []
        n_params = len(self.get_parameters())
        original_params = self.get_parameters()

        # Generate random initial points
        init_points = []
        for _ in range(n_initial_points):
            scale = np.random.choice([0.1, 0.5, 1.0, 2.0])
            init_points.append(np.random.randn(n_params) * scale)

        # Run Newton-Raphson search
        for i, current_params in enumerate(init_points):
            if i % 10 == 0:
                print(f"  Initialization {i}/{n_initial_points}")

            current_params = current_params.copy()

            for iteration in range(max_iter):
                self.set_parameters(current_params)
                H, grad = self.compute_hessian_torch(X, y)
                grad_norm = np.linalg.norm(grad)

                if grad_norm < grad_tol:
                    # Classify critical point
                    eigenvalues = np.linalg.eigvals(H)
                    
                    max_abs_eigen = np.max(np.abs(eigenvalues))
                    if max_abs_eigen > 0:
                        pos_threshold = 1e-12 * max_abs_eigen
                        neg_threshold = -1e-12 * max_abs_eigen
                    else:
                        pos_threshold = 0
                        neg_threshold = 0
                    
                    pos = np.sum(eigenvalues > pos_threshold)
                    neg = np.sum(eigenvalues < neg_threshold)
                    zero_count = len(eigenvalues) - pos - neg
                    
                    # Classify the critical point
                    if pos > 0 and neg > 0:
                        point_type = "Saddle Point"
                    elif neg == 0 and pos > 0:
                        point_type = "Local Minimum"
                    elif pos == 0 and neg > 0:
                        point_type = "Local Maximum"
                    else:
                        point_type = "Degenerate Critical Point"

                    # Check duplicates
                    duplicate = False
                    for cp in critical_points:
                        if np.linalg.norm(cp['parameters'] - current_params) < duplicate_tol:
                            duplicate = True
                            break
                    if duplicate:
                        break

                    loss = self.compute_loss_from_params(current_params, X, y)
                    critical_points.append({
                        'parameters': current_params.copy(),
                        'type': point_type,
                        'min_eigen': np.min(eigenvalues).real,
                        'max_eigen': np.max(eigenvalues).real,
                        'positive_eigen': pos,
                        'negative_eigen': neg,
                        'zero_eigen': zero_count,
                        'grad_norm': grad_norm,
                        'loss': loss
                    })
                    print(f"  → Found {point_type} at iter {iteration} (loss={loss:.6f})")
                    break

                # Newton-Raphson update (with damping)
                try:
                    step = np.linalg.solve(H + damping * np.eye(n_params), grad)
                except np.linalg.LinAlgError:
                    step = np.linalg.lstsq(H + damping * np.eye(n_params), grad, rcond=None)[0]

                current_params -= step

        self.set_parameters(original_params)
        print(f"Found {len(critical_points)} critical points total.")
        return critical_points

    def find_and_initialize_near_saddle(self, X, y, n_attempts=10, offset_magnitude=0.01):
        """
        Find a saddle point and initialize network parameters near it.
        Uses only n_attempts random initial points for quick saddle finding.
        """
        print(f"Quick saddle search with {n_attempts} random initial points...")
        
        # Clean up any old trajectory files to avoid architecture mismatch
        if os.path.exists("training_trajectory.csv"):
            os.remove("training_trajectory.csv")
        
        # Find saddle points with limited attempts
        saddle_points = self.find_saddle_points_newton(
            X, y, 
            n_initial_points=n_attempts,  # Use only n_attempts points
            max_iter=30,
            grad_tol=1e-4
        )
        
        if not saddle_points:
            print("No saddle points found. Using random initialization.")
            return False
        
        # Choose the saddle point with highest loss (more interesting dynamics)
        saddle_points.sort(key=lambda x: x['loss'], reverse=True)
        chosen_saddle = saddle_points[0]
        saddle_params = chosen_saddle['parameters']
        
        print(f"Found saddle point with loss {chosen_saddle['loss']:.6f}")
        print(f"Eigenvalue range: [{chosen_saddle['min_eigen']:.2e}, {chosen_saddle['max_eigen']:.2e}]")
        print(f"Positive/Negative eigenvalues: {chosen_saddle['positive_eigen']}/{chosen_saddle['negative_eigen']}")
        
        # Add small random perturbation to escape exact saddle
        perturbation = np.random.randn(len(saddle_params)) * offset_magnitude
        near_saddle_params = saddle_params + perturbation
        
        # Set network parameters to this near-saddle point
        self.set_parameters(near_saddle_params)
        
        # Verify we're near the saddle
        current_loss = self.compute_loss_from_params(near_saddle_params, X, y)
        grad_norm = np.linalg.norm(self.compute_analytical_gradient(X, y))
        
        print(f"Initialized near saddle point:")
        print(f"  - Distance from saddle: {np.linalg.norm(perturbation):.2e}")
        print(f"  - Initial loss: {current_loss:.6f}")
        print(f"  - Initial gradient norm: {grad_norm:.2e}")
        
        return True

    def compute_loss_from_params(self, params, X, y):
        original_params = self.get_parameters()
        self.set_parameters(params)
        output = self.forward(X)
        loss = self.compute_loss(y, output)
        self.set_parameters(original_params)
        return loss

    def train(self, X, y, epochs=1000, verbose=True, save_trajectory=True, trajectory_interval=100):
        losses = []
        self.trajectory_data = []  # Reset trajectory data
        
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            losses.append(loss)
            self.backward(X, y, output)
            
            # Save trajectory at specified intervals and at the beginning/end
            if save_trajectory and (epoch % trajectory_interval == 0 or epoch == epochs-1 or epoch == 0):
                # Compute gradient norm for the snapshot
                grad = self.compute_analytical_gradient(X, y)
                grad_norm = np.linalg.norm(grad)
                self.save_trajectory_snapshot(epoch, loss, grad_norm)
            
            if verbose and epoch % 100 == 0:
                grad_norm = np.linalg.norm(self.compute_analytical_gradient(X, y))
                print(f"Epoch {epoch}, Loss: {loss:.6f}, Grad Norm: {grad_norm:.2e}")
        
        return losses

    def save_trajectory_to_csv(self, filename="training_trajectory.csv"):
        """Save the trajectory data to CSV file with consistent layer-wise parameter order"""
        if not self.trajectory_data:
            print("No trajectory data to save.")
            return

        df_data = []
        for snapshot in self.trajectory_data:
            row = {
                'epoch': snapshot['epoch'],
                'loss': snapshot['loss'],
                'grad_norm': snapshot['grad_norm']
            }

            # Save parameters in a strict layer-wise format: W0, b0, W1, b1, ...
            param_index = 0
            for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
                w_size = w.size
                b_size = b.size
                w_params = snapshot['parameters'][param_index:param_index + w_size]
                b_params = snapshot['parameters'][param_index + w_size:param_index + w_size + b_size]
                for i, val in enumerate(w_params):
                    row[f'W{layer_idx}_{i}'] = val
                for i, val in enumerate(b_params):
                    row[f'b{layer_idx}_{i}'] = val
                param_index += w_size + b_size

            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(filename, index=False)
        print(f"Saved training trajectory with {len(df)} snapshots to '{filename}'")
        return df

    def predict(self, X, threshold=0.5):
        return (self.forward(X) > threshold).astype(int)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# =============================================================
# Experiment Function
# =============================================================
def experiment(
    n_samples=1000, 
    radius=100, 
    max_radius=150, 
    hidden_sizes=[4, 2], 
    learning_rate=0.1, 
    epochs=500,
    find_critical_points=True,
    n_initial_points=50,  # This is now used for the comprehensive search AFTER training
    save_trajectory=True,
    trajectory_interval=100,
    start_near_saddle=True,
    initial_saddle_search_points=10  # NEW: Separate parameter for initial saddle search
):
    np.random.seed(42)
    points = [(np.random.uniform(-max_radius, max_radius), np.random.uniform(-max_radius, max_radius))
              for _ in range(n_samples)]
    X = np.array(points)
    y = np.array([1.0 if x**2 + y**2 <= radius**2 else 0.0 for (x, y) in points]).reshape(-1, 1)
    Xn = (X + max_radius) / (2 * max_radius)
    X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42, stratify=y)

    nn = NeuralNetwork(2, hidden_sizes, 1, learning_rate)
    print(f"Network architecture: 2 -> {hidden_sizes} -> 1")
    print(f"Total parameters: {len(nn.get_parameters())}")
    print(f"Training with radius={radius}, lr={learning_rate}, hidden={hidden_sizes}, epochs={epochs}")

    # PHASE 1: Quick saddle search with only initial_saddle_search_points attempts
    if start_near_saddle:
        success = nn.find_and_initialize_near_saddle(
            X_train, y_train, 
            n_attempts=initial_saddle_search_points,  # Use the small number for initial search
            offset_magnitude=0.01
        )
        if success:
            print("✓ Training will start near a saddle point")
        else:
            print("✗ Could not find saddle point, using random initialization")

    # PHASE 2: Train the network
    print("\n" + "="*50)
    print("STARTING TRAINING FROM SADDLE POINT")
    print("="*50)
    losses = nn.train(X_train, y_train, epochs=epochs, save_trajectory=save_trajectory, 
                     trajectory_interval=trajectory_interval)
    print(f"Final Train Acc: {nn.accuracy(X_train, y_train):.4f} | Test Acc: {nn.accuracy(X_test, y_test):.4f}")

    # PHASE 3: Comprehensive critical point search with main n_initial_points
    if find_critical_points:
        print("\n" + "="*50)
        print(f"COMPREHENSIVE CRITICAL POINT SEARCH ({n_initial_points} initial points)")
        print("="*50)
        
        critical_points = nn.find_all_critical_points_newton(
            X_train, y_train, 
            n_initial_points=n_initial_points,
            max_iter=50,
            grad_tol=1e-6
        )
        
        print(f"\nFound {len(critical_points)} unique critical points.")

        # Prepare records for saving
        records = []
        for cp in critical_points:
            records.append({
                'type': cp['type'],
                'min_eigen': cp['min_eigen'],
                'max_eigen': cp['max_eigen'],
                'positive_eigen': cp['positive_eigen'],
                'negative_eigen': cp['negative_eigen'], 
                'zero_eigen': cp['zero_eigen'],
                'grad_norm': cp['grad_norm'],
                'loss': cp['loss'],
                'parameters': cp['parameters'].tolist()
            })

        # Add final trained model as "Final Destination"
        final_params = nn.get_parameters()
        final_loss = nn.compute_loss_from_params(final_params, X_train, y_train)
        final_grad_norm = np.linalg.norm(nn.compute_analytical_gradient(X_train, y_train))
        
        records.append({
            'type': 'Final Destination',
            'min_eigen': np.nan,
            'max_eigen': np.nan,
            'positive_eigen': np.nan,
            'negative_eigen': np.nan,
            'zero_eigen': np.nan,
            'grad_norm': final_grad_norm,
            'loss': final_loss,
            'parameters': final_params.tolist()
        })

        # Save all to CSV
        df = pd.DataFrame(records)
        df.to_csv("critical_points.csv", index=False)
        print(f"\nSaved {len(records)} critical points (including final destination) to 'critical_points.csv'.")

        # Frequency bar chart
        freq = df['type'].value_counts()
        plt.figure(figsize=(8, 5))
        plt.bar(freq.index, freq.values, color='skyblue')
        plt.title("Frequency of Critical Point Types\n(After Training from Saddle)")
        plt.xlabel("Critical Point Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Loss Curve (Training from Saddle)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    
    # Plot gradient norm if available
    if nn.trajectory_data:
        epochs_traj = [s['epoch'] for s in nn.trajectory_data]
        grad_norms = [s['grad_norm'] for s in nn.trajectory_data]
        plt.subplot(1, 2, 2)
        plt.plot(epochs_traj, grad_norms)
        plt.title("Gradient Norm During Training")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

    # Save trajectory to CSV
    if save_trajectory:
        nn.save_trajectory_to_csv("training_trajectory.csv")

# =============================================================
# Run Example
# =============================================================
if __name__ == "__main__":
    print("\n=== Running Saddle Point Initialization Experiment ===")
    
    # You can change these hidden sizes freely now!
    hidden_sizes = [8, 8]  # Change this to any architecture you want
    
    experiment(
        n_samples=2000,
        radius=50,
        max_radius=100,
        hidden_sizes=hidden_sizes,
        learning_rate=0.05,
        epochs=30000,
        find_critical_points=True,  # Now enabled for comprehensive search
        n_initial_points=100,        # Used for comprehensive search AFTER training
        save_trajectory=True,
        trajectory_interval=10,
        start_near_saddle=True,
        initial_saddle_search_points=10  # Quick initial saddle search
    )