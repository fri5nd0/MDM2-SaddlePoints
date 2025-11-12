import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torch

# =============================================================
# Neural Network (NumPy + PyTorch hybrid for analysis)
# =============================================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights, self.biases = [], []
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * np.sqrt(2.0 / hidden_sizes[i-1]))
            self.biases.append(np.zeros((1, hidden_sizes[i])))
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2.0 / hidden_sizes[-1]))
        self.biases.append(np.zeros((1, output_size)))
    
    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)
    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def forward(self, X):
        self.activations, self.z_values = [X], []
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))
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
            dZ = dA * self.relu_derivative(self.z_values[i])
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

    # =============================================================
    # PyTorch-accelerated gradient computation
    # =============================================================
    def compute_analytical_gradient(self, X, y):
        """Compute gradient using PyTorch autograd consistent with MSE loss."""
        X_t = torch.tensor(X, dtype=torch.float32, requires_grad=False)
        y_t = torch.tensor(y, dtype=torch.float32, requires_grad=False)
        params = self.get_parameters()
        theta = torch.tensor(params, dtype=torch.float32, requires_grad=True)

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

        # Forward pass (same network)
        a = X_t
        for i in range(len(weights) - 1):
            z = a @ weights[i] + biases[i]
            a = torch.relu(z)
        z = a @ weights[-1] + biases[-1]
        output = torch.sigmoid(z)

        # Use MSE loss to match compute_loss()
        loss = torch.mean((y_t - output) ** 2)

        # Backprop to get gradient wrt theta
        loss.backward()
        grad = theta.grad.detach().cpu().numpy()
        return grad


    # =============================================================
    # Hessian approximation and critical point analysis
    # =============================================================
    def compute_stochastic_hessian_eigenvalues(self, X, y, n_samples=100):
        """Approximate Hessian eigenvalues using random projections and central differences.

        Returns approximated eigenvalues of a small projected matrix which approximate
        the spectrum of the full Hessian.
        """
        original_params = self.get_parameters().copy()
        n_params = len(original_params)
        k = min(60, n_params)  # use more directions if possible for better coverage
        Hv_products = []

        eps = 1e-5
        for i in range(k):
            v = np.random.randn(n_params)
            v /= np.linalg.norm(v)

            # grad(theta + eps * v)
            self.set_parameters(original_params + eps * v)
            grad_plus = self.compute_analytical_gradient(X, y)

            # grad(theta - eps * v)
            self.set_parameters(original_params - eps * v)
            grad_minus = self.compute_analytical_gradient(X, y)

            # central difference for Hessian-vector product
            Hv = (grad_plus - grad_minus) / (2.0 * eps)
            Hv_products.append(Hv)

        # restore original params
        self.set_parameters(original_params)

        # Build Gram matrix of Hv vectors and compute small eigenvalues
        H_small = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                H_small[i, j] = np.dot(Hv_products[i], Hv_products[j])

        small_eigenvalues, _ = np.linalg.eig(H_small)
        # scale to approximate full Hessian eigenvalues (heuristic)
        scale_factor = np.sqrt(n_params / k)
        approx_eigenvalues = (small_eigenvalues * scale_factor).real
        return approx_eigenvalues


    def fast_analyze_critical_point(self, params, X, y, grad_tol=1e-6):
        gradient = self.compute_analytical_gradient(X, y)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > grad_tol:
            return None, None
        print("Found critical point! Computing approximate Hessian eigenvalues...")
        original_params = self.get_parameters()
        self.set_parameters(params)
        eigenvalues = self.compute_stochastic_hessian_eigenvalues(X, y)
        self.set_parameters(original_params)
        positive_ev = np.sum(eigenvalues > 1e-64)
        negative_ev = np.sum(eigenvalues < 1e-64)
        zero_ev = np.sum(np.abs(eigenvalues) < 1e-64)
        if negative_ev > 0 and positive_ev > 0:
            point_type = "Saddle Point"
        else:
            point_type = "Degenerate/Max/min Critical Point"
        print(f"Critical point type: {point_type}")
        return point_type, eigenvalues

    def find_critical_points_fast(self, X, y, n_initial_points=100, max_iter=200, grad_tol=1e-3, duplicate_tol=1e-4):
        print(f"Fast search for critical points using {n_initial_points} initializations...")
        critical_points = []
        n_params = len(self.get_parameters())
        original_params = self.get_parameters()

        for i in range(n_initial_points):
            if i % 10 == 0:
                print(f"Processing initial point {i}/{n_initial_points}")
            scale = np.random.choice([0.01, 0.1, 1.0, 10.0])
            current_params = np.random.randn(n_params) * scale

            for iteration in range(max_iter):
                self.set_parameters(current_params)
                grad = self.compute_analytical_gradient(X, y)
                grad_norm = np.linalg.norm(grad)
                if grad_norm < grad_tol:
                    # Check duplicates
                    duplicate = False
                    for cp in critical_points:
                        if np.linalg.norm(cp['parameters'] - current_params) < duplicate_tol:
                            duplicate = True
                            break
                    if duplicate:
                        break

                    loss = self.compute_loss_from_params(current_params, X, y)
                    point_type, eigenvalues = self.fast_analyze_critical_point(current_params, X, y, grad_tol)
                    if point_type:
                        critical_points.append({
                            'parameters': current_params.copy(),
                            'type': point_type,
                            'min_eigen': np.min(eigenvalues),
                            'max_eigen': np.max(eigenvalues),
                            'loss': loss,
                            'grad_norm': grad_norm
                        })
                    break
                current_params -= 0.1 * grad  # gradient descent
        self.set_parameters(original_params)
        return critical_points

    def compute_loss_from_params(self, params, X, y):
        original_params = self.get_parameters()
        self.set_parameters(params)
        output = self.forward(X)
        loss = self.compute_loss(y, output)
        self.set_parameters(original_params)
        return loss

    def train(self, X, y, epochs=1000, verbose=True):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            losses.append(loss)
            self.backward(X, y, output)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses

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
    n_initial_points=10
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

    losses = nn.train(X_train, y_train, epochs=epochs)
    print(f"Train Acc: {nn.accuracy(X_train, y_train):.4f} | Test Acc: {nn.accuracy(X_test, y_test):.4f}")

    if find_critical_points:
        critical_points = nn.find_critical_points_fast(X_train, y_train, n_initial_points=n_initial_points)
        print(f"\nFound {len(critical_points)} unique critical points.")

        records = [{
            'type': cp['type'],
            'min_eigen': cp['min_eigen'],
            'max_eigen': cp['max_eigen'],
            'grad_norm': cp['grad_norm'],
            'loss': cp['loss'],
            'parameters': cp['parameters'].tolist()
        } for cp in critical_points]

        # === Add final trained model parameters as "Final Destination" ===
        final_params = nn.get_parameters()
        final_loss = nn.compute_loss_from_params(final_params, X_train, y_train)
        records.append({
            'type': 'Final Destination',
            'min_eigen': np.nan,
            'max_eigen': np.nan,
            'grad_norm': np.nan,
            'loss': final_loss,
            'parameters': final_params.tolist()
        })

        # Save all to CSV
        df = pd.DataFrame(records)
        df.to_csv("critical_points.csv", index=False)
        print("\nSaved critical points + final destination to 'critical_points.csv'.")

        # === Frequency bar chart ===
        freq = df['type'].value_counts()
        plt.figure(figsize=(6,4))
        plt.bar(freq.index, freq.values, color='skyblue')
        plt.title("Frequency of Critical Point Types (Including Final Destination)")
        plt.xlabel("Critical Point Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

# =============================================================
# Run Example
# =============================================================
if __name__ == "__main__":
    print("\n=== Running Torch-Accelerated Critical Point Analysis (Unique Filtering) ===")
    experiment(
        n_samples=2000,
        radius=50,
        max_radius=100,
        hidden_sizes=[12,12],
        learning_rate=0.01,
        epochs=5500,
        find_critical_points=True,
        n_initial_points=200
    )
