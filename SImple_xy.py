import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def get_parameters(self):
        """Get all parameters as a flat vector"""
        params = []
        for w in self.weights:
            params.append(w.flatten())
        for b in self.biases:
            params.append(b.flatten())
        return np.concatenate(params)
    
    def set_parameters(self, params):
        """Set all parameters from a flat vector"""
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            self.weights[i] = params[idx:idx + w_size].reshape(self.weights[i].shape)
            idx += w_size
        for i in range(len(self.biases)):
            b_size = self.biases[i].size
            self.biases[i] = params[idx:idx + b_size].reshape(self.biases[i].shape)
            idx += b_size

    def compute_hessian(self, X, y, eps=1e-5):
        """Compute Hessian matrix using finite differences"""
        original_params = self.get_parameters()
        n_params = len(original_params)
        H = np.zeros((n_params, n_params))
        
        print(f"Computing Hessian for {n_params} parameters...")
        
        # Compute gradient at current point
        grad_0 = self.compute_gradient(X, y)
        
        for i in range(n_params):
            if i % 10 == 0:  # Progress indicator
                print(f"Computing column {i}/{n_params}")
                
            # Perturb parameter i
            params_plus = original_params.copy()
            params_plus[i] += eps
            self.set_parameters(params_plus)
            grad_plus = self.compute_gradient(X, y)
            
            # Central difference for better accuracy
            if i < n_params - 1:
                params_minus = original_params.copy()
                params_minus[i] -= eps
                self.set_parameters(params_minus)
                grad_minus = self.compute_gradient(X, y)
                H[:, i] = (grad_plus - grad_minus) / (2 * eps)
            else:
                H[:, i] = (grad_plus - grad_0) / eps
        
        # Restore original parameters
        self.set_parameters(original_params)
        
        # Make symmetric (finite differences can cause small asymmetries)
        H = 0.5 * (H + H.T)
        return H

    def compute_gradient(self, X, y):
        """Compute gradient of loss with respect to all parameters"""
        # Store original state
        original_weights = [w.copy() for w in self.weights]
        original_biases = [b.copy() for b in self.biases]
        
        # Forward pass
        output = self.forward(X)
        loss = self.compute_loss(y, output)
        
        # Backward pass to compute gradients
        m = X.shape[0]
        gradients = []
        
        # Output layer gradients
        dZ = output - y
        dW = (1/m) * np.dot(self.activations[-2].T, dZ)
        db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        gradients.append(dW.flatten())
        gradients.append(db.flatten())
        
        # Hidden layers gradients
        for i in range(len(self.weights) - 2, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)  
            dZ = dA * self.relu_derivative(self.z_values[i])
            dW = (1/m) * np.dot(self.activations[i].T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            gradients.append(dW.flatten())
            gradients.append(db.flatten())
        
        # Reverse to get correct order (input to output)
        gradients = gradients[::-1]
        
        # Restore original state
        self.weights = original_weights
        self.biases = original_biases
        
        return np.concatenate(gradients)

    def train(self, X, y, epochs=1000, verbose=True, compute_hessian_epochs=None):
        losses = []
        hessian_eigenvalues = []
        
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            losses.append(loss)
            self.backward(X, y, output)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
            # Compute Hessian at specified epochs
            if compute_hessian_epochs and epoch in compute_hessian_epochs:
                print(f"Computing Hessian at epoch {epoch}...")
                H = self.compute_hessian(X, y)
                eigenvalues = np.linalg.eigvals(H)
                hessian_eigenvalues.append((epoch, eigenvalues))
                print(f"Hessian eigenvalues at epoch {epoch}:")
                print(f"  Min: {np.min(eigenvalues):.6f}")
                print(f"  Max: {np.max(eigenvalues):.6f}")
                print(f"  Positive: {np.sum(eigenvalues > 0)}")
                print(f"  Negative: {np.sum(eigenvalues < 0)}")
                print(f"  Zero: {np.sum(np.abs(eigenvalues) < 1e-10)}")
        
        return losses, hessian_eigenvalues
    
    def predict(self, X, threshold=0.5):
        return (self.forward(X) > threshold).astype(int)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

# === INTERACTIVE FUNCTION ===
def experiment(
    n_samples=3000, 
    radius=100, 
    max_radius=150, 
    hidden_sizes=[8, 4], 
    learning_rate=0.1, 
    epochs=1000,
    compute_hessian=False
):
    np.random.seed(42)
    
    # Generate points in Cartesian coordinates
    points = []
    for _ in range(n_samples):
        x = np.random.uniform(-max_radius, max_radius)
        y = np.random.uniform(-max_radius, max_radius)
        points.append((x, y))
    
    X = np.array(points)
    y = np.array([1.0 if x**2 + y**2 <= radius**2 else 0.0 for (x, y) in points]).reshape(-1, 1)

    # Normalize to [0, 1] range
    Xn = X.copy()
    Xn = (Xn + max_radius) / (2 * max_radius)
    
    X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42, stratify=y)
    X_train_orig, X_test_orig = X_train * (2 * max_radius) - max_radius, X_test * (2 * max_radius) - max_radius
    
    nn = NeuralNetwork(2, hidden_sizes, 1, learning_rate)
    print(f"Training with radius={radius}, lr={learning_rate}, hidden={hidden_sizes}, epochs={epochs}")
    
    if compute_hessian:
        # Compute Hessian at specific epochs
        hessian_epochs = [0, epochs//4, epochs//2, epochs-1]
        losses, hessian_eigenvalues = nn.train(X_train, y_train, epochs=epochs, 
                                             compute_hessian_epochs=hessian_epochs)
        
        # Plot eigenvalue distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (epoch, eigenvalues) in enumerate(hessian_eigenvalues):
            axes[idx].hist(eigenvalues.real, bins=50, alpha=0.7)
            axes[idx].set_title(f'Epoch {epoch}: Eigenvalue Distribution\n'
                              f'Pos: {np.sum(eigenvalues > 0)}, Neg: {np.sum(eigenvalues < 0)}')
            axes[idx].set_xlabel('Eigenvalue')
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
    else:
        losses, _ = nn.train(X_train, y_train, epochs=epochs)  # Fix: unpack the tuple
    
    print(f"Train Acc: {nn.accuracy(X_train, y_train):.4f} | Test Acc: {nn.accuracy(X_test, y_test):.4f}")

    # Original visualizations
    fig = plt.figure(figsize=(18,5))
    ax1 = plt.subplot(1,3,1)
    ax1.plot(losses); ax1.set_title("Loss Curve")

    ax2 = plt.subplot(1,3,2)
    xt, yt = X_test_orig[:, 0], X_test_orig[:, 1]
    preds = nn.predict(X_test).flatten()
    ax2.scatter(xt, yt, c=['green' if p else 'red' for p in preds], s=10, alpha=0.6)
    ax2.add_artist(plt.Circle((0,0), radius, fill=False, color='blue', ls='--', lw=2))
    ax2.set_aspect('equal'); ax2.set_title("Test Predictions")
    ax2.set_xlim(-max_radius, max_radius); ax2.set_ylim(-max_radius, max_radius)

    ax3 = plt.subplot(1,3,3)
    x_g = np.linspace(0, 1, 150)
    y_g = np.linspace(0, 1, 150)
    X_grid, Y_grid = np.meshgrid(x_g, y_g)
    grid = np.c_[X_grid.ravel(), Y_grid.ravel()]
    Z = nn.predict(grid).reshape(X_grid.shape)
    X_o, Y_o = X_grid * (2 * max_radius) - max_radius, Y_grid * (2 * max_radius) - max_radius
    ax3.contourf(X_o, Y_o, Z, cmap='RdYlGn', alpha=0.3)
    ax3.contour(X_o, Y_o, Z, levels=[0.5], colors='black')
    ax3.add_artist(plt.Circle((0,0), radius, fill=False, color='blue', ls='--', lw=2))
    ax3.set_aspect('equal'); ax3.set_title("Decision Boundary")
    ax3.set_xlim(-max_radius, max_radius); ax3.set_ylim(-max_radius, max_radius)
    plt.tight_layout()
    plt.show()

# === Example usage ===
if __name__ == "__main__":
    # First, run with a smaller network to test Hessian computation
    print("=== Testing with Small Network ===")
    experiment(
        n_samples=1000,    # smaller dataset for faster computation
        radius=100,
        max_radius=250,
        hidden_sizes=[4, 2],  # smaller network
        learning_rate=0.05,
        epochs=500,
        compute_hessian=True  # Enable Hessian computation
    )
    
    # Then run with your original larger network
    print("\n=== Running with Larger Network ===")
    experiment(
        n_samples=10000,
        radius=100,
        max_radius=250,
        hidden_sizes=[12, 12],
        learning_rate=0.05,
        epochs=2500,
        compute_hessian=True  # Disable for larger network (too slow)
    )