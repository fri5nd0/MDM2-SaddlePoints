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
    
    def compute_hessian_eigenvalues(self, X, y, eps=1e-4):
        """
        Compute the Hessian matrix of the loss function with respect to all parameters
        using finite differences and return its eigenvalues
        """
        # Flatten all parameters into a single vector
        flat_params, shapes = self._flatten_parameters()
        n_params = len(flat_params)
        
        # Compute gradient at current point
        grad0 = self._compute_gradient(X, y)
        
        # Initialize Hessian matrix
        hessian = np.zeros((n_params, n_params))
        
        print(f"Computing Hessian matrix ({n_params}x{n_params})...")
        
        # Compute Hessian using finite differences
        for i in range(n_params):
            if i % 50 == 0:  # Progress indicator
                print(f"Computing column {i+1}/{n_params}")
            
            # Create parameter perturbation
            params_plus = flat_params.copy()
            params_plus[i] += eps
            
            # Temporarily set the perturbed parameters
            original_flat, _ = self._flatten_parameters()
            self._unflatten_parameters(params_plus, shapes)
            
            # Compute gradient at perturbed point
            grad_plus = self._compute_gradient(X, y)
            
            # Restore original parameters
            self._unflatten_parameters(original_flat, shapes)
            
            # Finite difference approximation of second derivative
            hessian[:, i] = (grad_plus - grad0) / eps
        
        # Make Hessian symmetric (average with its transpose)
        hessian = 0.5 * (hessian + hessian.T)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(hessian)
        
        return eigenvalues, hessian
    
    def _flatten_parameters(self):
        """Flatten all weights and biases into a single vector"""
        flat_params = []
        shapes = []
        
        # Store weights shapes and flatten
        weight_shapes = []
        for w in self.weights:
            weight_shapes.append(w.shape)
            flat_params.extend(w.flatten())
        
        # Store bias shapes and flatten  
        bias_shapes = []
        for b in self.biases:
            bias_shapes.append(b.shape)
            flat_params.extend(b.flatten())
        
        # Store both weight and bias shapes
        shapes = (weight_shapes, bias_shapes)
        return np.array(flat_params), shapes
    
    def _unflatten_parameters(self, flat_params, shapes):
        """Restore parameters from flattened vector"""
        weight_shapes, bias_shapes = shapes
        self.weights = []
        self.biases = []
        
        idx = 0
        
        # Restore weights
        for shape in weight_shapes:
            size = np.prod(shape)
            self.weights.append(flat_params[idx:idx+size].reshape(shape))
            idx += size
        
        # Restore biases
        for shape in bias_shapes:
            size = np.prod(shape)
            self.biases.append(flat_params[idx:idx+size].reshape(shape))
            idx += size
    
    def _compute_gradient(self, X, y):
        """Compute gradient of loss with respect to parameters at current state"""
        # Forward pass
        output = self.forward(X)
        
        # Backward pass to compute gradients
        m = X.shape[0]
        gradients = []
        
        # Output layer gradients
        dZ = output - y
        dW = (1/m) * np.dot(self.activations[-2].T, dZ)
        db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        
        # Store gradients in reverse order (to match flattening order)
        grad_list = []
        grad_list.append(db.flatten())
        grad_list.append(dW.flatten())
        
        # Hidden layers gradients
        for i in range(len(self.weights) - 2, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * self.relu_derivative(self.z_values[i])
            dW = (1/m) * np.dot(self.activations[i].T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            grad_list.append(db.flatten())
            grad_list.append(dW.flatten())
        
        # Reverse to match the flattening order (weights first, then biases)
        grad_list = grad_list[::-1]
        gradients = np.concatenate(grad_list)
        
        return gradients

# === INTERACTIVE FUNCTION ===
def experiment(
    n_samples=3000, 
    radius=100, 
    max_radius=150, 
    hidden_sizes=[8, 4], 
    learning_rate=0.1, 
    epochs=1000
):
    np.random.seed(42)
    
    # Generate points in Cartesian coordinates
    points = []
    for _ in range(n_samples):
        # Generate points within the square [-max_radius, max_radius]
        x = np.random.uniform(-max_radius, max_radius)
        y = np.random.uniform(-max_radius, max_radius)
        points.append((x, y))
    
    X = np.array(points)
    # Label points based on whether they're inside the circle of given radius
    y = np.array([1.0 if x**2 + y**2 <= radius**2 else 0.0 for (x, y) in points]).reshape(-1, 1)

    # Normalize to [0, 1] range
    Xn = X.copy()
    Xn = (Xn + max_radius) / (2 * max_radius)  # Map from [-max_radius, max_radius] to [0, 1]

    X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42, stratify=y)
    X_train_orig, X_test_orig = X_train * (2 * max_radius) - max_radius, X_test * (2 * max_radius) - max_radius
    
    nn = NeuralNetwork(2, hidden_sizes, 1, learning_rate)
    print(f"Training with radius={radius}, lr={learning_rate}, hidden={hidden_sizes}, epochs={epochs}")
    losses = nn.train(X_train, y_train, epochs=epochs)
    
    print(f"Train Acc: {nn.accuracy(X_train, y_train):.4f} | Test Acc: {nn.accuracy(X_test, y_test):.4f}")

    # Compute Hessian eigenvalues (use smaller subset for speed)
    print("\nComputing Hessian eigenvalues...")
    
    # Use even smaller subset for demonstration to avoid long computation
    sample_size = min(50, len(X_train))
    eigenvalues, hessian = nn.compute_hessian_eigenvalues(X_train[:sample_size], y_train[:sample_size])
    
    print(f"\nHessian Matrix Info:")
    print(f"Shape: {hessian.shape}")
    print(f"Min eigenvalue: {np.min(eigenvalues):.6f}")
    print(f"Max eigenvalue: {np.max(eigenvalues):.6f}")
    print(f"Condition number: {np.max(np.abs(eigenvalues)) / np.max([np.min(np.abs(eigenvalues)), 1e-10]):.6f}")
    print(f"Number of positive eigenvalues: {np.sum(eigenvalues > 0)}")
    print(f"Number of negative eigenvalues: {np.sum(eigenvalues < 0)}")
    print(f"Number of zero eigenvalues: {np.sum(np.abs(eigenvalues) < 1e-10)}")

    # Create separate windows for different plots
    
    # Window 1: Main results (loss, predictions, decision boundary)
    fig1 = plt.figure(figsize=(15, 5))
    fig1.canvas.manager.set_window_title('Neural Network Training Results')
    
    # Loss curve
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(losses)
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    # Test predictions
    ax2 = plt.subplot(1, 3, 2)
    xt, yt = X_test_orig[:, 0], X_test_orig[:, 1]
    preds = nn.predict(X_test).flatten()
    ax2.scatter(xt, yt, c=['green' if p else 'red' for p in preds], s=10, alpha=0.6)
    ax2.add_artist(plt.Circle((0,0), radius, fill=False, color='blue', ls='--', lw=2))
    ax2.set_aspect('equal')
    ax2.set_title("Test Predictions")
    ax2.set_xlim(-max_radius, max_radius)
    ax2.set_ylim(-max_radius, max_radius)
    
    # Decision boundary
    ax3 = plt.subplot(1, 3, 3)
    x_g = np.linspace(0, 1, 150)
    y_g = np.linspace(0, 1, 150)
    X_grid, Y_grid = np.meshgrid(x_g, y_g)
    grid = np.c_[X_grid.ravel(), Y_grid.ravel()]
    Z = nn.predict(grid).reshape(X_grid.shape)
    X_o, Y_o = X_grid * (2 * max_radius) - max_radius, Y_grid * (2 * max_radius) - max_radius
    ax3.contourf(X_o, Y_o, Z, cmap='RdYlGn', alpha=0.3)
    ax3.contour(X_o, Y_o, Z, levels=[0.5], colors='black')
    ax3.add_artist(plt.Circle((0,0), radius, fill=False, color='blue', ls='--', lw=2))
    ax3.set_aspect('equal')
    ax3.set_title("Decision Boundary")
    ax3.set_xlim(-max_radius, max_radius)
    ax3.set_ylim(-max_radius, max_radius)

    plt.tight_layout()
    plt.show()

    # Window 2: Eigenvalue histogram (separate window)
    fig2 = plt.figure(figsize=(10, 6))
    fig2.canvas.manager.set_window_title('Hessian Eigenvalues Analysis')
    
    # Filter out very small eigenvalues for better visualization
    mask = np.abs(eigenvalues) > 1e-10
    filtered_eigenvalues = eigenvalues[mask]
    
    if len(filtered_eigenvalues) > 0:
        plt.hist(filtered_eigenvalues, bins=50, alpha=0.7, edgecolor='black')
        plt.title("Hessian Eigenvalues Distribution")
        plt.xlabel("Eigenvalue")
        plt.ylabel("Frequency")
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Zero')
        plt.legend()
        
        # Add some statistics to the histogram plot
        plt.text(0.05, 0.95, 
                f'Min: {np.min(filtered_eigenvalues):.3e}\n'
                f'Max: {np.max(filtered_eigenvalues):.3e}\n'
                f'Positive: {np.sum(filtered_eigenvalues > 0)}\n'
                f'Negative: {np.sum(filtered_eigenvalues < 0)}\n'
                f'Zero: {np.sum(np.abs(eigenvalues) < 1e-10)}\n'
                f'Condition: {np.max(np.abs(eigenvalues)) / np.max([np.min(np.abs(eigenvalues)), 1e-10]):.3e}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        plt.text(0.5, 0.5, 'No significant eigenvalues\nfound above threshold', 
                horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title("Hessian Eigenvalues Distribution")

    plt.tight_layout()
    plt.show()
    
    return eigenvalues, hessian

# === Example usage with smaller network for faster computation ===
eigenvalues, hessian = experiment(
    n_samples=10000,    # total samples
    radius=100,         # circle radius threshold
    max_radius=200,     # overall range
    hidden_sizes=[10, 10],  # smaller hidden layers for faster Hessian computation
    learning_rate=0.05,    # LR
    epochs=2500         # training epochs
)