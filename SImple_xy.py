import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

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

    def compute_analytical_gradient(self, X, y):
        """Fast analytical gradient computation using backpropagation"""
        # Store original parameters
        original_weights = [w.copy() for w in self.weights]
        original_biases = [b.copy() for b in self.biases]
        
        # Forward pass
        output = self.forward(X)
        
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
        
        # Restore original parameters
        self.weights = original_weights
        self.biases = original_biases
        
        return np.concatenate(gradients)

    def compute_stochastic_hessian_eigenvalues(self, X, y, n_samples=100):
        """Fast approximate Hessian eigenvalues using random projection"""
        n_params = len(self.get_parameters())
        
        # Use random projection to estimate eigenvalue distribution
        k = min(50, n_params)  # Number of random vectors to use
        
        Hv_products = []
        for i in range(k):
            v = np.random.randn(n_params)
            v = v / np.linalg.norm(v)
            
            # Compute Hessian-vector product using finite differences on gradient
            eps = 1e-5
            grad_current = self.compute_analytical_gradient(X, y)
            
            self.set_parameters(self.get_parameters() + eps * v)
            grad_perturbed = self.compute_analytical_gradient(X, y)
            self.set_parameters(self.get_parameters() - eps * v)  # Reset
            
            Hv = (grad_perturbed - grad_current) / eps
            Hv_products.append(Hv)
        
        # Build small matrix for eigenvalue computation
        H_small = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                H_small[i, j] = np.dot(Hv_products[i], Hv_products[j])
        
        # Compute eigenvalues of the small matrix (much faster)
        small_eigenvalues = np.linalg.eigvals(H_small)
        
        # Scale eigenvalues to approximate true Hessian eigenvalues
        scale_factor = np.sqrt(n_params / k)
        approximated_eigenvalues = small_eigenvalues * scale_factor
        
        return approximated_eigenvalues.real

    def fast_analyze_critical_point(self, params, X, y, grad_tol=1e-3):
        """Fast critical point analysis using stochastic Hessian estimation"""
        gradient = self.compute_analytical_gradient(X, y)
        grad_norm = np.linalg.norm(gradient)
        
        if grad_norm > grad_tol:
            return None, None
        
        print("Found critical point! Computing approximate Hessian eigenvalues...")
        
        # Set parameters temporarily
        original_params = self.get_parameters()
        self.set_parameters(params)
        
        # Use fast stochastic method to estimate eigenvalues
        eigenvalues = self.compute_stochastic_hessian_eigenvalues(X, y)
        
        # Restore parameters
        self.set_parameters(original_params)
        
        eigenvalues_real = eigenvalues.real
        
        # Classify critical point
        positive_ev = np.sum(eigenvalues_real > 1e-6)
        negative_ev = np.sum(eigenvalues_real < -1e-6)
        zero_ev = np.sum(np.abs(eigenvalues_real) < 1e-6)
        
        print(f"Approximate eigenvalue analysis:")
        print(f"  Positive eigenvalues: {positive_ev}")
        print(f"  Negative eigenvalues: {negative_ev}")
        print(f"  Zero eigenvalues: {zero_ev}")
        print(f"  Min eigenvalue: {np.min(eigenvalues_real):.6e}")
        print(f"  Max eigenvalue: {np.max(eigenvalues_real):.6e}")
        
        if negative_ev == 0 and zero_ev == 0:
            point_type = "Local Minimum"
        elif positive_ev == 0 and zero_ev == 0:
            point_type = "Local Maximum"
        elif negative_ev > 0 and positive_ev > 0:
            point_type = "Saddle Point"
        else:
            point_type = "Degenerate Critical Point"
        
        print(f"Critical point type: {point_type}")
        return point_type, eigenvalues

    def find_critical_points_fast(self, X, y, n_initial_points=100, max_iter=200, grad_tol=1e-3):
        """Fast critical point finding using gradient-based search"""
        print(f"Fast search for critical points using {n_initial_points} initializations...")
        
        critical_points = []
        n_params = len(self.get_parameters())
        
        # Store original parameters
        original_params = self.get_parameters()
        
        for i in range(n_initial_points):
            if i % 10 == 0:
                print(f"Processing initial point {i}/{n_initial_points}")
            
            # Random initialization around different scales
            scale = np.random.choice([0.01, 0.1, 1.0, 10.0])
            random_params = np.random.randn(n_params)*scale
            
            try:
                # Simple gradient descent to find low-gradient regions
                current_params = random_params.copy()
                
                for iteration in range(max_iter):
                    self.set_parameters(current_params)
                    grad = self.compute_analytical_gradient(X, y)
                    grad_norm = np.linalg.norm(grad)
                    
                    if grad_norm < grad_tol:
                        # Found candidate critical point
                        loss = self.compute_loss_from_params(current_params, X, y)
                        
                        # Check if we already have a similar point
                        is_duplicate = False
                        for cp in critical_points:
                            if np.linalg.norm(cp['parameters'] - current_params) < 1e-3:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            point_type, eigenvalues = self.fast_analyze_critical_point(current_params, X, y, grad_tol)
                            
                            if point_type:
                                critical_points.append({
                                    'parameters': current_params.copy(),
                                    'type': point_type,
                                    'eigenvalues': eigenvalues,
                                    'loss': loss,
                                    'grad_norm': grad_norm
                                })
                        break
                    
                    # Gradient descent step
                    current_params -= 0.1 * grad
                    
            except Exception as e:
                continue
        
        # Restore original parameters
        self.set_parameters(original_params)
        
        return critical_points

    def compute_loss_from_params(self, params, X, y):
        """Compute loss for given parameters"""
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

def analyze_critical_points_distribution(critical_points):
    """Analyze and visualize the distribution of critical points"""
    if not critical_points:
        print("No critical points found!")
        return
    
    types = [cp['type'] for cp in critical_points]
    losses = [cp['loss'] for cp in critical_points]
    grad_norms = [cp['grad_norm'] for cp in critical_points]
    
    # Count by type
    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\n" + "="*50)
    print("CRITICAL POINT ANALYSIS RESULTS")
    print("="*50)
    for point_type, count in type_counts.items():
        print(f"{point_type}: {count} points ({count/len(critical_points)*100:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Critical point types
    axes[0,0].bar(type_counts.keys(), type_counts.values(), color=['red', 'blue', 'green', 'orange'])
    axes[0,0].set_title('Distribution of Critical Point Types')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Loss distribution by type
    type_colors = {'Saddle Point': 'red', 'Local Minimum': 'blue', 
                   'Local Maximum': 'green', 'Degenerate Critical Point': 'orange'}
    for point_type in set(types):
        type_losses = [loss for cp, loss in zip(critical_points, losses) if cp['type'] == point_type]
        axes[0,1].hist(type_losses, alpha=0.7, label=point_type, color=type_colors.get(point_type, 'gray'))
    axes[0,1].set_title('Loss Distribution by Critical Point Type')
    axes[0,1].set_xlabel('Loss')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    
    # Plot 3: Eigenvalue spectra for saddle points
    saddle_eigenvalues = [cp['eigenvalues'] for cp in critical_points if cp['type'] == 'Saddle Point']
    if saddle_eigenvalues:
        all_saddle_ev = np.concatenate([ev.real for ev in saddle_eigenvalues])
        axes[1,0].hist(all_saddle_ev, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1,0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[1,0].set_title('Eigenvalue Distribution for Saddle Points')
        axes[1,0].set_xlabel('Eigenvalue')
        axes[1,0].set_ylabel('Frequency')
    
    # Plot 4: Gradient norms vs loss
    scatter = axes[1,1].scatter(losses, grad_norms, c=[type_colors.get(t, 'gray') for t in types], alpha=0.6)
    axes[1,1].set_xlabel('Loss')
    axes[1,1].set_ylabel('Gradient Norm')
    axes[1,1].set_title('Gradient Norm vs Loss')
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return type_counts

# === INTERACTIVE FUNCTION ===
def experiment(
    n_samples=1000, 
    radius=100, 
    max_radius=150, 
    hidden_sizes=[4, 2], 
    learning_rate=0.1, 
    epochs=500,
    find_critical_points=True,
    n_initial_points=100  # Reduced for speed
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
    print(f"Network architecture: 2 -> {hidden_sizes} -> 1")
    print(f"Total parameters: {len(nn.get_parameters())}")
    print(f"Training with radius={radius}, lr={learning_rate}, hidden={hidden_sizes}, epochs={epochs}")
    
    # Standard training
    losses = nn.train(X_train, y_train, epochs=epochs)
    
    print(f"Train Acc: {nn.accuracy(X_train, y_train):.4f} | Test Acc: {nn.accuracy(X_test, y_test):.4f}")

    # Find critical points
    if find_critical_points:
        print("\n" + "="*60)
        print("FAST SEARCH FOR CRITICAL POINTS")
        print("="*60)
        
        critical_points = nn.find_critical_points_fast(
            X_train, y_train, 
            n_initial_points=n_initial_points,
            max_iter=50,
            grad_tol=1e-3
        )
        
        # Analyze results
        type_counts = analyze_critical_points_distribution(critical_points)
        
        # Show some examples of saddle points
        saddle_points = [cp for cp in critical_points if cp['type'] == 'Saddle Point']
        if saddle_points:
            print(f"\nFound {len(saddle_points)} saddle points!")
            print("Example saddle point analysis:")
            for i, saddle in enumerate(saddle_points[:3]):
                print(f"Saddle {i+1}:")
                print(f"  Loss: {saddle['loss']:.6f}")
                print(f"  Gradient norm: {saddle['grad_norm']:.6e}")
                evals = saddle['eigenvalues'].real
                print(f"  Eigenvalue range: [{np.min(evals):.6e}, {np.max(evals):.6e}]")
                print()

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
    x_g = np.linspace(0, 1, 100)
    y_g = np.linspace(0, 1, 100)
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
    print("\n=== Running Fast Critical Point Analysis ===")
    experiment(
        n_samples=10000,
        radius=100,
        max_radius=200,
        hidden_sizes=[4, 4],
        learning_rate=0.05,
        epochs=5500,
        find_critical_points=True,
        n_initial_points=2000  # Much faster but still finds good results
    )