import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights, self.biases = [], []
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights with He initialization
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * np.sqrt(2.0 / hidden_sizes[i-1]))
            self.biases.append(np.zeros((1, hidden_sizes[i])))
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2.0 / hidden_sizes[-1]))
        self.biases.append(np.zeros((1, output_size)))
        
        # For tracking training history
        self.training_history = {
            'epochs': [],
            'losses': [],
            'train_acc': [],
            'test_acc': [],
            'weights': [],
            'biases': [],
            'weights_epochs': []  # Track which epochs we stored weights at
        }
    
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

    def train(self, X, y, X_test=None, y_test=None, epochs=1000, verbose=True, store_history=True):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            losses.append(loss)
            self.backward(X, y, output)
            
            if store_history and (epoch % 10 == 0 or epoch == epochs-1):
                self.training_history['epochs'].append(epoch)
                self.training_history['losses'].append(loss)
                train_acc = self.accuracy(X, y)
                self.training_history['train_acc'].append(train_acc)
                
                if X_test is not None and y_test is not None:
                    test_acc = self.accuracy(X_test, y_test)
                    self.training_history['test_acc'].append(test_acc)
            
            # Store weights and biases (only store occasionally to save memory)
            if store_history and (epoch % 100 == 0 or epoch == epochs-1):
                self.training_history['weights'].append([w.copy() for w in self.weights])
                self.training_history['biases'].append([b.copy() for b in self.biases])
                self.training_history['weights_epochs'].append(epoch)
            
            if verbose and epoch % 100 == 0:
                test_acc_info = f" | Test Acc: {self.accuracy(X_test, y_test):.4f}" if X_test is not None else ""
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {self.accuracy(X, y):.4f}{test_acc_info}")
        
        return losses
    
    def predict(self, X, threshold=0.5):
        return (self.forward(X) > threshold).astype(int)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def visualize_network_structure(self):
        """Visualize the neural network as a graph/tree"""
        plt.figure(figsize=(12, 8))
        G = nx.Graph()
        pos = {}
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        # Create nodes and positions
        node_counter = 0
        for layer_idx, size in enumerate(layer_sizes):
            for i in range(size):
                node_id = f"L{layer_idx}_N{i}"
                G.add_node(node_id, layer=layer_idx)
                pos[node_id] = (layer_idx, i - size/2)
                node_counter += 1
        
        # Add edges with weights
        edge_weights = []
        for layer_idx in range(len(layer_sizes) - 1):
            weight_matrix = self.weights[layer_idx]
            for i in range(layer_sizes[layer_idx]):
                for j in range(layer_sizes[layer_idx + 1]):
                    source = f"L{layer_idx}_N{i}"
                    target = f"L{layer_idx+1}_N{j}"
                    weight = weight_matrix[i, j]
                    G.add_edge(source, target, weight=weight)
                    edge_weights.append(weight)
        
        # Normalize edge weights for visualization
        if edge_weights:
            max_weight = max(abs(w) for w in edge_weights)
            edge_weights_normalized = [abs(w) / max_weight for w in edge_weights]
            edge_colors = ['red' if w < 0 else 'blue' for w in edge_weights]
        else:
            edge_weights_normalized = [1] * len(G.edges())
            edge_colors = 'blue'
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', 
                              alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=edge_weights_normalized, 
                              edge_color=edge_colors, alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=6)
        
        plt.title("Neural Network Architecture\n(Red: Negative weights, Blue: Positive weights)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print weight statistics
        print(f"Weight Statistics:")
        for i, w in enumerate(self.weights):
            print(f"  Layer {i}: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}, std={w.std():.4f}")
    
    def visualize_hidden_activations(self, X_sample):
        """Visualize activations in hidden layers for a sample input"""
        # Forward pass to get activations
        output = self.forward(X_sample)
        
        n_plots = len(self.hidden_sizes) + 2  # Input + hidden layers + output
        fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        # Plot input
        axes[0].bar(range(X_sample.shape[1]), X_sample[0])
        axes[0].set_title('Input Layer')
        axes[0].set_xlabel('Input Feature')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Plot hidden layers
        for i, (activations, ax) in enumerate(zip(self.activations[1:-1], axes[1:1+len(self.hidden_sizes)])):
            ax.bar(range(activations.shape[1]), activations[0])
            ax.set_title(f'Hidden Layer {i+1}')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Activation Value')
            ax.grid(True, alpha=0.3)
        
        # Plot output
        axes[-1].bar([0], output[0], color='purple')
        axes[-1].set_title('Output Layer')
        axes[-1].set_xlabel('Output')
        axes[-1].set_ylabel('Activation')
        axes[-1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print activation statistics
        print("Activation Statistics:")
        for i, activation in enumerate(self.activations[1:-1]):
            act = activation[0]
            print(f"  Hidden Layer {i+1}: min={act.min():.4f}, max={act.max():.4f}, "
                  f"mean={act.mean():.4f}, % active={(act > 0).mean():.1%}")
    
    def visualize_weight_evolution(self):
        """Visualize how weights change during training"""
        if not self.training_history['weights']:
            print("No weight history stored. Run training with store_history=True")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Weight distributions at different epochs
        if len(self.training_history['weights']) >= 3:
            # Use safe indices
            epochs_to_plot = [
                0, 
                len(self.training_history['weights']) // 2,
                len(self.training_history['weights']) - 1
            ]
            colors = ['red', 'orange', 'green']
            labels = ['Start', 'Middle', 'End']
            
            for i, epoch_idx in enumerate(epochs_to_plot):
                if epoch_idx < len(self.training_history['weights']):
                    weights_flat = np.concatenate([w.flatten() for w in self.training_history['weights'][epoch_idx]])
                    epoch_num = self.training_history['weights_epochs'][epoch_idx]
                    axes[0].hist(weights_flat, bins=50, alpha=0.7, color=colors[i], 
                                label=f'{labels[i]} (Epoch {epoch_num})', 
                                density=True)
        
            axes[0].set_title('Weight Distribution Evolution')
            axes[0].set_xlabel('Weight Value')
            axes[0].set_ylabel('Density')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Weight magnitudes over layers
        if self.training_history['weights']:
            current_weights = self.training_history['weights'][-1]
            layer_weights_mean = []
            layer_weights_std = []
            
            for layer_idx, w in enumerate(current_weights):
                layer_weights_mean.append(np.mean(np.abs(w)))
                layer_weights_std.append(np.std(w))
            
            axes[1].bar(range(len(layer_weights_mean)), layer_weights_mean, 
                       yerr=layer_weights_std, capsize=5, alpha=0.7, color='skyblue')
            axes[1].set_title('Average Weight Magnitude by Layer')
            axes[1].set_xlabel('Layer Index')
            axes[1].set_ylabel('Average |Weight|')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: First layer weights as heatmap (if applicable)
        if (self.training_history['weights'] and self.input_size <= 20 and 
            self.hidden_sizes[0] <= 20 and len(self.weights) > 0):
            current_weights = self.training_history['weights'][-1]
            im = axes[2].imshow(current_weights[0], cmap='RdBu', aspect='auto')
            axes[2].set_title('Input-Hidden Layer 1 Weight Matrix')
            axes[2].set_xlabel('Hidden Layer 1 Neurons')
            axes[2].set_ylabel('Input Features')
            # Create colorbar properly
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axes[2])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)
        
        # Plot 4: Training curve
        if self.training_history['losses']:
            axes[3].plot(self.training_history['epochs'], self.training_history['losses'], 'b-')
            axes[3].set_title('Training Loss')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Loss')
            axes[3].grid(True, alpha=0.3)
            
            # Add accuracy curves if available
            if self.training_history['train_acc']:
                ax_twin = axes[3].twinx()
                ax_twin.plot(self.training_history['epochs'], self.training_history['train_acc'], 
                           'r-', alpha=0.7, label='Train Acc')
                if (self.training_history['test_acc'] and 
                    len(self.training_history['test_acc']) == len(self.training_history['epochs'])):
                    ax_twin.plot(self.training_history['epochs'], self.training_history['test_acc'], 
                               'g-', alpha=0.7, label='Test Acc')
                ax_twin.set_ylabel('Accuracy')
                ax_twin.legend()
        
        # Remove empty subplots if any
        for i in range(len(axes)):
            if not axes[i].has_data():
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        # Print training summary
        if self.training_history['losses']:
            print(f"Training Summary:")
            print(f"  Final Loss: {self.training_history['losses'][-1]:.4f}")
            if self.training_history['train_acc']:
                print(f"  Final Train Accuracy: {self.training_history['train_acc'][-1]:.4f}")
            if (self.training_history['test_acc'] and 
                len(self.training_history['test_acc']) == len(self.training_history['epochs'])):
                print(f"  Final Test Accuracy: {self.training_history['test_acc'][-1]:.4f}")

def create_training_animation(nn, X_test_orig, max_radius, radius):
    """Create animation of decision boundary evolution during training"""
    if not nn.training_history['weights']:
        print("No weight history available for animation")
        return None
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Get weights for this frame
        epoch_idx = min(frame, len(nn.training_history['weights'])-1)
        if epoch_idx < 0:
            return
        
        # Temporarily set network weights to historical values
        original_weights = nn.weights.copy()
        original_biases = nn.biases.copy()
        
        nn.weights = [w.copy() for w in nn.training_history['weights'][epoch_idx]]
        nn.biases = [b.copy() for b in nn.training_history['biases'][epoch_idx]]
        
        # Plot decision boundary
        x_g = np.linspace(0, 1, 100)
        y_g = np.linspace(0, 1, 100)
        X_grid, Y_grid = np.meshgrid(x_g, y_g)
        grid = np.c_[X_grid.ravel(), Y_grid.ravel()]
        Z = nn.predict(grid).reshape(X_grid.shape)
        X_o, Y_o = X_grid * (2 * max_radius) - max_radius, Y_grid * (2 * max_radius) - max_radius
        
        contour = ax1.contourf(X_o, Y_o, Z, cmap='RdYlGn', alpha=0.3)
        ax1.contour(X_o, Y_o, Z, levels=[0.5], colors='black')
        ax1.add_artist(plt.Circle((0,0), radius, fill=False, color='blue', ls='--', lw=2))
        ax1.set_aspect('equal')
        epoch_num = nn.training_history['weights_epochs'][epoch_idx]
        ax1.set_title(f'Decision Boundary - Epoch {epoch_num}')
        ax1.set_xlim(-max_radius, max_radius)
        ax1.set_ylim(-max_radius, max_radius)
        
        # Plot test points
        xt, yt = X_test_orig[:, 0], X_test_orig[:, 1]
        preds = nn.predict((X_test_orig + max_radius) / (2 * max_radius)).flatten()
        ax1.scatter(xt, yt, c=['green' if p else 'red' for p in preds], s=10, alpha=0.6)
        
        # Plot training curve
        current_epoch = epoch_num
        epochs_so_far = [e for e in nn.training_history['epochs'] if e <= current_epoch]
        losses_so_far = nn.training_history['losses'][:len(epochs_so_far)]
        
        ax2.plot(epochs_so_far, losses_so_far, 'b-')
        ax2.axvline(current_epoch, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        # Restore original weights
        nn.weights = original_weights
        nn.biases = original_biases
    
    anim = FuncAnimation(fig, update, frames=len(nn.training_history['weights']), 
                        interval=500, repeat=True)
    plt.close()
    return HTML(anim.to_jshtml())

# === ENHANCED INTERACTIVE FUNCTION ===
def experiment(
    n_samples=3000, 
    radius=100, 
    max_radius=150, 
    hidden_sizes=[8, 4], 
    learning_rate=0.1, 
    epochs=1000,
    visualize_network=True,
    visualize_training=True,
    visualize_activations=True
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
    
    # Train with history tracking
    losses = nn.train(X_train, y_train, X_test, y_test, epochs=epochs, store_history=True)
    
    print(f"Final Train Acc: {nn.accuracy(X_train, y_train):.4f} | Test Acc: {nn.accuracy(X_test, y_test):.4f}")

    # === COMPREHENSIVE VISUALIZATIONS ===
    
    # 1. Standard results
    fig = plt.figure(figsize=(18, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(nn.training_history['epochs'], nn.training_history['losses'])
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 3, 2)
    xt, yt = X_test_orig[:, 0], X_test_orig[:, 1]
    preds = nn.predict(X_test).flatten()
    ax2.scatter(xt, yt, c=['green' if p else 'red' for p in preds], s=10, alpha=0.6)
    ax2.add_artist(plt.Circle((0,0), radius, fill=False, color='blue', ls='--', lw=2))
    ax2.set_aspect('equal')
    ax2.set_title("Test Predictions")
    ax2.set_xlim(-max_radius, max_radius)
    ax2.set_ylim(-max_radius, max_radius)

    ax3 = plt.subplot(1, 3, 3)
    x_g = np.linspace(0, 1, 150)
    y_g = np.linspace(0, 1, 150)
    X_grid, Y_grid = np.meshgrid(x_g, y_g)
    grid = np.c_[X_grid.ravel(), Y_grid.ravel()]
    Z = nn.predict(grid).reshape(X_grid.shape)
    X_o, Y_o = X_grid * (2 * max_radius) - max_radius, Y_grid * (2 * max_radius) - max_radius
    
    contour = ax3.contourf(X_o, Y_o, Z, cmap='RdYlGn', alpha=0.3)
    ax3.contour(X_o, Y_o, Z, levels=[0.5], colors='black')
    ax3.add_artist(plt.Circle((0,0), radius, fill=False, color='blue', ls='--', lw=2))
    ax3.set_aspect('equal')
    ax3.set_title("Final Decision Boundary")
    ax3.set_xlim(-max_radius, max_radius)
    ax3.set_ylim(-max_radius, max_radius)
    
    # Add colorbar for the contour plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(contour, cax=cax, label='Prediction Confidence')
    
    plt.tight_layout()
    plt.show()

    # 2. Network structure visualization
    if visualize_network:
        print("\n=== Network Structure Visualization ===")
        nn.visualize_network_structure()
    
    # 3. Weight evolution visualization
    print("\n=== Weight Evolution Analysis ===")
    nn.visualize_weight_evolution()
    
    # 4. Hidden layer activations for sample points
    if visualize_activations:
        print("\n=== Hidden Layer Activations ===")
        # Sample a few points from different regions
        sample_points = np.array([
            [0, 0],           # Center (likely inside circle)
            [radius, 0],      # Edge
            [radius+50, 0],   # Outside
        ])
        sample_points_norm = (sample_points + max_radius) / (2 * max_radius)
        
        for i, point in enumerate(sample_points):
            print(f"\nSample point {i+1}: {point} (normalized: {sample_points_norm[i]})")
            nn.visualize_hidden_activations(sample_points_norm[i].reshape(1, -1))
    
    # 5. Training animation (optional - can be slow)
    if visualize_training and len(nn.training_history['weights']) > 1:
        print("\n=== Creating Training Animation ===")
        anim = create_training_animation(nn, X_test_orig, max_radius, radius)
        if anim:
            return anim
        else:
            print("Could not create animation - not enough weight history")

# === Example usage ===
if __name__ == "__main__":
    experiment(
        n_samples=10000,      # total samples
        radius=100,           # circle radius threshold
        max_radius=200,       # overall range
        hidden_sizes=[10, 5], # hidden layer configuration
        learning_rate=0.05,   # LR
        epochs=3500,          # training epochs
        visualize_network=True,
        visualize_training=True,  # Set to True if you want animation (can be slow)
        visualize_activations=True
    )