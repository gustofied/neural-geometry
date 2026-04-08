import numpy as np

# Simple backprop for one neuron learning y = 2*x

# 1. Data
x = np.array([1.0, 2.0, 3.0, 4.0])
y = 2 * x  # true outputs

# 2. Initialize parameters
w = 0.0  # weight
b = 0.0  # bias
lr = 0.01  # learning rate

print(f"{'Epoch':>5} {'Loss':>8} {'w':>8} {'b':>8}")
print("-" * 33)

# 3. Training loop
for epoch in range(1, 6):
    # Forward pass: compute predictions
    y_pred = w * x + b

    # Compute loss (mean squared error)
    loss = np.mean((y_pred - y) ** 2)

    # Backward pass: compute gradients
    dw = np.mean(2 * (y_pred - y) * x)  # ∂Loss/∂w
    db = np.mean(2 * (y_pred - y))      # ∂Loss/∂b

    # Update parameters
    w -= lr * dw
    b -= lr * db

    # Print progress
    print(f"{epoch:5d} {loss:8.4f} {w:8.4f} {b:8.4f}")