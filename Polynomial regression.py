import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



def plot_polynomial(coeffs, z_range, color='b', label='Polynomial fit'):
    z_min, z_max = z_range
    z_values = np.linspace(z_min, z_max, 500)  # Generate 500 points between the range
    # computing p(z) for each z_value
    p_values = polynomial(z_values, coeffs)
    # Plotting the polynomial
    plt.plot(z_values, p_values, color=color, label=label)


def create_dataset(coeffs, z_range, sample_size, sigma, seed=42):
    np.random.seed(seed)
    z_min, z_max = z_range
    z_values = np.random.uniform(z_min, z_max, sample_size)  # Generate z values

    # Compute the true polynomial values (without noise)
    y_values = polynomial(z_values, coeffs)

    # Adding Gaussian noise
    noise = np.random.normal(0, sigma, sample_size)
    y_noisy = y_values + noise

    # Return the dataset as torch tensors
    X = torch.tensor(np.column_stack([z_values**i for i in range(len(coeffs))]), dtype=torch.float32)
    y = torch.tensor(y_noisy, dtype=torch.float32)

    return X, y


def visualize_data(X, y, coeffs, z_range, title=""):
    z_values = np.linspace(z_range[0], z_range[1], X.shape[0])

    # Compute the true polynomial values without noise
    true_values = polynomial(z_values, coeffs)

    # Plot the true polynomial
    plt.plot(z_values, true_values, color='g', label='True Polynomial')

    # Plot the noisy data
    plt.scatter(z_values, y, color='r', alpha=0.5, label='Noisy Data')

    plt.title(f'{title}')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    def polynomial(x, coeffs):
        return coeffs[0] * x ** 4 + coeffs[1] * x ** 3 + coeffs[2] * x ** 2 + coeffs[3] * x + coeffs[4]

    coeffs = [1/30, -0.1, 5, -1, 1]  # Coefficients
    z_range = [-4, 4]  # Range for z values
    plot_polynomial(coeffs, z_range)


    #Create dataset with given parameters
    X_train, y_train = create_dataset(coeffs, [-2, 2], sample_size=500, sigma=0.5, seed=0)
    X_val, y_val = create_dataset(coeffs, [-2, 2], sample_size=500, sigma=0.5, seed=1)


    # Visualize training data
    visualize_data(X_train, y_train, coeffs, [-2, 2], title="Training Data Visualization")

    # Visualize validation data
    visualize_data(X_val, y_val, coeffs, [-2, 2], title="Validation Data Visualization")


    #Performing polynomial regression using linear regression
    class PolynomialRegression(nn.Module):
        def __init__(self, input_size):
            super(PolynomialRegression, self).__init__()
            # We don't need a bias term, because w_0 is already in the coefficients
            self.linear = nn.Linear(input_size, 1, bias=False)

        def forward(self, x):
            return self.linear(x)

    # Training the model with the provided dataset
    def train_polynomial_regression(X_train, y_train, X_val, y_val, learning_rate=0.001, epochs=300, weight_decay=1e-4):
        input_size = X_train.shape[1]
        model = PolynomialRegression(input_size)

        # Loss and optimizer (with L2 regularization, weight decay)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Tracking loss
        train_losses = []
        val_losses = []
        parameter_history = []  # Store parameter values for each epoch

        for epoch in range(epochs):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass for training data
            outputs = model(X_train)
            train_loss = criterion(outputs, y_train)

            # Backward pass and optimization
            train_loss.backward()
            optimizer.step()

            # Validation loss
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)

            # Record losses
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

            # Storing parameter values for each epoch
            current_parameters = model.linear.weight.data.cpu().numpy().flatten()
            parameter_history.append(current_parameters)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, Training Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}')

        return model, train_losses, val_losses, parameter_history

    # Parameters for the model training
    learning_rate = 0.001  # Reduced learning rate
    epochs = 300  # Increased number of epochs
    weight_decay = 1e-4  # L2 regularization term

    # reshaping y_train and y_val for training
    y_train = y_train.view(-1, 1)
    y_val = y_val.view(-1, 1)

    # Training the model and track parameter history
    model, train_losses, val_losses, parameter_history = train_polynomial_regression(X_train, y_train, X_val, y_val, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)


    # Plot the training and validation loss as a function of iterations

    plt.plot(range(epochs), train_losses, label="Training Loss", color='blue')
    plt.plot(range(epochs), val_losses, label="Validation Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss as a Function of Epochs")
    plt.grid(True)
    plt.savefig("training_validation_loss.png", dpi=300)
    plt.show()


   # Plot the estimated polynomial vs the original polynomial

    # Extracting learned coefficients from the model
    learned_coeffs = model.linear.weight.data.cpu().numpy().flatten()

    # Plot original polynomial
    plot_polynomial(coeffs, [-4, 4], color='blue', label='Original Polynomial')

    # Plot estimated polynomial using the learned coefficients
    plot_polynomial(learned_coeffs, [-4, 4], color='red', label='Estimated Polynomial')

    plt.title("Original vs Estimated Polynomial")
    plt.legend()
    plt.grid(True)
    plt.savefig("original_vs_estimated_polynomial.png", dpi=300)
    plt.show()


    # Evaluating Linear Regression Performance on a Logarithmic Function and checking for impact of Data Range

    # function f(x) = 2 * log(x + 1) + 3
    def f(x):
        return 2 * np.log(x + 1) + 3


    # Generate noisy data
    def generate_noisy_data(a, n_samples=100, noise_std=0.1):
        np.random.seed(42)  # For reproducibility
        x = np.random.uniform(-0.05, a, n_samples)
        y = f(x) + np.random.normal(0, noise_std, n_samples)
        return x, y


    # Perform linear regression and calculate loss
    def perform_linear_regression(a):
        x, y = generate_noisy_data(a)
        x = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        mse = mean_squared_error(y, y_pred)
        return model.coef_, model.intercept_, mse, x, y, y_pred


    # Case 1: a = 0.01
    coef_1, intercept_1, mse_1, x_1, y_1, y_pred_1 = perform_linear_regression(0.01)

    # Case 2: a = 10
    coef_2, intercept_2, mse_2, x_2, y_2, y_pred_2 = perform_linear_regression(10)

    # Plot the results for both cases
    plt.figure(figsize=(14, 6))

    # Plot for a = 0.01
    plt.subplot(1, 2, 1)
    plt.scatter(x_1, y_1, label='Noisy data', color='blue')
    plt.plot(x_1, y_pred_1, color='red', label='Linear fit')
    plt.title(f'Linear Regression (a = 0.01)\nMSE: {mse_1:.4f}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    # Plot for a = 10
    plt.subplot(1, 2, 2)
    plt.scatter(x_2, y_2, label='Noisy data', color='blue')
    plt.plot(x_2, y_pred_2, color='red', label='Linear fit')
    plt.title(f'Linear Regression (a = 10)\nMSE: {mse_2:.4f}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.tight_layout()
    plt.show()


