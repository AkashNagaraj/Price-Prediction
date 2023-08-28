import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            predicted = np.dot(X, self.weights) + self.bias
            gradient_weights = (1/num_samples) * np.dot(X.T, (predicted - y))
            gradient_bias = (1/num_samples) * np.sum(predicted - y)
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([2, 3, 4])

model = LinearRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)

new_data = np.array([[4, 5]])
predicted_values = model.predict(new_data)
print(predicted_values)
