from math import exp
import random

from data import load_adult_data


def logistic(x):
    """Calculate the logistic function."""
    return 1 / (1 + exp(-x))

def dot(x, y):
    """Calculate the dot product of two lists."""
    return sum(xi * yi for xi, yi in zip(x, y))

def predict(model, point):
    """Calculate prediction based on model."""
    return logistic(dot(model, point['features']))


def accuracy(data, predictions):
    """Calculate accuracy of predictions on data."""
    correct = sum((pred > 0.5) == point['label'] for point, pred in zip(data, predictions))
    return float(correct) / len(data)

def update(model, point, delta, rate, lam):
    """Update model using learning rate and L2 regularization."""
    for i in range(len(model)):
        grad = -delta * point['features'][i] + lam * model[i]
        model[i] -= rate * grad

def initialize_model(k):
    """Initialize the model with Gaussian-distributed values."""
    return [random.gauss(0, 1) for x in range(k)]


def train(data, epochs, rate, lam):
    """Train model using training data."""
    model = initialize_model(len(data[0]['features']))
    for epoch in range(epochs):
        for point in data:
            prediction = predict(model, point)
            delta = point['label'] - prediction
            update(model, point, delta, rate, lam)
    return model

def extract_features(raw):
    """Extract features from raw data."""
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')

        features = []
        features.append(1.)  # Bias term
        features.append(float(r['age']) / 100)  # Normalized age
        features.append(float(r['education_num']) / 20)  # Normalized education number
        features.append(r['marital'] == 'Married-civ-spouse')  # Binary marital status
        features.append(float(r['hr_per_week']) / 100)  # Normalized hours per week
        features.append(r['occupation'] == 'Exec-managerial')  # Binary occupation
        features.append(float(r['capital_gain']) / 10000)  # Normalized capital gain
        features.append(float(r['capital_loss']) / 10000)  # Normalized capital loss
        features.append(r['sex'] == 'Male')  # Binary gender
        features.append(r['race'] == 'White')  # Binary race feature
        features.append(r['type_employer'] == 'Private')  # Binary employer type

        point['features'] = features
        data.append(point)
    return data

def submission(data):
    """Tune parameters and train the model for final submission."""
    return train(data, epochs=10, rate=0.05, lam=0.01)

# Example usage with hypothetical raw data
raw_data = load_adult_data()

data = extract_features(raw_data)
model = submission(data)

# Example predictions
predictions = [predict(model, point) for point in data]
print("Model predictions:", predictions)
print("Accuracy:", accuracy(data, predictions))
