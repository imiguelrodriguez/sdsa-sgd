import random
from math import exp
from data import load_adult_data
from lithops import FunctionExecutor
import numpy as np

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
    return correct / len(data)

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

def update_model(model, point, learning_rate, lam):
    """Update model using stochastic gradient descent with L2 regularization."""
    prediction = predict(model, point)
    error = point['label'] - prediction
    for i in range(len(model)):
        model[i] += learning_rate * error * point['features'][i] - lam * model[i]  # L2 regularization term
    return model

def train_model(data, learning_rate, epochs, lam):
    """Train logistic regression model with SGD and L2 regularization."""
    model = [random.random() for _ in range(len(data[0]['features']))]
    for epoch in range(epochs):
        random.shuffle(data)  # Shuffle data before each epoch
        for point in data:
            model = update_model(model, point, learning_rate, lam)
        # Calculate and print accuracy after each epoch
        predictions = [predict(model, p) for p in data]
        epoch_accuracy = accuracy(data, predictions)
        print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {epoch_accuracy}")
    return model

def train_partition(data_partition, learning_rate, epochs, lam):
    return train_model(data_partition, learning_rate, epochs, lam)

def parallel_train_model(data, learning_rate, epochs, num_workers, lam):
    """Train logistic regression model in parallel using Lithops."""
    np.random.shuffle(data)
    partitions = np.array_split(data, num_workers)
    
    fexec = FunctionExecutor()
    futures = []
    for partition in partitions:
        futures.append(fexec.call_async(train_partition, (partition, learning_rate, epochs, lam)))
    models = fexec.get_result(futures)

    # Average the models
    final_model = [sum(weights) / len(weights) for weights in zip(*models)]
    return final_model

def submission(train_data, num_workers, lam, epochs):
    return parallel_train_model(train_data, learning_rate=0.01, epochs=epochs, num_workers=num_workers, lam=lam)