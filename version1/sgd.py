import random
import lithops
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

def train_worker(model, data_batch, rate, lam, worker_id):
    """Worker function to perform SGD on a subset of the data."""
    print(f"Worker {worker_id} started processing its batch with {len(data_batch)} points.")
    local_model = model[:]  # Copy model for local updates
    for point in data_batch:
        prediction = predict(local_model, point)
        delta = point['label'] - prediction
        update(local_model, point, delta, rate, lam)
    print(f"Worker {worker_id} finished processing its batch.")
    return local_model

def aggregate_models(models):
    """Aggregate models from different workers."""
    print(f"Aggregating models from {len(models)} workers.")
    num_workers = len(models)
    aggregated_model = [sum(weight) / num_workers for weight in zip(*models)]
    print("Model aggregation complete.")
    return aggregated_model

def distributed_train(data, epochs, rate, lam, num_workers=4):
    """Train model using parallel workers with Lithops."""
    model = initialize_model(len(data[0]['features']))
    
    # Create Lithops function executor once
    fexec = lithops.FunctionExecutor()

    # Split data into chunks for workers (ensure that the number of chunks doesn't overwhelm the system)
    chunk_size = len(data) // num_workers
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]

    print(f"Data divided into {num_workers} chunks, with {chunk_size} samples per chunk.")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} started.")

        # For each epoch, we distribute the computation to workers
        futures = []
        for worker_id, chunk in enumerate(chunks):
            print(f"Submitting task for Worker {worker_id + 1} with {len(chunk)} points.")
            futures.append(fexec.call_async(train_worker, (model, chunk, rate, lam, worker_id + 1)))
        
        # Wait for results from all workers and gather the models
        worker_models = [future.result() for future in futures]
        print(f"All workers completed their tasks for Epoch {epoch+1}.")

        # Aggregate models from workers
        model = aggregate_models(worker_models)
        print(f"Epoch {epoch+1} completed.")

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
    return distributed_train(data, epochs=10, rate=0.05, lam=0.01)

# Example usage with hypothetical raw data
raw_data = load_adult_data()

data = extract_features(raw_data)
model = submission(data)

# Example predictions
predictions = [predict(model, point) for point in data]
print("Model predictions:", predictions)
print("Accuracy:", accuracy(data, predictions))
