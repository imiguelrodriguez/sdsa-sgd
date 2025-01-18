from unittest import TestCase
import unittest
from sgd import logistic, dot, predict, accuracy, extract_features, DistributedSGD
from data import load_adult_train_data, load_adult_valid_data
import time

def evaluate_configuration(
        train_data,
        valid_data,
        test_data,
        num_workers,
        epochs,
        learning_rate,
        reg_lambda
):
    """Evaluate model with specified number of workers"""
    print(f"\nEvaluating with {num_workers} workers...")

    sgd = DistributedSGD(num_workers=num_workers)

    start_time = time.time()
    model = sgd.parallel_train_model(
        data = train_data,
        epochs=epochs,
        learning_rate=learning_rate,
        num_workers=num_workers,
        lam=reg_lambda
    )
    total_time = time.time() - start_time

    # Calculate training accuracy
    train_predictions = [predict(model, p) for p in train_data]
    train_accuracy = accuracy(train_data, train_predictions)
    
    # Calculate validation accuracy
    validation_predictions = [predict(model, p) for p in valid_data]
    validation_accuracy = accuracy(valid_data, validation_predictions)
    
    # Calculate test accuracy
    test_predictions = [predict(model, p) for p in test_data]
    test_accuracy = accuracy(test_data, test_predictions)

    print(f"\nResults for {num_workers} workers and {epochs} epochs:")
    print(f"Training time: {total_time:.2f} seconds")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    return model, total_time, train_accuracy, validation_accuracy, test_accuracy

class SGDTest(unittest.TestCase):

    # def test_logistic(self):
    #     self.assertAlmostEqual(logistic(1),  0.7310585786300049)
    #     self.assertAlmostEqual(logistic(2),  0.8807970779778823)
    #     self.assertAlmostEqual(logistic(-1),  0.2689414213699951)

    # def test_dot(self):
    #     d = dot([1.1,2,3.5], [-1,0.1,.08])
    #     self.assertAlmostEqual(d, -.62)

    # def test_predict(self):
    #     model = [1,2,1,0,1]
    #     point = {'features':[.4,1,3,.01,.1], 'label': 1}
    #     p = predict(model, point)
    #     self.assertAlmostEqual(p, 0.995929862284)
        
    # def test_accuracy(self):
    #     data = extract_features(load_adult_train_data())
    #     a = accuracy(data, [0]*len(data))
    #     self.assertAlmostEqual(a, 0.751077514754)

    def test_submission(self):
        # Load and extract features from the data
        raw_data = load_adult_train_data()
        data = extract_features(raw_data)
        
        # Split data into training and validation sets (80% train, 20% validation)
        split_index = int(0.8 * len(data))
        train_data = data[:split_index]
        valid_data = data[split_index:]
        
        # Load and extract features from the test data
        test_raw_data = load_adult_valid_data()
        test_data = extract_features(test_raw_data)
        
        # Configuration parameters
        worker_counts = [8, 12]  # Number of workers to test
        epoch_values = [15, 20, 25, 30]
        learning_rates = [0.1, 0.01]
        reg_lambdas = [ 0.0001, 0.00001, 0.000001]

        
        # Evaluate configurations and keep results
        # we create a list of tuples with the configuration and the results
        results = []

        for epochs in epoch_values:
            for num_workers in worker_counts:
                for learning_rate in learning_rates:
                    for reg_lambda in reg_lambdas:

                        model, time_taken, train_accuracy, valid_accuracy, test_accuracy = evaluate_configuration(
                            train_data,
                            valid_data,
                            test_data,
                            num_workers,
                            epochs,
                            learning_rate,
                            reg_lambda
                        )

                        results.append((num_workers, epochs, time_taken, train_accuracy, valid_accuracy, test_accuracy))

        # get best configuration checking best validation accuracy
        best_config = max(results, key=lambda x: x[4])
        num_workers, epochs, time_taken, train_accuracy, valid_accuracy, test_accuracy = best_config

            
        # print best configuration
        print(f"\nBest configuration:")
        print(f"Workers: {num_workers}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Regularization lambda: {reg_lambda}")
        print(f"Training time: {time_taken:.2f} seconds")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Validation accuracy: {valid_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")





if __name__ == '__main__':
    unittest.main()