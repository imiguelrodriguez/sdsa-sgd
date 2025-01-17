from unittest import TestCase
import unittest
from sgd import logistic, dot, predict, accuracy, submission, extract_features
from data import load_adult_train_data, load_adult_valid_data

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
        
        # epoch_values = [1, 5, 10, 20, 50, 100]  # List of different epoch values to test
        
        epoch_values = [100]

        for epochs in epoch_values:
            model = submission(train_data, num_workers=4, lam=0.01, epochs=epochs)  # Pass epochs here
            
            # Calculate training accuracy
            train_predictions = [predict(model, p) for p in train_data]
            train_accuracy = accuracy(train_data, train_predictions)
            
            # Calculate validation accuracy
            valid_predictions = [predict(model, p) for p in valid_data]
            valid_accuracy = accuracy(valid_data, valid_predictions)
            
            # Calculate test accuracy
            test_predictions = [predict(model, p) for p in test_data]
            test_accuracy = accuracy(test_data, test_predictions)
            
            print(f"Epochs: {epochs}")
            print("Training Accuracy:", train_accuracy)
            print("Validation Accuracy:", valid_accuracy)
            print("Test Accuracy:", test_accuracy)
            print()

if __name__ == '__main__':
    unittest.main()