import numpy as np
import pickle
from typing import List, Callable
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


class MultiLayerPerceptron:
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, activations: List[Callable]):
        self._input_size = input_size
        self._hidden_layers = hidden_layers
        self._output_size = output_size

        # Initialize weights and biases for hidden layers and output layer
        self._weights = [self.he_initialization(prev_size, curr_size)
                         for prev_size, curr_size in
                         zip([self._input_size] + self._hidden_layers, self._hidden_layers + [self._output_size])]
        self._biases = [np.zeros((1, size)) for size in self._hidden_layers + [self._output_size]]

        # Custom activation functions for each layer
        if len(activations) != len(hidden_layers) + 1:
            raise ValueError("Number of activation functions must match number of layers")
        self._activation_functions = activations
        self._auc_roc_scores = [0.9689714377591216, 0.9775300698616093, 0.970354089495319, 0.9847205490304853, 0.9754211265260184]
    def he_initialization(self, input_dim, output_dim):
        var = 2.0 / input_dim
        return np.random.normal(0.0, var, (input_dim, output_dim))

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _binary_crossentropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _binary_crossentropy_derivative(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def _feed_forward(self, x):
        activations = [x]
        for i in range(len(self._weights)):
            z = np.dot(activations[-1], self._weights[i]) + self._biases[i]
            a = self._activation_functions[i](z)
            activations.append(a)
        return activations

    def _clip_gradients(self, gradients, max_norm):
        total_norm = np.sqrt(sum(np.sum(np.square(g)) for g in gradients))
        if total_norm <= max_norm:
            return gradients
        scale = max_norm / total_norm
        clipped_gradients = [g * scale for g in gradients]
        return clipped_gradients

    def _backpropagation(self, y, activations, learning_rate, regularization_lambda):
        deltas = [self._binary_crossentropy_derivative(y, activations[-1])]
        for i in reversed(range(len(self._weights) - 1)):
            delta = np.dot(deltas[0], self._weights[i + 1].T) * (activations[i + 1] > 0)  # ReLU derivative
            deltas.insert(0, delta)

        # Update weights and biases
        gradients = [np.dot(activations[i].T, deltas[i]) for i in range(len(self._weights))]
        max_gradient_norm = 5.0
        clipped_gradients = self._clip_gradients(gradients, max_gradient_norm)
        m = y.shape[0]  # number of samples
        for i in range(len(self._weights)):
            self._weights[i] -= (learning_rate / m) * clipped_gradients[i] + (
                        regularization_lambda / m) * self._weights[i]
            self._biases[i] -= (learning_rate / m) * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X_train, y_train, learning_rate=0.001, epochs=30, batch_size=1, regularization_lambda=0.001):
        for epoch in range(epochs):
            # Mini-batch gradient descent
            for start_idx in range(0, len(X_train) - batch_size + 1, batch_size):
                end_idx = min(start_idx + batch_size, len(X_train))
                batch_x = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]

                activations = self._feed_forward(batch_x)
                self._backpropagation(batch_y, activations, learning_rate, regularization_lambda)
            print(f'Epoch {epoch} completed')

    def k_fold_cross_validation(self, X, y, k=5):
        fold_size = int(len(X) / k)
        scores = []

        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size
            X_val, y_val = X[start:end], y[start:end]
            X_train = np.concatenate([X[:start], X[end:]], axis=0)
            y_train = np.concatenate([y[:start], y[end:]], axis=0)

            self.train(X_train, y_train)  # Train the model
            score = self.evaluate_auc_roc(X_val, y_val)  # Evaluate the model
            scores.append(score)

        return scores

    def predict(self, x):
        return self._feed_forward(x)[-1]

    def calculate_confusion_matrix(self, X_test, y_test):
        y_pred = self.predict(X_test) > 0.5
        y_true = y_test
        TP = sum((y_true == 1) & (y_pred == 1))
        FP = sum((y_true == 0) & (y_pred == 1))
        FN = sum((y_true == 1) & (y_pred == 0))
        TN = sum((y_true == 0) & (y_pred == 0))

        return TP, FP, FN, TN

    def evaluate_accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        predicted_labels = predictions > 0.5
        accuracy = np.mean(predicted_labels == y_test)
        return accuracy

    def evaluate_precision(self, X_test, y_test):
        TP, FP, _, _ = self.calculate_confusion_matrix(X_test, y_test)
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    def evaluate_recall(self, X_test, y_test):
        TP, _, FN, _ = self.calculate_confusion_matrix(X_test, y_test)
        return TP / (TP + FN) if (TP + FN) != 0 else 0

    def evaluate_f1(self, X_test, y_test):
        precision = self.evaluate_precision(X_test, y_test)
        recall = self.evaluate_recall(X_test, y_test)
        return 2 * precision * recall / (precision + recall)

    def calculate_tpr_fpr(self, y_true, y_scores):
        # Concatenate true labels with predicted scores and sort by the scores
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted, y_scores_sorted = y_true[sorted_indices], y_scores[sorted_indices]

        # Initialize variables
        TPs, FPs = 0, 0
        total_positives = np.sum(y_true)
        total_negatives = len(y_true) - total_positives
        TPRs, FPRs = [0], [0]

        # Calculate TPR and FPR for each threshold
        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                TPs += 1
            else:
                FPs += 1

            # Update TPRs and FPRs only when the score changes
            if i == len(y_true_sorted) - 1 or y_scores_sorted[i] != y_scores_sorted[i + 1]:
                TPRs.append(TPs / total_positives)
                FPRs.append(FPs / total_negatives)

        return TPRs, FPRs

    def evaluate_auc_roc(self, X_test, y_test):
        predictions = self.predict(X_test)
        y_true = y_test.ravel()
        y_scores = predictions.ravel()
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        TPRs, FPRs = self.calculate_tpr_fpr(y_true, y_scores)
        auc = 0
        for i in range(1, len(TPRs)):
            # Calculate the area using the trapezoidal rule
            auc += (FPRs[i] - FPRs[i - 1]) * (TPRs[i] + TPRs[i - 1]) / 2
        return auc

    def bootstrap_auc_confidence_interval(self, n_bootstrap=1000, ci=95):
        bootstrap_samples = np.random.choice(self._auc_roc_scores, (n_bootstrap, len(self._auc_roc_scores)), replace=True)
        bootstrap_metrics = np.mean(bootstrap_samples, axis=1)
        lower_bound = np.percentile(bootstrap_metrics, (100 - ci) / 2)
        upper_bound = np.percentile(bootstrap_metrics, 100 - (100 - ci) / 2)

        return lower_bound, upper_bound

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump({'weights': self._weights, 'biases': self._biases}, file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self._weights = data['weights']
            self._biases = data['biases']
