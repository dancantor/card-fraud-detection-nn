import pandas as pd
import numpy as np
import Model

from Model.preprocessing import Preprocessing
from Model.multilayer_perceptron import MultiLayerPerceptron
# Our data is already scaled we should split our training and test sets
from sklearn.model_selection import train_test_split

# This is explicitly used for undersampling.

df = pd.read_csv('creditcard.csv')
df['Time'] = Preprocessing.standard_scale(df['Time'])
df['Amount'] = Preprocessing.standard_scale(df['Amount'])
df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()

# Define activation functions for each layer (including output layer)
activations = [MultiLayerPerceptron._relu, MultiLayerPerceptron._relu, MultiLayerPerceptron._sigmoid]

# Create an instance of the MLP
mlp = MultiLayerPerceptron(input_size=30, hidden_layers=[15, 15], output_size=1, activations=activations)

# Train the MLP (X_train and y_train should be numpy arrays of your dataset)
learning_rate = 0.001
epochs = 30
batch_size = 1
regularization_lambda = 0.001
X = new_df.drop('Class', axis=1)
y = new_df['Class']
X_all = df.drop('Class', axis=1)
Y_all = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)
y_train = np.array([np.array([value]) for value in y_train.values])
y_test = np.array([np.array([value]) for value in y_test.values])
y_train_all = np.array([np.array([value]) for value in y_train_all.values])
y_test_all = np.array([np.array([value]) for value in y_test_all.values])
mlp.train(X_train.values, y_train, learning_rate, epochs, batch_size, regularization_lambda)
# print(f"Cross Validation :{mlp.k_fold_cross_validation(X_train.values, y_train)}")

# Evaluate the model (X_test and y_test should be numpy arrays of your test dataset)
TP, FP, FN, TN = mlp.calculate_confusion_matrix(X_test, y_test)
print(f"Confusion Matrix: [{TP}, {FP}, {FN}, {TN}]")

accuracy = mlp.evaluate_accuracy(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

auc_roc = mlp.evaluate_auc_roc(X_test_all, y_test_all)
print(f"Model AUC-ROC: {auc_roc}")

precision = mlp.evaluate_precision(X_test_all, y_test_all)
print(f"Model Precision: {precision}")

recall = mlp.evaluate_recall(X_test_all, y_test_all)
print(f"Model Recall: {recall}")

f1 = mlp.evaluate_f1(X_test_all, y_test_all)
print(f"Model f1: {f1}")

lower_bound, upper_bound = mlp.bootstrap_auc_confidence_interval(n_bootstrap=1000, ci=80)
print(f"95% Confidence Interval for AUC-ROC: {lower_bound:.2f} to {upper_bound:.2f}")

# Save the model
#mlp.save_model("mlp_model.pkl")

# Load the model
#mlp.load_model("mlp_model.pkl")

# Predict new data
df = df[df['Class'] == 0]
sample = df.sample()
label = sample['Class']
sample = sample.drop('Class', axis=1)
predictions = mlp.predict(sample.values)
print(predictions)
print(label)
# print(df.head())

# [0.9689714377591216, 0.9775300698616093, 0.970354089495319, 0.9847205490304853, 0.9754211265260184]
