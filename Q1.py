# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.optimizers import Adagrad
from keras.optimizers import RMSprop
from keras.optimizers import Adam

# Load data from CSV file
num_classes = 5
tdata=pd.read_csv("train_data.csv",header=None)
train_data=tdata.values
tlabels=pd.read_csv("train_label.csv",header=None)
train_labels=tlabels.values
tedata=pd.read_csv("test_data.csv",header=None)
test_data=tedata.values
telabels=pd.read_csv("test_label.csv",header=None)
test_labels=telabels.values
vdata=pd.read_csv("val_data.csv",header=None)
val_data=vdata.values
vlabels=pd.read_csv("val_label.csv",header=None)
val_labels=vlabels.values
train_labels=to_categorical(train_labels, num_classes)
test_labels=to_categorical(test_labels, num_classes)
val_labels=to_categorical(val_labels, num_classes)

# Define parameters
input_size = train_data.shape[1]
num_classes = tlabels.iloc[:, 0].nunique()
learning_rate = 0.01
num_epochs = 50
batch_size = 32
stopping_threshold = 0.001

# Define weight update rules
update_rules = ['Delta', 'AdaGrad', 'RMSProp', 'AdaM']

# Define function to build model
def build_model():
    model = Sequential([
        Dense(64, input_shape=(input_size,), activation='tanh'),  # Hidden layer 1
        Dense(32, activation='tanh'),  # Hidden layer 2
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model

# Define function to train model with a specific weight update rule
def train_model(model, update_rule):
    # Compile model
    if update_rule == 'Delta':
        optimizer = SGD(learning_rate=learning_rate)
    elif update_rule == 'AdaGrad':
        optimizer = Adagrad(learning_rate=learning_rate, epsilon=1e-8)
    elif update_rule == 'RMSProp':
        optimizer = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-8)
    else:
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # # Define early stopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')
    
    # Train model
    history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=0, validation_data=(val_data, val_labels),callbacks=[early_stopping])
    
    return history

def accuracy_from_confusion_matrix(confusion_matrix):
    diagonal_sum = np.trace(confusion_matrix)
    total_sum = np.sum(confusion_matrix)
    accuracy = diagonal_sum / total_sum
    return accuracy

# Main loop to train models with different weight update rules
for update_rule in update_rules:
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, update_rule)
    
    # Plot average error vs Epoch
    plt.plot(history.history['loss'], label=update_rule)

    # Calculate confusion matrix for training data
    train_pred = model.predict(train_data)
    train_pred_labels = np.argmax(train_pred, axis=1)
    train_true_labels = np.argmax(train_labels, axis=1)
    train_conf_matrix = confusion_matrix(train_true_labels, train_pred_labels)
    print(f'Confusion matrix for training data with {update_rule}:\n{train_conf_matrix}')
    # Calculate accuracy for training data
    train_accuracy = accuracy_from_confusion_matrix(train_conf_matrix)
    print("Train Accuracy:", train_accuracy)

    # Calculate confusion matrix for test data
    test_pred = model.predict(test_data)
    test_pred_labels = np.argmax(test_pred, axis=1)
    test_true_labels = np.argmax(test_labels, axis=1)
    test_conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
    print(f'Confusion matrix for test data with {update_rule}:\n{test_conf_matrix}')
    # Calculate accuracy for test data
    test_accuracy = accuracy_from_confusion_matrix(test_conf_matrix)
    print("Test Accuracy:", test_accuracy)

# Plot settings
plt.xlabel('Epoch')
plt.ylabel('Average Error')
plt.title('Average Error on Training Data vs Epoch')
plt.legend()
plt.show()