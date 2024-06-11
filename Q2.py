import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam

# Define MLFFNN model with 2 hidden layers and tanh activation function
def build_model(normalization_method):
    model = Sequential()
    
    # Input layer
    model.add(Dense(64, input_dim=input_size, activation='tanh'))
    
    # Hidden layers
    if normalization_method == 'batch_norm':
        model.add(BatchNormalization())
    model.add(Dense(32, activation='tanh'))
    if normalization_method == 'batch_norm':
        model.add(BatchNormalization())

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Train the model
def train_model(model, train_data, train_labels, test_data, test_labels, num_epochs, batch_size, stopping_threshold):
    train_errors = []  # List to store average error on training data for each epoch
    test_errors = []   # List to store average error on test data for each epoch
    
    for epoch in range(num_epochs):
        # Shuffle training data and labels
        indices = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[indices]
        train_labels_shuffled = train_labels[indices]
        
        # Mini-batch training loop
        total_error = 0 
        num_batches = len(train_data) // batch_size
        for i in range(num_batches):
            batch_data = train_data_shuffled[i * batch_size: (i + 1) * batch_size]
            batch_labels = train_labels_shuffled[i * batch_size: (i + 1) * batch_size]
            
            # Train on batch
            batch_loss = model.train_on_batch(batch_data, batch_labels)
            total_error += batch_loss[0]
        
        # Calculate average error for training data
        average_train_error = total_error / num_batches
        train_errors.append(average_train_error)
        
        # Evaluate model on test data
        test_loss = model.evaluate(test_data, test_labels, verbose=0)
        test_errors.append(test_loss)
        
        #print("Epoch {}/{} - Training Error: {:.4f} - Test Error: {:.4f}".format(epoch+1, num_epochs, average_train_error[0], test_loss[0]))
        
        convergence_epoch=epoch
        # Check stopping criterion
        if epoch > 0 and abs(train_errors[-1] - train_errors[-2]) < stopping_threshold:
            print("Stopping criterion met. Training stopped.")
            break
    
    # Plot average error on training and test data vs. epoch
    plt.plot(range(1, len(train_errors) + 1), train_errors, label='Training Error')
    plt.plot(range(1, len(test_errors) + 1), [test_error[0] for test_error in test_errors], label='Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Average Error')
    plt.title('Average Error vs. Epoch')
    plt.legend()
    plt.show()

    # Generate predictions on training and test data
    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)

    # Calculate confusion matrices for training and test data
    train_confusion_matrix = confusion_matrix(np.argmax(train_labels, axis=1), np.argmax(train_predictions, axis=1))
    test_confusion_matrix = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(test_predictions, axis=1))

    return train_confusion_matrix, test_confusion_matrix, convergence_epoch

def accuracy_from_confusion_matrix(confusion_matrix):
  diagonal_sum = np.trace(confusion_matrix)
  total_sum = np.sum(confusion_matrix)
  accuracy = diagonal_sum / total_sum
  return accuracy

# Load data from CSV file
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
num_epochs = 100
batch_size = 32
stopping_threshold = 0.001

# Build and compile the model with no normalization
model_no_norm = build_model(normalization_method=None)
model_no_norm.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with no normalization
train_confusion_matrix_no_norm, test_confusion_matrix_no_norm, convergence_epoch_no_norm = train_model(model_no_norm, train_data, train_labels, test_data, test_labels, num_epochs, batch_size, stopping_threshold)

print(train_confusion_matrix_no_norm)
print(test_confusion_matrix_no_norm)
# Calculate accuracy for training data
train_accuracy = accuracy_from_confusion_matrix(train_confusion_matrix_no_norm)
print("Train Accuracy:", train_accuracy)
# Calculate accuracy for test data
test_accuracy = accuracy_from_confusion_matrix(test_confusion_matrix_no_norm)
print("Test Accuracy:", test_accuracy)

# Build and compile the model with batch normalization
model_batch_norm = build_model(normalization_method='batch_norm')
model_batch_norm.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with batch normalization
train_confusion_matrix_batch_norm, test_confusion_matrix_batch_norm, convergence_epoch_batch_norm = train_model(model_batch_norm, train_data, train_labels, test_data, test_labels, num_epochs, batch_size, stopping_threshold)

print(train_confusion_matrix_batch_norm)
print(test_confusion_matrix_batch_norm)
# Calculate accuracy for training data
train_accuracy = accuracy_from_confusion_matrix(train_confusion_matrix_batch_norm)
print("Train Accuracy:", train_accuracy)
# Calculate accuracy for test data
test_accuracy = accuracy_from_confusion_matrix(test_confusion_matrix_batch_norm)
print("Test Accuracy:", test_accuracy)

print("Convergence epoch (No normalization):", convergence_epoch_no_norm)
print("Convergence epoch (Batch normalization):", convergence_epoch_batch_norm)