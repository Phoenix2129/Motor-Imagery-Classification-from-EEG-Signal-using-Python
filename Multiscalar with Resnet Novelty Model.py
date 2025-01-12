import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and remove unnecessary columns from the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    eeg_data = data.drop(columns=["time", "Label"])
    labels = data["Label"].values

    # Check for NaN or Infinite values in the data
    if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
        raise ValueError("Data contains NaN or Inf values")

    # Standardize the EEG data
    scaler = StandardScaler()
    eeg_data_normalized = scaler.fit_transform(eeg_data)
    return np.array(eeg_data_normalized), labels

# Segment the data into non-overlapping windows
def segment_data(data, labels, window_size):
    num_samples = data.shape[0]
    segmented_data, segmented_labels = [], []

    for start_idx in range(0, num_samples - window_size + 1, window_size):
        end_idx = start_idx + window_size
        segmented_data.append(data[start_idx:end_idx])
        segmented_labels.append(labels[start_idx])  # First sample's label for each window

    return np.array(segmented_data), np.array(segmented_labels)

# Create a Multi-Scalar ResNet model
def create_multi_scalar_resnet_feature_extractor(input_shape):
    input_layer = layers.Input(shape=input_shape)

    def multi_scalar_resnet_block(x, filters=64, kernel_sizes=[3, 5, 7]):
        branch_outputs = []
        for kernel_size in kernel_sizes:
            branch = layers.Conv1D(filters, kernel_size, padding='same', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
            branch = layers.BatchNormalization()(branch)
            branch = layers.Conv1D(filters, kernel_size, padding='same', activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01))(branch)
            branch = layers.BatchNormalization()(branch)
            branch = layers.GlobalAveragePooling1D()(branch)
            branch_outputs.append(branch)

        merged = layers.concatenate(branch_outputs, axis=-1)
        return merged

    x = multi_scalar_resnet_block(input_layer)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)

    model = models.Model(inputs=input_layer, outputs=x)
    return model

# Training and evaluation function for each combination
def train_and_evaluate_for_combination(class_a, class_b, files, window_size=500, input_shape=(500, 22)):
    segmented_data, segmented_labels = [], []

    for file in files:
        data, labels = load_data(file)
        data, labels = segment_data(data, labels, window_size)
        segmented_data.append(data)
        segmented_labels.append(labels)

    X = np.concatenate(segmented_data)
    y = np.concatenate(segmented_labels)

    # Filter data for the given class combination
    mask = (y == class_a) | (y == class_b)
    X = X[mask]
    y = y[mask]

    # Map class_a to 0 and class_b to 1
    y = np.where(y == class_a, 0, 1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and compile feature extraction model
    feature_extractor = create_multi_scalar_resnet_feature_extractor(input_shape)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    feature_extractor.compile(optimizer=optimizer, loss='mean_squared_error')

    # Extract features
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)

    # Save features to CSV files
    train_features_df = pd.DataFrame(X_train_features)
    test_features_df = pd.DataFrame(X_test_features)
    train_features_df['Label'] = y_train
    test_features_df['Label'] = y_test

    train_features_df.to_csv(f'class_{class_a}vs{class_b}_train_features.csv', index=False)
    test_features_df.to_csv(f'class_{class_a}vs{class_b}_test_features.csv', index=False)

    # Define binary classifier
    binary_classifier = models.Sequential([
        layers.Input(shape=(X_train_features.shape[1],)),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.6),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    binary_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # Train the binary classifier
    history = binary_classifier.fit(
        X_train_features, y_train,
        validation_data=(X_test_features, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the classifier
    test_loss, test_accuracy = binary_classifier.evaluate(X_test_features, y_test)
    print(f"Test Accuracy for classes {class_a} vs {class_b}: {test_accuracy:.4f}")

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Learning Curve (Loss) for classes {class_a} vs {class_b}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Learning Curve (Accuracy) for classes {class_a} vs {class_b}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Predictions and confusion matrix
    y_pred = (binary_classifier.predict(X_test_features) > 0.5).astype("int32").flatten()

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for classes {class_a} vs {class_b}:\n", conf_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[class_a, class_b], yticklabels=[class_a, class_b])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for classes {class_a} vs {class_b}')
    plt.show()

    # Classification report
    print(classification_report(y_test, y_pred, target_names=[f"Class {class_a}", f"Class {class_b}"]))

    return test_accuracy

# File paths for the classes (adjust file paths as needed)
files = [
    '/content/drive/MyDrive/Colab Notebooks/class_1.0.csv',
    '/content/drive/MyDrive/Colab Notebooks/class_2.0.csv',
    '/content/drive/MyDrive/Colab Notebooks/class_3.0.csv',
    '/content/drive/MyDrive/Colab Notebooks/class_4.0.csv'
]

# List of class combinations to test
class_combinations = [
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4)
]

# Run the training and evaluation for each combination and calculate average accuracy
accuracies = []
for class_a, class_b in class_combinations:
    accuracy = train_and_evaluate_for_combination(class_a, class_b, files)
    accuracies.append(accuracy)

# Calculate and print the average accuracy
avg_accuracy = np.mean(accuracies)
print(f"Average Accuracy across all combinations: {avg_accuracy:.4f}")
