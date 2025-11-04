import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# Loading the dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values (0–255) to (0–1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define constants
img_height, img_width, num_channels = 32, 32, 3
num_classes = 10


def build_model():
    model = Sequential([
        Conv2D(128, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.3),  
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  
        metrics=['accuracy']
    )
    return model

# Initialization and training the model
model = build_model()
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=15,   
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Checking the accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

