import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, datasets
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the California housing dataset
(X_train, y_train), (X_test, y_test) = datasets.california_housing.load_data(
    version="large", path="california_housing.npz", test_split=0.2, seed=113
)

# Scaling using MinMax 
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the model
def build_model():
    model = models.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=losses.MeanAbsoluteError(),
        metrics=[metrics.MeanAbsolutePercentageError(name='mape')]
    )
    return model

# Train and evaluate
def main():
    model = build_model()

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1
    )

    test_loss, test_mape = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss (MSE): {test_loss:.4f}")
    print(f"Test MAPE: {test_mape:.4f}%")


if __name__ == "__main__":
    main()

