from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_model(model_name: str, lookback: int, n_features: int) -> tf.keras.Model:
    """Build a multitask model that predicts next-session log return and direction."""
    name = model_name.lower().replace("-", "_")
    inputs = layers.Input(shape=(lookback, n_features), name="market_sequence")
    x = inputs

    if name == "cnn":
        x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
        x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
    elif name == "rnn":
        x = layers.SimpleRNN(
            48, dropout=0.1, kernel_regularizer=regularizers.l2(1e-5)
        )(x)
    elif name == "lstm":
        x = layers.LSTM(
            48, dropout=0.1, kernel_regularizer=regularizers.l2(1e-5)
        )(x)
    elif name in {"cnn_lstm", "hybrid"}:
        x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(48, dropout=0.1)(x)
    else:
        raise ValueError("Unknown model. Choose cnn, rnn, lstm, or cnn_lstm.")

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    log_return = layers.Dense(1, name="log_return")(x)
    direction = layers.Dense(1, activation="sigmoid", name="direction")(x)

    model = models.Model(
        inputs=inputs,
        outputs={"log_return": log_return, "direction": direction},
        name=name,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"log_return": "mse", "direction": "binary_crossentropy"},
        loss_weights={"log_return": 1.0, "direction": 0.25},
        metrics={
            "log_return": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "direction": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        },
    )
    return model
