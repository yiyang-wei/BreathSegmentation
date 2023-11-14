import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

print(tf.config.list_physical_devices('GPU'))

def res_block(inputs, norm_type, activation, dropout, ff_dim):
  """Residual block of TSMixer."""

  norm = (
      layers.LayerNormalization
      if norm_type == 'L'
      else layers.BatchNormalization
  )

  # Temporal Linear
  x = norm(axis=[-2, -1])(inputs)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(x.shape[-1], activation=activation)(x)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
  x = layers.Dropout(dropout)(x)
  res = x + inputs

  # Feature Linear
  x = norm(axis=[-2, -1])(res)
  x = layers.Dense(ff_dim, activation=activation)(
      x
  )  # [Batch, Input Length, FF_Dim]
  x = layers.Dropout(dropout)(x)
  x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
  x = layers.Dropout(dropout)(x)
  return x + res

def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
  """Build TSMixer model."""

  inputs = tf.keras.Input(shape=input_shape)
  x = inputs  # [Batch, Input Length, Channel]
  for _ in range(n_block):
    x = res_block(x, norm_type, activation, dropout, ff_dim)

  if target_slice:
    x = x[:, :, target_slice]

  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
  outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

  return tf.keras.Model(inputs, outputs)

def train_model(
    model,
    train_dataset,
    val_dataset,
    loss_fn,
    optimizer,
    epochs,
    patience,
    min_delta,
    model_dir,
):
    """Train TSMixer model."""

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['mae'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_dir,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

    return model


if __name__ == "__main__":
    X = np.random.randn(60, 200, 10) + 10
    dX = np.concatenate((np.ones((60, 1, 10)), np.diff(X, axis=1)), axis=1)
    y = X ** (np.sqrt(abs(dX))) + np.log2(abs(dX) + 1) * (np.random.randn(60, 200, 10) + 10)

    print(X.shape)
    print(y.shape)

    model = build_model(
        input_shape=X.shape[1:],
        pred_len=200,
        norm_type='L',
        activation='relu',
        n_block=7,
        dropout=0.05,
        ff_dim=128,
        target_slice=None,
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((X[:-1], y[:-1])).batch(59)
    val_dataset = tf.data.Dataset.from_tensor_slices((X[-1:], y[-1:])).batch(1)

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    epochs = 200
    patience = 10
    min_delta = 0.00001
    model_dir = './model'

    # use cross validation
    model = train_model(
        model,
        train_dataset,
        val_dataset,
        loss_fn,
        optimizer,
        epochs,
        patience,
        min_delta,
        model_dir=model_dir
    )

    # evaluate
    model.evaluate(val_dataset)


