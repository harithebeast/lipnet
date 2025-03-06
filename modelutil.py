import os 
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam

def ctc_loss(y_true, y_pred):
    """CTC loss function."""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def load_lipnet_model() -> Sequential:
    """Load the pre-trained model from the h5 file."""
    try:
        model = load_model(os.path.join('models', 'model.h5'))
        # Compile the model with CTC loss and Adam optimizer
        model.compile(optimizer=Adam(learning_rate=0.0001), loss=ctc_loss)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None