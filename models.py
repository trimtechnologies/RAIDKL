import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


def build_1dcnn_large(input_shape, num_classes_device, num_classes_attack=None):
    """Build a large 1D-CNN for the teacher model."""
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    features = Dense(256, activation='relu', name='features')(x)
    x = Dropout(0.5)(features)

    if num_classes_attack:
        output_device = Dense(num_classes_device, activation='softmax', name='device')(x)
        output_attack = Dense(num_classes_attack, activation='softmax', name='attack')(x)
        model = Model(inputs, [output_device, output_attack])
    else:
        output = Dense(num_classes_device, activation='softmax')(x)
        model = Model(inputs, output)

    return model


def build_1dcnn_small(input_shape, num_classes_device, num_classes_attack=None):
    """Build a small 1D-CNN for the student model with feature output."""
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    features = Dense(128, activation='relu', name='features')(x)
    x = Dropout(0.3)(features)

    if num_classes_attack:
        output_device = Dense(num_classes_device, activation='softmax', name='device')(x)
        output_attack = Dense(num_classes_attack, activation='softmax', name='attack')(x)
        model = Model(inputs, [output_device, output_attack])
        feature_model = Model(inputs, features)
    else:
        output = Dense(num_classes_device, activation='softmax')(x)
        model = Model(inputs, output)
        feature_model = Model(inputs, features)

    return model, feature_model