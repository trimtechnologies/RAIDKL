import tensorflow as tf
from pathlib import Path
import os
from imblearn.over_sampling import SMOTE
import numpy as np

import tensorflow as tf
import os
import json

def count_flops(model, X_train=None, input_shape=None):
    """Manually compute the approximate FLOPs for the given 1D-CNN model."""
    if input_shape is None and X_train is None:
        raise ValueError("Either input_shape or X_train must be provided to compute FLOPs")
    if input_shape is None:
        input_shape = (None, X_train.shape[1], X_train.shape[2]) if X_train is not None else None

    # Build the model with the provided input shape if not already built
    if not model.built:
        model.build(input_shape)

    # Use real input shape from X_train if available
    if X_train is not None:
        batch_size = 1  # Use 1 for FLOPs, scales with batch size
        input_height = X_train.shape[1]
        input_channels = X_train.shape[2]
    else:
        batch_size = 1
        input_height = input_shape[1]
        input_channels = input_shape[2]

    flops = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            # Calculate output height: (input_height - kernel_size + 2 * padding) / stride + 1
            padding = layer.padding.upper() == 'SAME' and (layer.kernel_size[0] - 1) or 0
            stride = layer.strides[0]
            output_height = max(1, (input_height - layer.kernel_size[0] + 2 * padding) // stride + 1)

            # FLOPs = 2 * (input_channels * kernel_size * filters * output_height) + bias
            conv_flops = 2 * input_channels * layer.kernel_size[0] * layer.filters * output_height
            if layer.use_bias:
                conv_flops += layer.filters * output_height
            flops += conv_flops

            # Update input dimensions
            input_height = output_height
            input_channels = layer.filters
        elif isinstance(layer, tf.keras.layers.MaxPooling1D):
            # Calculate output height after pooling
            padding = layer.padding.upper() == 'SAME' and (layer.pool_size[0] - 1) or 0
            stride = layer.strides[0] if layer.strides else layer.pool_size[0]
            input_height = max(1, (input_height - layer.pool_size[0] + 2 * padding) // stride + 1)
            # Pooling itself has negligible FLOPs
        elif isinstance(layer, tf.keras.layers.Flatten):
            # FLOPs = number of elements (no computation, just reshaping)
            flatten_elements = input_height * input_channels
            flops += flatten_elements  # Count as 1 FLOP per element for data movement
            input_height = 1  # Flatten to 1D
            input_channels *= input_height  # Total elements become new input channels
        elif isinstance(layer, tf.keras.layers.Dense):
            # FLOPs = 2 * (input_units * output_units) + bias
            input_units = input_height * input_channels if input_channels else layer.input_shape[-1]
            output_units = layer.units
            dense_flops = 2 * input_units * output_units
            if layer.use_bias:
                dense_flops += output_units
            flops += dense_flops

            # Update input dimensions for the next layer
            input_height = 1
            input_channels = layer.units
        elif isinstance(layer, tf.keras.layers.Dropout):
            # Dropout has negligible FLOPs (just masking)
            continue
        elif isinstance(layer, tf.keras.layers.Activation):
            # FLOPs = number of output elements (e.g., ReLU)
            output_elements = input_height * input_channels
            flops += output_elements

    # Handle multi-output case (device and attack outputs)
    if isinstance(model.output, (list, tuple)):
        for output in model.output:
            if isinstance(output, tf.keras.layers.Dense):
                input_units = input_height * input_channels
                output_units = output.units
                output_flops = 2 * input_units * output_units
                if output.use_bias:
                    output_flops += output_units
                flops += output_flops
    else:
        # Single output case
        if isinstance(model.output, tf.keras.layers.Dense):
            input_units = input_height * input_channels
            output_units = model.output.units
            output_flops = 2 * input_units * output_units
            if model.output.use_bias:
                output_flops += output_units
            flops += output_flops

    return flops

def count_flops_1(model, X_train=None, input_shape=None):
    """Compute the approximate FLOPs for the given model using tf.profiler.experimental."""
    if input_shape is None and X_train is None:
        raise ValueError("Either input_shape or X_train must be provided to compute FLOPs")
    if input_shape is None:
        input_shape = (None, X_train.shape[1], X_train.shape[2]) if X_train is not None else None

    # Build the model with the provided input shape if not already built
    if not model.built:
        model.build(input_shape)

    # Use a small batch of real data if available, otherwise fall back to dummy input
    if X_train is not None and X_train.shape[0] > 0:
        # Take a small batch of real data (e.g., first 4 samples for better coverage)
        real_input = X_train[:4].astype(np.float32)
    else:
        # Fallback to dummy input if no real data is provided
        real_input = tf.zeros((4,) + input_shape[1:], dtype=tf.float32)

    # Wrap the model call in a tf.function for tracing
    @tf.function
    def model_pass(x):
        return model(x, training=False)

    # Warm up and trigger tracing with multiple passes
    for _ in range(3):  # Increased warm-up passes
        model_pass(real_input)

    # Profile the model with detailed options
    logdir = 'logs/profile'
    options = tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=3,  # Maximum detailed tracing
        python_tracer_level=2,
        device_tracer_level=2,
        delay_ms=0,  # Start immediately
        #duration_ms=1000  # Profile for 1 second
    )
    with tf.profiler.experimental.Profile(logdir, options=options):
        for _ in range(5):  # Increased profiling passes
            model_pass(real_input)

    # Load the profile data and extract FLOPs
    try:
        # Check for profile files
        profile_files = [f for f in os.listdir(logdir) if f.endswith('.json') or f.endswith('.profile')]
        if not profile_files:
            print(f"No profile data found in {logdir}")
            return 0

        # Load the first available profile file
        profile_path = os.path.join(logdir, profile_files[0])
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        print(f"Profile data keys: {profile_data.keys()}")  # Debug: Print available keys
        print(f"Full profile data: {json.dumps(profile_data, indent=2)}")  # Debug: Print full data

        # Extract FLOPs from the profile data
        flops = 0
        if 'traceEvents' in profile_data:
            for event in profile_data['traceEvents']:
                if 'args' in event and 'float_ops' in event['args']:
                    flops += event['args']['float_ops']
        elif 'run_metadata' in profile_data and 'step_stats' in profile_data['run_metadata']:
            flops = sum(step.get('float_ops', 0) for step in profile_data['run_metadata']['step_stats'])
        else:
            print(f"Unexpected profile data structure: {profile_data.keys()}")
            return 0
    except Exception as e:
        print(f"Error processing profile data: {e}")
        return 0

    return flops

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])


# def ensure_directory(directory):
#     """Ensure a directory exists and return its Path object."""
#     directory = Path(directory)
#     directory.mkdir(parents=True, exist_ok=True)
#     return directory

def ensure_directory(directory):
    """Create directory if it doesn't exist and verify writability."""
    directory = Path(directory)
    if not directory.exists():
        try:
            directory.mkdir(parents=True)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            raise
    if not os.access(directory, os.W_OK):
        print(f"Error: Directory {directory} is not writable.")
        raise PermissionError(f"Directory {directory} is not writable.")
    return directory

def apply_smote_balancing(X, y, output_mode, le_device=None, le_attack=None):
    """
    Apply SMOTE to balance the training dataset based on the specified output mode.

    Args:
        X (np.ndarray): Feature data.
        y (np.ndarray or tuple): Target labels (single array for 'device' or 'traffic', tuple of arrays for 'multi').
        output_mode (str): 'device', 'traffic', or 'multi' to determine balancing scope.
        le_device (LabelEncoder, optional): Encoder for device labels.
        le_attack (LabelEncoder, optional): Encoder for attack labels.

    Returns:
        tuple: Balanced X_balanced (np.ndarray) and y_balanced (np.ndarray or tuple).
    """
    if output_mode not in ['device', 'traffic', 'multi']:
        raise ValueError("output_mode must be 'device', 'traffic', or 'multi'")

    # Ensure X is 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    elif len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    if output_mode == 'multi':
        if not isinstance(y, (tuple)) or len(y) != 2:
            raise ValueError("For 'multi' mode, y must be a tuple of two arrays (e.g., (y_device, y_attack))")
        y_device, y_attack = y

        # Apply SMOTE to device labels
        smote_device = SMOTE(random_state=42)
        X_balanced_device, y_balanced_device = smote_device.fit_resample(X, y_device)

        # Apply SMOTE to attack labels
        smote_attack = SMOTE(random_state=42)
        X_balanced_attack, y_balanced_attack = smote_attack.fit_resample(X, y_attack)

        # Concatenate balanced datasets (using the union of indices)
        max_samples = max(X_balanced_device.shape[0], X_balanced_attack.shape[0])
        X_balanced = np.zeros((max_samples, X.shape[1]))
        y_balanced = [np.zeros(max_samples, dtype=int), np.zeros(max_samples, dtype=int)]

        # Fill with available data, padding with the majority class if needed
        min_samples = min(X_balanced_device.shape[0], X_balanced_attack.shape[0])
        X_balanced[:min_samples] = X_balanced_device[:min_samples]
        y_balanced[0][:min_samples] = y_balanced_device[:min_samples]
        X_balanced[min_samples:] = X_balanced_attack[min_samples:]
        y_balanced[1][min_samples:] = y_balanced_attack[min_samples:]

        # Adjust lengths if unequal
        if X_balanced_device.shape[0] > X_balanced_attack.shape[0]:
            y_balanced[1][:X_balanced_device.shape[0]] = np.pad(y_balanced_attack, (0, X_balanced_device.shape[0] - X_balanced_attack.shape[0]), mode='edge')
        elif X_balanced_attack.shape[0] > X_balanced_device.shape[0]:
            y_balanced[0][:X_balanced_attack.shape[0]] = np.pad(y_balanced_device, (0, X_balanced_attack.shape[0] - X_balanced_device.shape[0]), mode='edge')

        return X_balanced, tuple(y_balanced)

    else:
        # Single output mode ('device' or 'traffic')
        target_label = y
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, target_label)
        return X_balanced, y_balanced