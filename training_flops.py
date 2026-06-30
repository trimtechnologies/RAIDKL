import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import psutil
import time
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from evaluation import evaluate_model, save_predictions, compute_shap_explanations
from distillation_techniques import select_active_samples, feature_matching_loss, gradient_matching_loss
from distillation_techniques import knowledge_distillation_loss, jensen_shannon_divergence, combined_kl_js_loss
from distillation_techniques import uncertainty_weighted_loss, dynamic_weighted_loss, cosine_similarity_loss, focal_loss
from utils import count_parameters, ensure_directory, count_flops
from models import build_1dcnn_large, build_1dcnn_small
from distillation_techniques import select_coreset, generate_synthetic_data

def train_teacher_cnn(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, checkpoint_dir="checkpoints", output_mode='device'):
    """Train the teacher CNN model."""
    checkpoint_dir = ensure_directory(checkpoint_dir)
    start_time = time.time()
    process = psutil.Process()
    mem_before = process.memory_percent()
    cpu_before = psutil.cpu_percent(interval=None)

    if output_mode == 'multi':
        y_train_device, y_train_attack = y_train
        y_val_device, y_val_attack = y_val
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, {'device': y_train_device, 'attack': y_train_attack})).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, {'device': y_val_device, 'attack': y_val_attack})).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss={'device': 'sparse_categorical_crossentropy', 'attack': 'sparse_categorical_crossentropy'}, metrics=['accuracy'])
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)

    train_time = time.time() - start_time
    train_time_per_sample = train_time / len(X_train)
    mem_after = process.memory_percent()
    cpu_after = psutil.cpu_percent(interval=None)
    train_mem_usage = (mem_before + mem_after) / 2
    train_cpu_usage = (cpu_before + cpu_after) / 2

    checkpoint_path = checkpoint_dir / "teacher_cnn.weights.h5"
    model.save_weights(checkpoint_path)
    print(f"Saved teacher model weights to {checkpoint_path}")

    return model, {
        'train_time': train_time,
        'train_time_per_sample': train_time_per_sample,
        'train_mem_usage': train_mem_usage,
        'train_cpu_usage': train_cpu_usage
    }

def train_student_no_kd(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, checkpoint_dir="checkpoints", output_mode='device', fold=None):
    """Train the student CNN model without knowledge distillation."""
    checkpoint_dir = ensure_directory(checkpoint_dir)
    start_time = time.time()
    process = psutil.Process()
    mem_before = process.memory_percent()
    cpu_before = psutil.cpu_percent(interval=None)

    if output_mode == 'multi':
        y_train_device, y_train_attack = y_train
        y_val_device, y_val_attack = y_val
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, {'device': y_train_device, 'attack': y_train_attack})).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, {'device': y_val_device, 'attack': y_val_attack})).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss={'device': 'sparse_categorical_crossentropy', 'attack': 'sparse_categorical_crossentropy'}, metrics=['accuracy'])
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=1)

    train_time = time.time() - start_time
    train_time_per_sample = train_time / len(X_train)
    mem_after = process.memory_percent()
    cpu_after = psutil.cpu_percent(interval=None)
    train_mem_usage = (mem_before + mem_after) / 2
    train_cpu_usage = (cpu_before + cpu_after) / 2

    checkpoint_path = checkpoint_dir / f"student_no_kd_fold_{fold if fold else 'None'}.weights.h5"
    model.save_weights(checkpoint_path)
    print(f"Saved student (No KD) model weights to {checkpoint_path}")

    return model, {
        'train_time': train_time,
        'train_time_per_sample': train_time_per_sample,
        'train_mem_usage': train_mem_usage,
        'train_cpu_usage': train_cpu_usage
    }

def train_student_kd(teacher, student, feature_model, X_train, y_train, X_val, y_val, alpha=0.7, temperature=2.0,
                     epochs=10, batch_size=64, checkpoint_dir="checkpoints", output_mode='device', loss_type='kl',
                     distillation_method='standard', fold=None, gamma=None):
    """Train the student CNN model with knowledge distillation."""
    checkpoint_dir = ensure_directory(checkpoint_dir)
    start_time = time.time()
    process = psutil.Process()
    mem_before = process.memory_percent()
    cpu_before = psutil.cpu_percent(interval=None)

    # Apply active learning if specified
    if distillation_method == 'active':
        n_samples = min(10000, len(X_train))
        X_train, indices = select_active_samples(teacher, X_train, batch_size=batch_size, n_samples=n_samples)
        if output_mode == 'multi':
            y_train = (y_train[0][indices], y_train[1][indices])
        else:
            y_train = y_train[indices]
        print(f"Applied active learning distillation: selected {n_samples} samples.")

    print(f"Computing teacher logits for training and validation (loss_type={loss_type})...")
    teacher_logits = teacher.predict(X_train, batch_size=batch_size, verbose=0)
    teacher_val_logits = teacher.predict(X_val, batch_size=batch_size, verbose=0)

    if output_mode == 'multi':
        y_train_device, y_train_attack = y_train
        y_val_device, y_val_attack = y_val
        teacher_logits_device, teacher_logits_attack = teacher_logits
        teacher_val_logits_device, teacher_val_logits_attack = teacher_val_logits
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, (y_train_device, y_train_attack), (teacher_logits_device, teacher_logits_attack))
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, (y_val_device, y_val_attack), (teacher_val_logits_device, teacher_val_logits_attack))
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        print(f"Teacher logits shape: {teacher_logits.shape}")
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train, teacher_logits)
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val, teacher_val_logits)
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    if output_mode == 'multi':
        history.update({'device_loss': [], 'attack_loss': [], 'device_accuracy': [], 'attack_accuracy': [],
                        'val_device_loss': [], 'val_attack_loss': [], 'val_device_accuracy': [], 'val_attack_accuracy': []})

    loss_fn = {
        'kl': knowledge_distillation_loss,
        'js': jensen_shannon_divergence,
        'kl_js': combined_kl_js_loss,
        'uncertainty': uncertainty_weighted_loss,
        'dynamic': dynamic_weighted_loss,
        'cosine': cosine_similarity_loss,
        'focal': focal_loss
    }[loss_type]

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0
        if output_mode == 'multi':
            epoch_device_loss = 0
            epoch_attack_loss = 0
            epoch_device_acc = 0
            epoch_attack_acc = 0
            if distillation_method == 'feature':
                epoch_feature_loss = 0
            if distillation_method == 'gradient':
                epoch_gradient_loss = 0

        for X_batch, y_batch, teacher_logits_batch in train_dataset:
            with tf.GradientTape() as tape:
                y_pred = student(X_batch, training=True)
                loss_kwargs = {'alpha': alpha, 'temperature': temperature, 'output_mode': output_mode}
                if loss_type == 'dynamic':
                    loss_kwargs['epoch'] = epoch
                    loss_kwargs['max_epochs'] = epochs
                if loss_type == 'focal':
                    loss_kwargs['gamma'] = gamma

                if output_mode == 'multi':
                    y_pred_device, y_pred_attack = y_pred
                    y_batch_device, y_batch_attack = y_batch
                    teacher_logits_device, teacher_logits_attack = teacher_logits_batch
                    loss_device = loss_fn(y_batch_device, y_pred_device, teacher_logits_device, **loss_kwargs)
                    loss_attack = loss_fn(y_batch_attack, y_pred_attack, teacher_logits_attack, **loss_kwargs)
                    loss = 0.5 * (loss_device + loss_attack)
                    acc_device = tf.reduce_mean(
                        tf.cast(tf.equal(tf.argmax(y_pred_device, axis=-1), tf.cast(y_batch_device, tf.int64)), tf.float32))
                    acc_attack = tf.reduce_mean(
                        tf.cast(tf.equal(tf.argmax(y_pred_attack, axis=-1), tf.cast(y_batch_attack, tf.int64)), tf.float32))
                    acc = 0.5 * (acc_device + acc_attack)
                else:
                    print(f"Batch X shape: {X_batch.shape}, y shape: {y_batch.shape}, teacher_logits shape: {teacher_logits_batch.shape}")
                    loss = loss_fn(y_batch, y_pred, teacher_logits_batch, **loss_kwargs)
                    acc = tf.reduce_mean(
                        tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), tf.cast(y_batch, tf.int64)), tf.float32))

                if distillation_method == 'feature':
                    student_features = feature_model(X_batch, training=True)
                    teacher_features = teacher.get_layer('features')(X_batch, training=False)
                    feature_loss = feature_matching_loss(student_features, teacher_features)
                    loss += 0.1 * feature_loss
                if distillation_method == 'gradient':
                    gradient_loss = gradient_matching_loss(student, teacher, X_batch)
                    loss += 0.1 * gradient_loss

            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            epoch_loss += loss.numpy()
            epoch_acc += acc.numpy()
            if output_mode == 'multi':
                epoch_device_loss += loss_device.numpy()
                epoch_attack_loss += loss_attack.numpy()
                epoch_device_acc += acc_device.numpy()
                epoch_attack_acc += acc_attack.numpy()
                if distillation_method == 'feature':
                    epoch_feature_loss += feature_loss.numpy()
                if distillation_method == 'gradient':
                    epoch_gradient_loss += gradient_loss.numpy()
            batch_count += 1

        # Validation
        val_loss = 0
        val_acc = 0
        val_batch_count = 0
        if output_mode == 'multi':
            val_device_loss = 0
            val_attack_loss = 0
            val_device_acc = 0
            val_attack_acc = 0

        for X_val_batch, y_val_batch, teacher_val_logits_batch in val_dataset:
            y_val_pred = student(X_val_batch, training=False)
            if output_mode == 'multi':
                y_val_pred_device, y_val_pred_attack = y_val_pred
                y_val_batch_device, y_val_batch_attack = y_val_batch
                teacher_val_logits_device, teacher_val_logits_attack = teacher_val_logits_batch
                val_loss_device = loss_fn(y_val_batch_device, y_val_pred_device, teacher_val_logits_device, **loss_kwargs)
                val_loss_attack = loss_fn(y_val_batch_attack, y_val_pred_attack, teacher_val_logits_attack, **loss_kwargs)
                val_loss_batch = 0.5 * (val_loss_device + val_loss_attack)
                val_acc_device = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(y_val_pred_device, axis=-1), tf.cast(y_val_batch_device, tf.int64)), tf.float32))
                val_acc_attack = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(y_val_pred_attack, axis=-1), tf.cast(y_val_batch_attack, tf.int64)), tf.float32))
                val_acc_batch = 0.5 * (val_acc_device + val_acc_attack)
                val_device_loss += val_loss_device.numpy()
                val_attack_loss += val_loss_attack.numpy()
                val_device_acc += val_acc_device.numpy()
                val_attack_acc += val_acc_attack.numpy()
            else:
                val_loss_batch = loss_fn(y_val_batch, y_val_pred, teacher_val_logits_batch, **loss_kwargs)
                val_acc_batch = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(y_val_pred, axis=-1), tf.cast(y_val_batch, tf.int64)), tf.float32))
            val_loss += val_loss_batch.numpy()
            val_acc += val_acc_batch.numpy()
            val_batch_count += 1

        history['loss'].append(epoch_loss / batch_count)
        history['accuracy'].append(epoch_acc / batch_count)
        history['val_loss'].append(val_loss / val_batch_count)
        history['val_accuracy'].append(val_acc / val_batch_count)
        if output_mode == 'multi':
            history['device_loss'].append(epoch_device_loss / batch_count)
            history['attack_loss'].append(epoch_attack_loss / batch_count)
            history['device_accuracy'].append(epoch_device_acc / batch_count)
            history['attack_accuracy'].append(epoch_attack_acc / batch_count)
            history['val_device_loss'].append(val_device_loss / val_batch_count)
            history['val_attack_loss'].append(val_attack_loss / val_batch_count)
            history['val_device_accuracy'].append(val_device_acc / val_batch_count)
            history['val_attack_accuracy'].append(val_attack_acc / val_batch_count)
            print(f"Epoch {epoch + 1}, Device Loss: {epoch_device_loss / batch_count:.4f}, Attack Loss: {epoch_attack_loss / batch_count:.4f}, "
                  f"Device Accuracy: {epoch_device_acc / batch_count:.4f}, Attack Accuracy: {epoch_attack_acc / batch_count:.4f}, "
                  f"Val Device Loss: {val_device_loss / val_batch_count:.4f}, Val Attack Loss: {val_attack_loss / val_batch_count:.4f}, "
                  f"Val Device Accuracy: {val_device_acc / val_batch_count:.4f}, Val Attack Accuracy: {val_attack_acc / val_batch_count:.4f}")
            if distillation_method == 'feature':
                print(f"Feature Loss: {epoch_feature_loss / batch_count:.4f}")
            if distillation_method == 'gradient':
                print(f"Gradient Loss: {epoch_gradient_loss / batch_count:.4f}")
        else:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / batch_count:.4f}, Accuracy: {epoch_acc / batch_count:.4f}, "
                  f"Val Loss: {val_loss / val_batch_count:.4f}, Val Accuracy: {val_acc / val_batch_count:.4f}")

    train_time = time.time() - start_time
    train_time_per_sample = train_time / len(X_train)
    mem_after = process.memory_percent()
    cpu_after = psutil.cpu_percent(interval=None)
    train_mem_usage = (mem_before + mem_after) / 2
    train_cpu_usage = (cpu_before + cpu_after) / 2

    checkpoint_path = checkpoint_dir / f"student_{loss_type}_{distillation_method}_fold_{fold if fold else 'None'}.weights.h5"
    student.save_weights(checkpoint_path)
    print(f"Saved student ({loss_type.upper()}, {distillation_method}) model weights to {checkpoint_path}")

    return student, {
        'train_time': train_time,
        'train_time_per_sample': train_time_per_sample,
        'train_mem_usage': train_mem_usage,
        'train_cpu_usage': train_cpu_usage
    }

def train_and_evaluate_fold(X_train_val, y_train_val, X_test, y_test, feature_names, dataset_name, output_dir, checkpoint_dir,
                            output_mode, le, num_classes_device, num_classes_attack, n_splits=1, fold=None, distillation_method='standard'):
    """
    Train and evaluate models for a single fold.

    Args:
        X_train_val: Training and validation features.
        y_train_val: Training and validation labels.
        X_test: Test features.
        y_test: Test labels.
        feature_names: List of feature names for SHAP explanations.
        dataset_name: Name of the dataset.
        output_dir: Directory to save results.
        checkpoint_dir: Directory to save model checkpoints and SHAP plots.
        output_mode: 'device', 'traffic', or 'multi' for output type.
        le: LabelEncoder object(s) for decoding labels.
        num_classes_device: Number of device classes.
        num_classes_attack: Number of attack classes.
        n_splits: Number of cross-validation folds.
        fold: Current fold number (None for single run).
        distillation_method: Knowledge distillation method to use. Options:
            - 'standard': Standard knowledge distillation using KL divergence.
            - 'active': Active learning distillation, selecting high-uncertainty samples.
            - 'feature_matching': Aligns intermediate feature representations between teacher and student.
            - 'gradient_matching': Matches gradients of student and teacher for similar optimization paths.
            - 'combined': Combines standard, feature matching, and gradient matching losses.
            - 'coreset': Selects a representative subset (coreset) of training data using k-means clustering.
            - 'generative': Generates synthetic training data using a Variational Autoencoder (VAE).

    Returns:
        List of dictionaries containing evaluation metrics for each model.
    """
    results = []
    target_name = 'Device Type' if output_mode == 'device' else 'Attack Category'

    # Split validation set
    if fold is None:
        if output_mode == 'multi':
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=42,
                stratify=np.column_stack(y_train_val))
            y_train = (y_train[0], y_train[1])
            y_val = (y_val[0], y_val[1])
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val, y_val = X_test, y_test

    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1).astype(np.float32)
    X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1).astype(np.float32)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1).astype(np.float32)

    # Initialize LabelBinarizer for evaluation
    lb_device = None
    lb_attack = None
    if output_mode == 'multi':
        y_train_device, y_train_attack = y_train
        y_test_device, y_test_attack = y_test
        lb_device = LabelBinarizer()
        lb_attack = LabelBinarizer()
        # Fit with both train and test labels to handle all possible classes
        lb_device.fit(np.concatenate([y_train_device, y_test_device]))
        lb_attack.fit(np.concatenate([y_train_attack, y_test_attack]))
        le_device, le_attack = le
    else:
        lb_device = LabelBinarizer()
        lb_attack = LabelBinarizer()
        if output_mode == 'device':
            lb_device.fit(np.concatenate([y_train, y_test]))
        else:  # output_mode == 'traffic'
            lb_attack.fit(np.concatenate([y_train, y_test]))

    # Train teacher
    print(f"\nTraining Teacher 1D-CNN ({output_mode})...")
    # In train_and_evaluate_fold, teacher model section
    teacher_model = build_1dcnn_large((X_train.shape[1], 1), num_classes_device, num_classes_attack)
    teacher_params = count_parameters(teacher_model)
    teacher_flops = count_flops(teacher_model, X_train=X_train_cnn)  # Use real data
    print(f"Teacher 1D-CNN parameters: {teacher_params}, FLOPs: {teacher_flops}")
    teacher_model, teacher_train_metrics = train_teacher_cnn(
        teacher_model, X_train_cnn, y_train, X_val_cnn, y_val, epochs=20, batch_size=64,
        checkpoint_dir=checkpoint_dir, output_mode=output_mode)

    # Evaluate teacher
    start_time = time.time()
    process = psutil.Process()
    mem_before = process.memory_percent()
    cpu_before = psutil.cpu_percent(interval=None)
    y_pred_proba = teacher_model.predict(X_test_cnn, batch_size=128, verbose=0)
    test_time = time.time() - start_time
    test_time_per_sample = test_time / len(X_test)
    mem_after = process.memory_percent()
    cpu_after = psutil.cpu_percent(interval=None)
    test_mem_usage = (mem_before + mem_after) / 2
    test_cpu_usage = (cpu_before + cpu_after) / 2

    if output_mode == 'multi':
        y_pred_device_proba, y_pred_attack_proba = y_pred_proba
        y_pred_device = np.argmax(y_pred_device_proba, axis=1)
        y_pred_attack = np.argmax(y_pred_attack_proba, axis=1)
        y_pred = np.column_stack((y_pred_attack, y_pred_device))
        print(f"Predicted device type classes (Teacher 1D-CNN): {np.unique(y_pred_device)}")
        print(f"Predicted attack category classes (Teacher 1D-CNN): {np.unique(y_pred_attack)}")
        metrics_device = evaluate_model(
            y_test_device, y_pred_device, le_device, f"Device Type (Teacher 1D-CNN)",
            y_pred_device_proba, y_test, y_pred, lb_attack, lb_device, checkpoint_dir, fold, output_mode)
        metrics_attack = evaluate_model(
            y_test_attack, y_pred_attack, le_attack, f"Attack Category (Teacher 1D-CNN)",
            y_pred_attack_proba, y_test, y_pred, lb_attack, lb_device, checkpoint_dir, fold, output_mode)
        save_predictions(
            y_test_device, y_pred_device, y_pred_device_proba,
            y_test_attack, y_pred_attack, y_pred_attack_proba,
            le_device, le_attack, dataset_name, "Teacher_1D-CNN", output_dir, output_mode, fold)
        results.append({
            'Dataset': dataset_name,
            'Model': 'Teacher 1D-CNN (Device)',
            'Fold': fold if fold is not None else 'N/A',
            'Num Parameters': teacher_params,
            'FLOPs': teacher_flops,
            **metrics_device,
            'Train Time (s)': teacher_train_metrics['train_time'],
            'Test Time (s)': test_time,
            'Train Time per Sample (s)': teacher_train_metrics['train_time_per_sample'],
            'Test Time per Sample (s)': test_time_per_sample,
            'Train Memory Usage (%)': teacher_train_metrics['train_mem_usage'],
            'Test Memory Usage (%)': test_mem_usage,
            'Train CPU Usage (%)': teacher_train_metrics['train_cpu_usage'],
            'Test CPU Usage (%)': test_cpu_usage
        })
        results.append({
            'Dataset': dataset_name,
            'Model': 'Teacher 1D-CNN (Attack)',
            'Fold': fold if fold is not None else 'N/A',
            'Num Parameters': teacher_params,
            'FLOPs': teacher_flops,
            **metrics_attack,
            'Train Time (s)': teacher_train_metrics['train_time'],
            'Test Time (s)': test_time,
            'Train Time per Sample (s)': teacher_train_metrics['train_time_per_sample'],
            'Test Time per Sample (s)': test_time_per_sample,
            'Train Memory Usage (%)': teacher_train_metrics['train_mem_usage'],
            'Test Memory Usage (%)': test_mem_usage,
            'Train CPU Usage (%)': teacher_train_metrics['train_cpu_usage'],
            'Test CPU Usage (%)': test_cpu_usage
        })
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
        print(f"Predicted {target_name} classes (Teacher 1D-CNN): {np.unique(y_pred)}")
        metrics = evaluate_model(
            y_test, y_pred, le, f"{target_name} (Teacher 1D-CNN)",
            y_pred_proba, y_test, None, lb_attack, lb_device, checkpoint_dir, fold, output_mode)
        save_predictions(
            y_test, y_pred, y_pred_proba, le_device=le,
            dataset_name=dataset_name, model_name="Teacher_1D-CNN",
            output_dir=output_dir, output_mode=output_mode, fold=fold)
        results.append({
            'Dataset': dataset_name,
            'Model': f'Teacher 1D-CNN ({target_name})',
            'Fold': fold if fold is not None else 'N/A',
            'Num Parameters': teacher_params,
            'FLOPs': teacher_flops,
            **metrics,
            'Train Time (s)': teacher_train_metrics['train_time'],
            'Test Time (s)': test_time,
            'Train Time per Sample (s)': teacher_train_metrics['train_time_per_sample'],
            'Test Time per Sample (s)': test_time_per_sample,
            'Train Memory Usage (%)': teacher_train_metrics['train_mem_usage'],
            'Test Memory Usage (%)': test_mem_usage,
            'Train CPU Usage (%)': teacher_train_metrics['train_cpu_usage'],
            'Test CPU Usage (%)': test_cpu_usage
        })

    if output_mode != 'multi':
        try:
            compute_shap_explanations(
                teacher_model, X_test_cnn, feature_names, "teacher_1d_cnn",
                checkpoint_dir, output_mode, le_device=le if output_mode == 'device' else None,
                le_attack=le if output_mode == 'traffic' else None, y_test=y_test, fold=fold)
        except Exception as e:
            print(f"SHAP computation failed for teacher model: {e}. Continuing with pipeline.")

    # Train student with KD
    loss_types = ['kl', 'js', 'kl_js', 'uncertainty', 'dynamic', 'cosine']
    distillation_methods = ['standard', 'active', 'feature_matching', 'gradient_matching', 'combined', 'coreset', 'generative']
    for method in distillation_methods:
        for loss_type in loss_types:
            print(f"\nTraining Student 1D-CNN with KD ({loss_type.upper()}, {method}) ({output_mode})...")
            try:
                # Update the student with KD section
                # In student with KD section
                student_model, feature_model = build_1dcnn_small((X_train.shape[1], 1), num_classes_device,
                                                                 num_classes_attack)
                student_params = count_parameters(student_model)
                student_flops = count_flops(student_model, X_train=X_train_cnn)  # Use real data
                print(f"Student 1D-CNN KD ({loss_type.upper()}, {method}) parameters: {student_params}, FLOPs: {student_flops}")

                # Apply distillation method
                X_train_kd = X_train_cnn
                y_train_kd = y_train
                if method == 'active':
                    X_train_kd, indices = select_active_samples(teacher_model, X_train_cnn, batch_size=128, n_samples=10000)
                    if output_mode == 'multi':
                        y_train_kd = (y_train[0][indices], y_train[1][indices])
                    else:
                        y_train_kd = y_train[indices]
                elif method == 'coreset':
                    X_train_kd, indices = select_coreset(X_train_cnn, n_samples=10000)
                    if output_mode == 'multi':
                        y_train_kd = (y_train[0][indices], y_train[1][indices])
                    else:
                        y_train_kd = y_train[indices]
                elif method == 'generative':
                    X_train_kd, y_train_kd = generate_synthetic_data(X_train_cnn, y_train, output_mode=output_mode, n_synthetic=10000)
                    # Use teacher to label synthetic data
                    teacher_preds = teacher_model.predict(X_train_kd, batch_size=128, verbose=0)
                    if output_mode == 'multi':
                        y_train_kd = (np.argmax(teacher_preds[0], axis=1), np.argmax(teacher_preds[1], axis=1))
                    else:
                        y_train_kd = np.argmax(teacher_preds, axis=1)

                student_model, student_train_metrics = train_student_kd(
                    teacher_model, student_model, feature_model, X_train_kd, y_train_kd, X_val_cnn, y_val,
                    alpha=0.7, temperature=2.0, epochs=10, batch_size=64, checkpoint_dir=checkpoint_dir,
                    output_mode=output_mode, loss_type=loss_type, distillation_method=method, fold=fold,
                    gamma=2.0 if loss_type == 'focal' else None)

                start_time = time.time()
                process = psutil.Process()
                mem_before = process.memory_percent()
                cpu_before = psutil.cpu_percent(interval=None)
                y_pred_proba = student_model.predict(X_test_cnn, batch_size=128, verbose=0)
                test_time = time.time() - start_time
                test_time_per_sample = test_time / len(X_test)
                mem_after = process.memory_percent()
                cpu_after = psutil.cpu_percent(interval=None)
                test_mem_usage = (mem_before + mem_after) / 2
                test_cpu_usage = (cpu_before + cpu_after) / 2

                if output_mode == 'multi':
                    y_pred_device_proba, y_pred_attack_proba = y_pred_proba
                    y_pred_device = np.argmax(y_pred_device_proba, axis=1)
                    y_pred_attack = np.argmax(y_pred_attack_proba, axis=1)
                    y_pred = np.column_stack((y_pred_attack, y_pred_device))
                    print(f"Predicted device type classes (Student 1D-CNN KD {loss_type.upper()} {method}): {np.unique(y_pred_device)}")
                    print(f"Predicted attack category classes (Student 1D-CNN KD {loss_type.upper()} {method}): {np.unique(y_pred_attack)}")
                    metrics_device = evaluate_model(
                        y_test_device, y_pred_device, le_device, f"Device Type (Student 1D-CNN KD {loss_type.upper()} {method})",
                        y_pred_device_proba, y_test, y_pred, lb_attack, lb_device, checkpoint_dir, fold, output_mode)
                    metrics_attack = evaluate_model(
                        y_test_attack, y_pred_attack, le_attack, f"Attack Category (Student 1D-CNN KD {loss_type.upper()} {method})",
                        y_pred_attack_proba, y_test, y_pred, lb_attack, lb_device, checkpoint_dir, fold, output_mode)
                    save_predictions(
                        y_test_device, y_pred_device, y_pred_device_proba,
                        y_test_attack, y_pred_attack, y_pred_attack_proba,
                        le_device, le_attack, dataset_name, f"Student_1D-CNN_KD_{loss_type.upper()}_{method}", output_dir, output_mode, fold)
                    results.append({
                        'Dataset': dataset_name,
                        'Model': f'Student 1D-CNN KD {loss_type.upper()} {method} (Device)',
                        'Fold': fold if fold is not None else 'N/A',
                        'Num Parameters': student_params,
                        'FLOPs': student_flops,
                        **metrics_device,
                        'Train Time (s)': student_train_metrics['train_time'],
                        'Test Time (s)': test_time,
                        'Train Time per Sample (s)': student_train_metrics['train_time_per_sample'],
                        'Test Time per Sample (s)': test_time_per_sample,
                        'Train Memory Usage (%)': student_train_metrics['train_mem_usage'],
                        'Test Memory Usage (%)': test_mem_usage,
                        'Train CPU Usage (%)': student_train_metrics['train_cpu_usage'],
                        'Test CPU Usage (%)': test_cpu_usage
                    })
                    results.append({
                        'Dataset': dataset_name,
                        'Model': f'Student 1D-CNN KD {loss_type.upper()} {method} (Attack)',
                        'Fold': fold if fold is not None else 'N/A',
                        'Num Parameters': student_params,
                        'FLOPs': student_flops,
                        **metrics_attack,
                        'Train Time (s)': student_train_metrics['train_time'],
                        'Test Time (s)': test_time,
                        'Train Time per Sample (s)': student_train_metrics['train_time_per_sample'],
                        'Test Time per Sample (s)': test_time_per_sample,
                        'Train Memory Usage (%)': student_train_metrics['train_mem_usage'],
                        'Test Memory Usage (%)': test_mem_usage,
                        'Train CPU Usage (%)': student_train_metrics['train_cpu_usage'],
                        'Test CPU Usage (%)': test_cpu_usage
                    })
                else:
                    y_pred = np.argmax(y_pred_proba, axis=1)
                    print(f"Predicted {target_name} classes (Student 1D-CNN KD {loss_type.upper()} {method}): {np.unique(y_pred)}")
                    metrics = evaluate_model(
                        y_test, y_pred, le, f"{target_name} (Student 1D-CNN KD {loss_type.upper()} {method})",
                        y_pred_proba, y_test, None, lb_attack, lb_device, checkpoint_dir, fold, output_mode)
                    save_predictions(
                        y_test, y_pred, y_pred_proba, le_device=le,
                        dataset_name=dataset_name, model_name=f"Student_1D-CNN_KD_{loss_type.upper()}_{method}",
                        output_dir=output_dir, output_mode=output_mode, fold=fold)
                    results.append({
                        'Dataset': dataset_name,
                        'Model': f'Student 1D-CNN KD {loss_type.upper()} {method} ({target_name})',
                        'Fold': fold if fold is not None else 'N/A',
                        'Num Parameters': student_params,
                        'FLOPs': student_flops,
                        **metrics,
                        'Train Time (s)': student_train_metrics['train_time'],
                        'Test Time (s)': test_time,
                        'Train Time per Sample (s)': student_train_metrics['train_time_per_sample'],
                        'Test Time per Sample (s)': test_time_per_sample,
                        'Train Memory Usage (%)': student_train_metrics['train_mem_usage'],
                        'Test Memory Usage (%)': test_mem_usage,
                        'Train CPU Usage (%)': student_train_metrics['train_cpu_usage'],
                        'Test CPU Usage (%)': test_cpu_usage
                    })

                if output_mode != 'multi':
                    try:
                        compute_shap_explanations(
                            student_model, X_test_cnn, feature_names, f"student_1d_cnn_kd_{loss_type}_{method}",
                            checkpoint_dir, output_mode, le_device=le if output_mode == 'device' else None,
                            le_attack=le if output_mode == 'traffic' else None, y_test=y_test, fold=fold)
                    except Exception as e:
                        print(f"SHAP computation failed for student model KD {loss_type.upper()} {method}: {e}. Continuing with pipeline.")
            except Exception as e:
                print(f"Error training/evaluating Student 1D-CNN KD {loss_type.upper()} {method}: {e}")
                continue

    # Train student without KD
    print(f"\nTraining Student 1D-CNN without KD ({output_mode})...")
    try:
        # Update the student without KD section
        # In student without KD section
        student_no_kd, _ = build_1dcnn_small((X_train.shape[1], 1), num_classes_device, num_classes_attack)
        student_no_kd_params = count_parameters(student_no_kd)
        student_no_kd_flops = count_flops(student_no_kd, X_train=X_train_cnn)  # Use real data
        print(f"Student 1D-CNN (No KD) parameters: {student_no_kd_params}, FLOPs: {student_no_kd_flops}")
        student_no_kd, student_no_kd_train_metrics = train_student_no_kd(
            student_no_kd, X_train_cnn, y_train, X_val_cnn, y_val,
            epochs=10, batch_size=64, checkpoint_dir=checkpoint_dir, output_mode=output_mode, fold=fold)

        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_percent()
        cpu_before = psutil.cpu_percent(interval=None)
        y_pred_proba_no_kd = student_no_kd.predict(X_test_cnn, batch_size=128, verbose=0)
        test_time_no_kd = time.time() - start_time
        test_time_per_sample_no_kd = test_time_no_kd / len(X_test)
        mem_after = process.memory_percent()
        cpu_after = psutil.cpu_percent(interval=None)
        test_mem_usage_no_kd = (mem_before + mem_after) / 2
        test_cpu_usage_no_kd = (cpu_before + cpu_after) / 2

        if output_mode == 'multi':
            y_pred_device_proba_no_kd, y_pred_attack_proba_no_kd = y_pred_proba_no_kd
            y_pred_device_no_kd = np.argmax(y_pred_device_proba_no_kd, axis=1)
            y_pred_attack_no_kd = np.argmax(y_pred_attack_proba_no_kd, axis=1)
            y_pred_no_kd = np.column_stack((y_pred_attack_no_kd, y_pred_device_no_kd))
            print(f"Predicted device type classes (Student 1D-CNN No KD): {np.unique(y_pred_device_no_kd)}")
            print(f"Predicted attack category classes (Student 1D-CNN No KD): {np.unique(y_pred_attack_no_kd)}")
            metrics_device_no_kd = evaluate_model(
                y_test_device, y_pred_device_no_kd, le_device, f"Device Type (Student 1D-CNN No KD)",
                y_pred_device_proba_no_kd, y_test, y_pred_no_kd, lb_attack, lb_device, checkpoint_dir, fold, output_mode)
            metrics_attack_no_kd = evaluate_model(
                y_test_attack, y_pred_attack_no_kd, le_attack, f"Attack Category (Student 1D-CNN No KD)",
                y_pred_attack_proba_no_kd, y_test, y_pred_no_kd, lb_attack, lb_device, checkpoint_dir, fold, output_mode)
            save_predictions(
                y_test_device, y_pred_device_no_kd, y_pred_device_proba_no_kd,
                y_test_attack, y_pred_attack_no_kd, y_pred_attack_proba_no_kd,
                le_device, le_attack, dataset_name, "Student_1D-CNN_No_KD", output_dir, output_mode, fold)
            results.append({
                'Dataset': dataset_name,
                'Model': 'Student 1D-CNN No KD (Device)',
                'Fold': fold if fold is not None else 'N/A',
                'Num Parameters': student_no_kd_params,
                'FLOPs': student_no_kd_flops,
                **metrics_device_no_kd,
                'Train Time (s)': student_no_kd_train_metrics['train_time'],
                'Test Time (s)': test_time_no_kd,
                'Train Time per Sample (s)': student_no_kd_train_metrics['train_time_per_sample'],
                'Test Time per Sample (s)': test_time_per_sample_no_kd,
                'Train Memory Usage (%)': student_no_kd_train_metrics['train_mem_usage'],
                'Test Memory Usage (%)': test_mem_usage_no_kd,
                'Train CPU Usage (%)': student_no_kd_train_metrics['train_cpu_usage'],
                'Test CPU Usage (%)': test_cpu_usage_no_kd
            })
            results.append({
                'Dataset': dataset_name,
                'Model': 'Student 1D-CNN No KD (Attack)',
                'Fold': fold if fold is not None else 'N/A',
                'Num Parameters': student_no_kd_params,
                'FLOPs': student_no_kd_flops,
                **metrics_attack_no_kd,
                'Train Time (s)': student_no_kd_train_metrics['train_time'],
                'Test Time (s)': test_time_no_kd,
                'Train Time per Sample (s)': student_no_kd_train_metrics['train_time_per_sample'],
                'Test Time per Sample (s)': test_time_per_sample_no_kd,
                'Train Memory Usage (%)': student_no_kd_train_metrics['train_mem_usage'],
                'Test Memory Usage (%)': test_mem_usage_no_kd,
                'Train CPU Usage (%)': student_no_kd_train_metrics['train_cpu_usage'],
                'Test CPU Usage (%)': test_cpu_usage_no_kd
            })
        else:
            y_pred_no_kd = np.argmax(y_pred_proba_no_kd, axis=1)
            print(f"Predicted {target_name} classes (Student 1D-CNN No KD): {np.unique(y_pred_no_kd)}")
            metrics_no_kd = evaluate_model(
                y_test, y_pred_no_kd, le, f"{target_name} (Student 1D-CNN No KD)",
                y_pred_proba_no_kd, y_test, None, lb_attack, lb_device, checkpoint_dir, fold, output_mode)
            save_predictions(
                y_test, y_pred_no_kd, y_pred_proba_no_kd, le_device=le,
                dataset_name=dataset_name, model_name="Student_1D-CNN_No_KD",
                output_dir=output_dir, output_mode=output_mode, fold=fold)
            results.append({
                'Dataset': dataset_name,
                'Model': f'Student 1D-CNN No KD ({target_name})',
                'Fold': fold if fold is not None else 'N/A',
                'Num Parameters': student_no_kd_params,
                'FLOPs': student_no_kd_flops,
                **metrics_no_kd,
                'Train Time (s)': student_no_kd_train_metrics['train_time'],
                'Test Time (s)': test_time_no_kd,
                'Train Time per Sample (s)': student_no_kd_train_metrics['train_time_per_sample'],
                'Test Time per Sample (s)': test_time_per_sample_no_kd,
                'Train Memory Usage (%)': student_no_kd_train_metrics['train_mem_usage'],
                'Test Memory Usage (%)': test_mem_usage_no_kd,
                'Train CPU Usage (%)': student_no_kd_train_metrics['train_cpu_usage'],
                'Test CPU Usage (%)': test_cpu_usage_no_kd
            })

        if output_mode != 'multi':
            try:
                compute_shap_explanations(
                    student_no_kd, X_test_cnn, feature_names, "student_1d_cnn_no_kd",
                    checkpoint_dir, output_mode, le_device=le if output_mode == 'device' else None,
                    le_attack=le if output_mode == 'traffic' else None, y_test=y_test, fold=fold)
            except Exception as e:
                print(f"SHAP computation failed for student model (No KD): {e}. Continuing with pipeline.")
    except Exception as e:
        print(f"Error training/evaluating Student 1D-CNN (No KD): {e}")
        raise

    # Save fold results
    try:
        fold_results_df = pd.DataFrame(results)
        fold_results_path = output_dir / f"{dataset_name}_fold_{fold if fold else 'single'}_results.csv"
        fold_results_df.to_csv(fold_results_path, index=False)
        print(f"Saved fold results to {fold_results_path}")
    except Exception as e:
        print(f"Error saving fold results: {e}")

    return results