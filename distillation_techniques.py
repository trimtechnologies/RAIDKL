import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model


def knowledge_distillation_loss(y_true, y_pred, teacher_logits, alpha=0.7, temperature=2.0, output_mode='device'):
    """
    Compute knowledge distillation loss using KL divergence.

    Combines hard loss (cross-entropy with true labels) and soft loss (KL divergence with teacher's softened logits).

    Args:
        y_true: True labels.
        y_pred: Student model logits.
        teacher_logits: Teacher model logits.
        alpha: Weight for balancing hard and soft losses (default: 0.7).
        temperature: Softening factor for logits (default: 2.0).
        output_mode: 'device', 'traffic', or 'multi' (default: 'device').

    Returns:
        Combined loss value.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    teacher_logits = tf.cast(teacher_logits, tf.float32)

    hard_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False))
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_probs = tf.nn.softmax(y_pred / temperature, axis=-1)
    soft_loss = tf.reduce_mean(
        tf.keras.losses.kullback_leibler_divergence(teacher_probs, student_probs)) * (temperature ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss


def jensen_shannon_divergence(y_true, y_pred, teacher_logits, alpha=0.7, temperature=2.0, output_mode='device'):
    """
    Compute Jensen-Shannon divergence loss for knowledge distillation.

    Combines hard loss (cross-entropy) with Jensen-Shannon divergence between teacher and student softened probabilities.

    Args:
        y_true: True labels.
        y_pred: Student model logits.
        teacher_logits: Teacher model logits.
        alpha: Weight for balancing hard and soft losses (default: 0.7).
        temperature: Softening factor for logits (default: 2.0).
        output_mode: 'device', 'traffic', or 'multi' (default: 'device').

    Returns:
        Combined loss value.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    teacher_logits = tf.cast(teacher_logits, tf.float32)

    hard_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False))
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_probs = tf.nn.softmax(y_pred / temperature, axis=-1)
    m = 0.5 * (teacher_probs + student_probs)
    js_loss = 0.5 * tf.reduce_mean(
        tf.keras.losses.kullback_leibler_divergence(teacher_probs, m) +
        tf.keras.losses.kullback_leibler_divergence(student_probs, m)) * (temperature ** 2)
    return alpha * hard_loss + (1 - alpha) * js_loss


def combined_kl_js_loss(y_true, y_pred, teacher_logits, alpha=0.7, temperature=2.0, output_mode='device'):
    """
    Combine KL and JS divergence losses for knowledge distillation.

    Averages the KL divergence and Jensen-Shannon divergence losses for robust knowledge transfer.

    Args:
        y_true: True labels.
        y_pred: Student model logits.
        teacher_logits: Teacher model logits.
        alpha: Weight for balancing hard and soft losses (default: 0.7).
        temperature: Softening factor for logits (default: 2.0).
        output_mode: 'device', 'traffic', or 'multi' (default: 'device').

    Returns:
        Combined loss value.
    """
    kl_loss = knowledge_distillation_loss(y_true, y_pred, teacher_logits, alpha, temperature, output_mode)
    js_loss = jensen_shannon_divergence(y_true, y_pred, teacher_logits, alpha, temperature, output_mode)
    return 0.5 * (kl_loss + js_loss)


def uncertainty_weighted_loss(y_true, y_pred, teacher_logits, alpha=0.7, temperature=2.0, output_mode='device'):
    """
    Weight knowledge distillation loss by teacher uncertainty.

    Scales the KL divergence loss by the inverse of teacher prediction uncertainty (entropy).

    Args:
        y_true: True labels.
        y_pred: Student model logits.
        teacher_logits: Teacher model logits.
        alpha: Weight for balancing hard and soft losses (default: 0.7).
        temperature: Softening factor for logits (default: 2.0).
        output_mode: 'device', 'traffic', or 'multi' (default: 'device').

    Returns:
        Weighted loss value.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    teacher_logits = tf.cast(teacher_logits, tf.float32)

    hard_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False))
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_probs = tf.nn.softmax(y_pred / temperature, axis=-1)
    uncertainty = -tf.reduce_sum(teacher_probs * tf.math.log(teacher_probs + 1e-10), axis=-1)
    weights = 1.0 / (uncertainty + 1e-10)
    soft_loss = tf.reduce_mean(
        weights * tf.keras.losses.kullback_leibler_divergence(teacher_probs, student_probs)) * (temperature ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss


def dynamic_weighted_loss(y_true, y_pred, teacher_logits, alpha=0.7, temperature=2.0, epoch=0, max_epochs=10,
                          output_mode='device'):
    """
    Dynamically adjust alpha in knowledge distillation based on training epoch.

    Reduces the weight of hard loss (alpha) linearly over epochs to emphasize teacher knowledge.

    Args:
        y_true: True labels.
        y_pred: Student model logits.
        teacher_logits: Teacher model logits.
        alpha: Initial weight for hard loss (default: 0.7).
        temperature: Softening factor for logits (default: 2.0).
        epoch: Current training epoch.
        max_epochs: Total number of training epochs (default: 10).
        output_mode: 'device', 'traffic', or 'multi' (default: 'device').

    Returns:
        Loss value with dynamically adjusted alpha.
    """
    dynamic_alpha = alpha * (1 - epoch / max_epochs)
    return knowledge_distillation_loss(y_true, y_pred, teacher_logits, dynamic_alpha, temperature, output_mode)


def cosine_similarity_loss(y_true, y_pred, teacher_logits, alpha=0.7, temperature=2.0, output_mode='device'):
    """
    Compute cosine similarity loss for knowledge distillation.

    Combines cross-entropy with cosine similarity between softened teacher and student probabilities.

    Args:
        y_true: True labels.
        y_pred: Student model logits.
        teacher_logits: Teacher model logits.
        alpha: Weight for balancing hard and soft losses (default: 0.7).
        temperature: Softening factor for logits (default: 2.0).
        output_mode: 'device', 'traffic', or 'multi' (default: 'device').

    Returns:
        Combined loss value.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    teacher_logits = tf.cast(teacher_logits, tf.float32)

    hard_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False))
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_probs = tf.nn.softmax(y_pred / temperature, axis=-1)
    cosine_loss = -tf.reduce_mean(
        tf.keras.losses.cosine_similarity(teacher_probs, student_probs, axis=-1)) * (temperature ** 2)
    return alpha * hard_loss + (1 - alpha) * cosine_loss


def focal_loss(y_true, y_pred, teacher_logits, alpha=0.7, temperature=2.0, gamma=2.0, output_mode='device'):
    """
    Compute focal loss for knowledge distillation.

    Uses focal loss to focus on hard-to-classify samples, combined with KL divergence for teacher knowledge.

    Args:
        y_true: True labels.
        y_pred: Student model logits.
        teacher_logits: Teacher model logits.
        alpha: Weight for balancing hard and soft losses (default: 0.7).
        temperature: Softening factor for logits (default: 2.0).
        gamma: Focusing parameter for focal loss (default: 2.0).
        output_mode: 'device', 'traffic', or 'multi' (default: 'device').

    Returns:
        Combined loss value.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    teacher_logits = tf.cast(teacher_logits, tf.float32)

    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    pt = tf.exp(-ce_loss)
    focal_loss = tf.reduce_mean((1 - pt) ** gamma * ce_loss)
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_probs = tf.nn.softmax(y_pred / temperature, axis=-1)
    soft_loss = tf.reduce_mean(
        tf.keras.losses.kullback_leibler_divergence(teacher_probs, student_probs)) * (temperature ** 2)
    return alpha * focal_loss + (1 - alpha) * soft_loss


def select_active_samples(teacher, X, batch_size=128, n_samples=10000):
    """
    Select samples with high uncertainty for active learning distillation.

    Chooses samples based on high entropy in teacher predictions to focus on informative data points.

    Args:
        teacher: Teacher model for predicting logits.
        X: Input data (numpy array).
        batch_size: Batch size for prediction (default: 128).
        n_samples: Number of samples to select (default: 10000).

    Returns:
        Tuple of (selected samples, indices of selected samples).
    """
    logits = teacher.predict(X, batch_size=batch_size, verbose=0)
    if isinstance(logits, list):
        logits = logits[0]  # Use device logits for simplicity
    probs = tf.nn.softmax(logits, axis=-1)
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)
    indices = tf.argsort(entropy, direction='DESCENDING')[:n_samples]
    return X[indices], indices.numpy()


def select_coreset(X, n_samples=10000):
    """
    Select a coreset using k-means clustering for efficient distillation.

    Clusters the input data and selects the closest point to each cluster centroid to form a representative subset.

    Args:
        X: Input data (numpy array).
        n_samples: Number of samples to select for the coreset (default: 10000).

    Returns:
        Tuple of (selected coreset samples, indices of selected samples).
    """
    n_samples = min(n_samples, len(X))
    kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
    kmeans.fit(X.reshape(X.shape[0], -1))
    distances = kmeans.transform(X.reshape(X.shape[0], -1))
    indices = np.argmin(distances, axis=0)
    return X[indices], indices


def build_vae(input_shape, latent_dim=32):
    """
    Build a Variational Autoencoder (VAE) for generative model distillation.

    Creates an encoder and decoder to learn a latent representation of the input data, enabling synthetic data generation.

    Args:
        input_shape: Shape of input data (tuple, e.g., (num_features,)).
        latent_dim: Dimension of the latent space (default: 32).

    Returns:
        Tuple of (VAE model, encoder model, decoder model).
    """
    inputs = Input(shape=input_shape)
    h = Dense(64, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z])

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(64, activation='relu')(latent_inputs)
    outputs = Dense(input_shape[0], activation='linear')(x)
    decoder = Model(latent_inputs, outputs)

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs)
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(inputs, outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae.add_loss(reconstruction_loss + kl_loss)
    vae.compile(optimizer=Adam(learning_rate=0.001))
    return vae, encoder, decoder


def generate_synthetic_data(teacher, X, y, output_mode='device', n_synthetic=10000):
    """
    Generate synthetic data using a Variational Autoencoder (VAE) for generative model distillation.

    Trains a VAE on the input data and generates synthetic samples, with labels assigned by the teacher model.

    Args:
        teacher: Teacher model to predict labels for synthetic data.
        X: Input data (numpy array) to train the VAE.
        y: Original labels (not used for generation but included for compatibility).
        output_mode: 'device', 'traffic', or 'multi' (default: 'device').
        n_synthetic: Number of synthetic samples to generate (default: 10000).

    Returns:
        Tuple of (synthetic data, synthetic labels).
    """
    vae, encoder, decoder = build_vae(input_shape=(X.shape[1],))
    vae.fit(X, epochs=10, batch_size=128, verbose=0)
    z = tf.random.normal([n_synthetic, 32])
    X_synthetic = decoder(z).numpy()

    # Use teacher model to predict labels for synthetic data
    X_synthetic_cnn = X_synthetic.reshape(X_synthetic.shape[0], X_synthetic.shape[1], 1).astype(np.float32)
    teacher_preds = teacher.predict(X_synthetic_cnn, batch_size=128, verbose=0)
    if output_mode == 'multi':
        y_synthetic = (np.argmax(teacher_preds[0], axis=1), np.argmax(teacher_preds[1], axis=1))
    else:
        y_synthetic = np.argmax(teacher_preds, axis=1)

    return X_synthetic, y_synthetic


def feature_matching_loss(student_features, teacher_features):
    """
    Compute feature matching loss for prototype selection distillation.

    Aligns intermediate feature representations between teacher and student using mean squared error.

    Args:
        student_features: Feature maps from the student model.
        teacher_features: Feature maps from the teacher model.

    Returns:
        Mean squared error loss between feature maps.
    """
    return tf.reduce_mean(tf.square(student_features - teacher_features))


def gradient_matching_loss(student, teacher, X_batch):
    """
    Compute gradient matching loss for gradient-based distillation.

    Aligns gradients of the student and teacher models using cosine similarity to ensure similar optimization paths.

    Args:
        student: Student model.
        teacher: Teacher model.
        X_batch: Input batch for computing gradients.

    Returns:
        Cosine similarity loss between gradients.
    """
    with tf.GradientTape() as tape:
        teacher_outputs = teacher(X_batch, training=False)
        if isinstance(teacher_outputs, list):
            teacher_outputs = teacher_outputs[0]  # Use device output
        teacher_loss = tf.reduce_mean(teacher_outputs)
    teacher_grads = tape.gradient(teacher_loss, teacher.trainable_variables)

    with tf.GradientTape() as tape:
        student_outputs = student(X_batch, training=True)
        if isinstance(student_outputs, list):
            student_outputs = student_outputs[0]  # Use device output
        student_loss = tf.reduce_mean(student_outputs)
    student_grads = tape.gradient(student_loss, student.trainable_variables)

    grad_loss = 0.0
    for tg, sg in zip(teacher_grads, student_grads):
        if tg is not None and sg is not None:
            grad_loss += tf.reduce_mean(tf.keras.losses.cosine_similarity(tg, sg))
    return grad_loss