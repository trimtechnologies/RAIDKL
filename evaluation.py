import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, matthews_corrcoef, \
    roc_auc_score, hamming_loss, confusion_matrix
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate
from utils import ensure_directory
import re
#import scienceplots

#plt.style.use(['science', 'no-latex'])

def sanitize_filename(name):
    """Sanitize a string to be safe for use in filenames."""
    # Replace invalid characters with underscores, keep alphanumeric and a few safe characters
    return re.sub(r'[^a-zA-Z0-9_-]', '_', str(name)).lower()

def evaluate_model_1(y_true, y_pred, le, model_name, y_pred_proba=None, y_test=None, y_pred_multi=None, lb_attack=None,
                  lb_device=None, checkpoint_dir=None, fold=None, output_mode='device'):
    """Evaluate model performance with comprehensive metrics."""
    print(f"\nEvaluating {model_name}...")
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1 Score'] = f1
    metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    # Compute TNR, FPR, FNR
    try:
        if output_mode == 'multi':
            if lb_attack is None or lb_device is None:
                print(f"Error: LabelBinarizer not provided for {model_name} in multi-output mode.")
                metrics['TNR'] = 0
                metrics['FPR'] = 0
                metrics['FNR'] = 0
            else:
                lb = lb_device if 'Device' in model_name else lb_attack
                y_true_bin = lb.transform(y_true)
                y_pred_bin = lb.transform(y_pred)
                if y_true_bin.ndim > 1:
                    y_true_bin = y_true_bin[:, 0]
                    y_pred_bin = y_pred_bin[:, 0]
                tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
                fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
                fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
                tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
                metrics['TNR'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['FNR'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            # For 'device' or 'traffic' mode, use lb_device for device, lb_attack for traffic
            lb = lb_device if output_mode == 'device' else lb_attack
            if lb is None:
                print(f"Error: LabelBinarizer not provided for {model_name} in {output_mode} mode.")
                metrics['TNR'] = 0
                metrics['FPR'] = 0
                metrics['FNR'] = 0
            else:
                y_true_bin = lb.transform(y_true)
                y_pred_bin = lb.transform(y_pred)
                if y_true_bin.ndim > 1:
                    y_true_bin = y_true_bin[:, 0]
                    y_pred_bin = y_pred_bin[:, 0]
                tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
                fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
                fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
                tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
                metrics['TNR'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['FNR'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    except Exception as e:
        print(f"Error computing TNR/FPR/FNR for {model_name}: {e}")
        metrics['TNR'] = 0
        metrics['FPR'] = 0
        metrics['FNR'] = 0

    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            print(f"AUC for {model_name}: {metrics['AUC']:.4f}")
        except Exception as e:
            print(f"Error computing AUC for {model_name}: {e}")
            metrics['AUC'] = 0
    else:
        metrics['AUC'] = 0

    metrics['Hamming Loss'] = hamming_loss(y_true, y_pred)

    # Classification report
    try:
        labels = np.unique(y_true)
        class_names = le.inverse_transform(labels) if hasattr(le, 'inverse_transform') else labels
        report_data = []
        precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, labels=labels)
        for i, class_name in enumerate(class_names):
            report_data.append({
                'Class': str(class_name),
                'Precision': precisions[i],
                'Recall': recalls[i],
                'F1-Score': f1s[i],
                'Support': supports[i],
                'Accuracy': accuracy_score(y_true[y_true == labels[i]], y_pred[y_true == labels[i]])
            })
        print(f"\nClassification Report for {model_name}:")
        print(tabulate(report_data, headers='keys', tablefmt='grid', floatfmt='.4f'))
    except Exception as e:
        print(f"Error generating classification report for {model_name}: {e}")

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cm_df = pd.DataFrame(cm, index=le.inverse_transform(np.unique(y_true)),
                             columns=le.inverse_transform(np.unique(y_true)))
        print(f"\nConfusion Matrix for {model_name}:")
        print(tabulate(cm_df, headers='keys', tablefmt='grid', showindex=True))
        if checkpoint_dir:
            cm_df.to_csv(
                checkpoint_dir / f"confusion_matrix_{model_name.replace(' ', '_').lower()}_fold_{fold if fold else 'single'}.csv")
    except Exception as e:
        print(f"Error generating confusion matrix for {model_name}: {e}")

    return metrics

def evaluate_model(y_true, y_pred, le, model_name, y_pred_proba=None, y_test=None, y_pred_multi=None, lb_attack=None,
                  lb_device=None, checkpoint_dir=None, fold=None, output_mode='device'):
    """Evaluate model performance with comprehensive metrics."""
    print(f"\nEvaluating {model_name}...")
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1 Score'] = f1
    metrics['Kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    # Compute TNR, FPR, FNR
    try:
        if output_mode == 'multi':
            if lb_attack is None or lb_device is None:
                print(f"Error: LabelBinarizer not provided for {model_name} in multi-output mode.")
                metrics['TNR'] = 0
                metrics['FPR'] = 0
                metrics['FNR'] = 0
            else:
                lb = lb_device if 'Device' in model_name else lb_attack
                y_true_bin = lb.transform(y_true)
                y_pred_bin = lb.transform(y_pred)
                if y_true_bin.ndim > 1:
                    y_true_bin = y_true_bin[:, 0]
                    y_pred_bin = y_pred_bin[:, 0]
                tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
                fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
                fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
                tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
                metrics['TNR'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['FNR'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            # For 'device' or 'traffic' mode, use lb_device for device, lb_attack for traffic
            lb = lb_device if output_mode == 'device' else lb_attack
            if lb is None:
                print(f"Error: LabelBinarizer not provided for {model_name} in {output_mode} mode.")
                metrics['TNR'] = 0
                metrics['FPR'] = 0
                metrics['FNR'] = 0
            else:
                y_true_bin = lb.transform(y_true)
                y_pred_bin = lb.transform(y_pred)
                if y_true_bin.ndim > 1:
                    y_true_bin = y_true_bin[:, 0]
                    y_pred_bin = y_pred_bin[:, 0]
                tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
                fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
                fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
                tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
                metrics['TNR'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['FNR'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    except Exception as e:
        print(f"Error computing TNR/FPR/FNR for {model_name}: {e}")
        metrics['TNR'] = 0
        metrics['FPR'] = 0
        metrics['FNR'] = 0

    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            print(f"AUC for {model_name}: {metrics['AUC']:.4f}")
        except Exception as e:
            print(f"Error computing AUC for {model_name}: {e}")
            metrics['AUC'] = 0
    else:
        metrics['AUC'] = 0

    metrics['Hamming Loss'] = hamming_loss(y_true, y_pred)

    # Classification report
    try:
        labels = np.unique(y_true)
        class_names = le.inverse_transform(labels) if hasattr(le, 'inverse_transform') else labels
        report_data = []
        precisions, recalls, f1s, supports = precision_recall_fscore_support(y_true, y_pred, labels=labels)
        for i, class_name in enumerate(class_names):
            report_data.append({
                'Class': str(class_name),
                'Precision': precisions[i],
                'Recall': recalls[i],
                'F1-Score': f1s[i],
                'Support': supports[i],
                'Accuracy': accuracy_score(y_true[y_true == labels[i]], y_pred[y_true == labels[i]])
            })
        print(f"\nClassification Report for {model_name}:")
        print(tabulate(report_data, headers='keys', tablefmt='grid', floatfmt='.4f'))
    except Exception as e:
        print(f"Error generating classification report for {model_name}: {e}")

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cm_df = pd.DataFrame(cm, index=le.inverse_transform(np.unique(y_true)),
                             columns=le.inverse_transform(np.unique(y_true)))
        print(f"\nConfusion Matrix for {model_name}:")
        print(tabulate(cm_df, headers='keys', tablefmt='grid', showindex=True))
        if checkpoint_dir:
            cm_df.to_csv(
                checkpoint_dir / f"confusion_matrix_{model_name.replace(' ', '_').lower()}_fold_{fold if fold else 'single'}.csv")
    except Exception as e:
        print(f"Error generating confusion matrix for {model_name}: {e}")

    return metrics


def save_predictions(y_test_device, y_pred_device, y_pred_device_proba, y_test_attack=None, y_pred_attack=None,
                     y_pred_attack_proba=None,
                     le_device=None, le_attack=None, dataset_name=None, model_name=None, output_dir=None,
                     output_mode='device', fold=None):
    """Save model predictions to CSV."""
    try:
        output_dir = ensure_directory(output_dir)
        if output_mode == 'multi':
            predictions_df = pd.DataFrame({
                'True Device': le_device.inverse_transform(y_test_device),
                'Pred Device': le_device.inverse_transform(y_pred_device),
                'True Attack': le_attack.inverse_transform(y_test_attack),
                'Pred Attack': le_attack.inverse_transform(y_pred_attack)
            })
            for i in range(y_pred_device_proba.shape[1]):
                predictions_df[f'Device Prob Class {le_device.inverse_transform([i])[0]}'] = y_pred_device_proba[:, i]
            for i in range(y_pred_attack_proba.shape[1]):
                predictions_df[f'Attack Prob Class {le_attack.inverse_transform([i])[0]}'] = y_pred_attack_proba[:, i]
        else:
            predictions_df = pd.DataFrame({
                'True Label': le_device.inverse_transform(y_test_device),
                'Pred Label': le_device.inverse_transform(y_pred_device)
            })
            for i in range(y_pred_device_proba.shape[1]):
                predictions_df[f'Prob Class {le_device.inverse_transform([i])[0]}'] = y_pred_device_proba[:, i]

        predictions_path = output_dir / f"{dataset_name}_{model_name.replace(' ', '_').lower()}_predictions_fold_{fold if fold else 'single'}.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved predictions to {predictions_path}")
    except Exception as e:
        print(f"Error saving predictions for {model_name}: {e}")


def compute_shap_explanations_1(model, X_test, feature_names, model_name, checkpoint_dir, output_mode='device',
                              le_device=None, le_attack=None, y_test=None, fold=None):
    """Compute and save SHAP explanations using KernelExplainer."""
    try:
        checkpoint_dir = ensure_directory(checkpoint_dir)
        print(f"\nComputing SHAP explanations for {model_name}...")

        # Ensure X_test is 2D for SHAP
        X_test_shap = X_test.reshape(X_test.shape[0], -1)
        background_data = X_test_shap[
            np.random.choice(X_test_shap.shape[0], min(100, X_test_shap.shape[0]), replace=False)]

        # Wrap model prediction function
        def model_predict(X):
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            preds = model(X_reshaped, training=False)
            if output_mode == 'multi':
                preds = preds[0]  # Use device output for SHAP
            return preds.numpy()

        # Initialize KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background_data)

        # Compute SHAP values for a subset
        n_samples = min(1000, X_test_shap.shape[0])
        shap_indices = np.random.choice(X_test_shap.shape[0], n_samples, replace=False)
        X_test_subset = X_test_shap[shap_indices]
        shap_values = explainer.shap_values(X_test_subset, nsamples=100)

        # Handle SHAP values for multijoe_class
        if output_mode == 'device':
            num_classes = shap_values.shape[-1] if len(shap_values.shape) == 3 else len(shap_values)
            if le_device is not None and y_test is not None:
                class_names = [le_device.inverse_transform([i])[0] for i in range(num_classes)]
            else:
                class_names = [f"Class {i}" for i in range(num_classes)]

            model_name_clean = model_name.replace(' ', '_').lower()
            for class_idx in range(num_classes):
                try:
                    # Extract SHAP values for the current class
                    shap_values_class = shap_values[..., class_idx] if len(shap_values.shape) == 3 else shap_values[
                        class_idx]

                    # Summary plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_class, X_test_subset, feature_names=feature_names, show=False)
                    plt.title(f"SHAP Summary Plot for {model_name} - {class_names[class_idx]}")
                    summary_plot_path = checkpoint_dir / f"shap_summary_{model_name_clean}_class_{class_idx}_fold_{fold if fold else 'single'}.pdf"
                    plt.savefig(summary_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP summary plot for class {class_names[class_idx]} to {summary_plot_path}")

                    # Bar plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_class, X_test_subset, feature_names=feature_names, plot_type='bar',
                                      show=False)
                    plt.title(f"SHAP Bar Plot for {model_name} - {class_names[class_idx]}")
                    bar_plot_path = checkpoint_dir / f"shap_bar_{model_name_clean}_class_{class_idx}_fold_{fold if fold else 'single'}.pdf"
                    plt.savefig(bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP bar plot for class {class_names[class_idx]} to {bar_plot_path}")

                    # Save SHAP values
                    shap_df = pd.DataFrame(shap_values_class, columns=feature_names)
                    shap_df.to_csv(
                        checkpoint_dir / f"shap_values_{model_name_clean}_class_{class_idx}_fold_{fold if fold else 'single'}.csv",
                        index=False)
                    print()
                        #f"Saved SHAP values for class {class_names[class_idx]} to {checkpoint_dir / f'shap_values_{model_name_clean}_class_{class_idx}_fold_{fold if fold else 'single'}.csv")
                except Exception as e:
                    print(f"Error computing SHAP for class {class_names[class_idx]} in {model_name}: {e}")
                    continue
        else:
            print("SHAP explanations skipped for multi-output mode due to complexity.")
    except Exception as e:
        print(f"Error in SHAP computation for {model_name}: {e}")


def compute_shap_explanations_2(model, X_test, feature_names, model_name, checkpoint_dir, output_mode, y_test=None,
                              fold=None):
    """Compute and save SHAP explanations for the model."""
    try:
        print(f"Computing SHAP explanations for {model_name}...")
        checkpoint_dir = ensure_directory(checkpoint_dir)

        # Subsample for SHAP to reduce computation
        n_samples = min(1000, X_test.shape[0])
        indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
        X_test_sample = X_test[indices]

        # Compute SHAP values
        explainer = shap.KernelExplainer(model, X_test_sample)

        if output_mode == 'multi':
            shap_values = explainer.shap_values(X_test_sample)
            y_test_sample = (y_test[0][indices], y_test[1][indices]) if y_test is not None else None
            output_names = ['Device', 'Attack']
            shap_values_list = shap_values  # Tuple of (device_shap, attack_shap)
        else:
            shap_values = explainer.shap_values(X_test_sample)[0]
            y_test_sample = y_test[indices] if y_test is not None else None
            output_names = [model_name]
            shap_values_list = [shap_values]

        for idx, (shap_vals, output_name) in enumerate(zip(shap_values_list, output_names)):
            # Convert SHAP values to Explanation object for plotting
            shap_explanation = shap.Explanation(
                values=shap_vals,
                base_values=explainer.expected_value[idx] if isinstance(explainer.expected_value,
                                                                        list) else explainer.expected_value,
                data=X_test_sample,
                feature_names=feature_names
            )

            # Generate and save SHAP bar plot
            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_explanation.abs.mean(0), show=False)
            plt.title(f"SHAP Feature Importance (Mean Absolute) - {model_name} ({output_name})")
            bar_plot_path = checkpoint_dir / f"shap_bar_{model_name.replace(' ', '_').lower()}_{output_name.lower()}_fold_{fold if fold else 'single'}.png"
            plt.savefig(bar_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved SHAP bar plot to {bar_plot_path}")

            # Generate and save SHAP beeswarm plot
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_explanation, show=False)
            plt.title(f"SHAP Beeswarm Plot - {model_name} ({output_name})")
            beeswarm_plot_path = checkpoint_dir / f"shap_beeswarm_{model_name.replace(' ', '_').lower()}_{output_name.lower()}_fold_{fold if fold else 'single'}.png"
            plt.savefig(beeswarm_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved SHAP beeswarm plot to {beeswarm_plot_path}")

            # Generate and save SHAP summary plot (existing behavior)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_test_sample, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary Plot - {model_name} ({output_name})")
            summary_plot_path = checkpoint_dir / f"shap_summary_{model_name.replace(' ', '_').lower()}_{output_name.lower()}_fold_{fold if fold else 'single'}.png"
            plt.savefig(summary_plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved SHAP summary plot to {summary_plot_path}")

    except Exception as e:
        print(f"Error computing SHAP explanations for {model_name}: {e}")


def compute_shap_explanations_3(model, X_test, feature_names, model_name, checkpoint_dir, output_mode='device',
                             le_device=None, le_attack=None, y_test=None, fold=None):
    """Compute and save SHAP explanations using KernelExplainer."""
    try:
        checkpoint_dir = ensure_directory(checkpoint_dir)
        print(f"\nComputing SHAP explanations for {model_name}...")

        # Ensure X_test is 2D for SHAP
        X_test_shap = X_test.reshape(X_test.shape[0], -1)
        background_data = X_test_shap[
            np.random.choice(X_test_shap.shape[0], min(100, X_test_shap.shape[0]), replace=False)]

        # Wrap model prediction function
        def model_predict(X):
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            preds = model(X_reshaped, training=False)
            if output_mode == 'multi':
                return np.stack(preds, axis=1)  # Stack device and attack outputs
            return preds.numpy()

        # Initialize KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background_data)

        # Compute SHAP values for a subset
        n_samples = min(1000, X_test_shap.shape[0])
        shap_indices = np.random.choice(X_test_shap.shape[0], n_samples, replace=False)
        X_test_subset = X_test_shap[shap_indices]
        shap_values = explainer.shap_values(X_test_subset, nsamples=100)

        # Handle SHAP values for single-output and multi-output modes
        if output_mode == 'multi':
            # For multi-output, shap_values is a list: [device_shap, attack_shap]
            output_names = ['Device', 'Attack']
            le_list = [le_device, le_attack]
            y_test_list = [y_test[0][shap_indices], y_test[1][shap_indices]] if y_test is not None else [None, None]
        else:
            # For single-output, shap_values is a list of arrays (one per class) or a single array
            output_names = ['Device' if output_mode == 'device' else 'Traffic']
            le_list = [le_device if output_mode == 'device' else le_attack]
            y_test_list = [y_test[shap_indices] if y_test is not None else None]
            shap_values = [shap_values]  # Wrap in list for consistent handling

        model_name_clean = model_name.replace(' ', '_').lower()
        for output_idx, (output_name, le, y_test_subset) in enumerate(zip(output_names, le_list, y_test_list)):
            # Determine number of classes
            if output_mode == 'multi':
                shap_vals = shap_values[output_idx]  # Device or Attack SHAP values
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1
            else:
                shap_vals = shap_values[0]
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1

            # Get class names
            if le is not None and hasattr(le, 'inverse_transform'):
                class_names = [le.inverse_transform([i])[0] for i in range(num_classes)]
            else:
                class_names = [f"Class_{i}" for i in range(num_classes)]

            for class_idx in range(num_classes):
                try:
                    # Extract SHAP values for the current class
                    shap_values_class = shap_vals[..., class_idx] if len(shap_vals.shape) == 3 else shap_vals

                    # Create SHAP Explanation object for bar and beeswarm plots
                    shap_explanation = shap.Explanation(
                        values=shap_values_class,
                        base_values=explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                        data=X_test_subset,
                        feature_names=feature_names
                    )

                    # Sanitize class name for filename
                    class_name_safe = sanitize_filename(class_names[class_idx])

                    # Summary plot (existing)
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_class, X_test_subset, feature_names=feature_names, show=False)
                    plt.title(f"SHAP Summary Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    summary_plot_path = checkpoint_dir / f"shap_summary_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold if fold else 'single'}.pdf"
                    plt.savefig(summary_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP summary plot for {output_name} class {class_names[class_idx]} to {summary_plot_path}")

                    # Bar plot (mean absolute SHAP values)
                    # Global Feature Importance Plot
                    plt.figure(figsize=(10, 6))
                    shap.plots.bar(shap_explanation.abs.mean(0))
                    plt.title(f"SHAP Bar Plot (Mean Absolute) for {model_name} - {output_name} - {class_names[class_idx]}")
                    bar_plot_path = checkpoint_dir / f"shap_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold if fold else 'single'}.pdf"
                    plt.savefig(bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP bar plot for {output_name} class {class_names[class_idx]} to {bar_plot_path}")

                    # Beeswarm plot
                    plt.figure(figsize=(10, 6))
                    shap.plots.beeswarm(shap_explanation, show=False)
                    #plt.title(f"SHAP Beeswarm Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    plt.title(f"SHAP Plot for {class_names[class_idx]}")
                    beeswarm_plot_path = checkpoint_dir / f"shap_beeswarm_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold if fold else 'single'}.pdf"
                    plt.savefig(beeswarm_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP beeswarm plot for {output_name} class {class_names[class_idx]} to {beeswarm_plot_path}")

                    # Save SHAP values
                    shap_df = pd.DataFrame(shap_values_class, columns=feature_names)
                    shap_values_path = checkpoint_dir / f"shap_values_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold if fold else 'single'}.csv"
                    shap_df.to_csv(shap_values_path, index=False)
                    print(f"Saved SHAP values for {output_name} class {class_names[class_idx]} to {shap_values_path}")

                except Exception as e:
                    print(f"Error computing SHAP for {output_name} class {class_names[class_idx]} in {model_name}: {e}")
                    continue

    except Exception as e:
        print(f"Error in SHAP computation for {model_name}: {e}")

# This function include both local and global interpretations
def compute_shap_explanations_4(model, X_test, feature_names, model_name, checkpoint_dir, output_mode='device',
                             le_device=None, le_attack=None, y_test=None, fold=None):
    """
    Compute and save SHAP explanations using KernelExplainer.

    Generates multiple SHAP plots for each class and output:
    - Summary plot: Overall feature importance across samples.
    - Bar plot: Mean absolute SHAP values for feature importance.
    - Beeswarm plot: Detailed feature impact distribution.
    - Local bar plot: Feature contributions for the first sample.
    - Clustered bar plot: Feature importance with hierarchical clustering of features.
    - Partitioned bar plot: Feature importance with clustered features grouped by a cutoff.

    Args:
        model: Trained model for SHAP analysis.
        X_test: Test data for SHAP computation.
        feature_names: List of feature names.
        model_name: Name of the model (e.g., 'teacher_1d_cnn').
        checkpoint_dir: Directory to save SHAP plots and values.
        output_mode: 'device', 'traffic', or 'multi'.
        le_device: LabelEncoder for device classes (optional).
        le_attack: LabelEncoder for attack classes (optional).
        y_test: Test labels (optional, for clustering plot).
        fold: Current fold number or None for single run.
    """
    try:
        checkpoint_dir = ensure_directory(checkpoint_dir)
        print(f"\nComputing SHAP explanations for {model_name}...")

        # Ensure X_test is 2D for SHAP
        X_test_shap = X_test.reshape(X_test.shape[0], -1)
        background_data = X_test_shap[
            np.random.choice(X_test_shap.shape[0], min(100, X_test_shap.shape[0]), replace=False)]

        # Wrap model prediction function
        def model_predict(X):
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            preds = model(X_reshaped, training=False)
            if output_mode == 'multi':
                return np.stack(preds, axis=1)  # Stack device and attack outputs
            return preds.numpy()

        # Initialize KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background_data)

        # Compute SHAP values for a subset
        n_samples = min(1000, X_test_shap.shape[0])
        shap_indices = np.random.choice(X_test_shap.shape[0], n_samples, replace=False)
        X_test_subset = X_test_shap[shap_indices]
        shap_values = explainer.shap_values(X_test_subset, nsamples=100)

        # Handle SHAP values for single-output and multi-output modes
        if output_mode == 'multi':
            output_names = ['Device', 'Attack']
            le_list = [le_device, le_attack]
            y_test_list = [y_test[0][shap_indices], y_test[1][shap_indices]] if y_test is not None else [None, None]
        else:
            output_names = ['Device' if output_mode == 'device' else 'Traffic']
            le_list = [le_device if output_mode == 'device' else le_attack]
            y_test_list = [y_test[shap_indices] if y_test is not None else None]
            shap_values = [shap_values]  # Wrap in list for consistent handling

        model_name_clean = model_name.replace(' ', '_').lower()
        for output_idx, (output_name, le, y_test_subset) in enumerate(zip(output_names, le_list, y_test_list)):
            # Determine number of classes
            if output_mode == 'multi':
                shap_vals = shap_values[output_idx]  # Device or Attack SHAP values
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1
            else:
                shap_vals = shap_values[0]
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1

            # Get class names
            if le is not None and hasattr(le, 'inverse_transform'):
                class_names = [le.inverse_transform([i])[0] for i in range(num_classes)]
            else:
                class_names = [f"Class_{i}" for i in range(num_classes)]

            # Compute hierarchical clustering for clustering plots (once per output)
            if y_test_subset is not None:
                clustering = shap.utils.hclust(X_test_subset, y_test_subset)
            else:
                clustering = None

            for class_idx in range(num_classes):
                try:
                    # Extract SHAP values for the current class
                    shap_values_class = shap_vals[..., class_idx] if len(shap_vals.shape) == 3 else shap_vals

                    # Create SHAP Explanation object
                    shap_explanation = shap.Explanation(
                        values=shap_values_class,
                        base_values=explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                        data=X_test_subset,
                        feature_names=feature_names
                    )

                    # Sanitize class name for filename
                    class_name_safe = sanitize_filename(class_names[class_idx])
                    fold_str = fold if fold else 'single'

                    # Summary plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_class, X_test_subset, feature_names=feature_names, show=False)
                    # plt.title(f"SHAP Summary Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    plt.title(f"SHAP Summary Plot for {class_names[class_idx]}")
                    summary_plot_path = checkpoint_dir / f"shap_summary_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(summary_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP summary plot for {output_name} class {class_names[class_idx]} to {summary_plot_path}")

                    # Bar plot (mean absolute)
                    plt.figure(figsize=(10, 6))
                    shap.plots.bar(shap_explanation.abs.mean(0), show=False)
                    plt.title(f"SHAP Bar Plot (Mean Absolute) for {model_name} - {output_name} - {class_names[class_idx]}")
                    bar_plot_path = checkpoint_dir / f"shap_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP bar plot for {output_name} class {class_names[class_idx]} to {bar_plot_path}")

                    # Beeswarm plot
                    plt.figure(figsize=(10, 6))
                    shap.plots.beeswarm(shap_explanation, show=False)
                    plt.title(f"SHAP Beeswarm Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    beeswarm_plot_path = checkpoint_dir / f"shap_beeswarm_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(beeswarm_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP beeswarm plot for {output_name} class {class_names[class_idx]} to {beeswarm_plot_path}")

                    # Local bar plot (for first sample)
                    plt.figure(figsize=(10, 6))
                    shap.plots.bar(shap_explanation[0], show=False)
                    plt.title(f"SHAP Local Bar Plot for {model_name} - {output_name} - {class_names[class_idx]} (First Sample)")
                    local_bar_plot_path = checkpoint_dir / f"shap_local_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(local_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP local bar plot for {output_name} class {class_names[class_idx]} to {local_bar_plot_path}")

                    # Clustering plots (if y_test_subset is available)
                    if clustering is not None:
                        # Clustered bar plot
                        plt.figure(figsize=(10, 6))
                        shap.plots.bar(shap_explanation, clustering=clustering, show=False)
                        plt.title(f"SHAP Clustered Bar Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                        clustered_bar_plot_path = checkpoint_dir / f"shap_clustered_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(clustered_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP clustered bar plot for {output_name} class {class_names[class_idx]} to {clustered_bar_plot_path}")

                        # Partitioned bar plot
                        plt.figure(figsize=(10, 6))
                        shap.plots.bar(shap_explanation, clustering=clustering, clustering_cutoff=2, show=False)
                        plt.title(f"SHAP Partitioned Bar Plot for {model_name} - {output_name} - {class_names[class_idx]} (Cutoff=2)")
                        partitioned_bar_plot_path = checkpoint_dir / f"shap_partitioned_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(partitioned_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP partitioned bar plot for {output_name} class {class_names[class_idx]} to {partitioned_bar_plot_path}")

                    # Save SHAP values
                    shap_df = pd.DataFrame(shap_values_class, columns=feature_names)
                    shap_values_path = checkpoint_dir / f"shap_values_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    shap_df.to_csv(shap_values_path, index=False)
                    print(f"Saved SHAP values for {output_name} class {class_names[class_idx]} to {shap_values_path}")

                except Exception as e:
                    print(f"Error computing SHAP for {output_name} class {class_names[class_idx]} in {model_name}: {e}")
                    continue

    except Exception as e:
        print(f"Error in SHAP computation for {model_name}: {e}")


def compute_shap_explanations_5(model, X_test, feature_names, model_name, checkpoint_dir, output_mode='device',
                             le_device=None, le_attack=None, y_test=None, fold=None):
    """
    Compute and save SHAP explanations using KernelExplainer.

    Generates multiple SHAP plots for each class and output:
    - Summary plot: Overall feature importance across samples.
    - Bar plot: Mean absolute SHAP values for feature importance, with feature values shown.
    - Beeswarm plot: Detailed feature impact distribution.
    - Local bar plot: Feature contributions for the first sample, with feature values shown.
    - Clustered bar plot: Feature importance with hierarchical clustering of features, with feature values shown.
    - Partitioned bar plot: Feature importance with clustered features grouped by a cutoff, with feature values shown.

    Args:
        model: Trained model for SHAP analysis.
        X_test: Test data for SHAP computation.
        feature_names: List of feature names.
        model_name: Name of the model (e.g., 'teacher_1d_cnn').
        checkpoint_dir: Directory to save SHAP plots and values.
        output_mode: 'device', 'traffic', or 'multi'.
        le_device: LabelEncoder for device classes (optional).
        le_attack: LabelEncoder for attack classes (optional).
        y_test: Test labels (optional, for clustering plot).
        fold: Current fold number or None for single run.
    """
    try:
        checkpoint_dir = ensure_directory(checkpoint_dir)
        print(f"\nComputing SHAP explanations for {model_name}...")

        # Ensure X_test is 2D for SHAP
        X_test_shap = X_test.reshape(X_test.shape[0], -1)
        background_data = X_test_shap[
            np.random.choice(X_test_shap.shape[0], min(100, X_test_shap.shape[0]), replace=False)]

        # Wrap model prediction function
        def model_predict(X):
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            preds = model(X_reshaped, training=False)
            if output_mode == 'multi':
                return np.stack(preds, axis=1)  # Stack device and attack outputs
            return preds.numpy()

        # Initialize KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background_data)

        # Compute SHAP values for a subset
        n_samples = min(200, X_test_shap.shape[0])
        shap_indices = np.random.choice(X_test_shap.shape[0], n_samples, replace=False)
        X_test_subset = X_test_shap[shap_indices]
        shap_values = explainer.shap_values(X_test_subset, nsamples=100)

        # Handle SHAP values for single-output and multi-output modes
        if output_mode == 'multi':
            output_names = ['Device', 'Attack']
            le_list = [le_device, le_attack]
            y_test_list = [y_test[0][shap_indices], y_test[1][shap_indices]] if y_test is not None else [None, None]
        else:
            output_names = ['Device' if output_mode == 'device' else 'Traffic']
            le_list = [le_device if output_mode == 'device' else le_attack]
            y_test_list = [y_test[shap_indices] if y_test is not None else None]
            shap_values = [shap_values]  # Wrap in list for consistent handling

        model_name_clean = model_name.replace(' ', '_').lower()
        for output_idx, (output_name, le, y_test_subset) in enumerate(zip(output_names, le_list, y_test_list)):
            # Determine number of classes
            if output_mode == 'multi':
                shap_vals = shap_values[output_idx]  # Device or Attack SHAP values
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1
            else:
                shap_vals = shap_values[0]
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1

            # Get class names
            if le is not None and hasattr(le, 'inverse_transform'):
                class_names = [le.inverse_transform([i])[0] for i in range(num_classes)]
            else:
                class_names = [f"Class_{i}" for i in range(num_classes)]

            # Compute hierarchical clustering for clustering plots (once per output)
            if y_test_subset is not None:
                clustering = shap.utils.hclust(X_test_subset, y_test_subset)
            else:
                clustering = None

            for class_idx in range(num_classes):
                try:
                    # Extract SHAP values for the current class
                    shap_values_class = shap_vals[..., class_idx] if len(shap_vals.shape) == 3 else shap_vals

                    # Create SHAP Explanation object
                    shap_explanation = shap.Explanation(
                        values=shap_values_class,
                        base_values=explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                        data=X_test_subset,
                        feature_names=feature_names
                    )

                    # Sanitize class name for filename
                    class_name_safe = sanitize_filename(class_names[class_idx])
                    fold_str = fold if fold else 'single'

                    # Summary plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_class, X_test_subset, feature_names=feature_names, show=False)
                    # plt.title(f"SHAP Summary Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    summary_plot_path = checkpoint_dir / f"shap_summary_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(summary_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP summary plot for {output_name} class {class_names[class_idx]} to {summary_plot_path}")

                    # Bar plot (mean absolute) with feature values
                    plt.figure(figsize=(12, 6))  # Slightly wider to accommodate feature values
                    # shap.plots.bar(shap_explanation.abs.mean(0), max_display=len(feature_names), show=False)
                    shap.plots.bar(shap_explanation.abs.mean(0), max_display=10, show=False)
                    # plt.title(f"SHAP Bar Plot (Mean Absolute) for {model_name} - {output_name} - {class_names[class_idx]}")
                    # plt.title(f"SHAP Feature Importance for {class_names[class_idx]}")
                    bar_plot_path = checkpoint_dir / f"shap_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP bar plot for {output_name} class {class_names[class_idx]} to {bar_plot_path}")

                    # Beeswarm plot
                    plt.figure(figsize=(10, 6))
                    shap.plots.beeswarm(shap_explanation, show=False)
                    # plt.title(f"SHAP Beeswarm Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    # plt.title(f"SHAP Feature Importance for {class_names[class_idx]}")
                    beeswarm_plot_path = checkpoint_dir / f"shap_beeswarm_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(beeswarm_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP beeswarm plot for {output_name} class {class_names[class_idx]} to {beeswarm_plot_path}")

                    # Local bar plot (for first sample) with feature values
                    plt.figure(figsize=(12, 6))  # Wider for feature values
                    shap.plots.bar(shap_explanation[0], max_display=10, show=False)
                    # plt.title(f"SHAP Local Bar Plot for {model_name} - {output_name} - {class_names[class_idx]} (First Sample)")
                    # plt.title(f"SHAP Local Feature Importance for {class_names[class_idx]} (First Sample)")
                    local_bar_plot_path = checkpoint_dir / f"shap_local_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(local_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP local bar plot for {output_name} class {class_names[class_idx]} to {local_bar_plot_path}")

                    # Clustering plots (if y_test_subset is available)
                    if clustering is not None:
                        # Clustered bar plot with feature values
                        plt.figure(figsize=(12, 6))  # Wider for feature values
                        shap.plots.bar(shap_explanation, clustering=clustering, max_display=10, show=False)
                        # plt.title(f"SHAP Clustered Bar Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                        # plt.title(
                        #     f"SHAP Importance for {class_names[class_idx]}")
                        clustered_bar_plot_path = checkpoint_dir / f"shap_clustered_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(clustered_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP clustered bar plot for {output_name} class {class_names[class_idx]} to {clustered_bar_plot_path}")

                        # Partitioned bar plot with feature values
                        plt.figure(figsize=(12, 6))  # Wider for feature values
                        shap.plots.bar(shap_explanation, clustering=clustering, clustering_cutoff=2,
                                       max_display=10, show=False)
                        # shap.plots.bar(shap_explanation, clustering=clustering, clustering_cutoff=2, max_display=len(feature_names), show=False)
                        # plt.title(f"SHAP Importance for {class_names[class_idx]}")
                        partitioned_bar_plot_path = checkpoint_dir / f"shap_partitioned_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(partitioned_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP partitioned bar plot for {output_name} class {class_names[class_idx]} to {partitioned_bar_plot_path}")

                    # Save SHAP values
                    shap_df = pd.DataFrame(shap_values_class, columns=feature_names)
                    shap_values_path = checkpoint_dir / f"shap_values_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.csv"
                    shap_df.to_csv(shap_values_path, index=False)
                    print(f"Saved SHAP values for {output_name} class {class_names[class_idx]} to {shap_values_path}")

                except Exception as e:
                    print(f"Error computing SHAP for {output_name} class {class_names[class_idx]} in {model_name}: {e}")
                    continue

    except Exception as e:
        print(f"Error in SHAP computation for {model_name}: {e}")


def compute_shap_explanations_6(model, X_test, feature_names, model_name, checkpoint_dir, output_mode='device',
                             le_device=None, le_attack=None, y_test=None, fold=None):
    """
    Compute and save SHAP explanations using KernelExplainer.

    Generates multiple SHAP plots for each class and output:
    - Summary plot: Overall feature importance across samples.
    - Bar plot: Mean absolute SHAP values for feature importance, with feature values shown.
    - Beeswarm plot: Detailed feature impact distribution.
    - Local bar plot: Feature contributions for the first sample, with feature values shown.
    - Clustered bar plot: Feature importance with hierarchical clustering of features, with feature values shown.
    - Partitioned bar plot: Feature importance with clustered features grouped by a cutoff, with feature values shown.
    - Waterfall plot: Contribution of features to the prediction for the first sample.
    - Decision plot: Cumulative effect of features on the prediction.
    - Force plot: Forces influencing predictions for misclassified samples.
    - Force summary plot: Stacked force plot summarizing misclassified samples.
    - Multioutput decision plot: Decision plot for multi-output models.

    Args:
        model: Trained model for SHAP analysis.
        X_test: Test data for SHAP computation.
        feature_names: List of feature names.
        model_name: Name of the model (e.g., 'teacher_1d_cnn').
        checkpoint_dir: Directory to save SHAP plots and values.
        output_mode: 'device', 'traffic', or 'multi'.
        le_device: LabelEncoder for device classes (optional).
        le_attack: LabelEncoder for attack classes (optional).
        y_test: Test labels (optional, for clustering and force plots).
        fold: Current fold number or None for single run.
    """
    try:
        checkpoint_dir = ensure_directory(checkpoint_dir)
        print(f"\nComputing SHAP explanations for {model_name}...")

        # Ensure X_test is 2D for SHAP
        X_test_shap = X_test.reshape(X_test.shape[0], -1)
        background_data = X_test_shap[
            np.random.choice(X_test_shap.shape[0], min(100, X_test_shap.shape[0]), replace=False)]

        # Wrap model prediction function
        def model_predict(X):
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            preds = model(X_reshaped, training=False)
            if output_mode == 'multi':
                return np.stack(preds, axis=1)  # Stack device and attack outputs
            return preds.numpy()

        # Initialize KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background_data)

        # Compute SHAP values for a subset
        n_samples = min(200, X_test_shap.shape[0])
        shap_indices = np.random.choice(X_test_shap.shape[0], n_samples, replace=False)
        X_test_subset = X_test_shap[shap_indices]
        shap_values = explainer.shap_values(X_test_subset, nsamples=100)

        # Handle SHAP values for single-output and multi-output modes
        if output_mode == 'multi':
            output_names = ['Device', 'Attack']
            le_list = [le_device, le_attack]
            y_test_list = [y_test[0][shap_indices], y_test[1][shap_indices]] if y_test is not None else [None, None]
        else:
            output_names = ['Device' if output_mode == 'device' else 'Traffic']
            le_list = [le_device if output_mode == 'device' else le_attack]
            y_test_list = [y_test[shap_indices] if y_test is not None else None]
            shap_values = [shap_values]  # Wrap in list for consistent handling

        model_name_clean = model_name.replace(' ', '_').lower()
        for output_idx, (output_name, le, y_test_subset) in enumerate(zip(output_names, le_list, y_test_list)):
            # Determine number of classes
            if output_mode == 'multi':
                shap_vals = shap_values[output_idx]  # Device or Attack SHAP values
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1
            else:
                shap_vals = shap_values[0]
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1

            # Get class names
            if le is not None and hasattr(le, 'inverse_transform'):
                class_names = [le.inverse_transform([i])[0] for i in range(num_classes)]
            else:
                class_names = [f"Class_{i}" for i in range(num_classes)]

            # Compute hierarchical clustering for clustering plots (once per output)
            if y_test_subset is not None:
                clustering = shap.utils.hclust(X_test_subset, y_test_subset)
            else:
                clustering = None

            # Get predictions for misclassification detection
            X_test_subset_reshaped = X_test_subset.reshape(X_test_subset.shape[0], X_test_subset.shape[1], 1)
            y_pred_proba = model(X_test_subset_reshaped, training=False)
            if output_mode == 'multi':
                y_pred = np.argmax(y_pred_proba[output_idx], axis=1)
                # Ensure y_test_subset is aligned (assuming one-hot or multi-label)
                if y_test_subset is not None and y_test_subset.ndim > 1:
                    y_test_subset = np.argmax(y_test_subset, axis=1)
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
                if y_test_subset is not None and y_test_subset.ndim > 1:
                    y_test_subset = np.argmax(y_test_subset, axis=1)

            # Debug shape check
            if y_test_subset is not None and y_pred.shape != y_test_subset.shape:
                print(f"Warning: Shape mismatch - y_pred: {y_pred.shape}, y_test_subset: {y_test_subset.shape}")
                y_test_subset = y_test_subset[:y_pred.shape[0]]  # Truncate to match if necessary

            misclassified = np.where(y_pred != y_test_subset)[0] if y_test_subset is not None else []
            print(f"Debug: Number of misclassified samples for {output_name} class {class_names[0] if num_classes > 0 else 'all'}: {len(misclassified)}")

            for class_idx in range(num_classes):
                try:
                    # Extract SHAP values for the current class
                    shap_values_class = shap_vals[..., class_idx] if len(shap_vals.shape) == 3 else shap_vals

                    # Create SHAP Explanation object
                    shap_explanation = shap.Explanation(
                        values=shap_values_class,
                        base_values=explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                        data=X_test_subset,
                        feature_names=feature_names
                    )

                    # Sanitize class name for filename
                    class_name_safe = sanitize_filename(class_names[class_idx])
                    fold_str = fold if fold else 'single'

                    # Summary plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_class, X_test_subset, feature_names=feature_names, show=False)
                    # plt.title(f"SHAP Summary Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    summary_plot_path = checkpoint_dir / f"shap_summary_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(summary_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP summary plot for {output_name} class {class_names[class_idx]} to {summary_plot_path}")

                    # Bar plot (mean absolute) with feature values
                    plt.figure(figsize=(12, 6))
                    shap.plots.bar(shap_explanation.abs.mean(0), max_display=len(feature_names), show=False)
                    # plt.title(f"SHAP Bar Plot (Mean Absolute) for {model_name} - {output_name} - {class_names[class_idx]}")
                    bar_plot_path = checkpoint_dir / f"shap_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP bar plot for {output_name} class {class_names[class_idx]} to {bar_plot_path}")

                    # Beeswarm plot
                    plt.figure(figsize=(10, 6))
                    shap.plots.beeswarm(shap_explanation, show=False)
                    # plt.title(f"SHAP Beeswarm Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    beeswarm_plot_path = checkpoint_dir / f"shap_beeswarm_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(beeswarm_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP beeswarm plot for {output_name} class {class_names[class_idx]} to {beeswarm_plot_path}")

                    # Local bar plot (for first sample) with feature values
                    plt.figure(figsize=(12, 6))
                    shap.plots.bar(shap_explanation[0], max_display=len(feature_names), show=False)
                    # plt.title(f"SHAP Local Bar Plot for {model_name} - {output_name} - {class_names[class_idx]} (First Sample)")
                    local_bar_plot_path = checkpoint_dir / f"shap_local_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(local_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP local bar plot for {output_name} class {class_names[class_idx]} to {local_bar_plot_path}")

                    # Clustering plots (if y_test_subset is available)
                    if clustering is not None:
                        # Clustered bar plot with feature values
                        plt.figure(figsize=(12, 6))
                        shap.plots.bar(shap_explanation, clustering=clustering, max_display=len(feature_names), show=False)
                        # plt.title(f"SHAP Clustered Bar Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                        clustered_bar_plot_path = checkpoint_dir / f"shap_clustered_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(clustered_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP clustered bar plot for {output_name} class {class_names[class_idx]} to {clustered_bar_plot_path}")

                        # Partitioned bar plot with feature values
                        plt.figure(figsize=(12, 6))
                        shap.plots.bar(shap_explanation, clustering=clustering, clustering_cutoff=2, max_display=len(feature_names), show=False)
                        # plt.title(f"SHAP Partitioned Bar Plot for {model_name} - {output_name} - {class_names[class_idx]} (Cutoff=2)")
                        partitioned_bar_plot_path = checkpoint_dir / f"shap_partitioned_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(partitioned_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP partitioned bar plot for {output_name} class {class_names[class_idx]} to {partitioned_bar_plot_path}")

                    # Waterfall plot (for first sample)
                    plt.figure(figsize=(12, 6))
                    shap.plots.waterfall(shap_explanation[0], max_display=20, show=False)
                    # plt.title(f"SHAP Waterfall Plot for {model_name} - {output_name} - {class_names[class_idx]} (First Sample)")
                    waterfall_plot_path = checkpoint_dir / f"shap_waterfall_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(waterfall_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP waterfall plot for {output_name} class {class_names[class_idx]} to {waterfall_plot_path}")

                    # Decision plot
                    plt.figure(figsize=(12, 6))
                    shap.decision_plot(
                        explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                        shap_values_class,
                        feature_names,
                        show=False
                    )
                    # plt.title(f"SHAP Decision Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    decision_plot_path = checkpoint_dir / f"shap_decision_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(decision_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP decision plot for {output_name} class {class_names[class_idx]} to {decision_plot_path}")

                    # Force plot (for misclassified samples)
                    if len(misclassified) > 0 and y_test_subset is not None:
                        print(f"Debug: Processing {len(misclassified)} misclassified samples for force plot")
                        fig = plt.figure(figsize=(12, 6))
                        shap.force_plot(
                            explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                            shap_values_class[misclassified],
                            X_test_subset[misclassified],
                            link="logit",
                            matplotlib=True
                        )
                        # plt.title(f"SHAP Force Plot for {model_name} - {output_name} - {class_names[class_idx]} (Misclassified)")
                        force_plot_path = checkpoint_dir / f"shap_force_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(force_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP force plot for {output_name} class {class_names[class_idx]} to {force_plot_path}")

                    # Force summary plot (for misclassified samples)
                    if len(misclassified) > 0 and y_test_subset is not None:
                        print(f"Debug: Processing {len(misclassified)} misclassified samples for force summary plot")
                        max_samples = min(50, len(misclassified))  # Limit to 50 samples to avoid clutter
                        fig = plt.figure(figsize=(12, 6))
                        shap.force_plot(
                            explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                            shap_values_class[misclassified[:max_samples]],
                            X_test_subset[misclassified[:max_samples]],
                            link="logit",
                            matplotlib=True
                        )
                        # plt.title(f"SHAP Force Summary Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                        force_summary_plot_path = checkpoint_dir / f"shap_force_summary_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(force_summary_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP force summary plot for {output_name} class {class_names[class_idx]} to {force_summary_plot_path}")

                    # Multioutput decision plot (for multi-output mode)
                    if output_mode == 'multi' and len(shap_values) > 1:
                        fig = plt.figure(figsize=(12, 6))
                        shap.multioutput_decision_plot(
                            [explainer.expected_value[i][class_idx] for i in range(len(shap_values))],
                            [shap_values[i][..., class_idx] for i in range(len(shap_values))],
                            feature_names=feature_names,
                            show=False
                        )
                        # plt.title(f"SHAP Multioutput Decision Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                        multioutput_decision_plot_path = checkpoint_dir / f"shap_multioutput_decision_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(multioutput_decision_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP multioutput decision plot for {output_name} class {class_names[class_idx]} to {multioutput_decision_plot_path}")

                    # Save SHAP values
                    shap_df = pd.DataFrame(shap_values_class, columns=feature_names)
                    shap_values_path = checkpoint_dir / f"shap_values_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.csv"
                    shap_df.to_csv(shap_values_path, index=False)
                    print(f"Saved SHAP values for {output_name} class {class_names[class_idx]} to {shap_values_path}")

                except Exception as e:
                    print(f"Error computing SHAP for {output_name} class {class_names[class_idx]} in {model_name}: {e}")
                    continue

    except Exception as e:
        print(f"Error in SHAP computation for {model_name}: {e}")


def compute_shap_explanations(model, X_test, feature_names, model_name, checkpoint_dir, output_mode='device',
                             le_device=None, le_attack=None, y_test=None, fold=None):
    """
    Compute and save SHAP explanations using KernelExplainer.

    Generates multiple SHAP plots for each class and output:
    - Summary plot: Overall feature importance across samples.
    - Bar plot: Mean absolute SHAP values for feature importance, with feature values shown.
    - Beeswarm plot: Detailed feature impact distribution.
    - Local bar plot: Feature contributions for the first sample, with feature values shown.
    - Clustered bar plot: Feature importance with hierarchical clustering of features, with feature values shown.
    - Partitioned bar plot: Feature importance with clustered features grouped by a cutoff, with feature values shown.
    - Waterfall plot: Contribution of features to the prediction for the first sample.
    - Decision plot: Cumulative effect of features on the prediction.
    - Force plot: Forces influencing predictions for misclassified samples.

    Args:
        model: Trained model for SHAP analysis.
        X_test: Test data for SHAP computation.
        feature_names: List of feature names.
        model_name: Name of the model (e.g., 'teacher_1d_cnn').
        checkpoint_dir: Directory to save SHAP plots and values.
        output_mode: 'device', 'traffic', or 'multi'.
        le_device: LabelEncoder for device classes (optional).
        le_attack: LabelEncoder for attack classes (optional).
        y_test: Test labels (optional, for clustering and force plots).
        fold: Current fold number or None for single run.
    """
    try:
        checkpoint_dir = ensure_directory(checkpoint_dir)
        print(f"\nComputing SHAP explanations for {model_name}...")

        # Ensure X_test is 2D for SHAP
        X_test_shap = X_test.reshape(X_test.shape[0], -1)
        background_data = X_test_shap[
            np.random.choice(X_test_shap.shape[0], min(100, X_test_shap.shape[0]), replace=False)]

        # Wrap model prediction function
        def model_predict(X):
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            preds = model(X_reshaped, training=False)
            if output_mode == 'multi':
                return np.stack(preds, axis=1)  # Stack device and attack outputs
            return preds.numpy()

        # Initialize KernelExplainer
        explainer = shap.KernelExplainer(model_predict, background_data)

        # Compute SHAP values for a subset
        n_samples = min(200, X_test_shap.shape[0])
        shap_indices = np.random.choice(X_test_shap.shape[0], n_samples, replace=False)
        X_test_subset = X_test_shap[shap_indices]
        shap_values = explainer.shap_values(X_test_subset, nsamples=100)

        # Handle SHAP values for single-output and multi-output modes
        if output_mode == 'multi':
            output_names = ['Device', 'Attack']
            le_list = [le_device, le_attack]
            y_test_list = [y_test[0][shap_indices], y_test[1][shap_indices]] if y_test is not None else [None, None]
        else:
            output_names = ['Device' if output_mode == 'device' else 'Traffic']
            le_list = [le_device if output_mode == 'device' else le_attack]
            y_test_list = [y_test[shap_indices] if y_test is not None else None]
            shap_values = [shap_values]  # Wrap in list for consistent handling

        model_name_clean = model_name.replace(' ', '_').lower()
        for output_idx, (output_name, le, y_test_subset) in enumerate(zip(output_names, le_list, y_test_list)):
            # Determine number of classes
            if output_mode == 'multi':
                shap_vals = shap_values[output_idx]  # Device or Attack SHAP values
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1
            else:
                shap_vals = shap_values[0]
                num_classes = shap_vals.shape[-1] if len(shap_vals.shape) == 3 else 1

            # Get class names
            if le is not None and hasattr(le, 'inverse_transform'):
                class_names = [le.inverse_transform([i])[0] for i in range(num_classes)]
            else:
                class_names = [f"Class_{i}" for i in range(num_classes)]

            # Compute hierarchical clustering for clustering plots (once per output)
            if y_test_subset is not None:
                clustering = shap.utils.hclust(X_test_subset, y_test_subset)
            else:
                clustering = None

            # Get predictions for misclassification detection
            X_test_subset_reshaped = X_test_subset.reshape(X_test_subset.shape[0], X_test_subset.shape[1], 1)
            y_pred_proba = model(X_test_subset_reshaped, training=False)
            if output_mode == 'multi':
                y_pred = np.argmax(y_pred_proba[output_idx], axis=1)
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
            misclassified = y_pred != y_test_subset if y_test_subset is not None else []

            for class_idx in range(num_classes):
                try:
                    # Extract SHAP values for the current class
                    shap_values_class = shap_vals[..., class_idx] if len(shap_vals.shape) == 3 else shap_vals

                    # Create SHAP Explanation object
                    shap_explanation = shap.Explanation(
                        values=shap_values_class,
                        base_values=explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                        data=X_test_subset,
                        feature_names=feature_names
                    )

                    # Sanitize class name for filename
                    class_name_safe = sanitize_filename(class_names[class_idx])
                    fold_str = fold if fold else 'single'

                    # Summary plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_class, X_test_subset, feature_names=feature_names, show=False)
                    # plt.title(f"SHAP Summary Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    summary_plot_path = checkpoint_dir / f"shap_summary_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(summary_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP summary plot for {output_name} class {class_names[class_idx]} to {summary_plot_path}")

                    # Bar plot (mean absolute) with feature values
                    plt.figure(figsize=(12, 6))
                    shap.plots.bar(shap_explanation.abs.mean(0), max_display=12, show=False)
                    # plt.title(f"SHAP Bar Plot (Mean Absolute) for {model_name} - {output_name} - {class_names[class_idx]}")
                    bar_plot_path = checkpoint_dir / f"shap_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP bar plot for {output_name} class {class_names[class_idx]} to {bar_plot_path}")

                    # Beeswarm plot
                    plt.figure(figsize=(10, 6))
                    shap.plots.beeswarm(shap_explanation, show=False)
                    # plt.title(f"SHAP Beeswarm Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    beeswarm_plot_path = checkpoint_dir / f"shap_beeswarm_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(beeswarm_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP beeswarm plot for {output_name} class {class_names[class_idx]} to {beeswarm_plot_path}")

                    # Local bar plot (for first sample) with feature values
                    plt.figure(figsize=(12, 6))
                    shap.plots.bar(shap_explanation[0], max_display=12, show=False)
                    # plt.title(f"SHAP Local Bar Plot for {model_name} - {output_name} - {class_names[class_idx]} (First Sample)")
                    local_bar_plot_path = checkpoint_dir / f"shap_local_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(local_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP local bar plot for {output_name} class {class_names[class_idx]} to {local_bar_plot_path}")

                    # Clustering plots (if y_test_subset is available)
                    if clustering is not None:
                        # Clustered bar plot with feature values
                        plt.figure(figsize=(12, 6))
                        shap.plots.bar(shap_explanation, clustering=clustering, max_display=12, show=False)
                        # plt.title(f"SHAP Clustered Bar Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                        clustered_bar_plot_path = checkpoint_dir / f"shap_clustered_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(clustered_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP clustered bar plot for {output_name} class {class_names[class_idx]} to {clustered_bar_plot_path}")

                        # Partitioned bar plot with feature values
                        plt.figure(figsize=(12, 6))
                        shap.plots.bar(shap_explanation, clustering=clustering, clustering_cutoff=2, max_display=12, show=False)
                        # plt.title(f"SHAP Partitioned Bar Plot for {model_name} - {output_name} - {class_names[class_idx]} (Cutoff=2)")
                        partitioned_bar_plot_path = checkpoint_dir / f"shap_partitioned_bar_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(partitioned_bar_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP partitioned bar plot for {output_name} class {class_names[class_idx]} to {partitioned_bar_plot_path}")

                    # Waterfall plot (for first sample)
                    plt.figure(figsize=(12, 6))
                    shap.plots.waterfall(shap_explanation[0], max_display=12, show=False)
                    # plt.title(f"SHAP Waterfall Plot for {model_name} - {output_name} - {class_names[class_idx]} (First Sample)")
                    waterfall_plot_path = checkpoint_dir / f"shap_waterfall_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(waterfall_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP waterfall plot for {output_name} class {class_names[class_idx]} to {waterfall_plot_path}")

                    # Decision plot
                    plt.figure(figsize=(12, 6))
                    shap.decision_plot(
                        explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                        shap_values_class,
                        feature_names,
                        show=False
                    )
                    # plt.title(f"SHAP Decision Plot for {model_name} - {output_name} - {class_names[class_idx]}")
                    decision_plot_path = checkpoint_dir / f"shap_decision_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                    plt.savefig(decision_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved SHAP decision plot for {output_name} class {class_names[class_idx]} to {decision_plot_path}")

                    # Force plot (for misclassified samples)
                    if misclassified and y_test_subset is not None:
                        plt.figure(figsize=(12, 6))
                        shap.force_plot(
                            explainer.expected_value[output_idx][class_idx] if output_mode == 'multi' and len(explainer.expected_value) > output_idx else explainer.expected_value[class_idx] if len(shap_vals.shape) == 3 else explainer.expected_value,
                            shap_values_class[misclassified],
                            X_test_subset[misclassified],
                            link="logit",
                            matplotlib=True,
                            show=False
                        )
                        # plt.title(f"SHAP Force Plot for {model_name} - {output_name} - {class_names[class_idx]} (Misclassified)")
                        force_plot_path = checkpoint_dir / f"shap_force_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.pdf"
                        plt.savefig(force_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved SHAP force plot for {output_name} class {class_names[class_idx]} to {force_plot_path}")

                    # Save SHAP values
                    shap_df = pd.DataFrame(shap_values_class, columns=feature_names)
                    shap_values_path = checkpoint_dir / f"shap_values_{model_name_clean}_{output_name.lower()}_{class_name_safe}_fold_{fold_str}.csv"
                    shap_df.to_csv(shap_values_path, index=False)
                    print(f"Saved SHAP values for {output_name} class {class_names[class_idx]} to {shap_values_path}")

                except Exception as e:
                    print(f"Error computing SHAP for {output_name} class {class_names[class_idx]} in {model_name}: {e}")
                    continue

    except Exception as e:
        print(f"Error in SHAP computation for {model_name}: {e}")