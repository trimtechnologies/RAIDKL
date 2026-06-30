import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabulate import tabulate
import pickle
from data_preprocessing import load_and_preprocess_data
from training import train_and_evaluate_fold
from utils import ensure_directory

def main(csv_path, output_mode='device', n_splits=1, target_labels=None, target_traffic_types=None, distillation_method='standard'):
    """Main function to orchestrate data loading, preprocessing, training, and evaluation."""
    output_dir = Path("results")
    checkpoint_dir = Path("checkpoints")
    ensure_directory(output_dir)
    ensure_directory(checkpoint_dir)

    dataset_name = Path(csv_path).stem
    print(f"\nProcessing dataset: {dataset_name} with output mode: {output_mode}")

    # Load and preprocess data
    try:
        X, y, le, feature_names, scaler, num_classes_device, num_classes_attack, output_dir, checkpoint_dir = load_and_preprocess_data(
            csv_path, target_labels=target_labels, target_traffic_types=target_traffic_types,
            output_mode=output_mode, distillation_method=distillation_method)
        if X is None or y is None:
            print(f"Error: Failed to load or preprocess data from {csv_path}")
            return
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        return

    # Save feature names and scaler
    with open(checkpoint_dir / f"{dataset_name}_feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)
    with open(checkpoint_dir / f"{dataset_name}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    all_results = []
    if n_splits > 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if output_mode == 'multi':
            y_for_stratify = np.column_stack(y)
        else:
            y_for_stratify = y

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_for_stratify[:, 0] if output_mode == 'multi' else y)):
            print(f"\nProcessing Fold {fold + 1}/{n_splits}")
            X_train_val = X[train_idx]
            X_test = X[test_idx]
            if output_mode == 'multi':
                y_train_val = (y[0][train_idx], y[1][train_idx])
                y_test = (y[0][test_idx], y[1][test_idx])
            else:
                y_train_val = y[train_idx]
                y_test = y[test_idx]

            fold_results = train_and_evaluate_fold(
                X_train_val, y_train_val, X_test, y_test, feature_names, dataset_name,
                output_dir, checkpoint_dir, output_mode, le, num_classes_device, num_classes_attack,
                n_splits, fold=fold + 1, distillation_method=distillation_method)
            all_results.extend(fold_results)
    else:
        if output_mode == 'multi':
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=np.column_stack(y))
            y_train_val = (y_train_val[0], y_train_val[1])
            y_test = (y_test[0], y_test[1])
        else:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

        fold_results = train_and_evaluate_fold(
            X_train_val, y_train_val, X_test, y_test, feature_names, dataset_name,
            output_dir, checkpoint_dir, output_mode, le, num_classes_device, num_classes_attack,
            n_splits=1, fold=None, distillation_method=distillation_method)
        all_results.extend(fold_results)

    # Aggregate and save results
    results_df = pd.DataFrame(all_results)
    results_csv_path = output_dir / f"{dataset_name}_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved all results to {results_csv_path}")

    # Generate summary plots
    try:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Model', y='Accuracy', data=results_df)
        plt.title(f"Model Accuracy Comparison ({dataset_name})")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_accuracy_boxplot.pdf", format='pdf', dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Model', y='Train Time (s)', data=results_df)
        plt.title(f"Model Training Time Comparison ({dataset_name})")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_train_time_boxplot.pdf", format='pdf', dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Model', y='Num Parameters', data=results_df)
        plt.title(f"Model Complexity Comparison ({dataset_name})")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_parameters_boxplot.pdf", format='pdf', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating summary plots: {e}")

    # Print average metrics
    print("\nAverage Metrics Across Folds:")
    avg_metrics = results_df.groupby('Model').mean(numeric_only=True)
    print(tabulate(avg_metrics.reset_index(), headers='keys', tablefmt='grid', floatfmt=".4f"))

if __name__ == "__main__":
    csv_path = r"F:\Datasets_Project\NIM LAB IoT Dataset 2025\slowloriss\small\processed\NIMLABIoT_processed.csv"
    main(csv_path, output_mode='device', n_splits=1, target_labels=None, target_traffic_types=None, distillation_method='standard')

# import argparse
# from pathlib import Path
#
#   def main(csv_path, output_mode='device', n_splits=1, target_labels=None, target_traffic_types=None, distillation_method='standard'):
#       # Existing main function code
#       pass
#  if __name__ == "__main__":
#       parser = argparse.ArgumentParser(description="IoMT Classification Pipeline")
#       parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
#       parser.add_argument("--output_mode", type=str, choices=['device', 'traffic', 'multi'], default='device', help="Output mode")
#       parser.add_argument("--n_splits", type=int, default=1, help="Number of cross-validation folds")
#       parser.add_argument("--distillation_method", type=str, choices=['standard', 'active', 'feature_matching', 'gradient_matching', 'combined', 'coreset', 'generative'], default='standard', help="Distillation method")
#       args = parser.parse_args()
#       main(args.csv_path, output_mode=args.output_mode, n_splits=args.n_splits, distillation_method=args.distillation_method)