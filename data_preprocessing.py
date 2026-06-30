import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from distillation_techniques import select_coreset, generate_synthetic_data


def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(np.prod(var.shape) for var in model.trainable_variables)


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


def shorten_label(label, max_length=10):
    """Shorten a label to the first max_length characters."""
    return str(label)[:max_length]


def sample_40_percent_by_labels(df, target_column=None):
    """Sample 40% of the rows for each unique value or combination of 'Label' and 'Traffic Type' columns."""
    if target_column:
        if target_column not in df.columns:
            print(f"Error: '{target_column}' column not found in DataFrame for sampling")
            return df
        print(f"Initial {target_column} value counts before sampling:")
        print(df[target_column].value_counts())
        sampled_df = df.groupby(target_column, group_keys=False).apply(
            lambda x: x.sample(frac=0.4, random_state=42)
        ).reset_index(drop=True)
        print(f"\nSampled {target_column} value counts (40% of each):")
        print(sampled_df[target_column].value_counts())
    else:
        if 'Label' not in df.columns or 'Traffic Type' not in df.columns:
            print("Error: 'Label' or 'Traffic Type' column not found in DataFrame for sampling")
            return df
        print("Initial (Label, Traffic Type) value counts before sampling:")
        print(df.groupby(['Label', 'Traffic Type']).size())
        sampled_df = df.groupby(['Label', 'Traffic Type'], group_keys=False).apply(
            lambda x: x.sample(frac=0.4, random_state=42)
        ).reset_index(drop=True)
        print("\nSampled (Label, Traffic Type) value counts (40% of each):")
        print(sampled_df.groupby(['Label', 'Traffic Type']).size())

    print(f"\nOriginal rows: {len(df)}, Sampled rows: {len(sampled_df)}")
    return sampled_df


def load_and_preprocess_data(csv_path, target_labels=None, target_traffic_types=None, output_mode='multi',
                             distillation_method='standard'):
    """Load CSV, filter by specified labels and traffic types, preprocess data, and return features and labels.
    Drops classes with only one sample. Handles non-numeric columns, NaN, and extreme values.
    Supports distillation methods: 'standard', 'coreset', and 'generative'."""
    cols_to_use = [
        'Entropy', 'Burst_count', 'TCP_response', 'TCP_mss_values', 'STUN_method', 'MQTT_type', 'IP_tos',
        'IP_len', 'IP_id', 'IP_flags', 'IP_ttl', 'IP_chksum', 'IP_padding', 'TCP_dport', 'TCP_ack',
        'TCP_dataofs', 'TCP_flags', 'TCP_SYN', 'TCP_RST', 'TCP_ACK', 'TCP_ECE', 'TCP_CWR', 'TCP_options',
        'TCP_window_scaling', 'UDP_chksum', 'ICMP_type', 'ICMP_code', 'ICMP_id', 'ICMP_seq', 'BOOTP_xid',
        'BOOTP_options', 'DNS_qr', 'DNS_rd', 'DNS_ra', 'DNS_ancount', 'DNS_arcount', 'Portcl_src',
        'Portcl_dst', 'sport23', 'dport_bare', 'NTP', 'Packet_freq', 'Multicast', 'TTL_default',
        'TLS_selected_cipher', 'HTTP_user_agent', 'HTTP_content_type', 'TLS_cipher_suites', 'TLS_ja3',
        'MQTT_topics', 'Protocol_UDP', 'TLS_Cipher_Suites_Len', 'DNS_Query_Len', 'DNS_Interval',
        'HTTP_URI', 'HTTP_Content_Len', 'ICMP_Data_Size', 'Min_Elapsed_Time', 'Pck_Size_Max',
        'Pck_Size_Med', 'Pck_Size_Avg', 'Pck_Size_Var', 'Pck_Size_Q1', 'Pck_Size_Q3', 'Entropy_Min',
        'Pck_Size_Skew', 'Pck_Size_Kurt', 'Protocol_ICMP_Ratio', 'IP_Dst_Entropy', 'Payload_Len_Q1',
        'Payload_Len_Q3', 'Payload_Len_IQR', 'Traffic Type',
    ] # 'Label',

    try:
        df = pd.read_csv(csv_path,
                         usecols=[col for col in cols_to_use if col in pd.read_csv(csv_path, nrows=1).columns],
                         low_memory=False)
        df['Traffic Type'] = df['Traffic Type'].replace({
            'MQTT-Malformed_Data': 'MQTT-Malformed', 'TCP_IP-DoS-ICMP1': 'TCPIP-DoS',
            'Recon-Port_Scan': 'Recon-PortScan', 'Recon-OS_Scan': 'Recon-OSScan',
            'TCP_IP-DDoS-UDP1': 'TCPIP-DDoS', 'TCP_IP-DoS-UDP1': 'TCPIP-DoS',
            'Recon-Ping_Sweep': 'Recon-PingSweep', 'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS',
            'TCP_IP-DDoS-ICMP1': 'TCPIP-DDoS', 'MQTT-DoS-Publish_Flood': 'MQTT-DoS',
            'MQTT-DoS-Connect_Flood': 'MQTT-DoS', 'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS',
            'TCP_IP-DDoS-SYN1': 'TCPIP-DDoS'})

        # df['Label'] = df['Label'].replace({
        #     'Blink mini Amazon Techno': 'Blink Mini',
        #     'COOSPO_HW807_Armband_Power': 'Armband',
        #     'CheckmeO2_Oximeter_Power': 'Oximeter',
        #     'Checkme_BP2A_Power': 'Checkme BP',
        #     'Checkme_O2_Oximeter_Power': 'Oximeter',
        #     'Ecobee Camera': 'Ecobee Camera',
        #     'Expressif Sense-U Baby Monitor': 'Baby Monitor',
        #     'Ipad': 'Ipad',
        #     'Lookee_O2_Ring_Power': 'Lookee 02 Ring',
        #     'Lookee_Sleep_ring_Power': 'Sleep Ring',
        #     'MIT Camera Laxihub Altobeam': 'MIT Camera',
        #     'Multifunctional Pager Tuya Smart': 'MF Pager',
        #     'Plugable Tech': 'Plugable Tech',
        #     'Powerlabs_HR_Monitor_Power': 'HR Monitor',
        #     'Realtek Semic': 'Realtek Semic',
        #     'Rhythm+_Power': 'Rhythm+',
        #     'SINGCALL SOS Button Tuya Smart': 'Singcall SOS Btn',
        #     'SleepU_Sleep_Oxygen_Monitor_Power': 'Sleep 02 Monitor',
        #     'TPLink': 'TPLink',
        #     'Tuya Smart Owltron': 'Owltron',
        #     'Wellue_O2_Ring_Power': 'Wellue 02 Ring'
        # })
        # df = df[~df['Label'].isin(['TPLink', 'Ipad', 'Realtek Semic', 'Plugable Tech'])]
        print(f"Loaded {Path(csv_path).name} with {len(df)} rows and {len(df.columns)} columns.")

        # Get directories
        script_dir = Path(__file__).parent
        output_dir = ensure_directory(script_dir / 'results')
        checkpoint_dir = ensure_directory(script_dir / 'checkpoints')

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_inf_cols = [col for col in numeric_cols if df[col].isna().any() or np.isinf(df[col]).any()]
        if nan_inf_cols:
            print(f"\nWarning: NaN or inf values detected in columns: {nan_inf_cols}")
            for col in nan_inf_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
            print("Replaced NaN/inf with median values.")

        if target_labels is not None and 'Label' in df.columns:
            invalid_labels = [lbl for lbl in target_labels if lbl not in df['Label'].values]
            if invalid_labels:
                print(f"Warning: The following target labels not found in data: {invalid_labels}")
            df = df[df['Label'].isin(target_labels)]
            print(f"\nFiltered by target labels {target_labels}. Rows remaining: {len(df)}")

        if target_traffic_types is not None and 'Traffic Type' in df.columns:
            invalid_traffic_types = [tt for tt in target_traffic_types if tt not in df['Traffic Type'].values]
            if invalid_traffic_types:
                print(f"Warning: The following target traffic types not found in data: {invalid_traffic_types}")
            df = df[df['Traffic Type'].isin(target_traffic_types)]
            print(f"\nFiltered by target traffic types {target_traffic_types}. Rows remaining: {len(df)}")

        if df.empty:
            print("Error: No data remains after filtering by target labels and/or traffic types.")
            return None, None, None, None, None, None, None, None, None

        print("\nNull values per column after preprocessing:")
        print(df.isna().sum())

        duplicates = df.duplicated().sum()
        print(f"\nNumber of duplicate rows: {duplicates}")
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"Dropped {duplicates} duplicate rows.")

        print("\nColumn data types:")
        print(df.dtypes)

        if output_mode == 'traffic' and 'Traffic Type' not in df.columns:
            print("Warning: 'Traffic Type' column not found. Assuming binary attack category based on 'Label'.")
            df['Traffic Type'] = df['Label'].apply(lambda x: 'Attack' if 'attack' in str(x).lower() else 'Benign')

        target_column = 'Label' if output_mode == 'device' else 'Traffic Type' if output_mode == 'traffic' else None
        #df = sample_40_percent_by_labels(df, target_column)

        if output_mode == 'device' and 'Label' not in df.columns:
            print("Error: 'Label' not found in dataset, cannot train for device type.")
            return None, None, None, None, None, None, None, None, None
        if output_mode == 'traffic' and 'Traffic Type' not in df.columns:
            print("Error: 'Traffic Type' not found in dataset, cannot train for traffic type.")
            return None, None, None, None, None, None, None, None, None

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if output_mode == 'multi':
            numeric_cols = [col for col in numeric_cols if col not in ['Label', 'Traffic Type']]
        else:
            numeric_cols = [col for col in numeric_cols if col != target_column]

        nan_counts = df[numeric_cols].isna().sum()
        print("\nNaN values in numeric columns before imputation:")
        print(nan_counts[nan_counts > 0])

        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=True)
                print(f"Imputed NaN in {col} with mean value: {mean_value:.4f}")

        if output_mode == 'multi':
            print("\nDevice type class distribution after sampling:")
            print(pd.Series(df['Label']).value_counts())
            print("\nAttack category class distribution after sampling:")
            print(pd.Series(df['Traffic Type']).value_counts())
            print("\nCombined (Label, Traffic Type) distribution after sampling:")
            print(df.groupby(['Label', 'Traffic Type']).size())

            print("\nDevice type class distribution before dropping rare classes:")
            device_counts = pd.Series(df['Label']).value_counts()
            print(device_counts)
            print("\nAttack category class distribution before dropping rare classes:")
            attack_counts = pd.Series(df['Traffic Type']).value_counts()
            print(attack_counts)
            print("\nCombined (Label, Traffic Type) distribution before dropping rare classes:")
            combined_counts = df.groupby(['Label', 'Traffic Type']).size()
            print(combined_counts)

            rare_device_classes = device_counts[device_counts == 1].index
            rare_attack_classes = attack_counts[attack_counts == 1].index
            rare_combined = combined_counts[combined_counts == 1].index

            print("\nDropping rare device type classes (1 sample):", rare_device_classes.tolist())
            print("Dropping rare attack category classes (1 sample):", rare_attack_classes.tolist())
            print("Dropping rare combined (Label, Traffic Type) classes (1 sample):", rare_combined.tolist())

            df = df[~df['Label'].isin(rare_device_classes)]
            df = df[~df['Traffic Type'].isin(rare_attack_classes)]
            df = df[~df[['Label', 'Traffic Type']].apply(tuple, axis=1).isin(rare_combined)]
        else:
            print(f"\n{target_column} class distribution after sampling:")
            print(pd.Series(df[target_column]).value_counts())
            print(f"\n{target_column} class distribution before dropping rare classes:")
            target_counts = pd.Series(df[target_column]).value_counts()
            print(target_counts)
            rare_classes = target_counts[target_counts == 1].index
            print(f"Dropping rare {target_column} classes (1 sample):", rare_classes.tolist())
            df = df[~df[target_column].isin(rare_classes)]

        print(f"\nAfter dropping rare classes, dataset has {len(df)} rows.")

        if 'MAC' in df.columns:
            df = df.drop(columns=['MAC'])

        if output_mode == 'multi':
            X = df.drop(columns=['Label', 'Traffic Type'])
            y_device = df['Label']
            y_attack = df['Traffic Type']
        else:
            X = df.drop(columns=[target_column])
            y = df[target_column]

        non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
        print(f"\nNon-numeric columns: {list(non_numeric_cols)}")

        label_encoders = {}
        for col in non_numeric_cols:
            print(f"Encoding non-numeric column: {col}")
            print(f"Sample values: {X[col].head().tolist()}")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            print("Error: Some columns are still non-numeric after encoding:")
            print(X.dtypes)
            raise ValueError("Non-numeric columns detected after preprocessing.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.clip(X_scaled, -10, 10)
        print("\nClipped feature matrix to range [-10, 10] to prevent numerical instability.")

        if output_mode == 'multi':
            le_device = LabelEncoder()
            le_attack = LabelEncoder()
            y_device_encoded = le_device.fit_transform(y_device)
            y_attack_encoded = le_attack.fit_transform(y_attack)
            y_encoded = (y_device_encoded, y_attack_encoded)
            le = (le_device, le_attack)
            num_classes_device = len(le_device.classes_)
            num_classes_attack = len(le_attack.classes_)
            if np.any(y_device_encoded < 0) or np.any(y_attack_encoded < 0):
                print("Error: Negative labels detected in y_encoded.")
                return None, None, None, None, None, None, None, None, None
        else:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            num_classes_device = len(le.classes_)
            num_classes_attack = None
            if np.any(y_encoded < 0):
                print("Error: Negative labels detected in y_encoded.")
                return None, None, None, None, None, None, None, None, None

        # Apply distillation-specific preprocessing
        if distillation_method == 'coreset':
            n_samples = min(10000, len(X_scaled))
            X_scaled, indices = select_coreset(X_scaled, n_samples=n_samples)
            if output_mode == 'multi':
                y_encoded = (y_encoded[0][indices], y_encoded[1][indices])
            else:
                y_encoded = y_encoded[indices]
            print(f"Applied coreset distillation: reduced to {n_samples} samples.")
        elif distillation_method == 'generative':
            n_synthetic = min(10000, len(X_scaled))
            X_synthetic, y_synthetic = generate_synthetic_data(X_scaled, y_encoded, output_mode=output_mode,
                                                               n_synthetic=n_synthetic)
            X_scaled = np.concatenate([X_scaled, X_synthetic], axis=0)
            if output_mode == 'multi':
                y_encoded = (
                    np.concatenate([y_encoded[0], y_synthetic[0]], axis=0),
                    np.concatenate([y_encoded[1], y_synthetic[1]], axis=0)
                )
            else:
                y_encoded = np.concatenate([y_encoded, y_synthetic], axis=0)
            print(f"Applied generative distillation: added {n_synthetic} synthetic samples.")

        X_scaled = np.array(X_scaled)
        print(f"\nFeature matrix shape: {X_scaled.shape}")
        if output_mode == 'multi':
            print(f"Device labels shape: {y_encoded[0].shape}")
            print(f"Attack labels shape: {y_encoded[1].shape}")
        else:
            print(f"Target labels shape: {y_encoded.shape}")

        return X_scaled, y_encoded, le, X.columns, scaler, num_classes_device, num_classes_attack, output_dir, checkpoint_dir

    except Exception as e:
        print(f"Error in load_and_preprocess_data: {e}")
        return None, None, None, None, None, None, None, None, None