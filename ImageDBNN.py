import torch
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Union
from collections import defaultdict
import requests
from io import StringIO
import os,re
import json
from itertools import combinations
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import normaltest
import numpy as np
from itertools import combinations
import torch
import os
import pickle
import configparser
import traceback  # Add to provide debug
#from Invertible_DBNN import InvertibleDBNN
#------------------------------------------------------------------------Declarations---------------------
# Device configuration - set this first since other classes need it
Train_device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Default device
Trials = 100  # Number of epochs to wait for improvement in training
cardinality_threshold =0.9
cardinality_tolerance=4 #Use when the features are likely to be extremly diverse and deciimal values;4 means, precison restricted to 4 decimal places
LearningRate =0.1
TrainingRandomSeed=42  #None # 42
Epochs=1000
TestFraction=0.2
Train=True #True #False #
Train_only=False #True #
Predict=True
Gen_Samples=False
EnableAdaptive = True  # New parameter to control adaptive training
# Assume no keyboard control by default. If you have X11 running and want to be interactive, set nokbd = False
nokbd =  False # Enables interactive keyboard when training (q and Q will not have any effect)
display = None  # Initialize display variable
#----------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
import traceback
from typing import Dict, List, Union, Optional
from collections import defaultdict
import requests
from io import StringIO
import os
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from io import StringIO
import zipfile
import tarfile
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

class DatasetProcessor:
    """A class to handle dataset-related operations such as downloading, processing, and formatting."""

    def __init__(self, data_dir: str = 'data', config_dir: str = 'config'):
        """
        Initialize the DatasetProcessor.

        Args:
            data_dir: Directory to store datasets.
            config_dir: Directory to store configuration files.
        """
        self.data_dir = data_dir
        self.config_dir = config_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        self.base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        self.compressed_extensions = ['.zip', '.gz', '.tar', '.7z', '.rar']
        self.colors = Colors()


    def download_uci_dataset(self, dataset_name: str, url: str) -> str:
        """
        Download a dataset from the UCI repository.

        Args:
            dataset_name: Name of the dataset.
            url: URL to the dataset.

        Returns:
            Path to the downloaded dataset.
        """
        dataset_path = os.path.join(self.data_dir, f"{dataset_name}.csv")
        if os.path.exists(dataset_path):
            print("\033[K" +f"Dataset {dataset_name} already exists at {dataset_path}.")
            return dataset_path

        print("\033[K" +f"Downloading dataset {dataset_name} from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        # Save the dataset
        with open(dataset_path, 'wb') as f:
            f.write(response.content)

        print("\033[K" +f"Dataset saved to {dataset_path}.")
        return dataset_path


    def search_uci_repository(self, query: str) -> List[Dict[str, str]]:
        """
        Search the UCI repository for datasets matching a query.

        Args:
            query: Search query.

        Returns:
            List of dictionaries containing dataset information.
        """
        search_url = f"https://archive.ics.uci.edu/ml/datasets.php?format=json&query={query}"
        response = requests.get(search_url)
        response.raise_for_status()

        datasets = response.json()
        print("\033[K" +f"Found {len(datasets)} datasets matching query '{query}'.")
        return datasets


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def color_value(current_value, previous_value=None, higher_is_better=True):
        """Color a value based on whether it improved or declined"""
        if previous_value is None:
            return f"{current_value:.4f}"

        if higher_is_better:
            if current_value > previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value < previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"
        else:  # lower is better
            if current_value < previous_value:
                return f"{Colors.GREEN}{current_value:.4f}{Colors.ENDC}"
            elif current_value > previous_value:
                return f"{Colors.RED}{current_value:.4f}{Colors.ENDC}"

        return f"{current_value:.4f}"  # No color if equal

    @staticmethod
    def highlight_dataset(name):
        """Highlight dataset name in red"""
        return f"{Colors.RED}{name}{Colors.ENDC}"

    @staticmethod
    def highlight_time(time_value):
        """Color time values based on threshold"""
        if time_value < 10:
            return f"{Colors.GREEN}{time_value:.2f}{Colors.ENDC}"
        elif time_value < 30:
            return f"{Colors.YELLOW}{time_value:.2f}{Colors.ENDC}"
        else:
            return f"{Colors.RED}{time_value:.2f}{Colors.ENDC}"

class DatasetConfig:
    """Enhanced dataset configuration handling with support for column names and URLs"""

    DEFAULT_CONFIG = {
        "file_path": None,
        "column_names": None,
        "target_column": "target",
        "separator": ",",
        "has_header": True,
        "likelihood_config": {
            "feature_group_size": 2,
            "max_combinations": 100000,
            "bin_sizes": [128]
        },
        "active_learning": {
            "tolerance": 1.0,
            "cardinality_threshold_percentile": 95
        },
        "training_params": {
            "save_plots": True,  # Parameter to save plots
            "Save_training_epochs": False,  # Save the epochs parameter
            "training_save_path": "data"  # Save epochs path parameter
        }
    }


    @staticmethod
    def is_url(path: str) -> bool:
        """Check if the given path is a URL"""
        return path.startswith(('http://', 'https://'))

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate that a URL is accessible"""
        try:
            response = requests.head(url)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def validate_columns(config: Dict) -> bool:
        """Validate column configuration"""
        if 'column_names' in config and config['column_names']:
            if not isinstance(config['column_names'], list):
                print("\033[K" +"Error: column_names must be a list")
                return False

            # Validate target column is in column names
            if config['target_column'] not in config['column_names']:
                print("\033[K" +f"Error: target_column '{config['target_column']}' not found in column_names")
                return False

        return True

    @staticmethod
    def create_default_config(dataset_name: str) -> Dict:
        """Create a default configuration file with enhanced defaults"""
        config = DatasetConfig.DEFAULT_CONFIG.copy()
        config['file_path'] = f"{dataset_name}.csv"

        # Try to infer column names from CSV if it exists
        if os.path.exists(config['file_path']):
            try:
                with open(config['file_path'], 'r') as f:
                    header = f.readline().strip()
                    config['column_names'] = header.split(config['separator'])
                    if config['column_names']:
                        config['target_column'] = config['column_names'][-1]
            except Exception as e:
                print("\033[K" +f"Warning: Could not read header from {config['file_path']}: {str(e)}")

        # Add model type configuration
        config['modelType'] = "Histogram"  # Default to Histogram model

        # Add training parameters
        config['training_params'] = {
            "trials": 100,
            "cardinality_threshold": 0.9,
            "minimum_training_accuracy": 0.95,
            "cardinality_tolerance": 4,
            "learning_rate": 0.001,
            "random_seed": 42,
            "epochs": 1000,
            "test_fraction": 0.2,
            "n_bins_per_dim": 21,
            "enable_adaptive": True,
            "compute_device": "auto",
            "invert_DBNN": True,
            "reconstruction_weight": 0.5,
            "feedback_strength": 0.3,
            "inverse_learning_rate": 0.001,
            "save_plots": True
        }
        config["active_learning"]= {
            "tolerance": 1.0,
            "cardinality_threshold_percentile": 95,
            "strong_margin_threshold": 0.3,
            "marginal_margin_threshold": 0.1,
            "min_divergence": 0.1
        }
        config["execution_flags"]= {
            "train": true,
            "train_only": false,
            "predict": true,
            "fresh_start": false,
            "use_previous_model": true
        }
        # Save the configuration
        config_path = f"data/{dataset_name}/{dataset_name}.conf"
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print("\033[K" +f"Created default configuration file: {config_path}")
        except Exception as e:
            print("\033[K" +f"Warning: Could not save configuration file: {str(e)}")

        return config


    @staticmethod
    def load_config(dataset_name: str) -> Dict:
        """Enhanced configuration loading with URL handling and comment removal"""
        if not dataset_name or not isinstance(dataset_name, str):
            print("\033[K" +"Error: Invalid dataset name provided.")
            return None

        config_path = os.path.join('data', dataset_name,f"{dataset_name}.conf")

        try:
            # Check if configuration file exists
            if not os.path.exists(config_path):
                print("\033[K" +f"Configuration file {config_path} not found.")
                print("\033[K" +f"Creating default configuration for {dataset_name}")
                return DatasetConfig.create_default_config(dataset_name)

            # Read and parse configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config_text = f.read()
            # Remove comments and parse
            def remove_comments(json_str):
                lines = []
                in_multiline_comment = False
                for line in json_str.split('\n'):
                    if '_comment' in line:
                        continue
                    if '/*' in line and '*/' in line:
                        line = line[:line.find('/*')] + line[line.find('*/') + 2:]
                    elif '/*' in line:
                        in_multiline_comment = True
                        line = line[:line.find('/*')]
                    elif '*/' in line:
                        in_multiline_comment = False
                        line = line[line.find('*/') + 2:]
                    elif in_multiline_comment:
                        continue
                    if '//' in line and not ('http://' in line or 'https://' in line):
                        line = line.split('//')[0]
                    stripped = line.strip()
                    if stripped and not stripped.startswith('_comment'):
                        lines.append(stripped)
                return '\n'.join(lines)

            # Remove comments and parse JSON
            clean_config = remove_comments(config_text)
            config = json.loads(clean_config)
            # Validate configuration
            validated_config = DatasetConfig.DEFAULT_CONFIG.copy()
            validated_config.update(config)

            # Handle file path
            if validated_config.get('file_path'):
                # If path is relative to data directory, update it
                if not os.path.exists(validated_config['file_path']):
                    alt_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
                    if os.path.exists(alt_path):
                        validated_config['file_path'] = alt_path
                        print("\033[K" +f"Using data file: {alt_path}")

            # If still no file path, try default location
            if not validated_config.get('file_path'):
                default_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
                if os.path.exists(default_path):
                    validated_config['file_path'] = default_path
                    print("\033[K" +f"Using default data file: {default_path}")

            # If URL, handle download
            if DatasetConfig.is_url(validated_config.get('file_path', '')):
                url = validated_config['file_path']
                local_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")

                if not os.path.exists(local_path):
                    print("\033[K" +f"Downloading dataset from {url}")
                    if not DatasetConfig.download_dataset(url, local_path):
                        print("\033[K" +f"Failed to download dataset from {url}")
                        return None
                    print("\033[K" +f"Downloaded dataset to {local_path}")

                validated_config['file_path'] = local_path

            # Verify data file exists
            if not validated_config.get('file_path') or not os.path.exists(validated_config['file_path']):
                print("\033[K" +f"Warning: Data file not found")
                return None

            # If no column names provided, try to infer from CSV header
            if not validated_config.get('column_names'):
                try:
                    df = pd.read_csv(validated_config['file_path'], nrows=0)
                    validated_config['column_names'] = df.columns.tolist()
                except Exception as e:
                    print("\033[K" +f"Warning: Could not infer column names: {str(e)}")
                    return None

            return validated_config

        except Exception as e:
            print("\033[K" +f"Error loading configuration for {dataset_name}: {str(e)}")
            traceback.print_exc()
            return None

    @staticmethod
    def download_dataset(url: str, local_path: str) -> bool:
        """Download dataset from URL to local path with proper error handling"""
        try:
            print("\033[K" +f"Downloading dataset from {url}")
            response = requests.get(url, timeout=30)  # Add timeout
            response.raise_for_status()  # Check for HTTP errors

            # Handle potential text/csv content
            content = response.content.decode('utf-8')

            # Save to local file
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print("\033[K" +f"Dataset downloaded successfully to {local_path}")
            return True
        except requests.exceptions.RequestException as e:
            print("\033[K" +f"Error downloading dataset: {str(e)}")
            return False
        except UnicodeDecodeError:
            # Handle binary content
            try:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print("\033[K" +f"Dataset downloaded successfully to {local_path}")
                return True
            except Exception as e:
                print("\033[K" +f"Error saving binary content: {str(e)}")
                return False
        except Exception as e:
            print("\033[K" +f"Unexpected error downloading dataset: {str(e)}")
            return False



    @staticmethod
    def get_available_datasets(create_configs: bool = False) -> List[str]:
        """Get list of available dataset configurations with better filename handling"""
        # Get all config and CSV files
        conf_files = {f.split('.')[0] for f in os.listdir()
                     if f.endswith('.conf') and f != 'adaptive_dbnn.conf'}  # Exclude own config
        csv_files = {f.split('.')[0] for f in os.listdir()
                    if f.endswith('.csv')}

        # Filter out derived filenames
        exclude_suffixes = [
            '_last_testing', '_Last_testing',
            '_last_training', '_Last_training',
            '_predictions', '_training_metrics',
            '_training_metrics_metrics'
        ]

        # Filter CSV files that don't have config
        csv_without_conf = csv_files - conf_files
        csv_without_conf = {name for name in csv_without_conf
                           if not any(name.endswith(suffix) for suffix in exclude_suffixes)}

        # Start with datasets that have config files
        datasets = conf_files

        # If requested, ask about creating configs for remaining CSVs
        if create_configs and csv_without_conf:
            print("\033[K" +"Found CSV files without configuration:")
            for csv_name in sorted(csv_without_conf):
                response = input(f"Create configuration for {csv_name}.csv? (y/n): ")
                if response.lower() == 'y':
                    try:
                        DatasetConfig.create_default_config(csv_name)
                        datasets.add(csv_name)
                    except Exception as e:
                        print("\033[K" +f"Error creating config for {csv_name}: {str(e)}")

        return sorted(list(datasets))


    @staticmethod
    def validate_dataset(dataset_name: str) -> bool:
        """Validate dataset with better name handling"""
        # Check if this is a derived filename
        exclude_suffixes = [
            '_last_testing', '_Last_testing',
            '_last_training', '_Last_training',
            '_predictions', '_training_metrics',
            '_training_metrics_metrics'
        ]

        if any(dataset_name.endswith(suffix) for suffix in exclude_suffixes):
            print("\033[K" +f"Skipping validation for derived dataset: {dataset_name}")
            return False

        config = DatasetConfig.load_config(dataset_name)
        file_path = config['file_path']

        # Handle URL-based datasets
        if DatasetConfig.is_url(file_path):
            if not DatasetConfig.validate_url(file_path):
                print("\033[K" +f"Warning: Dataset URL {file_path} is not accessible")
                return False

            # Download to local cache if needed
            local_path = f"{dataset_name}.csv"
            if not os.path.exists(local_path):
                if not DatasetConfig.download_dataset(file_path, local_path):
                    return False
            file_path = local_path

        if not os.path.exists(file_path):
            print("\033[K" +f"Warning: Dataset file {file_path} not found")
            return False

        return True

#---------------------------------------Feature Filter with a #------------------------------------
def _filter_features_from_config(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Filter DataFrame to only include specified columns from config

    Args:
        df: Input DataFrame
        config: Configuration dictionary containing column names

    Returns:
        DataFrame with only the specified columns
    """
    # If no column names in config, return original DataFrame
    if 'column_names' not in config or not config['column_names']:
        print("\033[K" +"No column names specified in config. Keeping all columns.")
        return df

    # Get current DataFrame columns
    current_cols = df.columns.tolist()
    #print("\033[K" +f"Current DataFrame columns: {current_cols}")

    # Get column names from config (only those not commented out)
    requested_columns = [
        name.strip() for name in config['column_names']
        if not name.strip().startswith('#')
    ]

    # If no uncommented columns found in config, return original DataFrame
    if not requested_columns:
        print("\033[K" +"No uncommented column names found in config. Returning original DataFrame.")
        return df

    # Check if any requested columns exist in the DataFrame
    valid_columns = [col for col in requested_columns if col in current_cols]

    # If no valid columns found, return original DataFrame
    if not valid_columns:
        print("\033[K" +"None of the requested columns exist in the DataFrame. Returning original DataFrame.")
        return df

    # Return DataFrame with only the columns to keep
    #print("\033[K" +f"Keeping only these features: {valid_columns}")
    return df[valid_columns]
#-------------------------------------------------
class ComputationCache:
    """Cache for frequently used computations"""
    def __init__(self, device):
        self.device = device
        self.bin_edges_cache = {}
        self.bin_indices_cache = {}
        self.feature_group_cache = {}
        self.max_cache_size = 1000

    def get_bin_edges(self, group_idx, edges):
        if group_idx not in self.bin_edges_cache:
            self.bin_edges_cache[group_idx] = torch.stack([
                edge.contiguous() for edge in edges
            ]).to(self.device)
        return self.bin_edges_cache[group_idx]

    def get_feature_group(self, features, group_idx, feature_pairs):
        key = (features.shape[0], group_idx)
        if key not in self.feature_group_cache:
            if len(self.feature_group_cache) >= self.max_cache_size:
                self.feature_group_cache.clear()
            self.feature_group_cache[key] = features[:, feature_pairs[group_idx]].contiguous()
        return self.feature_group_cache[key]

class BinWeightUpdater:
    def __init__(self, n_classes, feature_pairs, n_bins_per_dim=5,batch_size=128):
        self.n_classes = n_classes
        self.feature_pairs = feature_pairs
        self.n_bins_per_dim = n_bins_per_dim
        self.device=Train_device
        # Initialize histogram_weights as empty dictionary first
        self.histogram_weights = {}
        self.batch_size=batch_size

        # Create weights for each class and feature pair
        for class_id in range(n_classes):
            self.histogram_weights[class_id] = {}
            for pair_idx in range(len(feature_pairs)):
                # Initialize with default weight of 0.1
                #print("\033[K" +f"[DEBUG] Creating weights for class {class_id}, pair {pair_idx}")
                self.histogram_weights[class_id][pair_idx] = torch.full(
                    (n_bins_per_dim, n_bins_per_dim),
                    0.1,
                    dtype=torch.float32,
                    device=self.device  # Ensure weights are created on correct device
                ).contiguous()

        # Initialize weights for each class and feature pair
        self.gaussian_weights = {}
        for class_id in range(n_classes):
            self.gaussian_weights[class_id] = {}
            for pair_idx in range(len(feature_pairs)):
                # Initialize with default weight of 0.1
                self.gaussian_weights[class_id][pair_idx] = torch.tensor(0.1,
                    dtype=torch.float32,
                    device=self.device
                ).contiguous()

        # Verify initialization
        print("\033[K" +f"[DEBUG] Weight initialization complete. Structure:")
        print("\033[K" +f"- Number of classes: {len(self.histogram_weights)}")
        for class_id in self.histogram_weights:
            print("\033[K" +f"- Class {class_id}: {len(self.histogram_weights[class_id])} feature pairs")

        # Use a single contiguous tensor for all weights
        self.weights = torch.full(
            (n_classes, len(feature_pairs), n_bins_per_dim, n_bins_per_dim),
            0.1,
            dtype=torch.float32,
            device=self.device  # Ensure weights are created on correct device
        ).contiguous()

        # Pre-allocate update buffers
        self.update_indices = torch.zeros((3, 1000), dtype=torch.long)  # [dim, max_updates]
        self.update_values = torch.zeros(1000, dtype=torch.float32)
        self.update_count = 0


    def batch_update_weights(self, class_indices, pair_indices, bin_indices, adjustments):
            """Batch update with compatibility and proper shape handling"""
            n_updates = len(class_indices)

            # Process in batches for memory efficiency
            batch_size = self.batch_size  # Adjust based on available memory
            for i in range(0, n_updates, batch_size):
                end_idx = min(i + batch_size, n_updates)

                for idx in range(i, end_idx):
                    class_id = int(class_indices[idx])
                    pair_idx = int(pair_indices[idx])

                    # Handle bin indices properly based on their structure
                    if isinstance(bin_indices[idx], tuple):
                        bin_i, bin_j = bin_indices[idx]
                    else:
                        # If bin_indices is a tensor or array
                        bin_i = bin_indices[idx][0] if len(bin_indices[idx].shape) > 1 else bin_indices[idx]
                        bin_j = bin_indices[idx][1] if len(bin_indices[idx].shape) > 1 else bin_indices[idx]

                    # Ensure indices are properly shaped scalars
                    bin_i = int(bin_i.item() if torch.is_tensor(bin_i) else bin_i)
                    bin_j = int(bin_j.item() if torch.is_tensor(bin_j) else bin_j)

                    adjustment = float(adjustments[idx].item() if torch.is_tensor(adjustments[idx]) else adjustments[idx])

                    # Update weight with proper shape handling
                    self.histogram_weights[class_id][pair_idx][bin_i, bin_j] += adjustment


    def get_histogram_weights(self, class_id: int, pair_idx: int) -> torch.Tensor:
        """Get weights ensuring proper dimensions"""
        class_id = int(class_id)
        pair_idx = int(pair_idx)

        if class_id not in self.histogram_weights:
            raise KeyError(f"Invalid class_id: {class_id}")
        if pair_idx not in self.histogram_weights[class_id]:
            raise KeyError(f"Invalid pair_idx: {pair_idx}")

        weights = self.histogram_weights[class_id][pair_idx]
        if len(weights.shape) != 2:
            raise ValueError(f"Invalid weight shape: {weights.shape}, expected 2D tensor")
        if weights.shape[0] != self.n_bins_per_dim or weights.shape[1] != self.n_bins_per_dim:
            raise ValueError(f"Invalid weight dimensions: {weights.shape}, expected ({self.n_bins_per_dim}, {self.n_bins_per_dim})")

        return weights


    def _ensure_buffers(self, batch_size):
        """Ensure buffers exist and are the right size"""
        if (self.batch_indices_buffer is None or
            self.batch_indices_buffer.shape[0] < batch_size):
            self.batch_indices_buffer = torch.zeros(
                (batch_size, 2),
                dtype=torch.long,
                device=next(iter(next(iter(self.histogram_weights.values())).values())).device
            ).contiguous()

            self.batch_adjustments_buffer = torch.zeros(
                batch_size,
                dtype=torch.float32,
                device=self.batch_indices_buffer.device
            ).contiguous()

    def update_weight(self, class_id: int, pair_idx: int, bin_i: int, bin_j: int, adjustment: float):
        """Single weight update with proper error handling"""
        try:
            # Convert all indices to integers
            class_id = int(class_id)
            pair_idx = int(pair_idx)
            bin_i = int(bin_i)
            bin_j = int(bin_j)

            # Ensure indices are within bounds
            bin_i = min(max(0, bin_i), self.n_bins_per_dim - 1)
            bin_j = min(max(0, bin_j), self.n_bins_per_dim - 1)

            # Update the weight
            self.histogram_weights[class_id][pair_idx][bin_i, bin_j] += adjustment

        except Exception as e:
            print("\033[K" +f"Error updating weight: {str(e)}")
            print("\033[K" +f"class_id: {class_id}, pair_idx: {pair_idx}")
            print("\033[K" +f"bin_i: {bin_i}, bin_j: {bin_j}")
            print("\033[K" +f"adjustment: {adjustment}")
            raise

    def update_histogram_weights(self, failed_case, true_class, pred_class,
                               bin_indices, posteriors, learning_rate):
        """Update weights with proper type checking"""
        try:
            # Ensure proper types
            true_class = int(true_class)
            pred_class = int(pred_class)

            # Get the posterior values needed for adjustment
            true_posterior = float(posteriors[true_class])
            pred_posterior = float(posteriors[pred_class])

            # Calculate weight adjustment
            adjustment = learning_rate * (1.0 - (true_posterior / pred_posterior))

            for pair_idx, (bin_i, bin_j) in bin_indices.items():
                # Ensure integer indices
                pair_idx = int(pair_idx)
                bin_i = int(bin_i)
                bin_j = int(bin_j)

                # Get and update weights
                weights = self.histogram_weights[true_class][pair_idx]
                weights[bin_i, bin_j] += adjustment

        except Exception as e:
            DEBUG.log(f"Error updating histogram weights:")
            DEBUG.log(f"- True class: {true_class}")
            DEBUG.log(f"- Pred class: {pred_class}")
            DEBUG.log(f"- Adjustment: {adjustment}")
            DEBUG.log(f"- Error: {str(e)}")
            raise

    def update_gaussian_weights(self, failed_case, true_class, pred_class,
                               component_responsibilities, posteriors, learning_rate):
        """Update weights for Gaussian components with improved efficiency"""
        DEBUG.log(f" Updating Gaussian weights for class {true_class}")

        try:
            # Convert tensor values to Python types and validate
            true_class = int(true_class) if isinstance(true_class, torch.Tensor) else true_class
            pred_class = int(pred_class) if isinstance(pred_class, torch.Tensor) else pred_class

            if true_class not in self.gaussian_weights:
                raise ValueError(f"Invalid true_class: {true_class}")

            # Get posterior values with type checking
            true_posterior = float(posteriors[true_class]) if isinstance(posteriors, torch.Tensor) else posteriors[true_class]
            pred_posterior = float(posteriors[pred_class]) if isinstance(posteriors, torch.Tensor) else posteriors[pred_class]

            # Calculate adjustment based on posterior ratio with stability check
            adjustment = learning_rate * (1.0 - max(min(true_posterior / pred_posterior, 10), 0.1))
            DEBUG.log(f" Weight adjustment: {adjustment}")

            # Process each feature pair efficiently
            for pair_idx in range(len(self.feature_pairs)):
                # Get and validate responsibility matrix
                resp_matrix = component_responsibilities[pair_idx]
                if not isinstance(resp_matrix, torch.Tensor):
                    resp_matrix = torch.tensor(resp_matrix)
                resp_matrix = resp_matrix.to(self.gaussian_weights[true_class][pair_idx].device)

                # Ensure shapes match
                current_weights = self.gaussian_weights[true_class][pair_idx]
                if resp_matrix.shape != current_weights.shape:
                    DEBUG.log(f" Shape mismatch - resp_matrix: {resp_matrix.shape}, weights: {current_weights.shape}")
                    resp_matrix = resp_matrix[:current_weights.shape[0], :current_weights.shape[1]]

                # Update weights with stability check
                weight_update = resp_matrix * adjustment
                weight_update = torch.clamp(weight_update, -1.0, 1.0)  # Prevent extreme updates
                self.gaussian_weights[true_class][pair_idx] += weight_update

                # Apply non-negativity constraint
                self.gaussian_weights[true_class][pair_idx].clamp_(min=0.0)

        except Exception as e:
            DEBUG.log(f" Error updating Gaussian weights: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            raise

    def get_gaussian_weights(self, class_id, pair_idx):
        """Get Gaussian weights with proper type conversion and validation"""
        try:
            # Convert tensor values to Python integers
            class_id = int(class_id) if isinstance(class_id, torch.Tensor) else class_id
            pair_idx = int(pair_idx) if isinstance(pair_idx, torch.Tensor) else pair_idx

            if class_id not in self.gaussian_weights:
                raise KeyError(f"Invalid class_id: {class_id}")
            if pair_idx not in self.gaussian_weights[class_id]:
                raise KeyError(f"Invalid pair_idx: {pair_idx}")

            weights = self.gaussian_weights[class_id][pair_idx]
            DEBUG.log(f" Retrieved Gaussian weights for class {class_id}, pair {pair_idx}, shape: {weights.shape}")
            return weights

        except Exception as e:
            DEBUG.log(f" Error retrieving Gaussian weights: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            raise

    # Modified posterior computation for Histogram model
    def compute_histogram_posterior(self, features, bin_indices):
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        for group_idx, feature_group in enumerate(self.likelihood_params['feature_pairs']):
            bin_edges = self.likelihood_params['bin_edges'][group_idx]
            bin_probs = self.likelihood_params['bin_probs'][group_idx]

            # Get bin-specific weights
            bin_weights = self.weight_updater.get_histogram_weights(
                class_idx,
                group_idx
            )[bin_indices[group_idx]]

            # Apply bin-specific weights to probabilities
            weighted_probs = bin_probs * bin_weights.unsqueeze(0)

            # Continue with regular posterior computation...
            group_log_likelihoods = torch.log(weighted_probs + epsilon)
            log_likelihoods.add_(group_log_likelihoods)

        return log_likelihoods

    # Modified posterior computation for Gaussian model
    def compute_gaussian_posterior(self, features, component_responsibilities):
        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        for group_idx, feature_group in enumerate(self.likelihood_params['feature_pairs']):
            # Get component-specific weights
            component_weights = self.weight_updater.get_gaussian_weights(
                class_idx,
                group_idx
            )

            # Weight the Gaussian components
            weighted_resp = component_responsibilities[group_idx] * component_weights

            # Continue with regular posterior computation...
            group_log_likelihoods = torch.log(weighted_resp.sum() + epsilon)
            log_likelihoods.add_(group_log_likelihoods)

        return log_likelihoods
#----------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm

class InvertibleDBNN(nn.Module):
    """Invertible DBNN for reconstructing input features from classification probabilities."""

    def __init__(self, forward_model: nn.Module, feature_dims: int, n_classes: int, hidden_dims: int = 128, device: str = 'cuda'):
        """
        Initialize the Invertible DBNN.

        Args:
            forward_model (nn.Module): The forward DBNN model.
            feature_dims (int): Number of input feature dimensions.
            n_classes (int): Number of classes in the classification task.
            hidden_dims (int): Number of hidden dimensions in the inverse model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        super(InvertibleDBNN, self).__init__()
        self.forward_model = forward_model
        self.feature_dims = feature_dims
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.device = device

        # Define the inverse model architecture
        self.inverse_model = nn.Sequential(
            nn.Linear(n_classes, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, feature_dims))

        # Move model to the appropriate device
        self.to(device)

    def forward(self, class_probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the inverse model to reconstruct features.

        Args:
            class_probs (torch.Tensor): Classification probabilities (batch_size, n_classes).

        Returns:
            torch.Tensor: Reconstructed features (batch_size, feature_dims).
        """
        return self.inverse_model(class_probs)

    def reconstruct_features(self, class_probs: torch.Tensor, original_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct features from classification probabilities.

        Args:
            class_probs (torch.Tensor): Classification probabilities (batch_size, n_classes).
            original_features (Optional[torch.Tensor]): Original features for validation (batch_size, feature_dims).

        Returns:
            torch.Tensor: Reconstructed features (batch_size, feature_dims).
        """
        reconstructed_features = self.forward(class_probs)

        if original_features is not None:
            # Calculate reconstruction error
            reconstruction_error = torch.mean((reconstructed_features - original_features) ** 2)
            print("\033[K" +f"Reconstruction Error: {reconstruction_error.item():.4f}")

        return reconstructed_features

    def train_inverse_model(self, class_probs: torch.Tensor, original_features: torch.Tensor, epochs: int = 100, lr: float = 0.001):
        """
        Train the inverse model to reconstruct features from classification probabilities.

        Args:
            class_probs (torch.Tensor): Classification probabilities (batch_size, n_classes).
            original_features (torch.Tensor): Original features (batch_size, feature_dims).
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in tqdm(range(epochs), desc="Training Inverse Model"):
            optimizer.zero_grad()

            # Forward pass
            reconstructed_features = self.forward(class_probs)

            # Compute reconstruction loss
            loss = criterion(reconstructed_features, original_features)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print("\033[K" +f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def save_model(self, path: str):
        """Save the inverse model to a file."""
        torch.save(self.state_dict(), path)
        print("\033[K" +f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the inverse model from a file."""
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        print("\033[K" +f"Model loaded from {path}")
#----------------------------------------------DBNN class-------------------------------------------------------------
class GPUDBNN:
    """GPU-Optimized Deep Bayesian Neural Network with Parallel Feature Pair Processing"""

    def __init__(self, dataset_name: str, learning_rate: float = LearningRate,
                 max_epochs: int = Epochs, test_size: float = TestFraction,
                 random_state: int = TrainingRandomSeed, device: str = None,
                 fresh: bool = False, use_previous_model: bool = True,
                 n_bins_per_dim: int = 64, model_type: str = "Histogram"):
        """Initialize GPUDBNN with support for continued training with fresh data"""

        # Set dataset_name and model type first
        self.dataset_name = dataset_name
        self.model_type = model_type  # Store model type as instance variable
        self.device = Train_device
        self.computation_cache = ComputationCache(self.device)
        # Initialize train/test indices
        self.train_indices = []
        self.test_indices = None
        self._last_metrics_printed =False
        # Add new attribute for bin-specific weights
        self.n_bins_per_dim = n_bins_per_dim
        self.weight_updater = None  # Will be initialized after computing likelihood params

        # Load configuration before potential cleanup
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.feature_bounds = None  # Store global min/max for each

        # Initialize other attributes

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.test_size = test_size
        if random_state !=-1:
            self.random_state = random_state
            self.shuffle_state =1
        else:
            self.random_state = -1
            self.shuffle_state =-1
        #self.compute_dtype = torch.float64  # Use double precision for computations
        self.cardinality_tolerance = cardinality_tolerance  # Only for feature grouping
        self.fresh_start = fresh
        self.use_previous_model = use_previous_model
        # Create Model directory
        os.makedirs('Model', exist_ok=True)

        # Load configuration and data
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.target_column = self.config['target_column']

        # Initialize model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.bin_edges = None  # Add bin_edges attribute
        self.gaussian_params = None  # Add gaussian_params attribute for Gaussian model
        self.best_W = None
        self.best_error = float('inf')
        self.current_W = None

        # Enable cuDNN autotuner if using CUDA
        if self.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True

        # Pin memory for faster CPU->GPU transfer
        if torch.cuda.is_available():
            self.pin_memory = True
            torch.cuda.empty_cache()

        # Handle fresh start after configuration is loaded
        # Handle model state based on flags
        if not fresh and use_previous_model:
            # Load previous model state
            self._load_model_components()
            self._load_best_weights()
            self._load_categorical_encoders()
        elif fresh and use_previous_model:
            # Use previous model weights but start with fresh data
            self._load_best_weights()
            self._load_categorical_encoders()
        else:
            # Complete fresh start
            self._clean_existing_model()


        #------------------------------------------Adaptive Learning--------------------------------------
        super().__init__()
        self.adaptive_learning = True
        self.base_save_path = './data'
        os.makedirs(self.base_save_path, exist_ok=True)
        self.in_adaptive_fit=False # Set when we are in adaptive learning process
        #------------------------------------------Adaptive Learning--------------------------------------
        # Automatically select device if none specified

        print("\033[K" +f"Using device: {self.device}")

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.test_size = test_size

        # Model components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.likelihood_params = None
        self.feature_pairs = None
        self.best_W = None
        self.best_error = float('inf')
        self.current_W = None

        # Categorical feature handling
        self.categorical_encoders = {}

        # Create Model directory
        os.makedirs('Model', exist_ok=True)

        # Load dataset configuration and data
        self.config = DatasetConfig.load_config(self.dataset_name)
        self.data = self._load_dataset()

        self.target_column = self.config['target_column']

        # Load saved weights and encoders
        self._load_best_weights()
        self._load_categorical_encoders()

    def _compute_bin_edges(self, dataset: torch.Tensor, bin_sizes: List[int]) -> List[torch.Tensor]:
        """
        Compute bin edges for all feature pairs once during initialization.

        Args:
            dataset: Input tensor of shape [n_samples, n_features]
            bin_sizes: List of integers specifying bin sizes for each dimension

        Returns:
            List of tensors containing bin edges for each feature pair
        """
        DEBUG.log("Starting _compute_bin_edges")
        print("\033[K" +f"Dataset shape: {dataset.shape}")
        print("\033[K" +f"Bin sizes: {bin_sizes}")

        bin_edges = []
        for feature_group in self.feature_pairs:
            feature_group = [int(x) for x in feature_group]
            group_data = dataset[:, feature_group].contiguous()
            n_dims = len(feature_group)

            # Get appropriate bin sizes for this group
            group_bin_sizes = bin_sizes[:n_dims] if len(bin_sizes) > 1 else [bin_sizes[0]] * n_dims

            # Compute bin edges for all dimensions
            group_bin_edges = []
            for dim in range(n_dims):
                dim_data = group_data[:, dim].contiguous()
                dim_min, dim_max = dim_data.min(), dim_data.max()
                padding = (dim_max - dim_min) * 0.01

                # Ensure bin size is valid
                if group_bin_sizes[dim] <= 1:
                    raise ValueError(f"Bin size must be > 1, got {group_bin_sizes[dim]}")

                # Compute bin edges
                try:
                    edges = torch.linspace(
                        dim_min - padding,
                        dim_max + padding,
                        group_bin_sizes[dim] + 1,  # Use configured bin size for this dimension
                        device=self.device
                    ).contiguous()
                    group_bin_edges.append(edges)
                    DEBUG.log(f" Dimension {dim} edges range: {edges[0].item():.3f} to {edges[-1].item():.3f}")
                except Exception as e:
                    print("\033[K" +f"Error computing bin edges for dimension {dim}: {str(e)}")
                    print("\033[K" +f"dim_min: {dim_min}, dim_max: {dim_max}, padding: {padding}, bin_size: {group_bin_sizes[dim]}")
                    raise

            bin_edges.append(group_bin_edges)

        return bin_edges
#----------------------
    def _compute_balanced_accuracy(self, y_true, y_pred):
        """Compute class-balanced accuracy"""
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        return np.mean(per_class_acc[~np.isnan(per_class_acc)])

    def _select_balanced_samples(self, misclassified_indices, y_test, n_samples=2):
        """Select misclassified samples with balanced class representation and error margins"""
        class_indices = defaultdict(list)
        y_test_cpu = y_test.cpu()

        # Get probability info for each misclassified sample
        probs_info = {}
        for idx in misclassified_indices:
            true_class = y_test_cpu[idx].item()
            probs = self._compute_batch_posterior(self.X_tensor[idx].unsqueeze(0))[0]
            pred_class = torch.argmax(probs).item()
            true_prob = probs[true_class].item()
            pred_prob = probs[pred_class].item()
            error_margin = pred_prob - true_prob  # Higher means more confident wrong prediction

            class_indices[true_class].append(idx)
            probs_info[idx] = {
                'true_class': true_class,
                'pred_class': pred_class,
                'error_margin': error_margin,
                'true_prob': true_prob,
                'pred_prob': pred_prob
            }

        # Calculate class-wise statistics
        class_stats = {}
        for cls in class_indices:
            cls_samples = y_test_cpu == cls
            total = cls_samples.sum().item()
            misclassified = len(class_indices[cls])
            error_rate = misclassified / total if total > 0 else 1.0
            class_stats[cls] = {
                'error_rate': error_rate,
                'total': total,
                'misclassified': misclassified
            }

        selected_indices = []
        remaining_samples = n_samples

        # First, ensure at least one sample from each failing class
        for true_class, stats in class_stats.items():
            if stats['error_rate'] > 0:  # Class has errors
                # Get samples from this class sorted by error margin
                class_samples = [(idx, probs_info[idx]['error_margin'])
                               for idx in class_indices[true_class]]
                class_samples.sort(key=lambda x: x[1], reverse=True)  # Highest error margin first

                # Select the sample with highest error margin
                if class_samples:
                    idx = class_samples[0][0]
                    selected_indices.append(idx)
                    remaining_samples -= 1

                    # Print selection info
                    info = probs_info[idx]
                    true_class_name = self.label_encoder.inverse_transform([info['true_class']])[0]
                    pred_class_name = self.label_encoder.inverse_transform([info['pred_class']])[0]
                    print("\033[K" +f"Adding sample from class {true_class_name} (misclassified as {pred_class_name}, "
                          f"error margin: {info['error_margin']:.3f})")

        # If we still have samples to select, choose based on error rates and margins
        if remaining_samples > 0:
            # Create pool of remaining samples with weights
            remaining_pool = []
            for idx in misclassified_indices:
                if idx not in selected_indices:
                    info = probs_info[idx]
                    cls_stats = class_stats[info['true_class']]

                    # Weight based on class error rate and individual error margin
                    weight = cls_stats['error_rate'] * (1 + info['error_margin'])
                    remaining_pool.append((idx, weight))

            # Sort by weight and select top remaining_samples
            remaining_pool.sort(key=lambda x: x[1], reverse=True)
            for idx, weight in remaining_pool[:remaining_samples]:
                selected_indices.append(idx)
                info = probs_info[idx]
                true_class_name = self.label_encoder.inverse_transform([info['true_class']])[0]
                pred_class_name = self.label_encoder.inverse_transform([info['pred_class']])[0]
                print("\033[K" +f"Adding additional sample from class {true_class_name} (misclassified as {pred_class_name}, "
                      f"error margin: {info['error_margin']:.3f})")

        # Print summary
        print("\033[K" +f"Selection Summary:")
        print("\033[K" +f"Total failing classes: {len(class_stats)}")
        print("\033[K" +f"Selected {len(selected_indices)} samples total")
        for cls in sorted(class_stats.keys()):
            cls_name = self.label_encoder.inverse_transform([cls])[0]
            stats = class_stats[cls]
            selected_from_class = sum(1 for idx in selected_indices
                                    if probs_info[idx]['true_class'] == cls)
            print("\033[K" +f"Class {cls_name}: {selected_from_class} samples selected out of {stats['misclassified']} "
                  f"misclassified (error rate: {stats['error_rate']:.3f})")

        return selected_indices

    def _print_detailed_metrics(self, y_true, y_pred, prefix=""):
        """Print detailed performance metrics with color coding"""
        # Compute metrics
        balanced_acc = self._compute_balanced_accuracy(y_true, y_pred)
        raw_acc = np.mean(y_true == y_pred)

        # Print metrics with colors
        print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}{prefix}Detailed Metrics:{Colors.ENDC}")

        # Raw accuracy
        acc_color = Colors.GREEN if raw_acc >= 0.9 else Colors.YELLOW if raw_acc >= 0.7 else Colors.BLUE
        print("\033[K" +f"{Colors.BOLD}Raw Accuracy:{Colors.ENDC} {acc_color}{raw_acc:.4%}{Colors.ENDC}")

        # Balanced accuracy
        bal_color = Colors.GREEN if balanced_acc >= 0.9 else Colors.YELLOW if balanced_acc >= 0.7 else Colors.BLUE
        print("\033[K" +f"{Colors.BOLD}Balanced Accuracy:{Colors.ENDC} {bal_color}{balanced_acc:.4%}{Colors.ENDC}")

        # Per-class metrics
        print("\033[K" +f"{Colors.BOLD}Per-class Performance:{Colors.ENDC}")
        cm = confusion_matrix(y_true, y_pred)
        class_labels = np.unique(y_true)

        for i, label in enumerate(class_labels):
            class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
            color = Colors.GREEN if class_acc >= 0.9 else Colors.YELLOW if class_acc >= 0.7 else Colors.BLUE
            samples = cm[i].sum()
            print("\033[K" +f"Class {label}: {color}{class_acc:.4%}{Colors.ENDC} ({samples:,} samples)")

        return balanced_acc
#---------------------- -------------------------------------DBNN Class -------------------------------
class DBNNConfig:
    """Configuration class for DBNN parameters"""
    def __init__(self, **kwargs):
        # Training parameters
        self.trials = kwargs.get('trials', 100)
        self.cardinality_threshold = kwargs.get('cardinality_threshold', 0.9)
        self.cardinality_tolerance = kwargs.get('cardinality_tolerance', 4)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.random_seed = kwargs.get('random_seed', 42)
        self.epochs = kwargs.get('epochs', 1000)
        self.test_fraction = kwargs.get('test_fraction', 0.2)
        self.enable_adaptive = kwargs.get('enable_adaptive', True)
        self.batch_size = kwargs.get('batch_size', 128)

        # Model parameters
        self.model_type = kwargs.get('model_type', 'Histogram')  # or 'Gaussian'
        self.n_bins_per_dim = kwargs.get('n_bins_per_dim', 128)

        # Execution flags
        self.train = kwargs.get('train', True)
        self.train_only = kwargs.get('train_only', False)
        self.predict = kwargs.get('predict', True)
        self.fresh_start = kwargs.get('fresh_start', False)
        self.use_previous_model = kwargs.get('use_previous_model', True)

        # Device configuration
        self.device = kwargs.get('device', 'auto')
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Paths configuration
        self.model_dir = kwargs.get('model_dir', 'Model')
        self.training_data_dir = kwargs.get('training_data_dir', 'training_data')

        # Debug configuration
        self.debug = kwargs.get('debug', False)

    @classmethod
    def from_file(cls, config_path: str) -> 'DBNNConfig':
        """Create configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

    def save(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items()}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

class DBNN(GPUDBNN):
    """Enhanced DBNN class that builds on GPUDBNN implementation"""

    def __init__(self, config: Optional[Union[DBNNConfig, dict]] = None,
                 dataset_name: Optional[str] = None):

        """
        Initialize DBNN with configuration

        Args:
            config: DBNNConfig object or dictionary of parameters
            dataset_name: Name of the dataset (optional)
        """
        # Initialize configuration
        if config is None:
            config = DBNNConfig()
        elif isinstance(config, dict):
            config = DBNNConfig(**config)

        # First load the dataset configuration
        self.data_config = DatasetConfig.load_config(dataset_name) if dataset_name else None

        # Map DBNNConfig to GPUDBNN parameters
        super().__init__(
            dataset_name=dataset_name,
            learning_rate=config.learning_rate,
            max_epochs=config.epochs,
            test_size=config.test_fraction,
            random_state=config.random_seed,
            fresh=config.fresh_start,
            use_previous_model=config.use_previous_model,
            model_type=config.model_type  # Pass model type from config
        )
        self.cardinality_threshold = self.config.get('training_params', {}).get('cardinality_threshold', 0.9)

        # Store model configuration
        self.model_config = config
        self.training_log = pd.DataFrame()
        self.save_plots = self.config.get('training_params', {}).get('save_plots', False)

        # Add new attributes to track the best round
        self.best_round = None  # Track the best round number
        self.best_round_initial_conditions = None  # Save initial conditions of the best round
        self.best_combined_accuracy = 0.00
        self.best_model_weights = None

        self.global_mean = None  # Store global mean
        self.global_std = None   # Store global standard deviation
        self.global_stats_computed = False  # Flag to track if stats are computed

        # Validate dataset_name
        if not dataset_name or not isinstance(dataset_name, str):
            raise ValueError("Invalid dataset_name provided. Must be a non-empty string.")

        # Load configuration
        self.config = DatasetConfig.load_config(dataset_name)
        if self.config is None:
            raise ValueError(f"Failed to load configuration for dataset: {dataset_name}")

        # Initialize other attributes
        self.target_column = self.config['target_column']
        self.batch_size = self.config.get('batch_size',128)
        self.invertible_model = None
        # Preprocess data once during initialization
        self._is_preprocessed = False  # Flag to track preprocessing
        self._preprocess_and_split_data()  # Call preprocessing only once

    def compute_global_statistics(self, X: pd.DataFrame):
        """Compute global statistics (e.g., mean, std) for normalization."""
        batch_size = self.batch_size  # Adjust based on available memory
        n_samples = len(X)
        n_features = X.shape[1]

        # Initialize accumulators
        sum_features = np.zeros(n_features)
        sum_squared_features = np.zeros(n_features)

        # Compute sum and sum of squares in batches
        for i in range(0, n_samples, batch_size):
            batch_X = X.iloc[i:i + batch_size]
            batch_X_numeric = batch_X.select_dtypes(include=[np.number])  # Only numeric features
            sum_features += batch_X_numeric.sum(axis=0)
            sum_squared_features += (batch_X_numeric ** 2).sum(axis=0)

        # Compute mean and standard deviation
        self.global_mean = sum_features / n_samples
        self.global_std = np.sqrt((sum_squared_features / n_samples) - (self.global_mean ** 2))

        # Handle zero standard deviation (replace with 1 to avoid division by zero)
        self.global_std[self.global_std == 0] = 1.0

    def _preprocess_and_split_data(self):
        """Preprocess data and split into training and testing sets."""
        # Load dataset
        self.data = self._load_dataset()

        # Preprocess features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Compute global statistics for normalization
        self.compute_global_statistics(X)

        # Encode labels if not already done
        if not hasattr(self.label_encoder, 'classes_'):
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)

        # Preprocess features
        X_processed = self._preprocess_data(X, is_training=True)

        # Convert to tensors on CPU first, then move to device
        self.X_tensor =  X_processed.clone().detach().to(self.device)
        self.y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(self.device)

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = self._get_train_test_split(
            self.X_tensor, self.y_tensor)

        self._is_preprocessed = True  # Mark preprocessing as complete

    def create_invertible_model(self, reconstruction_weight: float = 0.5, feedback_strength: float = 0.3):
        """Create an invertible DBNN model"""
        if self.invertible_model is None:
            self.invertible_model = InvertibleDBNN(
                forward_model=self,
                feature_dims=self.data.shape[1] - 1,  # Exclude target column
                reconstruction_weight=reconstruction_weight,
                feedback_strength=feedback_strength
            )
        return self.invertible_model

    def process_dataset(self, config_path: str) -> Dict:
        """
        Process dataset according to configuration file specifications

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Dictionary containing processing results
        """
        # Load and validate configuration
        try:
            with open(config_path, 'r') as f:
                config_text = f.read()
            # Remove comments starting with _comment
            config_lines = [line for line in config_text.split('\n') if not '"_comment"' in line]
            clean_config = '\n'.join(config_lines)

            self.data_config = json.loads(clean_config)
        except Exception as e:
            raise ValueError(f"Error reading configuration file: {str(e)}")
        dataset_name = os.path.splitext(os.path.basename(config_path))[0]
        # Ensure file_path is set
        if not self.data_config.get('file_path'):
            default_path = os.path.join('data', dataset_name, f"{dataset_name}.csv")
            if os.path.exists(default_path):
                self.data_config['file_path'] = default_path
                print("\033[K" +f"Using default data file: {default_path}")
            else:
                raise ValueError(f"No data file found for {dataset_name}")
        # Load or create the configuration file and update global variables
        config = load_or_create_config(config_path=f'data/{dataset_name}/{dataset_name}.conf')
        # Update global variables based on the configuration file
        global Train_device, Trials, cardinality_threshold, cardinality_tolerance, LearningRate, TrainingRandomSeed, Epochs, TestFraction, Train, Train_only, Predict, Gen_Samples, EnableAdaptive, nokbd, display
        Train_device = config.get("compute_device", Train_device)
        print(''f"The train device is set as {Train_device}")
        self.device=Train_device
        Trials = config.get("trials", Trials)
        cardinality_threshold = config.get("cardinality_threshold", cardinality_threshold)
        cardinality_tolerance = config.get("cardinality_tolerance", cardinality_tolerance)
        LearningRate = config.get("learning_rate", LearningRate)
        TrainingRandomSeed = config.get("random_seed", TrainingRandomSeed)
        Epochs = config.get("epochs", Epochs)
        TestFraction = config.get("test_fraction", TestFraction)
        Train = config.get("train", Train)
        Train_only = config.get("train_only", Train_only)
        Predict = config.get("predict", Predict)
        Gen_Samples = config.get("gen_samples", Gen_Samples)
        EnableAdaptive = config.get("enable_adaptive", EnableAdaptive)
        nokbd = config.get("nokbd", nokbd)
        display = config.get("display", display)

        # Convert dictionary config to DBNNConfig object
        config_params = {
            'epochs': self.data_config.get('training_params', {}).get('epochs', Epochs),
            'learning_rate': self.data_config.get('training_params', {}).get('learning_rate', LearningRate),
            'model_type': self.data_config.get('modelType', 'Histogram'),
            'enable_adaptive': self.data_config.get('training_params', {}).get('enable_adaptive', EnableAdaptive),
            'batch_size': self.data_config.get('training_params', {}).get('batch_size', 128),
            'training_data_dir': self.data_config.get('training_params', {}).get('training_save_path', 'training_data')
        }
        self.model_config = DBNNConfig(**config_params)

        # Create output directory structure
        dataset_name = os.path.splitext(os.path.basename(self.data_config['file_path']))[0]
        output_dir = os.path.join(self.model_config.training_data_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Update dataset name
        self.dataset_name = dataset_name

         # Load data using existing GPUDBNN method
        self.data =self._load_dataset()
        self.X_Orig =self.Original_data.drop(columns=[self.data_config['target_column']])
        self.batch_size=self._calculate_optimal_batch_size(len(self.data))
        # Add row tracking
        self.data['original_index'] = range(len(self.data))

        # Extract features and target

        if 'target_column' not in self.data_config:
            self.data_config['target_column'] = 'target'  # Set default target column
            print("\033[K" +f"Using default target column: 'target'")

        X = self.data.drop(columns=[self.data_config['target_column']])
        y = self.data[self.data_config['target_column']]

        # Initialize training log
        log_file = os.path.join(output_dir, f'{dataset_name}_log.csv')
        self.training_log = pd.DataFrame(columns=[
            'timestamp', 'round', 'train_size', 'test_size',
            'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy',
            'training_time'
        ])

        # Train model using existing GPUDBNN methods
        if self.model_config.enable_adaptive:
            results = self.adaptive_fit_predict(max_rounds=self.model_config.epochs)
        else:
            results = self.fit_predict()

        # Preprocess features and convert to tensors
        X_tensor = self._preprocess_data(X, is_training=True)  # Preprocess data
        y_tensor = torch.tensor(self.label_encoder.transform(y), dtype=torch.long).to(self.device)  # Encode labels

        # Generate predictions and true labels
        predictions = self.predict(X_tensor, batch_size=self.batch_size)
        true_labels = y_tensor.cpu().numpy()

        # Generate detailed predictions
        #predictions_df = self._generate_detailed_predictions(X_tensor, predictions, true_labels)
        predictions_df = self._generate_detailed_predictions(self.X_Orig, predictions, true_labels)

        # Save results
        results_path = os.path.join(output_dir, f'{dataset_name}_predictions.csv')
        predictions_df.to_csv(results_path, index=False)

        # Save training log
        self.training_log.to_csv(log_file, index=False)

        # Count number of features actually used (excluding high cardinality and excluded features)
        n_features = len(X.columns)
        n_excluded = len(getattr(self, 'high_cardinality_columns', []))

        return {
            'results_path': results_path,
            'log_path': log_file,
            'n_samples': len(self.data),
            'n_features': n_features,
            'n_excluded': n_excluded,
            'training_results': results
        }

    def _generate_detailed_predictions(self, X: Union[pd.DataFrame, torch.Tensor], predictions: torch.Tensor, true_labels: Union[torch.Tensor, np.ndarray] = None) -> pd.DataFrame:
        """
        Generate detailed predictions with confidence metrics and metadata.

        Args:
            X: Input data (can be a Pandas DataFrame or PyTorch tensor).
            predictions: Tensor containing the predicted class labels.
            true_labels: Tensor or NumPy array containing the true labels (optional).

        Returns:
            DataFrame containing the complete input data, predictions, and posterior probabilities.
        """
        # Convert predictions to original class labels
        pred_labels = self.label_encoder.inverse_transform(predictions)

        # Handle input data (X) based on its type
        if isinstance(X, torch.Tensor):
            # Convert tensor to NumPy array, copy it, and then back to a DataFrame
            X_np = X.cpu().numpy()  # Move tensor to CPU and convert to NumPy
            results_df = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
        elif isinstance(X, pd.DataFrame):
            # If X is already a DataFrame, just copy it
            results_df = X.copy()
        else:
            raise TypeError(f"Unsupported input type for X: {type(X)}. Expected Pandas DataFrame or PyTorch tensor.")

        # Add predictions
        results_df['predicted_class'] = pred_labels

        # Add true labels if provided
        if true_labels is not None:
            # Handle true_labels based on its type
            if isinstance(true_labels, torch.Tensor):
                true_labels_np = true_labels.cpu().numpy()  # Convert tensor to NumPy array
            elif isinstance(true_labels, np.ndarray):
                true_labels_np = true_labels  # Use as-is
            else:
                raise TypeError(f"Unsupported type for true_labels: {type(true_labels)}. Expected PyTorch tensor or NumPy array.")

            results_df['true_class'] = self.label_encoder.inverse_transform(true_labels_np)

        # Preprocess features for probability computation
        X_processed = self._preprocess_data(X, is_training=False)
        if isinstance(X_processed, torch.Tensor):
            X_tensor = X_processed.clone().detach().to(self.device)
        else:
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)

        # Compute probabilities in batches
        batch_size = 128
        all_probabilities = []

        for i in range(0, len(X_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_tensor))
            batch_X = X_tensor[i:batch_end]

            try:
                if self.model_type == "Histogram":
                    batch_probs, _ = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    batch_probs, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    raise ValueError(f"{self.model_type} is invalid")

                all_probabilities.append(batch_probs.cpu().numpy())

            except Exception as e:
                print("\033[K" +f"Error computing probabilities for batch {i}: {str(e)}")
                return None

        if all_probabilities:
            all_probabilities = np.vstack(all_probabilities)
        else:
            print("\033[K" +"No probabilities were computed successfully", end="\r", flush=True)
            return None

        # Ensure we're only using valid class indices
        valid_classes = self.label_encoder.classes_
        n_classes = len(valid_classes)

        # Verify probability array shape matches number of classes
        if all_probabilities.shape[1] != n_classes:
            print("\033[K" +f"Warning: Probability array shape ({all_probabilities.shape}) doesn't match number of classes ({n_classes})")
            # Adjust probabilities array if necessary
            if all_probabilities.shape[1] > n_classes:
                all_probabilities = all_probabilities[:, :n_classes]
            else:
                # Pad with zeros if needed
                pad_width = ((0, 0), (0, n_classes - all_probabilities.shape[1]))
                all_probabilities = np.pad(all_probabilities, pad_width, mode='constant')

        # Add probability columns for each valid class
        for i, class_name in enumerate(valid_classes):
            if i < all_probabilities.shape[1]:  # Safety check
                results_df[f'prob_{class_name}'] = all_probabilities[:, i]

        # Add maximum probability
        results_df['max_probability'] = all_probabilities.max(axis=1)

        return results_df



    def _update_training_log(self, round_num: int, metrics: Dict):
        """Update training log with current metrics"""
        self.training_log = self.training_log.append({
            'timestamp': pd.Timestamp.now(),
            'round': round_num,
            'train_size': metrics['train_size'],
            'test_size': metrics['test_size'],
            'train_loss': metrics['train_loss'],
            'test_loss': metrics['test_loss'],
            'train_accuracy': metrics['train_accuracy'],
            'test_accuracy': metrics['test_accuracy'],
            'training_time': metrics['training_time']
        }, ignore_index=True)
#--------------------------------------------------------------Class Ends ------------------------------------
    def prepare_batch(self, features):
        """Efficient batch preparation"""
        if not features.is_contiguous():
            features = features.contiguous()

        if self.device.startswith('cuda') and not features.is_cuda:
            features = features.cuda(non_blocking=True)

        return features

    def _load_dataset(self) -> pd.DataFrame:
        """Load and preprocess dataset with improved error handling"""
        DEBUG.log(f" Loading dataset from config: {self.config}")
        try:
            # Validate configuration
            if self.config is None:
                raise ValueError(f"No configuration found for dataset: {self.dataset_name}")
            file_path = self.config.get('file_path')
            if file_path is None:
                raise ValueError(f"No file path specified in configuration for dataset: {self.dataset_name}")
            # Handle URL or local file
            try:
                if file_path.startswith(('http://', 'https://')):
                    DEBUG.log(f" Loading from URL: {file_path}")
                    response = requests.get(file_path)
                    response.raise_for_status()
                    data = StringIO(response.text)
                else:
                    DEBUG.log(f" Loading from local file: {file_path}")
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Dataset file not found: {file_path}")
                    data = file_path

                # First, read the CSV to get the actual headers
                has_header = self.config.get('has_header', True)

                read_params = {
                    'sep': self.config.get('separator', ','),
                    'header': 0 if has_header else None,
                }

                # DO NOT include 'names' parameter for the initial read
                # This allows us to read the actual headers from the file
                DEBUG.log(f" Reading CSV with parameters: {read_params}")
                df = pd.read_csv(data, **read_params)
                self.Original_data=df.copy()

                if df is None or df.empty:
                    raise ValueError(f"Empty dataset loaded from {file_path}")

                DEBUG.log(f" Loaded DataFrame shape: {df.shape}")
                DEBUG.log(f" Original DataFrame columns: {df.columns.tolist()}")

                # Filter features based on config after reading the actual data
                if 'column_names' in self.config:
                    DEBUG.log(" Filtering features based on config")
                    df = _filter_features_from_config(df, self.config)
                    DEBUG.log(f" Shape after filtering: {df.shape}")

                # Handle target column
                target_column = self.config.get('target_column')

                if target_column is None:
                    raise ValueError(f"No target column specified for dataset: {self.dataset_name}")

                if isinstance(target_column, int):
                    cols = df.columns.tolist()
                    if target_column >= len(cols):
                        raise ValueError(f"Target column index {target_column} is out of range")
                    target_column = cols[target_column]
                    self.config['target_column'] = target_column
                    DEBUG.log(f" Using target column: {target_column}")

                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset")

                DEBUG.log(f" Dataset loaded successfully. Shape: {df.shape}")
                DEBUG.log(f" Columns: {df.columns.tolist()}")
                DEBUG.log(f" Data types:\n{df.dtypes}")

                # Create data directory path
                dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
                base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
                data_dir = os.path.join(base_path, dataset_folder, 'data')
                shuffled_file = os.path.join(data_dir, 'shuffled_data.csv')

                # Check if this is a fresh start with random shuffling
                if self.fresh_start and self.random_state == -1:
                    print("\033[K" +"Fresh start with random shuffling enabled")
                    # Perform 3 rounds of truly random shuffling
                    for _ in range(3):
                        df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
                    # Ensure directory exists before saving
                    os.makedirs(data_dir, exist_ok=True)
                    # Save shuffled data
                    df.to_csv(shuffled_file, index=False)
                    print("\033[K" +f"Saved shuffled data to {shuffled_file}")
                elif os.path.exists(shuffled_file):
                    print("\033[K" +f"Loading previously shuffled data from {shuffled_file}")
                    df = pd.read_csv(shuffled_file)
                else:
                    print("\033[K" +"Using original data order (no shuffling required)")

                return df

            except requests.exceptions.RequestException as e:
                DEBUG.log(f" Error downloading dataset from URL: {str(e)}")
                raise RuntimeError(f"Failed to download dataset from URL: {str(e)}")
            except pd.errors.EmptyDataError:
                DEBUG.log(f" Error: Dataset file is empty")
                raise ValueError(f"Dataset file is empty: {file_path}")
            except pd.errors.ParserError as e:
                DEBUG.log(f" Error parsing CSV file: {str(e)}")
                raise ValueError(f"Invalid CSV format: {str(e)}")
        except Exception as e:
            DEBUG.log(f" Error loading dataset: {str(e)}")
            DEBUG.log(" Stack trace:", traceback.format_exc())
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

    def _compute_batch_posterior(self, features: torch.Tensor, epsilon: float = 1e-10):
        """Optimized batch posterior with vectorized operations"""
        # Ensure input features are on the correct device
        features = features.to(self.device)

        # Safety checks
        if self.weight_updater is None:
            DEBUG.log(" Weight updater not initialized, initializing now...")
            self._initialize_bin_weights()
            if self.weight_updater is None:
                raise RuntimeError("Failed to initialize weight updater")

        if self.likelihood_params is None:
            raise RuntimeError("Likelihood parameters not initialized")

        batch_size = features.shape[0]
        n_classes = len(self.likelihood_params['classes'])

        # Pre-allocate tensors on the correct device
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Process all feature pairs at once
        feature_groups = torch.stack([
            features[:, pair].contiguous().to(self.device)  # Ensure on correct device
            for pair in self.likelihood_params['feature_pairs']
        ]).transpose(0, 1)  # [batch_size, n_pairs, 2]

        # Compute all bin indices at once
        bin_indices_dict = {}
        for group_idx in range(len(self.likelihood_params['feature_pairs'])):
            bin_edges = self.likelihood_params['bin_edges'][group_idx]
            edges = torch.stack([edge.contiguous().to(self.device) for edge in bin_edges])  # Ensure on correct device

            # Vectorized binning with contiguous tensors
            indices = torch.stack([
                torch.bucketize(
                    feature_groups[:, group_idx, dim].contiguous(),
                    edges[dim].contiguous()
                ).sub_(1).clamp_(0, self.n_bins_per_dim - 1)
                for dim in range(2)
            ])  # Shape: (2, batch_size)
            bin_indices_dict[group_idx] = indices


        # Process all classes simultaneously
        for group_idx in range(len(self.likelihood_params['feature_pairs'])):
            bin_probs = self.likelihood_params['bin_probs'][group_idx]  # [n_classes, n_bins, n_bins]
            indices = bin_indices_dict[group_idx]  # [2, batch_size]

            # Get all weights at once
            weights = torch.stack([
                self.weight_updater.get_histogram_weights(c, group_idx)
                for c in range(n_classes)
            ])  # [n_classes, n_bins, n_bins]

            # Ensure weights are contiguous
            if not weights.is_contiguous():
                weights = weights.contiguous()

            # Apply weights to probabilities
            weighted_probs = bin_probs * weights  # [n_classes, n_bins, n_bins]

            # Gather probabilities for all samples and classes at once
            probs = weighted_probs[:, indices[0], indices[1]]  # [n_classes, batch_size]
            log_likelihoods += torch.log(probs.t() + epsilon)

        # Compute posteriors efficiently
        max_log_likelihood = log_likelihoods.max(dim=1, keepdim=True)[0]
        posteriors = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors /= posteriors.sum(dim=1, keepdim=True) + epsilon

        return posteriors, bin_indices_dict if self.model_type == "Histogram" else None


#----------------------

    def set_feature_bounds(self, dataset):
        """Initialize global feature bounds from complete dataset"""
        if self.feature_bounds is None:
            self.feature_bounds = {}
            numerical_columns = dataset.select_dtypes(include=['int64', 'float64']).columns

            for feat_name in numerical_columns:
                feat_data = dataset[feat_name]
                min_val = feat_data.min()
                max_val = feat_data.max()
                padding = (max_val - min_val) * 0.01
                self.feature_bounds[feat_name] = {
                    'min': min_val - padding,
                    'max': max_val + padding
                }

    def _clean_existing_model(self):
        """Remove existing model files for a fresh start"""
        try:
            files_to_remove = [
                self._get_weights_filename(),
                self._get_encoders_filename(),
                self._get_model_components_filename()
            ]
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
                    print("\033[K" +f"Removed existing model file: {file}")
        except Exception as e:
            print("\033[K" +f"Warning: Error cleaning model files: {str(e)}")


    #------------------------------------------Adaptive Learning--------------------------------------
    def save_epoch_data(self, epoch: int, train_indices: list, test_indices: list):
        """
        Save training and testing indices for each epoch if enabled in config
        """
        # Check if epoch saving is enabled
        save_epochs = self.config.get('training_params', {}).get('Save_training_epochs', False)
        if not save_epochs:
            return

        # Use dataset name as subfolder
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        save_path = os.path.join(base_path, dataset_folder)

        # Create epoch directory
        epoch_dir = os.path.join(save_path, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        # Save indices
        try:
            with open(os.path.join(epoch_dir, f'{self.model_type}_train_indices.pkl'), 'wb') as f:
                pickle.dump(train_indices, f)
            with open(os.path.join(epoch_dir, f'{self.model_type}_test_indices.pkl'), 'wb') as f:
                pickle.dump(test_indices, f)
            print("\033[K" +f"Saved epoch {epoch} data to {epoch_dir}")
        except Exception as e:
            print("\033[K" +f"Error saving epoch data: {str(e)}")

    def load_epoch_data(self, epoch: int):
        """
        Load training and testing indices for a specific epoch
        """
        epoch_dir = os.path.join(self.base_save_path, f'epoch_{epoch}')

        with open(os.path.join(epoch_dir, f'{self.model_type}_train_indices.pkl'), 'rb') as f:
            train_indices = pickle.load(f)
        with open(os.path.join(epoch_dir, f'{self.model_type}_test_indices.pkl'), 'rb') as f:
            test_indices = pickle.load(f)

        return train_indices, test_indices

    def _compute_cardinality_threshold(self, cardinalities, config=None):
        """
        Compute the cardinality threshold using percentile-based analysis.

        Args:
            cardinalities: List of cardinality values
            config: Configuration dictionary with threshold parameters

        Returns:
            float: Computed cardinality threshold
        """
        # Get active learning parameters from config
        active_learning_config = self.config.get('active_learning', {})
        percentile = active_learning_config.get('cardinality_threshold_percentile', 95)

        # If no cardinalities provided, create a synthetic distribution
        if not cardinalities:
            print("\033[K" +"Warning: No cardinality data available. Using synthetic distribution based on percentile.")
            # Create a synthetic distribution around the percentile threshold
            cardinalities = np.array([1.0, 2.0, 5.0, 10.0, 20.0])  # Synthetic values
        else:
            cardinalities = np.array(cardinalities)

        # Compute basic statistics
        min_card = np.min(cardinalities)
        max_card = np.max(cardinalities)
        mean_card = np.mean(cardinalities)
        median_card = np.median(cardinalities)

        # Compute threshold using percentile
        threshold = np.percentile(cardinalities, percentile)

        # Print detailed analysis
        print("\033[K" +f"Cardinality Analysis:")
        print("\033[K" +f"- Using {percentile}th percentile threshold")
        print("\033[K" +f"- Distribution statistics:")
        print("\033[K" +f"  - Min: {min_card:.2f}")
        print("\033[K" +f"  - Max: {max_card:.2f}")
        print("\033[K" +f"  - Mean: {mean_card:.2f}")
        print("\033[K" +f"  - Median: {median_card:.2f}")
        print("\033[K" +f"  - Threshold: {threshold:.2f}")

        # Print number of samples that would be included
        n_included = sum(c <= threshold for c in cardinalities)
        print("\033[K" +f"- {n_included} out of {len(cardinalities)} samples below threshold "
              f"({(n_included/len(cardinalities))*100:.1f}%)")

        return threshold

    def _compute_sample_divergence(self, sample_data: torch.Tensor, feature_pairs: List[Tuple]) -> torch.Tensor:
        """
        Vectorized computation of pairwise feature divergence.
        """
        n_samples = sample_data.shape[0]
        if n_samples <= 1:
            return torch.zeros((1, 1), device=self.device)

        # Pre-allocate tensor for pair distances
        pair_distances = torch.zeros((len(feature_pairs), n_samples, n_samples),
                                   device=self.device)

        # Compute distances for all pairs in one batch
        for i, pair in enumerate(feature_pairs):
            pair_data = sample_data[:, pair]
            # Vectorized pairwise difference computation
            diff = pair_data.unsqueeze(1) - pair_data.unsqueeze(0)
            pair_distances[i] = torch.norm(diff, dim=2)

        # Average across feature pairs
        distances = torch.mean(pair_distances, dim=0)

        # Normalize
        if distances.max() > 0:
            distances /= distances.max()

        return distances

    def _compute_feature_cardinalities(self, samples_data: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of feature cardinalities.
        """
        cardinalities = torch.zeros(len(samples_data), device=self.device)

        # Process feature pairs in batches
        batch_size = self.batch_size  # Adjust based on memory constraints
        for i in range(0, len(samples_data), batch_size):
            batch_end = min(i + batch_size, len(samples_data))
            batch_data = samples_data[i:batch_end]

            # Compute cardinalities for each feature pair
            batch_cardinalities = torch.zeros(batch_end - i, device=self.device)
            for feat_pair in self.feature_pairs:
                pair_data = batch_data[:, feat_pair]
                # Compute unique values efficiently
                _, counts = torch.unique(pair_data, dim=0, return_counts=True)
                batch_cardinalities += len(counts)

            cardinalities[i:batch_end] = batch_cardinalities

        return cardinalities

    def _calculate_optimal_batch_size(self, sample_tensor_size):
        """
        Calculate optimal batch size based on available GPU memory and sample size.

        Args:
            sample_tensor_size: Size of one sample tensor in bytes

        Returns:
            optimal_batch_size: int
        """
        if not torch.cuda.is_available():
            print(f"{Colors.RED} The GPU cuda library is unavailable. setting batch size to 128{Colors.ENDC}")
            return 128  # Default for CPU

        # Get total and reserved GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)

        # Calculate available memory (leaving 20% as buffer)
        available_memory = (total_memory - reserved_memory - allocated_memory) * 0.8

        # Calculate memory needed per sample (with buffer for intermediate computations)
        memory_per_sample = sample_tensor_size * 4  # Factor of 4 for intermediate computations

        # Calculate optimal batch size
        optimal_batch_size = int(available_memory / memory_per_sample)

        # Enforce minimum and maximum bounds
        optimal_batch_size = max(32, min(optimal_batch_size, 4096))

        DEBUG.log(f" Memory Analysis:")
        DEBUG.log(f" - Total GPU Memory: {total_memory / 1e9:.2f} GB")
        DEBUG.log(f" - Reserved Memory: {reserved_memory / 1e9:.2f} GB")
        DEBUG.log(f" - Allocated Memory: {allocated_memory / 1e9:.2f} GB")
        DEBUG.log(f" - Available Memory: {available_memory / 1e9:.2f} GB")
        DEBUG.log(f" - Memory per sample: {memory_per_sample / 1e6:.2f} MB")
        print(f" {Colors.GREEN}Batch size updated to- Optimal batch size: {optimal_batch_size}{Colors.ENDC}")

        return optimal_batch_size


    def _select_samples_from_failed_classes(self, test_predictions, y_test, test_indices):
        """
        Memory-efficient implementation of sample selection using batched processing
        """
        # Configuration parameters
        active_learning_config = self.config.get('active_learning', {})
        tolerance = active_learning_config.get('tolerance', 1.0) / 100.0
        min_divergence = active_learning_config.get('min_divergence', 0.1)
        strong_margin_threshold = active_learning_config.get('strong_margin_threshold', 0.3)
        marginal_margin_threshold = active_learning_config.get('marginal_margin_threshold', 0.1)
        max_class_addition_percent = active_learning_config.get('max_class_addition_percent', 5)  # Default to 5%

        # Calculate optimal batch size based on sample size
        sample_size = self.X_tensor[0].element_size() * self.X_tensor[0].nelement()
        self.batch_size = self._calculate_optimal_batch_size(sample_size)
        batch_size=self.batch_size
        print("\033[K" +f"{Colors.GREEN}Upadated batch size to  {batch_size} using dynamic batchsize updater{Colors.ENDC}", end="\r", flush=True)

        test_predictions = torch.as_tensor(test_predictions, device=self.device)
        y_test = torch.as_tensor(y_test, device=self.device)
        test_indices = torch.as_tensor(test_indices, device=self.device)

        misclassified_mask = (test_predictions != y_test)
        misclassified_indices = torch.nonzero(misclassified_mask).squeeze()

        if misclassified_indices.dim() == 0:
            return []

        final_selected_indices = []
        unique_classes = torch.unique(y_test[misclassified_indices])

        for class_id in unique_classes:
            class_mask = y_test[misclassified_indices] == class_id
            class_indices = misclassified_indices[class_mask]

            if len(class_indices) == 0:
                continue

            # Calculate the maximum number of samples to add from this class
            total_class_samples = (y_test == class_id).sum().item()
            max_samples_to_add = int(total_class_samples * (max_class_addition_percent / 100.0))

            # Process class samples in batches
            for batch_start in range(0, len(class_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(class_indices))
                batch_indices = class_indices[batch_start:batch_end]

                # Get batch data
                batch_samples = self.X_tensor[test_indices[batch_indices]]

                # Compute probabilities for batch
                if self.model_type == "Histogram":
                    probs, _ = self._compute_batch_posterior(batch_samples)
                else:
                    probs, _ = self._compute_batch_posterior_std(batch_samples)

                # Compute error margins for batch
                true_probs = probs[:, class_id]
                pred_classes = torch.argmax(probs, dim=1)
                pred_probs = probs[torch.arange(len(pred_classes)), pred_classes]
                error_margins = pred_probs - true_probs

                # Split into strong and marginal failures
                strong_failures = error_margins >= strong_margin_threshold
                marginal_failures = (error_margins > 0) & (error_margins < marginal_margin_threshold)

                # Process each failure type
                for failure_type, failure_mask in [("strong", strong_failures), ("marginal", marginal_failures)]:
                    if not failure_mask.any():
                        continue

                    # Get failure samples for this batch
                    failure_samples = batch_samples[failure_mask]
                    failure_margins = error_margins[failure_mask]
                    failure_indices = test_indices[batch_indices[failure_mask]]

                    # Compute cardinalities for these samples
                    cardinalities = self._compute_feature_cardinalities(failure_samples)

                    # Use dynamic threshold based on distribution
                    cardinality_threshold = torch.median(cardinalities)
                    low_card_mask = cardinalities <= cardinality_threshold

                    if not low_card_mask.any():
                        continue

                    # Process samples meeting cardinality criteria
                    low_card_samples = failure_samples[low_card_mask]
                    low_card_margins = failure_margins[low_card_mask]
                    low_card_indices = failure_indices[low_card_mask]

                    # Compute divergences only for low cardinality samples
                    divergences = self._compute_sample_divergence(low_card_samples, self.feature_pairs)

                    # Select diverse samples efficiently
                    selected_mask = torch.zeros(len(low_card_samples), dtype=torch.bool, device=self.device)

                    # Initialize with best margin sample
                    if failure_type == "strong":
                        best_idx = torch.argmax(low_card_margins)
                    else:
                        best_idx = torch.argmin(low_card_margins)
                    selected_mask[best_idx] = True

                    # Add diverse samples meeting divergence criterion
                    while True:
                        # Compute minimum divergence to selected samples
                        min_divs = divergences[:, selected_mask].min(dim=1)[0]
                        candidate_mask = (~selected_mask) & (min_divs >= min_divergence)

                        if not candidate_mask.any():
                            break

                        # Select next sample based on margin type
                        candidate_margins = low_card_margins.clone()
                        candidate_margins[~candidate_mask] = float('inf') if failure_type == "marginal" else float('-inf')

                        best_idx = torch.argmin(candidate_margins) if failure_type == "marginal" else torch.argmax(candidate_margins)
                        selected_mask[best_idx] = True

                    # Add selected indices, but ensure we don't exceed the maximum allowed for this class
                    selected_indices = low_card_indices[selected_mask]
                    if len(final_selected_indices) + len(selected_indices) > max_samples_to_add:
                        # If adding these samples would exceed the limit, only add enough to reach the limit
                        remaining_samples = max_samples_to_add - len(final_selected_indices)
                        selected_indices = selected_indices[:remaining_samples]

                    final_selected_indices.extend(selected_indices.cpu().tolist())

                    # Print selection info
                    true_class_name = self.label_encoder.inverse_transform([class_id.item()])[0]
                    n_selected = selected_mask.sum().item()
                    DEBUG.log(f" Selected {n_selected} {failure_type} failure samples from class {true_class_name}")
                    DEBUG.log(f" - Cardinality threshold: {cardinality_threshold:.3f}")
                    DEBUG.log(f" - Average margin: {low_card_margins[selected_mask].mean().item():.3f}")

                # Clear cache after processing each batch
                torch.cuda.empty_cache()

        print("\033[K" +f"Total samples selected: {len(final_selected_indices)}                                                           ")
        return final_selected_indices

    def _save_reconstruction_plots(self, original_features: np.ndarray,
                                reconstructed_features: np.ndarray,
                                true_labels: np.ndarray,
                                save_path: str):
        """Generate visualization plots for reconstruction analysis"""
        plt.figure(figsize=(15, 5))

        # Feature-wise reconstruction error
        plt.subplot(131)
        errors = np.mean((original_features - reconstructed_features) ** 2, axis=0)
        plt.bar(range(len(errors)), errors)
        plt.title('Feature-wise Reconstruction Error')
        plt.xlabel('Feature Index')
        plt.ylabel('MSE')

        # Class-wise reconstruction quality
        plt.subplot(132)
        unique_classes = np.unique(true_labels)
        class_errors = []
        for class_label in unique_classes:
            mask = (true_labels == class_label)
            error = np.mean((original_features[mask] - reconstructed_features[mask]) ** 2)
            class_errors.append(error)

        plt.bar(unique_classes, class_errors)
        plt.title('Class-wise Reconstruction Error')
        plt.xlabel('Class')
        plt.ylabel('MSE')

        # Error distribution
        plt.subplot(133)
        all_errors = np.mean((original_features - reconstructed_features) ** 2, axis=1)
        plt.hist(all_errors, bins=30)
        plt.title('Error Distribution')
        plt.xlabel('MSE')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(f"{save_path}_reconstruction_plots.png")
        plt.close()

    def reset_to_initial_state(self):
        """Reset the model's weights and parameters to their initial state."""
        DEBUG.log("Resetting model to initial state for fresh training...")

        # Reset weights to uniform priors
        n_classes = len(self.label_encoder.classes_)
        n_pairs = len(self.feature_pairs) if self.feature_pairs is not None else 0

        if n_pairs > 0:
            self.current_W = torch.full(
                (n_classes, n_pairs),
                0.1,  # Default uniform prior
                device=self.device,
                dtype=torch.float32
            )
            self.best_W = self.current_W.clone()
            self.best_error = float('inf')

        # Reset likelihood parameters (if applicable)
        if self.model_type == "Histogram":
            self.likelihood_params = self._compute_pairwise_likelihood_parallel(
                self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
            )
        elif self.model_type == "Gaussian":
            self.likelihood_params = self._compute_pairwise_likelihood_parallel_std(
                self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
            )

        # Reset weight updater
        if self.weight_updater is not None:
            self.weight_updater = BinWeightUpdater(
                n_classes=n_classes,
                feature_pairs=self.feature_pairs,
                n_bins_per_dim=self.n_bins_per_dim,
                batch_size=self.batch_size
            )


        DEBUG.log("Model reset to initial state.")

    def adaptive_fit_predict(self, max_rounds: int = 10,
                            improvement_threshold: float = 0.001,
                            load_epoch: int = None,
                            batch_size: int = 128):
        """Modified adaptive training strategy with proper fresh start handling"""
        DEBUG.log(" Starting adaptive_fit_predict")
        if not EnableAdaptive:
            print("\033[K" +"Adaptive learning is disabled. Using standard training.")
            return self.fit_predict(batch_size=batch_size)

        # Record the start time
        start_time = time.time()
        start_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Adaptive training started at: {start_clock}{Colors.ENDC}")

        self.in_adaptive_fit = True
        train_indices = []
        test_indices = None

        try:
            # Get initial data
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            print(self.target_column)
            print("\033[K" +f" Initial data shape: X={X.shape}, y={len(y)}")
            print("\033[K" +f"Number of classes in data = {np.unique(y)}")
            print(self.data.head)
            # Initialize label encoder if not already done
            if not hasattr(self.label_encoder, 'classes_'):
                self.label_encoder.fit(y)

            # Use existing label encoder
            y_encoded = self.label_encoder.transform(y)

            # Process features and initialize model components if needed
            X_processed = self._preprocess_data(X, is_training=True)
            self.X_tensor =  X_processed.clone().detach().to(self.device)
            self.y_tensor = torch.LongTensor(y_encoded).to(self.device)

            # Handle model state based on flags
            model_loaded = False
            if self.use_previous_model:
                print("\033[K" +"Loading previous model state")
                if self._load_model_components():
                    self._load_best_weights()
                    self._load_categorical_encoders()
                    model_loaded = True

                    if not self.fresh_start:
                        # Load previous training data
                        print("\033[K" +"Loading previous training data...")
                        prev_train_file = f'{self.dataset_name}_Last_training.csv'
                        if os.path.exists(prev_train_file):
                            prev_train_data = pd.read_csv(prev_train_file)

                            # Match rows between previous training data and current data
                            train_indices = []
                            prev_features = prev_train_data.drop(columns=[self.target_column])
                            current_features = X

                            # Ensure columns match
                            common_cols = list(set(prev_features.columns) & set(current_features.columns))

                            # Find matching rows
                            for idx, row in current_features[common_cols].iterrows():
                                # Check if this row exists in previous training data
                                matches = (prev_features[common_cols] == row).all(axis=1)
                                if matches.any():
                                    train_indices.append(idx)

                            print("\033[K" +f"Loaded {len(train_indices)} previous training samples")

                            # Initialize test indices as all indices not in training
                            test_indices = list(set(range(len(X))) - set(train_indices))
                        else:
                            print("\033[K" +"No previous training data found - starting fresh")
                            train_indices = []
                            test_indices = list(range(len(X)))
                else:
                    print("\033[K" +"No previous model found - starting fresh")

            if not model_loaded:
                print("\033[K" +"Initializing fresh model")
                self._clean_existing_model()
                train_indices = []
                test_indices = list(range(len(X)))

                # Initialize feature pairs for fresh start
                self.feature_pairs = self._generate_feature_combinations(
                    self.X_tensor.shape[1],
                    self.config.get('likelihood_config', {}).get('feature_group_size', 2),
                    self.config.get('likelihood_config', {}).get('max_combinations', None)
                )

            # Initialize test indices if still None
            if test_indices is None:
                test_indices = list(range(len(X)))

            # Initialize likelihood parameters if needed
            if self.likelihood_params is None:
                DEBUG.log(" Initializing likelihood parameters")
                if self.model_type == "Histogram":
                    self.likelihood_params = self._compute_pairwise_likelihood_parallel(
                        self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
                    )
                elif self.model_type == "Gaussian":
                    self.likelihood_params = self._compute_pairwise_likelihood_parallel_std(
                        self.X_tensor, self.y_tensor, self.X_tensor.shape[1]
                    )
                DEBUG.log(" Likelihood parameters computed")

            # Initialize weights if needed
            if self.weight_updater is None:
                DEBUG.log(" Initializing weight updater")
                self._initialize_bin_weights()
                DEBUG.log(" Weight updater initialized")

            # Initialize model weights if needed
            if self.current_W is None:
                DEBUG.log(" Initializing model weights")
                n_classes = len(self.label_encoder.classes_)
                n_pairs = len(self.feature_pairs) if self.feature_pairs is not None else 0
                if n_pairs == 0:
                    raise ValueError("Feature pairs not initialized")
                self.current_W = torch.full(
                    (n_classes, n_pairs),
                    0.1,
                    device=self.device,
                    dtype=torch.float32
                )
                if self.best_W is None:
                    self.best_W = self.current_W.clone()

            # Initialize training set if empty
            if len(train_indices) == 0:
                print("\033[K" +"Initializing new training set with minimum samples")
                # Select minimum samples from each class for initial training
                unique_classes = self.label_encoder.classes_
                for class_label in unique_classes:
                    class_indices = np.where(y_encoded == self.label_encoder.transform([class_label])[0])[0]
                    if len(class_indices) < 2:
                        selected_indices = class_indices
                    else:
                        selected_indices = class_indices[:2]
                    train_indices.extend(selected_indices)

                # Update test indices
                test_indices = list(set(range(len(X))) - set(train_indices))

            DEBUG.log(f" Initial training set size: {len(train_indices)}")
            DEBUG.log(f" Initial test set size: {len(test_indices)}")

            # Continue with training loop...
            for round_num in range(max_rounds):
                print("\033[K" +f"Round {round_num + 1}/{max_rounds}")
                print("\033[K" +f"Training set size: {len(train_indices)}")
                print("\033[K" +f"Test set size: {len(test_indices)}")

                # Reset model to initial state for fresh training
                self.reset_to_initial_state()

                # Save indices for this epoch
                self.save_epoch_data(round_num, train_indices, test_indices)

                # Create feature tensors for training
                X_train = self.X_tensor[train_indices]
                y_train = self.y_tensor[train_indices]

                # Train the model
                save_path = f"data/{self.dataset_name}/Predictions/"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.train_indices = train_indices
                self.test_indices = test_indices
                results = self.fit_predict(batch_size=batch_size, save_path=save_path)

                # Check training accuracy
                print("\033[K" +f"{Colors.GREEN}Predctions on Training data{Colors.ENDC}", end="\r", flush=True)
                train_accuracy=results['train_accuracy']
                print("\033[K" +f"Training accuracy: {train_accuracy:.4f}         ")

                # Get test accuracy from results
                test_accuracy = results['test_accuracy']

                # Check if we're improving overall
                improved = False
                if 'best_train_accuracy' not in locals():
                    best_train_accuracy = self.best_combined_accuracy
                    improved = True
                elif self.best_combined_accuracy > best_train_accuracy + improvement_threshold:
                    best_train_accuracy = self.best_combined_accuracy
                    improved = True
                    print("\033[K" +f"Improved training accuracy to {train_accuracy:.4f}")

                # Reset adaptive patience if improved
                if improved:
                    adaptive_patience_counter = 0
                    # Save the last training and test data
                    self.save_last_split(self.train_indices, self.test_indices)
                    print("\033[K" + "Saved model and data due to improved training accuracy")
                else:
                    adaptive_patience_counter += 1
                    print("\033[K" +f"No significant overall improvement. Adaptive patience: {adaptive_patience_counter}/5")
                    if adaptive_patience_counter >= 5:  # Using fixed value of 5 for adaptive patience
                        print("\033[K" +f"No improvement in accuracy after 5 rounds of adding samples.")
                        print("\033[K" +f"Best training accuracy achieved: {best_train_accuracy:.4f}")
                        print("\033[K" +"Stopping adaptive training.")
                        break

                # Evaluate test data using combined predictions from fit_predict
                test_predictions = results['test_predictions']['predicted_class']
                y_test = self.y_tensor[test_indices].cpu().numpy()

                # Convert test_predictions to a NumPy array if it's a Pandas Series
                if isinstance(test_predictions, pd.Series):
                    test_predictions = test_predictions.to_numpy()

                # Ensure test_predictions is numeric
                if test_predictions.dtype == np.object_:
                    # If predictions are class labels, convert them to numeric indices
                    test_predictions = self.label_encoder.transform(test_predictions)
                else:
                    # If predictions are numeric but stored as object, cast to int64
                    test_predictions = test_predictions.astype(np.int64)

                # Check if we've achieved perfect accuracy
                if train_accuracy == 1.0:
                    if len(test_indices) == 0:
                        print("\033[K" +"No more test samples available. Training complete.")
                        break

                    # Get new training samples from misclassified examples
                    new_train_indices = self._select_samples_from_failed_classes(
                        test_predictions, y_test, test_indices
                    )

                    if not new_train_indices:
                        print("\033[K" +"Achieved 100% accuracy on all data. Training complete.                                           ")
                        self.in_adaptive_fit = False
                        return {'train_indices': [], 'test_indices': []}

                else:
                    # Training did not achieve 100% accuracy, select new samples
                    new_train_indices = self._select_samples_from_failed_classes(
                        test_predictions, y_test, test_indices
                    )

                    if not new_train_indices:
                        print("\033[K" +"No suitable new samples found. Training complete.")
                        break


                if new_train_indices:
                    # Reset to the best round's initial conditions
                    if self.best_round_initial_conditions is not None:
                        print("\033[K" +f"Resetting to initial conditions of best round {self.best_round}")
                        self.current_W = self.best_round_initial_conditions['weights'].clone()
                        self.likelihood_params = self.best_round_initial_conditions['likelihood_params']
                        self.feature_pairs = self.best_round_initial_conditions['feature_pairs']
                        self.bin_edges = self.best_round_initial_conditions['bin_edges']
                        self.gaussian_params = self.best_round_initial_conditions['gaussian_params']


                # Update training and test sets with new samples
                train_indices.extend(new_train_indices)
                test_indices = list(set(test_indices) - set(new_train_indices))
                print("\033[K" +f"Added {len(new_train_indices)} new samples to training set")


            # Record the end time
            end_time = time.time()
            end_clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            elapsed_time = end_time - start_time

            # Print the timing information
            print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Adaptive training started at: {start_clock}{Colors.ENDC}")
            print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Adaptive training ended at: {end_clock}{Colors.ENDC}")
            print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Total adaptive training time: {elapsed_time:.2f} seconds{Colors.ENDC}")

            self.in_adaptive_fit = False
            return {'train_indices': train_indices, 'test_indices': test_indices}

        except Exception as e:
            DEBUG.log(f" Error in adaptive_fit_predict: {str(e)}")
            DEBUG.log(" Traceback:", traceback.format_exc())
            self.in_adaptive_fit = False
            raise

    #------------------------------------------Adaptive Learning--------------------------------------


    def _calculate_cardinality_threshold(self):
        """Calculate appropriate cardinality threshold based on dataset characteristics"""
        n_samples = len(self.data)
        n_classes = len(self.data[self.target_column].unique())

        # Base threshold from config
        base_threshold = cardinality_threshold

        # Adjust threshold based on dataset size and number of classes
        adjusted_threshold = min(
            base_threshold,
            max(0.1, 1.0 / np.sqrt(n_classes))  # Lower bound of 0.1
        )

        DEBUG.log(f"\nCardinality Threshold Calculation:")
        DEBUG.log(f"- Base threshold: {base_threshold}")
        DEBUG.log(f"- Number of samples: {n_samples}")
        DEBUG.log(f"- Number of classes: {n_classes}")
        DEBUG.log(f"- Adjusted threshold: {adjusted_threshold}")

        return adjusted_threshold


    def _round_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round features based on cardinality_tolerance"""
        if cardinality_tolerance == -1:
            return df
        return df.round(cardinality_tolerance)

    def _remove_high_cardinality_columns(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Remove high cardinality columns with more conservative approach"""
        DEBUG.log(f"Starting cardinality analysis with threshold {threshold}")

        df_filtered = df.copy()
        columns_to_drop = []
        cardinality_info = {}

        # First pass: calculate cardinality for all columns
        for column in df.columns:
            if column == self.target_column:
                continue

            unique_count = len(df[column].unique())
            unique_ratio = unique_count / len(df)
            cardinality_info[column] = {
                'unique_count': unique_count,
                'ratio': unique_ratio
            }

            DEBUG.log(f"Column {column}: {unique_count} unique values, ratio {unique_ratio:.4f}")

        # Determine adaptive threshold
        ratios = [info['ratio'] for info in cardinality_info.values()]
        if ratios:
            median_ratio = np.median(ratios)
            adaptive_threshold = min(threshold, max(median_ratio * 2, 0.1))
            DEBUG.log(f"Adaptive threshold: {adaptive_threshold} (original: {threshold})")
        else:
            adaptive_threshold = threshold

        # Second pass: mark columns for dropping
        for column, info in cardinality_info.items():
            if info['ratio'] > adaptive_threshold:
                columns_to_drop.append(column)
                DEBUG.log(f"Marking {column} for removal (ratio: {info['ratio']:.4f})")

        # Ensure we keep at least some features
        if len(columns_to_drop) == len(cardinality_info):
            DEBUG.log("Would remove all features - keeping lowest cardinality ones")
            sorted_columns = sorted(cardinality_info.items(), key=lambda x: x[1]['ratio'])
            keep_count = max(2, len(cardinality_info) // 5)  # Keep at least 2 or 20%
            columns_to_drop = [col for col, _ in sorted_columns[keep_count:]]

        # Drop columns
        if columns_to_drop:
            df_filtered = df_filtered.drop(columns=columns_to_drop)
            DEBUG.log(f"Dropped columns: {columns_to_drop}")

        DEBUG.log(f"Features after cardinality filtering: {df_filtered.columns.tolist()}")
        return df_filtered

    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns with improved debugging"""
        DEBUG.log(" Starting categorical column detection")
        categorical_columns = []

        for column in df.columns:
            if column != self.target_column:
                dtype = df[column].dtype
                unique_count = len(df[column].unique())
                DEBUG.log(f" Column {column}: dtype={dtype}, unique values={unique_count}")

                if dtype == 'object' or dtype.name == 'category':
                    DEBUG.log(f" Adding {column} as categorical (object/category type)")
                    categorical_columns.append(column)
                elif dtype in ['int64', 'float64']:
                    threshold = min(50, len(df) * 0.05)
                    if unique_count < threshold:
                        DEBUG.log(f" Adding {column} as categorical (numeric with few unique values: {unique_count} < {threshold})")
                        categorical_columns.append(column)
                    else:
                        DEBUG.log(f" Keeping {column} as numeric (unique values: {unique_count} >= {threshold})")

        DEBUG.log(f" Detected categorical columns: {categorical_columns}")
        return categorical_columns

    def _preprocess_data(self, X: Union[pd.DataFrame, torch.Tensor], is_training: bool = True) -> torch.Tensor:
        """Preprocess data with improved error handling and column consistency."""
        DEBUG.log(f"Starting preprocessing (is_training={is_training})")

        # Check if X is a DataFrame or a tensor
        if isinstance(X, pd.DataFrame):
            DEBUG.log(f"Input shape: {X.shape}")
            DEBUG.log(f"Input columns: {X.columns.tolist()}")
            DEBUG.log(f"Input dtypes:\n{X.dtypes}")
        else:
            DEBUG.log("Input is a tensor, skipping column-specific operations")

        # Make a copy to avoid modifying original data
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            X = X.clone().detach()

        # Step 1: Compute global statistics only once during training
        if is_training and not self.global_stats_computed:
            DEBUG.log("Training mode preprocessing - computing global statistics")
            self.compute_global_statistics(X)  # Compute global stats only once
            self.global_stats_computed = True  # Mark stats as computed
        elif is_training:
            DEBUG.log("Training mode preprocessing - using precomputed global statistics")

        # Step 2: Handle high cardinality columns (only during training)
        if is_training:
            self.original_columns = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
            cardinality_threshold = self._calculate_cardinality_threshold()
            DEBUG.log(f"Cardinality threshold: {cardinality_threshold}")
            if isinstance(X, pd.DataFrame):
                X = self._remove_high_cardinality_columns(X, cardinality_threshold)
            DEBUG.log(f"Shape after cardinality filtering: {X.shape}")

            # Store the features we'll actually use
            self.feature_columns = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
            DEBUG.log(f"Selected feature columns: {self.feature_columns}")

            # Store high cardinality columns for future reference
            if isinstance(X, pd.DataFrame):
                self.high_cardinality_columns = list(set(self.original_columns) - set(self.feature_columns))
                if self.high_cardinality_columns:
                    DEBUG.log(f"Removed high cardinality columns: {self.high_cardinality_columns}")

        # Step 3: Handle categorical features
        DEBUG.log("Starting categorical encoding")
        try:
            if isinstance(X, pd.DataFrame):
                X_encoded = self._encode_categorical_features(X, is_training)
            else:
                X_encoded = X  # Skip encoding if X is already a tensor
            DEBUG.log(f"Shape after categorical encoding: {X_encoded.shape}")
            if isinstance(X_encoded, pd.DataFrame):
                DEBUG.log(f"Encoded dtypes:\n{X_encoded.dtypes}")
        except Exception as e:
            DEBUG.log(f"Error in categorical encoding: {str(e)}")
            raise

        # Step 4: Convert to numpy and check for issues
        try:
            if isinstance(X_encoded, pd.DataFrame):
                X_numpy = X_encoded.to_numpy()
            else:
                X_numpy = X_encoded.cpu().numpy()
            DEBUG.log(f"Numpy array shape: {X_numpy.shape}")
            DEBUG.log(f"Any NaN: {np.isnan(X_numpy).any()}")
            DEBUG.log(f"Any Inf: {np.isinf(X_numpy).any()}")
        except Exception as e:
            DEBUG.log(f"Error converting to numpy: {str(e)}")
            raise

        # Step 5: Scale the features using precomputed global statistics
        try:
            X_scaled = np.zeros_like(X_numpy)
            #batch_size = 1024  # Adjust based on available memory

            for i in range(0, len(X_numpy), batch_size):
                batch_end = min(i + batch_size, len(X_numpy))
                batch_X = X_numpy[i:batch_end]

                # Scale the batch using precomputed global statistics
                X_scaled[i:batch_end] = (batch_X - self.global_mean) / self.global_std

            DEBUG.log("Scaling successful")
        except Exception as e:
            DEBUG.log(f"Standard scaling failed: {str(e)}. Using manual scaling")
            if X_numpy.size == 0:
                print("\033[K" +"[WARNING] Empty feature array! Returning original data")
                X_scaled = X_numpy
            else:
                means = np.nanmean(X_numpy, axis=0)
                stds = np.nanstd(X_numpy, axis=0)
                stds[stds == 0] = 1
                X_scaled = (X_numpy - means) / stds

        # Step 6: Convert scaled data to a PyTorch tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        # Step 7: Compute feature pairs and bin edges (only during training)
        if is_training:
            remaining_feature_indices = list(range(len(self.feature_columns))) if self.feature_columns else list(range(X_scaled.shape[1]))
            DEBUG.log(f"Computing feature pairs from {len(remaining_feature_indices)} features")

            # Generate feature pairs using the updated set of features
            self.feature_pairs = self._generate_feature_combinations(
                remaining_feature_indices,
                self.config.get('likelihood_config', {}).get('feature_group_size', 2),
                self.config.get('likelihood_config', {}).get('max_combinations', None)
            )
            DEBUG.log(f"Generated {len(self.feature_pairs)} feature pairs")

            # Compute bin edges using the preprocessed data (now a tensor)
            self.bin_edges = self._compute_bin_edges(X_tensor, self.config.get('likelihood_config', {}).get('bin_sizes', [128]))
            DEBUG.log(f"Computed bin edges for {len(self.bin_edges)} feature pairs")

        DEBUG.log(f"Final preprocessed shape: {X_scaled.shape}")
        return X_tensor

    def _generate_feature_combinations(self, feature_indices: Union[List[int], int], group_size: int = None, max_combinations: int = None) -> torch.Tensor:
        """Generate and save/load consistent feature combinations, treating groups as unique sets."""
        # Ensure feature_indices is a list
        if isinstance(feature_indices, int):
            feature_indices = list(range(feature_indices))  # Convert integer to list of indices

        # Get parameters directly from the root of the config file
        group_size = self.config.get('feature_group_size', 2)
        max_combinations = max_combinations or self.config.get('max_combinations', None)
        bin_sizes = self.config.get('bin_sizes', [128])

        # Debug: Print parameters
        print("\033[K" +f"[DEBUG] Generating feature combinations after filtering out features with high cardinality set by the conf file:")
        print("\033[K" +f"- n_features: {len(feature_indices)}")
        print("\033[K" +f"- group_size: {group_size}")
        print("\033[K" +f"- max_combinations: {max_combinations}")
        print("\033[K" +f"- bin_sizes: {bin_sizes}")

        # Create path for storing feature combinations
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        combinations_path = os.path.join(base_path, dataset_folder, 'feature_combinations.pkl')

        # Check if combinations already exist
        if os.path.exists(combinations_path):
            print("\033[K" +"---------------------BEWARE!! Remove if you get Error on retraining------------------------")
            print("\033[K" +f"[DEBUG] Loading cached feature combinations from {combinations_path}")
            print("\033[K" +"---------------------BEWARE!! Remove if you get Error on retraining------------------------")
            with open(combinations_path, 'rb') as f:
                combinations_tensor = pickle.load(f)
            print("\033[K" +f"[DEBUG] Loaded feature combinations: {combinations_tensor.shape}")
            return combinations_tensor.to(self.device)

        # Generate new combinations if none exist
        print("\033[K" +f"[DEBUG] Generating new feature combinations for {self.dataset_name}")
        from itertools import combinations
        all_combinations = list(combinations(feature_indices, group_size))
        unique_combinations = list(set([tuple(sorted(comb)) for comb in all_combinations]))
        unique_combinations = sorted(unique_combinations)[:max_combinations]
        combinations_tensor = torch.tensor(unique_combinations, device=self.device)

        # Save the new combinations
        os.makedirs(os.path.dirname(combinations_path), exist_ok=True)
        with open(combinations_path, 'wb') as f:
            pickle.dump(combinations_tensor.cpu(), f)
        print("\033[K" +f"[DEBUG] Saved {len(unique_combinations)} unique feature combinations to {combinations_path}")

        return combinations_tensor
#-----------------------------------------------------------------------------Bin model ---------------------------

    def _compute_pairwise_likelihood_parallel(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Optimized non-parametric likelihood computation with precomputed feature pairs and bin edges"""
        DEBUG.log(" Starting _compute_pairwise_likelihood_parallel")
        print("\033[K" +"Computing pairwise likelihoods...")

        # Input validation and preparation
        dataset = torch.as_tensor(dataset, device=self.device).contiguous()
        labels = torch.as_tensor(labels, device=self.device).contiguous()

        # Initialize progress tracking
        n_pairs = len(self.feature_pairs)
        pair_pbar = tqdm(total=n_pairs, desc="Processing feature pairs")

        # Pre-compute unique classes once
        unique_classes, class_counts = torch.unique(labels, return_counts=True)
        n_classes = len(unique_classes)
        n_samples = len(dataset)

        # Get bin sizes from configuration
        bin_sizes = self.config.get('likelihood_config', {}).get('bin_sizes', [128])
        if len(bin_sizes) == 1:
            # If single bin size provided, use it for all dimensions
            n_bins = bin_sizes[0]
            self.n_bins_per_dim = n_bins  # Store for reference
            DEBUG.log(f" Using uniform {n_bins} bins per dimension")
        else:
            DEBUG.log(f" Using variable bin sizes: {bin_sizes}")

        # Use precomputed feature pairs and bin edges
        if not hasattr(self, 'feature_pairs') or not hasattr(self, 'bin_edges'):
            self.feature_pairs = self._generate_feature_combinations(
                feature_dims,
                self.config.get('likelihood_config', {}).get('feature_group_size', 2),  # Use group_size from config
                self.config.get('likelihood_config', {}).get('max_combinations', None)
            ).to(self.device)
            self.bin_edges = self._compute_bin_edges(dataset, bin_sizes)

        # Ensure bin_edges is not None
        if self.bin_edges is None:
            raise ValueError("bin_edges is not initialized. Ensure _compute_bin_edges is called before this method.")

        # Initialize storage for bin counts and probabilities
        all_bin_counts = []  # Initialize the list to store bin counts for all feature pairs
        all_bin_probs = []   # Initialize the list to store bin probabilities for all feature pairs

        # Process each feature group
        for group_idx, feature_group in enumerate(self.feature_pairs):
            feature_group = [int(x) for x in feature_group]
            DEBUG.log(f" Processing feature group: {feature_group}")

            # Extract group data
            group_data = dataset[:, feature_group].contiguous()
            n_dims = len(feature_group)

            # Get appropriate bin sizes for this group
            group_bin_sizes = bin_sizes[:n_dims] if len(bin_sizes) > 1 else [bin_sizes[0]] * n_dims

            # Use precomputed bin edges
            bin_edges = self.bin_edges[group_idx]

            # Initialize bin counts with appropriate shape for variable bin sizes
            bin_shape = [n_classes] + [size for size in group_bin_sizes]
            bin_counts = torch.zeros(bin_shape, device=self.device, dtype=torch.float32)

            # Process each class
            for class_idx, class_label in enumerate(unique_classes):
                class_mask = labels == class_label
                if class_mask.any():
                    class_data = group_data[class_mask].contiguous()

                    # Compute bin indices for all dimensions
                    bin_indices = torch.stack([
                        torch.bucketize(
                            class_data[:, dim].contiguous(),
                            bin_edges[dim].contiguous()
                        ).sub_(1).clamp_(0, group_bin_sizes[dim] - 1)
                        for dim in range(n_dims)
                    ])  # Shape: (n_dims, batch_size)

                    # Use scatter_add_ for efficient counting
                    counts = torch.zeros(np.prod(group_bin_sizes), device=self.device)
                    flat_indices = torch.sum(
                        bin_indices * torch.tensor(
                            [np.prod(group_bin_sizes[i+1:]) for i in range(n_dims)],
                            device=self.device
                        ).unsqueeze(1),  # Add a new dimension for broadcasting
                        dim=0
                    ).long()  # Cast to int64
                    counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
                    bin_counts[class_idx] = counts.reshape(*group_bin_sizes)

            # Apply Laplace smoothing and compute probabilities
            smoothed_counts = bin_counts + 1.0
            bin_probs = smoothed_counts / smoothed_counts.sum(dim=tuple(range(1, n_dims + 1)), keepdim=True)

            # Store results
            all_bin_counts.append(smoothed_counts)
            all_bin_probs.append(bin_probs)

            DEBUG.log(f" Bin counts shape: {smoothed_counts.shape}")
            DEBUG.log(f" Bin probs shape: {bin_probs.shape}")
            pair_pbar.update(1)

        pair_pbar.close()
        return {
            'bin_counts': all_bin_counts,
            'bin_probs': all_bin_probs,
            'bin_edges': self.bin_edges,  # Include bin_edges in the returned dictionary
            'feature_pairs': self.feature_pairs,
            'classes': unique_classes.to(self.device)
        }

    def _compute_gaussian_params(self, dataset: torch.Tensor, labels: torch.Tensor):
        """
        Compute Gaussian parameters (means and covariances) for all feature pairs once during initialization.

        Args:
            dataset: Input tensor of shape [n_samples, n_features]
            labels: Target tensor of shape [n_samples]

        Returns:
            Dictionary containing means and covariances for each feature pair and class
        """
        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)
        n_pairs = len(self.feature_pairs)

        # Initialize storage for means and covariances
        means = torch.zeros((n_classes, n_pairs, 2), device=self.device)
        covs = torch.zeros((n_classes, n_pairs, 2, 2), device=self.device)

        # Process each class
        for class_idx, class_id in enumerate(unique_classes):
            class_mask = (labels == class_id)
            class_data = dataset[class_mask]

            # Process each feature pair
            for pair_idx, pair in enumerate(self.feature_pairs):
                pair_data = class_data[:, pair]

                # Compute mean efficiently
                means[class_idx, pair_idx] = torch.mean(pair_data, dim=0)

                # Compute covariance with stability check
                if len(pair_data) > 1:
                    centered_data = pair_data - means[class_idx, pair_idx].unsqueeze(0)
                    cov = torch.matmul(centered_data.T, centered_data) / (len(pair_data) - 1)

                    # Add stability term
                    min_eig = torch.linalg.eigvals(cov).real.min()
                    if min_eig < 1e-6:
                        reg_term = (1e-6 - min_eig) + 1e-6
                        cov += torch.eye(2, device=self.device) * reg_term
                else:
                    # Single sample case - use identity matrix
                    cov = torch.eye(2, device=self.device)

                covs[class_idx, pair_idx] = cov

        return {
            'means': means,
            'covs': covs,
            'classes': unique_classes,
            'feature_pairs': self.feature_pairs
        }


    def _compute_pairwise_likelihood_parallel_std(self, dataset: torch.Tensor, labels: torch.Tensor, feature_dims: int):
        """Optimized Gaussian likelihood computation with precomputed feature pairs and Gaussian parameters"""
        DEBUG.log(" Starting _compute_pairwise_likelihood_parallel_std")
        print("\033[K" +"Computing Gaussian likelihoods...")

        # Input validation and preparation
        dataset = torch.as_tensor(dataset, device=self.device).contiguous()
        labels = torch.as_tensor(labels, device=self.device).contiguous()

        # Initialize progress tracking
        n_pairs = len(self.feature_pairs)
        pair_pbar = tqdm(total=n_pairs, desc="Processing feature pairs")

        # Pre-compute unique classes once
        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)

        # Use precomputed feature pairs and Gaussian parameters
        if not hasattr(self, 'feature_pairs') or not hasattr(self, 'gaussian_params'):
            self.feature_pairs = self._generate_feature_combinations(
                feature_dims,
                self.config.get('likelihood_config', {}).get('feature_group_size', 2),  # Use group_size from config
                self.config.get('likelihood_config', {}).get('max_combinations', None)
            ).to(self.device)
            self.gaussian_params = self._compute_gaussian_params(dataset, labels)

        # Ensure gaussian_params is not None
        if self.gaussian_params is None:
            raise ValueError("gaussian_params is not initialized. Ensure _compute_gaussian_params is called before this method.")
        # Pre-allocate storage arrays
        log_likelihoods = torch.zeros((len(dataset), n_classes), device=self.device)

        # Process each feature group
        for pair_idx, pair in enumerate(self.feature_pairs):
            pair_data = dataset[:, pair]

            # Process each class
            for class_idx, class_id in enumerate(unique_classes):
                mean = self.gaussian_params['means'][class_idx, pair_idx]
                cov = self.gaussian_params['covs'][class_idx, pair_idx]

                # Center the data
                centered = pair_data - mean.unsqueeze(0)

                # Compute class likelihood
                try:
                    # Add minimal regularization
                    reg_cov = cov + torch.eye(2, device=self.device) * 1e-6
                    prec = torch.inverse(reg_cov)

                    # Quadratic term
                    quad = torch.sum(
                        torch.matmul(centered.unsqueeze(1), prec).squeeze(1) * centered,
                        dim=1
                    )

                    # Log likelihood (excluding constant terms that are same for all classes)
                    class_ll = -0.5 * quad

                except RuntimeError:
                    # Handle numerical issues by setting very low likelihood
                    class_ll = torch.full_like(quad, -1e10)

                log_likelihoods[:, class_idx] += class_ll

            pair_pbar.update(1)

        # Convert to probabilities using softmax
        max_log_ll = torch.max(log_likelihoods, dim=1, keepdim=True)[0]
        exp_ll = torch.exp(log_likelihoods - max_log_ll)
        posteriors = exp_ll / (torch.sum(exp_ll, dim=1, keepdim=True) + 1e-10)

        pair_pbar.close()
        return posteriors, None

    def _compute_batch_posterior_std(self, features: torch.Tensor, epsilon: float = 1e-10):
        """Gaussian posterior computation focusing on relative class probabilities"""
        features = features.to(self.device)
        batch_size = len(features)
        n_classes = len(self.likelihood_params['classes'])

        # Initialize log likelihoods
        log_likelihoods = torch.zeros((batch_size, n_classes), device=self.device)

        # Process each feature pair
        for pair_idx, pair in enumerate(self.feature_pairs):
            pair_data = features[:, pair]

            # Get weights for this pair (same as histogram mode)
            pair_weights = [
                self.weight_updater.get_gaussian_weights(class_idx, pair_idx)
                for class_idx in range(n_classes)
            ]

            # Compute class contributions for this pair
            for class_idx in range(n_classes):
                mean = self.likelihood_params['means'][class_idx, pair_idx]
                cov = self.likelihood_params['covs'][class_idx, pair_idx]
                weight = pair_weights[class_idx]

                # Center the data
                centered = pair_data - mean.unsqueeze(0)

                # Compute class likelihood
                try:
                    # Add minimal regularization
                    reg_cov = cov + torch.eye(2, device=self.device) * 1e-6
                    prec = torch.inverse(reg_cov)

                    # Quadratic term
                    quad = torch.sum(
                        torch.matmul(centered.unsqueeze(1), prec).squeeze(1) * centered,
                        dim=1
                    )

                    # Log likelihood (excluding constant terms that are same for all classes)
                    class_ll = -0.5 * quad + torch.log(weight + epsilon)

                except RuntimeError:
                    # Handle numerical issues by setting very low likelihood
                    class_ll = torch.full_like(quad, -1e10)

                log_likelihoods[:, class_idx] += class_ll

        # Convert to probabilities using softmax
        max_log_ll = torch.max(log_likelihoods, dim=1, keepdim=True)[0]
        exp_ll = torch.exp(log_likelihoods - max_log_ll)
        posteriors = exp_ll / (torch.sum(exp_ll, dim=1, keepdim=True) + epsilon)

        return posteriors, None

    def _initialize_bin_weights(self):
        """Initialize weights for either histogram bins or Gaussian components"""
        n_classes = len(self.label_encoder.classes_)
        if self.model_type == "Histogram":
            self.weight_updater = BinWeightUpdater(
                n_classes=n_classes,
                feature_pairs=self.feature_pairs,
                n_bins_per_dim=self.n_bins_per_dim,
                batch_size=self.batch_size
            )
        elif self.model_type == "Gaussian":
            # Use same weight structure but for Gaussian components
            self.weight_updater = BinWeightUpdater(
                n_classes=n_classes,
                feature_pairs=self.feature_pairs,
                n_bins_per_dim=self.n_bins_per_dim  # Number of Gaussian components
            )

    def _update_priors_parallel(self, failed_cases: List[Tuple], batch_size: int = 128):
        """Vectorized weight updates with proper error handling"""
        n_failed = len(failed_cases)
        if n_failed == 0:
            self.consecutive_successes += 1
            return

        self.consecutive_successes = 0
        self.learning_rate = max(self.learning_rate / 2, 1e-6)

        # Stack all features and convert classes at once
        features = torch.stack([case[0] for case in failed_cases]).to(self.device)
        true_classes = torch.tensor([int(case[1]) for case in failed_cases], device=self.device)

        # Compute posteriors for all cases at once
        if self.model_type == "Histogram":
            posteriors, bin_indices = self._compute_batch_posterior(features)
        else:  # Gaussian model
            posteriors, _ = self._compute_batch_posterior_std(features)
            return  # Gaussian model doesn't need bin-based updates

        pred_classes = torch.argmax(posteriors, dim=1)

        # Compute adjustments for all cases at once
        true_posteriors = posteriors[torch.arange(n_failed), true_classes]
        pred_posteriors = posteriors[torch.arange(n_failed), pred_classes]
        adjustments = self.learning_rate * (1.0 - (true_posteriors / pred_posteriors))

        # Update weights for each feature group
        if bin_indices is not None:  # Only proceed if we have bin indices (Histogram model)
            for group_idx in bin_indices:
                bin_i, bin_j = bin_indices[group_idx]

                # Group updates by class for vectorization
                for class_id in range(self.weight_updater.n_classes):
                    class_mask = true_classes == class_id
                    if not class_mask.any():
                        continue

                    # Get relevant indices and adjustments for this class
                    class_bin_i = bin_i[class_mask]
                    class_bin_j = bin_j[class_mask]
                    class_adjustments = adjustments[class_mask]

                    # Update weights for this class
                    weights = self.weight_updater.histogram_weights[class_id][group_idx]
                    for idx in range(len(class_adjustments)):
                        i, j = class_bin_i[idx], class_bin_j[idx]
                        weights[i, j] += class_adjustments[idx]
#------------------------------------------Boost weights------------------------------------------

    def _compute_custom_bin_edges(self, data: torch.Tensor, bin_sizes: List[int]) -> List[torch.Tensor]:
        """
        Compute bin edges based on custom bin sizes.
        Supports both uniform and non-uniform binning.

        Args:
            data: Input tensor of shape [n_samples, n_features]
            bin_sizes: List of integers specifying bin sizes for each dimension

        Returns:
            List of tensors containing bin edges for each dimension
        """
        n_dims = data.shape[1]
        bin_edges = []

        # If single bin size provided, use it for all dimensions
        if len(bin_sizes) == 1:
            bin_sizes = bin_sizes * n_dims

        # Ensure we have enough bin sizes
        if len(bin_sizes) < n_dims:
            raise ValueError(f"Not enough bin sizes provided. Need {n_dims}, got {len(bin_sizes)}")

        for dim in range(n_dims):
            dim_data = data[:, dim]
            dim_min, dim_max = dim_data.min(), dim_data.max()
            padding = (dim_max - dim_min) * 0.01

            # Create edges based on specified bin size
            if bin_sizes[dim] <= 1:
                raise ValueError(f"Bin size must be > 1, got {bin_sizes[dim]}")

            edges = torch.linspace(
                dim_min - padding,
                dim_max + padding,
                bin_sizes[dim] + 1,
                device=self.device
            )
            bin_edges.append(edges)

        return bin_edges


#---------------------------------------------------------Save Last data -------------------------
    def save_last_split(self, train_indices: list, test_indices: list):
        """Save the last training/testing split to CSV files"""
        dataset_name = self.dataset_name

        # Get full dataset
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Save training data
        train_data = pd.concat([X.iloc[train_indices], y.iloc[train_indices]], axis=1)
        train_data.to_csv(f'{dataset_name}_Last_training.csv',header=True, index=True)

        # Save testing data
        test_data = pd.concat([X.iloc[test_indices], y.iloc[test_indices]], axis=1)
        test_data.to_csv(f'{dataset_name}_Last_testing.csv', header=True, index=True)
        print("\033[K" +f"{Colors.GREEN}Last testing data is saved to {dataset_name}_Last_testing.csv{Colors.ENDC}")
        print("\033[K" +f"{Colors.GREEN}Last training data is saved to {dataset_name}_Last_training.csv{Colors.ENDC}")

    def load_last_known_split(self):
        """Load the last known good training/testing split with proper column alignment"""
        dataset_name = self.dataset_name
        train_file = f'{dataset_name}_Last_training.csv'
        test_file = f'{dataset_name}_Last_testing.csv'

        if os.path.exists(train_file) and os.path.exists(test_file):
            try:
                # Load the saved splits
                train_data = pd.read_csv(train_file)
                test_data = pd.read_csv(test_file)

                # Get current feature columns excluding target
                X = self.data.drop(columns=[self.target_column])
                current_columns = X.columns

                # Ensure train_data has same columns as current data
                train_features = train_data.drop(columns=[self.target_column])
                train_features = train_features[current_columns]

                # Initialize indices lists
                train_indices = []
                test_indices = []

                # Match rows using selected columns
                for idx, row in X.iterrows():
                    # Align the row with train_features columns
                    row = row[current_columns]

                    # Compare with train data first
                    match_mask = (train_features == row).all(axis=1)
                    if match_mask.any():
                        train_indices.append(idx)
                    else:
                        test_indices.append(idx)

                if train_indices or test_indices:
                    print("\033[K" +f"Loaded previous split - Training: {len(train_indices)}, Testing: {len(test_indices)}")
                    return train_indices, test_indices
                else:
                    print("\033[K" +"No valid indices found in previous split")
                    return None, None

            except Exception as e:
                print("\033[K" +f"Error loading previous split: {str(e)}")
                return None, None

        return None, None


    def predict(self, X: torch.Tensor, batch_size: int = 128):
        """
        Make predictions in batches using the best model weights.
        This function ensures all batches are processed and predictions are concatenated correctly.
        """
        # Store current weights temporarily
        #print("\033[K" +"Making predictions...")
        temp_W = self.current_W
        n_batches = (len(X) + batch_size - 1) // batch_size
        pred_pbar = tqdm(total=n_batches, desc="Prediction batches")

        # Use best weights for prediction
        self.current_W = self.best_W.clone() if self.best_W is not None else self.current_W

        X = X.to(self.device)
        predictions = []

        try:
            for i in range(0, len(X), batch_size):
                batch_X = X[i:min(i + batch_size, len(X))]
                if self.model_type == "Histogram":
                    # Get posteriors only, ignore bin indices
                    posteriors, _ = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    # Get posteriors only, ignore component responsibilities
                    posteriors, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    raise ValueError(f"{self.model_type} is an invalid model type. Please edit the configuration file.")

                # Get predictions for the current batch
                batch_predictions = torch.argmax(posteriors, dim=1)
                predictions.append(batch_predictions)
                pred_pbar.update(1)

        finally:
            # Restore current weights
            self.current_W = temp_W

        pred_pbar.close()

        # Concatenate predictions from all batches
        return torch.cat(predictions).cpu()


    def _save_best_weights(self):
        """Save the best weights to file"""
        if self.best_W is not None:
            # Convert tensor to numpy for saving
            weights_array = self.best_W.cpu().numpy()

            weights_dict = {
                'version': 2,  # Add version to track format
                'weights': weights_array.tolist(),
                'shape': list(weights_array.shape)
            }

            with open(self._get_weights_filename(), 'w') as f:
                json.dump(weights_dict, f)

    def _load_best_weights(self):
        """Load the best weights from file if they exist"""
        weights_file = self._get_weights_filename()
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                weights_dict = json.load(f)

            try:
                if 'version' in weights_dict and weights_dict['version'] == 2:
                    # New format (tensor-based)
                    weights_array = np.array(weights_dict['weights'])
                    self.best_W = torch.tensor(
                        weights_array,
                        dtype=torch.float32,
                        device=self.device
                    )
                else:
                    # Old format (dictionary-based)
                    # Convert old format to tensor format
                    class_ids = sorted([int(k) for k in weights_dict.keys()])
                    max_class_id = max(class_ids)

                    # Get number of feature pairs from first class
                    first_class = weights_dict[str(class_ids[0])]
                    n_pairs = len(first_class)

                    # Initialize tensor
                    weights_array = np.zeros((max_class_id + 1, n_pairs))

                    # Fill in weights from old format
                    for class_id in class_ids:
                        class_weights = weights_dict[str(class_id)]
                        for pair_idx, (pair, weight) in enumerate(class_weights.items()):
                            weights_array[class_id, pair_idx] = float(weight)

                    self.best_W = torch.tensor(
                        weights_array,
                        dtype=torch.float32,
                        device=self.device
                    )

                print("\033[K" +f"Loaded best weights from {weights_file}")
            except Exception as e:
                print("\033[K" +f"Warning: Could not load weights from {weights_file}: {str(e)}")
                self.best_W = None

    def _init_keyboard_listener(self):
        """Initialize keyboard listener with shared display connection"""
        if not hasattr(self, '_display'):
            try:
                import Xlib.display
                self._display = Xlib.display.Display()
            except Exception as e:
                print("\033[K" +f"Warning: Could not initialize X display: {e}")
                return None

        try:
            from pynput import keyboard
            return keyboard.Listener(
                on_press=self._on_key_press,
                _display=self._display  # Pass shared display connection
            )
        except Exception as e:
            print("\033[K" +f"Warning: Could not create keyboard listener: {e}")
            return None

    def _cleanup_keyboard(self):
        """Clean up keyboard resources"""
        if hasattr(self, '_display'):
            try:
                self._display.close()
                del self._display
            except:
                pass

    def print_colored_confusion_matrix(self, y_true, y_pred, class_labels=None,header=None):
        # Decode numeric labels back to original alphanumeric labels
        y_true_labels = self.label_encoder.inverse_transform(y_true)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        # Get unique classes from both true and predicted labels
        unique_true = np.unique(y_true_labels)
        unique_pred = np.unique(y_pred_labels)

        # Use provided class labels or get from label encoder
        if class_labels is None:
            class_labels = self.label_encoder.classes_

        # Ensure all classes are represented in confusion matrix
        all_classes = np.unique(np.concatenate([unique_true, unique_pred, class_labels]))
        n_classes = len(all_classes)

        # Create class index mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

        # Initialize confusion matrix with zeros
        cm = np.zeros((n_classes, n_classes), dtype=int)

        # Fill confusion matrix
        for t, p in zip(y_true_labels, y_pred_labels):
            if t in class_to_idx and p in class_to_idx:
                cm[class_to_idx[t], class_to_idx[p]] += 1

        # Print confusion matrix with colors
        print("\033[K" +f"{Colors.BOLD}Confusion Matrix and Class-wise Accuracy for [{header}]:{Colors.ENDC}")
        print("\033[K" +f"{'Actual/Predicted':<15}", end='')
        for label in all_classes:
            print("\033[K" +f"{str(label):<8}", end='')
        print("\033[K" +"Accuracy")
        print("\033[K" +"-" * (15 + 8 * n_classes + 10))

        # Print matrix with colors
        for i in range(n_classes):
            # Print actual class label
            print("\033[K" +f"{Colors.BOLD}{str(all_classes[i]):<15}{Colors.ENDC}", end='')

            # Print confusion matrix row
            for j in range(n_classes):
                if i == j:
                    # Correct predictions in green
                    color = Colors.GREEN
                else:
                    # Incorrect predictions in red
                    color = Colors.RED
                print("\033[K" +f"{color}{cm[i, j]:<8}{Colors.ENDC}", end='')

            # Print class accuracy with color based on performance
            acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
            if acc >= 0.9:
                color = Colors.GREEN
            elif acc >= 0.7:
                color = Colors.YELLOW
            else:
                color = Colors.BLUE
            print("\033[K" +f"{color}{acc:>7.2%}{Colors.ENDC}")

        # Print overall accuracy
        total_correct = np.diag(cm).sum()
        total_samples = cm.sum()
        if total_samples > 0:
            overall_acc = total_correct / total_samples
            print("\033[K" +"-" * (15 + 8 * n_classes + 10))
            color = Colors.GREEN if overall_acc >= 0.9 else Colors.YELLOW if overall_acc >= 0.7 else Colors.BLUE
            print("\033[K" +f"{Colors.BOLD}Overall Accuracy: {color}{overall_acc:.2%}{Colors.ENDC}")
            print("\033[K" +f"Best Overall Accuracy till now is: {Colors.GREEN}{self.best_combined_accuracy:.2%}{Colors.ENDC}")

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, batch_size: int = 128):
        """Training loop with proper weight handling and enhanced progress tracking"""
        print("\033[K" +"Starting training..." , end="\r", flush=True)
        # Initialize best combined accuracy if not already set
        if not hasattr(self, 'best_combined_accuracy'):
            self.best_combined_accuracy = 0.0

        # Initialize best model weights if not already set
        if not hasattr(self, 'best_model_weights'):
            self.best_model_weights = None

        # Store initial conditions at the start of training
        if self.best_round_initial_conditions is None:
            self.best_round_initial_conditions = {
                'weights': self.current_W.clone(),
                'likelihood_params': self.likelihood_params,
                'feature_pairs': self.feature_pairs,
                'bin_edges': self.bin_edges,
                'gaussian_params': self.gaussian_params
            }


        # Initialize progress bar for epochs
        epoch_pbar = tqdm(total=self.max_epochs, desc="Training epochs")

        # Store current weights for prediction during training
        train_weights = self.current_W.clone() if self.current_W is not None else None

        # Pre-allocate tensors for batch processing
        n_samples = len(X_train)
        predictions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        batch_mask = torch.empty(batch_size, dtype=torch.bool, device=self.device)

        # Initialize tracking variables
        error_rates = []
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        prev_train_error = float('inf')
        prev_train_accuracy = 0.0
        prev_test_accuracy = 0.0
        patience_counter = 0
        best_train_accuracy = 0.0
        best_test_accuracy = 0.0

        if self.in_adaptive_fit:
            patience = 5
        else:
            patience = Trials

        for epoch in range(self.max_epochs):
            # Save epoch data
            self.save_epoch_data(epoch, self.train_indices, self.test_indices)

            Trstart_time = time.time()
            failed_cases = []
            n_errors = 0

            # Process training data in batches
            n_batches = (len(X_train) + batch_size - 1) // batch_size
            batch_pbar = tqdm(total=n_batches, desc=f"Epoch {epoch+1} batches", leave=False)

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                current_batch_size = batch_end - i

                batch_X = X_train[i:batch_end]
                batch_y = y_train[i:batch_end]

                # Compute posteriors for batch
                if self.model_type == "Histogram":
                    posteriors, bin_indices = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    posteriors, comp_resp = self._compute_batch_posterior_std(batch_X)

                predictions[:current_batch_size] = torch.argmax(posteriors, dim=1)
                batch_mask[:current_batch_size] = (predictions[:current_batch_size] != batch_y)

                n_errors += batch_mask[:current_batch_size].sum().item()

                if batch_mask[:current_batch_size].any():
                    failed_indices = torch.where(batch_mask[:current_batch_size])[0]
                    for idx in failed_indices:
                        failed_cases.append((
                            batch_X[idx],
                            batch_y[idx].item(),
                            posteriors[idx].cpu().numpy()
                        ))
                batch_pbar.update(1)

            batch_pbar.close()

            # Calculate training error rate
            train_error_rate = n_errors / n_samples
            error_rates.append(train_error_rate)

            # Calculate metrics using current weights
            with torch.no_grad():
                # Temporarily set current_W for training metrics
                orig_weights = self.current_W
                self.current_W = train_weights

                # Training metrics
                print("\033[K" +f"{Colors.GREEN}Predctions on Training data{Colors.ENDC}", end="\r", flush=True)
                train_predictions = self.predict(X_train, batch_size=batch_size)
                train_accuracy = (train_predictions == y_train.cpu()).float().mean()
                train_loss = n_errors / n_samples

                # Restore original weights
                self.current_W = orig_weights

            # Update best accuracies
            best_train_accuracy = max(best_train_accuracy, train_accuracy)

            # Store metrics
            train_losses.append(train_loss)

            train_accuracies.append(train_accuracy)


            # Calculate training time
            Trend_time = time.time()
            training_time = Trend_time - Trstart_time

            # Update progress display
            epoch_pbar.update(1)
            epoch_pbar.set_postfix({
                'train_err': f"{train_error_rate:.4f} (best: {1-best_train_accuracy:.4f})",
                'train_acc': f"{train_accuracy:.4f} (best: {best_train_accuracy:.4f})"
            })

            #print("\033[K" +f"Epoch {epoch + 1}/{self.max_epochs}:", end="\r", flush=True)
            #print("\033[K" +f"Training time: {Colors.highlight_time(training_time)} seconds", end="\r", flush=True)
            #print("\033[K" +f"Train error rate: {Colors.color_value(train_error_rate, prev_train_error, False)} (best: {1-best_train_accuracy:.4f})", end="\r", flush=True)
            #print("\033[K" +f"Train accuracy: {Colors.color_value(train_accuracy, prev_train_accuracy, True)} (best: {Colors.GREEN}{best_train_accuracy:.4f}{Colors.ENDC})", end="\r", flush=True)

            # Update previous values for next iteration
            prev_train_error = train_error_rate
            prev_train_accuracy = train_accuracy

            # Check if this is the best round so far
            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy
                self.best_round = epoch
                self.best_round_initial_conditions = {
                    'weights': self.current_W.clone(),
                    'likelihood_params': self.likelihood_params,
                    'feature_pairs': self.feature_pairs,
                    'bin_edges': self.bin_edges,
                    'gaussian_params': self.gaussian_params
                }

            # Update best model if improved
            if train_error_rate <= self.best_error:
                improvement = self.best_error - train_error_rate
                self.best_error = train_error_rate
                self.best_W = self.current_W.clone()  # Save current weights as best
                #self._save_best_weights()   # move it to train_fit

                if improvement <= 0.001:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    self.learning_rate = LearningRate
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= patience or  train_accuracy ==1.00:
                print("\033[K" +f"{Colors.YELLOW} Early stopping.{Colors.ENDC}")
                break

            # Update weights if there were failures
            if failed_cases:
                self._update_priors_parallel(failed_cases, batch_size)
            # Save reconstruction plots if enabled
            if self.save_plots:
                # Reconstruct features from predictions
                reconstructed_features = self.reconstruct_features(posteriors)
                save_path=f"data/{self.dataset_name}/plots/epoch_{epoch+1}"
                os.makedirs(save_path, exist_ok=True)
                self._save_reconstruction_plots(
                    original_features=X_train.cpu().numpy(),
                    reconstructed_features=reconstructed_features.cpu().numpy(),
                    true_labels=y_train.cpu().numpy(),
                    save_path=save_path
                )


        # Training complete
        epoch_pbar.close()
        #self._save_model_components()
        return self.current_W.cpu(), error_rates

    #---------------------------------Train InvertableDBNN on the fly ------------------------------------
    def load_inverse_model(self, custom_path: str = None) -> bool:
       try:
           load_dir = custom_path or os.path.join('Model', f'Best_inverse_{self.forward_model.dataset_name}')
           model_path = os.path.join(load_dir, 'inverse_model.pt')
           config_path = os.path.join(load_dir, 'inverse_config.json')

           if not (os.path.exists(model_path) and os.path.exists(config_path)):
               print("\033[K" +f"No saved inverse model found at {load_dir}")
               return False

           model_state = torch.load(model_path, map_location=self.device, weights_only=True)

           with open(config_path, 'r') as f:
               config = json.load(f)

           if config['feature_dims'] != self.feature_dims or config['n_classes'] != self.n_classes:
               raise ValueError("Model architecture mismatch")

           # Load parameters
           self.weight_linear.data = model_state['weight_linear']
           self.weight_nonlinear.data = model_state['weight_nonlinear']
           self.bias_linear.data = model_state['bias_linear']
           self.bias_nonlinear.data = model_state['bias_nonlinear']
           self.feature_attention.data = model_state['feature_attention']
           self.layer_norm.load_state_dict(model_state['layer_norm'])

           # Safely update or register buffers
           for param in ['min_vals', 'max_vals', 'scale_factors', 'inverse_feature_pairs']:
               if param in model_state:
                   buffer_data = model_state[param]
                   if buffer_data is not None:
                       if hasattr(self, param) and getattr(self, param) is not None:
                           getattr(self, param).copy_(buffer_data)
                       else:
                           self.register_buffer(param, buffer_data)

           # Restore other attributes
           self.metrics = model_state.get('metrics', {})
           self.reconstruction_weight = model_state.get('reconstruction_weight', 0.5)
           self.feedback_strength = model_state.get('feedback_strength', 0.3)

           print("\033[K" +f"Loaded inverse model from {load_dir}")
           return True

       except Exception as e:
           print("\033[K" +f"Error loading inverse model: {str(e)}")
           traceback.print_exc()
           return False

    def save_reconstruction_features(self,
                                     reconstructed_features: torch.Tensor,
                                     original_features: torch.Tensor,
                                     predictions: torch.Tensor,
                                     true_labels: torch.Tensor = None,
                                     class_probs: torch.Tensor = None) -> Dict:
        """Save reconstruction features and return JSON-compatible output.

        Args:
            reconstructed_features: Reconstructed feature tensor
            original_features: Original input feature tensor
            predictions: Model predictions tensor
            true_labels: True labels tensor (optional)
            class_probs: Class probabilities tensor (optional)

        Returns:
            Dict containing reconstruction data and paths
        """
        # Create reconstruction directory
        dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
        recon_dir = os.path.join('data', dataset_name, 'reconstruction')
        os.makedirs(recon_dir, exist_ok=True)

        # Convert tensors to numpy arrays
        recon_np = reconstructed_features.cpu().numpy()
        orig_np = original_features.cpu().numpy()
        pred_np = predictions.cpu().numpy()

        # Create DataFrame with original and reconstructed features
        feature_cols = [f'feature_{i}' for i in range(orig_np.shape[1])]
        recon_cols = [f'reconstructed_{i}' for i in range(recon_np.shape[1])]

        df = pd.DataFrame(orig_np, columns=feature_cols)
        df = pd.concat([df, pd.DataFrame(recon_np, columns=recon_cols)], axis=1)

        # Add predictions
        df['predicted_class'] = self.label_encoder.inverse_transform(pred_np)

        # Add true labels if provided
        if true_labels is not None:
            true_np = true_labels.cpu().numpy()
            df['true_class'] = self.label_encoder.inverse_transform(true_np)

        # Add class probabilities if provided
        if class_probs is not None:
            probs_np = class_probs.cpu().numpy()
            for i, class_name in enumerate(self.label_encoder.classes_):
                df[f'prob_{class_name}'] = probs_np[:, i]

        # Add reconstruction error
        df['reconstruction_error'] = np.mean((orig_np - recon_np) ** 2, axis=1)

        # Save to CSV
        csv_path = os.path.join(recon_dir, f'data/{dataset_name}/{dataset_name}_reconstruction.csv')
        df.to_csv(csv_path, index=False)

        # Create JSON-compatible output
        output = {
            'dataset': dataset_name,
            'reconstruction_path': csv_path,
            'feature_count': orig_np.shape[1],
            'sample_count': len(df),
            'mean_reconstruction_error': float(df['reconstruction_error'].mean()),
            'std_reconstruction_error': float(df['reconstruction_error'].std()),
            'features': {
                'original': feature_cols,
                'reconstructed': recon_cols
            },
            'class_mapping': dict(zip(
                range(len(self.label_encoder.classes_)),
                self.label_encoder.classes_
            ))
        }

        # Save metadata as JSON
        json_path = os.path.join(recon_dir, f'{dataset_name}_reconstruction_meta.json')
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

        return output

    #------------------------------End Train InvertableDBNN on the fly ------------------------------------

    def plot_training_metrics(self, train_loss, test_loss, train_acc, test_acc, save_path=None):
        """Plot training and testing metrics over epochs"""
        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label='Train Loss', marker='o')
        plt.plot(test_loss, label='Test Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(train_acc, label='Train Accuracy', marker='o')
        plt.plot(test_acc, label='Test Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            # Also save metrics to CSV
            metrics_df = pd.DataFrame({
                'epoch': range(1, len(train_loss) + 1),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            })
            metrics_df.to_csv(save_path.replace('.png', '_metrics.csv'), index=False)

        plt.close()

    def verify_classifications(self, X: pd.DataFrame, true_labels: pd.Series, predictions: torch.Tensor) -> None:
        """
        Verify classification accuracy computation with proper index handling
        """
        # Convert predictions to numpy array if it's a tensor
        pred_labels = self.label_encoder.inverse_transform(predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions)

        # Ensure true_labels is a numpy array
        true_labels_array = true_labels.to_numpy() if isinstance(true_labels, pd.Series) else true_labels

        # Calculate accuracy metrics
        n_total = len(true_labels_array)
        correct_mask = (true_labels_array == pred_labels)
        n_correct = correct_mask.sum()

        # Print basic metrics with colors
        print("\033[K" +f"{Colors.BOLD}{Colors.BLUE}Detailed Classification Analysis:{Colors.ENDC}")
        print("\033[K" +f"{Colors.BOLD}Total samples:{Colors.ENDC} {Colors.YELLOW}{n_total:,}{Colors.ENDC}")

        # Color code for correctly classified
        correct_color = Colors.GREEN if (n_correct/n_total) >= 0.9 else \
                       Colors.YELLOW if (n_correct/n_total) >= 0.7 else \
                       Colors.BLUE
        print("\033[K" +f"{Colors.BOLD}Correctly classified:{Colors.ENDC} {correct_color}{n_correct:,}{Colors.ENDC}")

        # Color code for incorrectly classified
        incorrect = n_total - n_correct
        incorrect_color = Colors.GREEN if (incorrect/n_total) <= 0.1 else \
                         Colors.YELLOW if (incorrect/n_total) <= 0.3 else \
                         Colors.RED
        print("\033[K" +f"{Colors.BOLD}Incorrectly classified:{Colors.ENDC} {incorrect_color}{incorrect:,}{Colors.ENDC}")

        # Color code for raw accuracy
        accuracy = n_correct/n_total
        accuracy_color = Colors.GREEN if accuracy >= 0.9 else \
                        Colors.YELLOW if accuracy >= 0.7 else \
                        Colors.BLUE
        print("\033[K" +f"{Colors.BOLD}Raw accuracy:{Colors.ENDC} {accuracy_color}{accuracy:.4%}{Colors.ENDC}\n")

        # Print confusion matrix with colors
        self.print_colored_confusion_matrix(true_labels_array, pred_labels,header="Test data")

        # Save detailed analysis to file
        analysis_file = f"classification_analysis_{self.dataset_name}.txt"
        with open(analysis_file, 'w') as f:
            f.write("Per-class breakdown:\n")
            for cls in np.unique(true_labels_array):
                cls_mask = (true_labels_array == cls)
                n_cls = cls_mask.sum()
                n_correct_cls = (correct_mask & cls_mask).sum()

                f.write(f"\nClass {cls}:\n")
                f.write(f"Total samples: {n_cls}\n")
                f.write(f"Correctly classified: {n_correct_cls}\n")
                f.write(f"Class accuracy: {n_correct_cls/n_cls:.4f}\n")

                if n_cls - n_correct_cls > 0:
                    # Find misclassified examples
                    misclassified_mask = (~correct_mask & cls_mask)
                    mis_predictions = pred_labels[misclassified_mask]
                    unique_mispred, counts = np.unique(mis_predictions, return_counts=True)

                    f.write("\nMisclassified as:\n")
                    for pred_cls, count in zip(unique_mispred, counts):
                        f.write(f"{pred_cls}: {count}\n")

                    # Save examples of misclassified instances
                    f.write("\nSample misclassified instances:\n")
                    misclassified_indices = np.where(misclassified_mask)[0]
                    for idx in misclassified_indices[:5]:  # Show up to 5 examples
                        f.write(f"\nInstance {idx}:\n")
                        f.write(f"True class: {true_labels_array[idx]}\n")
                        f.write(f"Predicted class: {pred_labels[idx]}\n")
                        f.write("Feature values:\n")
                        for col, val in X.iloc[idx].items():
                            f.write(f"{col}: {val}\n")

        print("\033[K" +f"Detailed analysis saved to {analysis_file}")

#------------------------------------------------------------End of PP code ---------------------------------------------------
    def _compute_pairwise_likelihood(self, dataset, labels, feature_dims):
        """Compute pairwise likelihood PDFs"""
        unique_classes = torch.unique(labels)
        feature_pairs = list(combinations(range(feature_dims), 2))
        likelihood_pdfs = {}

        for class_id in unique_classes:
            class_mask = (labels == class_id)
            class_data = dataset[class_mask]
            likelihood_pdfs[class_id.item()] = {}

            for feat_i, feat_j in feature_pairs:
                pair_data = torch.stack([
                    class_data[:, feat_i],
                    class_data[:, feat_j]
                ], dim=1)

                mean = torch.mean(pair_data, dim=0)
                centered_data = pair_data - mean
                cov = torch.mm(centered_data.T, centered_data) / (len(pair_data) - 1)
                cov = cov + torch.eye(2) * 1e-6

                likelihood_pdfs[class_id.item()][(feat_i, feat_j)] = {
                    'mean': mean,
                    'cov': cov
                }

        return likelihood_pdfs


    def _get_weights_filename(self):
        """Get the filename for saving/loading weights"""
        return os.path.join('Model', f'Best_{self.model_type}_{self.dataset_name}_weights.json')

    def _get_encoders_filename(self):
        """Get the filename for saving/loading categorical encoders"""
        return os.path.join('Model', f'Best_{self.model_type}_{self.dataset_name}_encoders.json')




    def _remove_high_cardinality_columns(self, df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """Remove high cardinality columns and round features with detailed debugging"""
        DEBUG.log(f" Starting high cardinality removal with threshold {threshold}")
        DEBUG.log(f" Initial columns: {df.columns.tolist()}")

        # Round all features first if cardinality_tolerance is not -1
        if self.cardinality_tolerance != -1:
            DEBUG.log(f" Rounding features with tolerance {self.cardinality_tolerance}")
            df_rounded = self._round_features(df)
        else:
            df_rounded = df.copy()

        df_filtered = df_rounded.copy()
        columns_to_drop = []

        for column in df.columns:
            if column == self.target_column:
                continue

            # Use rounded data for cardinality check
            unique_count = len(df_rounded[column].unique())
            unique_ratio = unique_count / len(df)
            DEBUG.log(f" Column {column}: {unique_count} unique values, ratio {unique_ratio:.4f}")

            if unique_ratio > threshold:
                columns_to_drop.append(column)
                DEBUG.log(f" Marking {column} for removal (ratio {unique_ratio:.4f} > {threshold})")

        if columns_to_drop:
            DEBUG.log(f" Dropping columns: {columns_to_drop}")
            df_filtered = df_filtered.drop(columns=columns_to_drop)

        DEBUG.log(f" Columns after filtering: {df_filtered.columns.tolist()}")
        DEBUG.log(f" Remaining features: {len(df_filtered.columns)}")

        if len(df_filtered.columns) == 0:
            print("\033[K" +"[WARNING] All features were removed! Reverting to original features with warnings.")
            return df.copy()

        return df_filtered

    def _detect_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect categorical columns with improved debugging"""
        DEBUG.log(" Starting categorical column detection")
        categorical_columns = []

        for column in df.columns:
            if column != self.target_column:
                dtype = df[column].dtype
                unique_count = len(df[column].unique())
                DEBUG.log(f" Column {column}: dtype={dtype}, unique values={unique_count}")

                if dtype == 'object' or dtype.name == 'category':
                    DEBUG.log(f" Adding {column} as categorical (object/category type)")
                    categorical_columns.append(column)
                elif dtype in ['int64', 'float64']:
                    threshold = min(50, len(df) * 0.05)
                    if unique_count < threshold:
                        DEBUG.log(f" Adding {column} as categorical (numeric with few unique values: {unique_count} < {threshold})")
                        categorical_columns.append(column)
                    else:
                        DEBUG.log(f" Keeping {column} as numeric (unique values: {unique_count} >= {threshold})")

        DEBUG.log(f" Detected categorical columns: {categorical_columns}")
        return categorical_columns

    def _get_train_test_split(self, X_tensor, y_tensor):
        """Get or create consistent train-test split using smaller chunks to avoid memory issues."""
        dataset_folder = os.path.splitext(os.path.basename(self.dataset_name))[0]
        base_path = self.config.get('training_params', {}).get('training_save_path', 'training_data')
        split_path = os.path.join(base_path, dataset_folder, 'train_test_split.pkl')

        if os.path.exists(split_path):
            with open(split_path, 'rb') as f:
                split_indices = pickle.load(f)
                train_idx, test_idx = split_indices['train'], split_indices['test']
                return (X_tensor[train_idx], X_tensor[test_idx],
                        y_tensor[train_idx], y_tensor[test_idx])

        # If no saved split exists, create one using smaller chunks
        X_cpu = X_tensor.cpu().numpy()  # Move data to CPU for sklearn compatibility
        y_cpu = y_tensor.cpu().numpy()

        # Perform train-test split on CPU in smaller chunks
        X_train, X_test, y_train, y_test = train_test_split(
            X_cpu,  # Use NumPy array for sklearn compatibility
            y_cpu,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=(self.shuffle_state != -1)
        )

        # Convert back to PyTorch tensors and move to the appropriate device
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)

        # Save split indices to avoid recomputing
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        split_indices = {
            'train': torch.arange(len(X_train)),  # Indices for training set
            'test': torch.arange(len(X_train), len(X_train) + len(X_test))  # Indices for test set
        }
        with open(split_path, 'wb') as f:
            pickle.dump(split_indices, f)

        # Store test indices for later use
        self.test_indices = split_indices['test']

        return X_train, X_test, y_train, y_test

    def _train_test_split_tensor(self, X, y, test_size, random_state):
        """Split data consistently using fixed indices"""
        num_samples = len(X)

        # Generate fixed permutation
        if random_state == -1:
            # Use numpy's random permutation directly
            indices = torch.from_numpy(np.random.permutation(num_samples))
        else:
            rng = np.random.RandomState(random_state)
            indices = torch.from_numpy(rng.permutation(num_samples))

        split = int(num_samples * (1 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def _multivariate_normal_pdf(self, x, mean, cov):
        """Compute multivariate normal PDF"""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        dim = 2
        centered_x = x - mean.unsqueeze(0)
        inv_cov = torch.inverse(cov)
        det = torch.det(cov)
        quad_form = torch.sum(torch.mm(centered_x, inv_cov) * centered_x, dim=1)
        norm_const = 1.0 / (torch.sqrt((2 * torch.tensor(np.pi)) ** dim * det))
        return norm_const * torch.exp(-0.5 * quad_form)

    def _initialize_priors(self):
        """Initialize weights"""
        if self.best_W is not None:
            return self.best_W

        W = {}
        for class_id in self.likelihood_pdfs.keys():
            W[class_id] = {}
            for feature_pair in self.likelihood_pdfs[class_id].keys():
                W[class_id][feature_pair] = torch.tensor(0.1, dtype=torch.float32)
        return W

    def compute_posterior(self, feature_data, class_id=None, epsilon=1e-10):
        """Compute posterior probabilities"""
        classes = list(self.likelihood_pdfs.keys())
        n_classes = len(classes)
        feature_pairs = list(self.likelihood_pdfs[classes[0]].keys())
        log_likelihoods = torch.zeros(n_classes, dtype=torch.float32)

        for idx, c_id in enumerate(classes):
            class_log_likelihood = 0.0

            for feat_i, feat_j in feature_pairs:
                pair_data = torch.tensor([
                    feature_data[feat_i].item(),
                    feature_data[feat_j].item()
                ], dtype=torch.float32).reshape(1, 2)

                pdf_params = self.likelihood_pdfs[c_id][(feat_i, feat_j)]
                pair_likelihood = self._multivariate_normal_pdf(
                    pair_data,
                    pdf_params['mean'],
                    pdf_params['cov']
                ).squeeze()

                prior = self.current_W[c_id][(feat_i, feat_j)].item()
                likelihood_term = (pair_likelihood * prior + epsilon).item()
                class_log_likelihood += torch.log(torch.tensor(likelihood_term))

            log_likelihoods[idx] = class_log_likelihood

        max_log_likelihood = torch.max(log_likelihoods)
        likelihoods = torch.exp(log_likelihoods - max_log_likelihood)
        posteriors = likelihoods / (likelihoods.sum() + epsilon)

        return {c_id: posteriors[idx].item() for idx, c_id in enumerate(classes)}


    def fit_predict(self, batch_size: int = 128, save_path: str = None):
        """Full training and prediction pipeline with GPU optimization and optional prediction saving"""
        try:
            # Set a flag to indicate we're printing metrics
            self._last_metrics_printed = True

            # If this is a fresh training round, reset to the best round's initial conditions
            if self.best_round_initial_conditions is not None:
                print("\033[K" + "Starting fresh training with best round's initial conditions", end='\r', flush=True)
                self.current_W = self.best_round_initial_conditions['weights'].clone()
                self.likelihood_params = self.best_round_initial_conditions['likelihood_params']
                self.feature_pairs = self.best_round_initial_conditions['feature_pairs']
                self.bin_edges = self.best_round_initial_conditions['bin_edges']
                self.gaussian_params = self.best_round_initial_conditions['gaussian_params']

            # Handle data preparation based on whether we're in adaptive training or final evaluation
            if self.in_adaptive_fit:
                if not hasattr(self, 'X_tensor') or not hasattr(self, 'y_tensor'):
                    raise ValueError("X_tensor or y_tensor not found. Initialize them in adaptive_fit_predict first.")

                if not hasattr(self, 'train_indices') or not hasattr(self, 'test_indices'):
                    raise ValueError("train_indices or test_indices not found")

                # Use stored tensors and indices, but verify sizes match
                try:
                    X_train = self.X_tensor[self.train_indices]
                    X_test = self.X_tensor[self.test_indices]
                    y_train = self.y_tensor[self.train_indices]
                    y_test = self.y_tensor[self.test_indices]
                except Exception as e:
                    # If there's any issue with indices, fall back to regular training path
                    DEBUG.log(f"Error using stored indices: {str(e)}. Falling back to regular training.")
                    self.in_adaptive_fit = False
                    # Reset indices and proceed with regular path
                    self.train_indices = None
                    self.test_indices = None
                    return self.fit_predict(batch_size=batch_size, save_path=save_path)

            else:
                # Regular training path
                X = self.data.drop(columns=[self.target_column])
                y = self.data[self.target_column]

                # Check if label encoder is already fitted
                if not hasattr(self.label_encoder, 'classes_'):
                    y_encoded = self.label_encoder.fit_transform(y)
                else:
                    y_encoded = self.label_encoder.transform(y)

                # Preprocess features including categorical encoding
                X_processed = self._preprocess_data(X, is_training=True)

                # Convert to tensors and move to device
                X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)
                y_tensor = torch.LongTensor(y_encoded).to(self.device)

                # Split data
                # Get consistent train-test split
                X_train, X_test, y_train, y_test = self._get_train_test_split(
                    X_tensor, y_tensor)

                # Convert split data back to tensors
                X_train = torch.from_numpy(X_train).to(self.device, dtype=torch.float32)
                X_test = torch.from_numpy(X_test).to(self.device, dtype=torch.float32)
                y_train = torch.from_numpy(y_train).to(self.device, dtype=torch.long)
                y_test = torch.from_numpy(y_test).to(self.device, dtype=torch.long)

            # Verify tensor sizes match before training
            if X_train.size(0) != y_train.size(0) or X_test.size(0) != y_test.size(0):
                raise ValueError(f"Tensor size mismatch. X_train: {X_train.size(0)}, y_train: {y_train.size(0)}, "
                               f"X_test: {X_test.size(0)}, y_test: {y_test.size(0)}")

            # Train model
            final_W, error_rates = self.train(X_train, y_train, X_test, y_test, batch_size=batch_size)

            # Save categorical encoders
            self._save_categorical_encoders()

            # Make predictions on the entire dataset
            print("\033[K" + f"{Colors.YELLOW}Generating predictions for the entire dataset{Colors.ENDC}", end='\r', flush=True)
            X_all = torch.cat([X_train, X_test], dim=0)
            y_all = torch.cat([y_train, y_test], dim=0)
            all_predictions = self.predict(X_all, batch_size=batch_size)
            # Ensure all tensors are on the same device (GPU)
            all_pr = all_predictions.to(self.device)  # Move predictions to GPU
            y_all_pr = y_all.to(self.device)  # Ensure y_all is on GPU (though it likely already is)

            combined_accuracy = len(X_all[y_all_pr == all_pr]) / len(X_all)
            if combined_accuracy > self.best_combined_accuracy:
                print("\033[K" + f"{Colors.RED}---------------------------------------------------------------------------------------{Colors.ENDC}")
                print("\033[K" +  f"{Colors.GREEN}The best combined accuracy has improved from {self.best_combined_accuracy} to {combined_accuracy}{Colors.ENDC}")
                print("\033[K" +  f"{Colors.RED}---------------------------------------------------------------------------------------{Colors.ENDC}")
                self.best_combined_accuracy = combined_accuracy
                self._save_model_components()
                self._save_best_weights()


            # Extract predictions for training and test data using stored indices
            y_train_pred = all_predictions[:len(y_train)]  # Predictions for training data
            y_test_pred = all_predictions[len(y_train):]   # Predictions for test data

           # Generate detailed predictions for the entire dataset
            print("\033[K" + "Computing detailed predictions for the whole data", end='\r', flush=True)
            all_results = self._generate_detailed_predictions(self.X_Orig, all_predictions, y_all)
            train_results = all_results.iloc[self.train_indices]
            test_results = all_results.iloc[self.test_indices]
            # Filter failed examples (where predicted class != true class)
            failed_examples = all_results[all_results['predicted_class'] != all_results['true_class']]
            # Filter passed examples (where predicted class == true class)
            passed_examples = all_results[all_results['predicted_class'] == all_results['true_class']]

            # Save results if path is provided
            if save_path:

                # Save training predictions
                print("\033[K" + "Saving Train predictions", end='\r', flush=True)
                train_results.to_csv(f"{save_path}/train_predictions.csv", index=False)

                # Save test predictions
                print("\033[K" + "Saving Test predictions", end='\r', flush=True)
                test_results.to_csv(f"{save_path}/test_predictions.csv", index=False)

                # Save all predictions
                print("\033[K" + "Saving Combined predictions", end='\r', flush=True)
                all_results.to_csv(f"{save_path}/combined_predictions.csv", index=False)

                # Save failed examples
                print("\033[K" + "Saving Failed examples", end='\r', flush=True)
                failed_examples.to_csv(f"{save_path}/failed_examples.csv", index=False)

                # Save passed examples
                print("\033[K" + "Saving Passed examples", end='\r', flush=True)
                passed_examples.to_csv(f"{save_path}/passed_examples.csv", index=False)

                # Save metadata
                print("\033[K" + "Saving Metadata", end='\r', flush=True)
                metadata = {
                    'rejected_columns': self.high_cardinality_columns,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'preprocessing_details': {
                        'cardinality_threshold': self.cardinality_threshold,
                        'cardinality_tolerance': self.cardinality_tolerance,
                        'categorical_columns': list(self.categorical_encoders.keys())
                    }
                }
                with open(f"{save_path}/metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=4)

            # Calculate metrics for test set
            y_test_cpu = y_test.cpu().numpy()
            y_train_cpu = y_train.cpu().numpy()

            # Convert numerical labels back to original classes
            y_test_labels = self.label_encoder.inverse_transform(y_test_cpu)
            y_train_labels = self.label_encoder.inverse_transform(y_train_cpu)
            y_test_pred_labels = self.label_encoder.inverse_transform(y_test_pred)
            y_train_pred_labels = self.label_encoder.inverse_transform(y_train_pred)

            # Prepare results
            print("\033[K" + "Preparing results of training", end='\r', flush=True)
            results = {
                'all_predictions': all_results,
                'train_predictions': train_results,
                'test_predictions': test_results,
                'metadata': metadata,
                'classification_report': classification_report(y_test_labels, y_test_pred_labels),
                'confusion_matrix': confusion_matrix(y_test_labels, y_test_pred_labels),
                'error_rates': error_rates,
                'test_accuracy': (np.asarray(y_test_pred) ==np.asarray(y_test_cpu)).astype(float).mean().item(),  # Convert to float before mean
                'train_accuracy': (np.asarray(y_train_pred) == np.asarray(y_train_cpu)).astype(float).mean().item()  # Convert to float before mean
            }

            # Generate point-colored confusion matrices for train, test, and combined data
            print("\033[K" + f"{Colors.BOLD}Generating Confusion Matrices:{Colors.ENDC}", end='\r', flush=True)

            # Confusion matrix for training data
            self.print_colored_confusion_matrix(y_train_cpu, y_train_pred, header="Training Data")

            # Confusion matrix for test data
            self.print_colored_confusion_matrix(y_test_cpu, y_test_pred, header="Test Data")

            # Confusion matrix for combined data
            y_all_cpu = y_all.cpu().numpy()
            self.print_colored_confusion_matrix(y_all_cpu, all_predictions.cpu().numpy(), header="Combined Data")

            return results

        except Exception as e:
            DEBUG.log(f"Error in fit_predict: {str(e)}")
            DEBUG.log(f"Traceback: {traceback.format_exc()}")
            raise

    def _get_model_components_filename(self):
        """Get filename for model components"""
        return os.path.join('Model', f'Best{self.model_type}_{self.dataset_name}_components.pkl')
#----------------Handling categorical variables across sessions -------------------------
    def _save_categorical_encoders(self):
        """Save categorical feature encoders"""
        if self.categorical_encoders:
            # Create a serializable dictionary structure
            encoders_dict = {
                'encoders': {
                    column: {
                        str(k): v for k, v in mapping.items()
                    } for column, mapping in self.categorical_encoders.items()
                }
            }

            # Add metadata
            if hasattr(self, 'original_columns'):
                if isinstance(self.original_columns, list):
                    column_types = {col: str(self.data[col].dtype) for col in self.original_columns if col in self.data.columns}
                else:
                    column_types = {col: str(dtype) for col, dtype in self.original_columns.items()}

                encoders_dict['metadata'] = {
                    'column_types': column_types,
                    'timestamp': pd.Timestamp.now().isoformat()
                }

            with open(self._get_encoders_filename(), 'w') as f:
                json.dump(encoders_dict, f, indent=2)

    def _load_categorical_encoders(self):
        """Load categorical feature encoders from file"""
        encoders_file = self._get_encoders_filename()
        if os.path.exists(encoders_file):
            try:
                with open(encoders_file, 'r') as f:
                    data = json.load(f)

                # Extract encoders from the loaded data
                if 'encoders' in data:
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data['encoders'].items()
                    }
                else:
                    # Handle legacy format where encoders were at top level
                    self.categorical_encoders = {
                        column: {
                            k: int(v) if isinstance(v, (str, int, float)) else v
                            for k, v in mapping.items()
                        }
                        for column, mapping in data.items()
                    }

                print("\033[K" +f"Loaded categorical encoders from {encoders_file}", end="\r", flush=True)
            except Exception as e:
                print("\033[K" +f"Warning: Failed to load categorical encoders: {str(e)}")
                self.categorical_encoders = {}

    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True):
        """Encode categorical features with proper dtype handling"""
        DEBUG.log("Starting categorical encoding")
        df_encoded = df.copy()
        categorical_columns = self._detect_categorical_columns(df)

        for column in categorical_columns:
            if is_training:
                if column not in self.categorical_encoders:
                    # Create new encoder
                    unique_values = df[column].fillna('MISSING').unique()
                    self.categorical_encoders[column] = {
                        value: idx for idx, value in enumerate(unique_values)
                    }

            if column not in self.categorical_encoders:
                continue

            # Get original dtype
            original_dtype = df[column].dtype
            mapping = self.categorical_encoders[column]

            # Handle missing values and new categories
            df_encoded[column] = df[column].fillna('MISSING').map(
                lambda x: mapping.get(x, -1)
            )

            # Handle unmapped values
            unmapped = df_encoded[df_encoded[column] == -1].index
            if len(unmapped) > 0:
                DEBUG.log(f"Found {len(unmapped)} unmapped values in column {column}")

                # Calculate mean value
                mapped_values = [v for v in mapping.values() if isinstance(v, (int, float))]
                if mapped_values:
                    mean_value = float(np.mean(mapped_values))

                    # Convert to proper dtype based on original column type
                    if pd.api.types.is_integer_dtype(original_dtype):
                        mean_value = int(round(mean_value))

                    # Update unmapped values with proper type casting
                    df_encoded.loc[unmapped, column] = pd.Series([mean_value] * len(unmapped), index=unmapped).astype(original_dtype)

        # Verify no categorical columns remain
        remaining_object_cols = df_encoded.select_dtypes(include=['object']).columns
        if len(remaining_object_cols) > 0:
            DEBUG.log(f"Remaining object columns after encoding: {remaining_object_cols}")
            # Convert any remaining object columns to numeric
            for col in remaining_object_cols:
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)

        DEBUG.log(f"Categorical encoding complete. Shape: {df_encoded.shape}")
        return df_encoded

    def save_predictions(self, X: pd.DataFrame, predictions: torch.Tensor, output_file: str, true_labels: pd.Series = None):
        """Save predictions with proper class handling and probability computation"""
        predictions = predictions.cpu()

        # Create a copy of the original dataset to preserve all columns
        result_df = X.copy()

        # Convert predictions to original class labels
        pred_labels = self.label_encoder.inverse_transform(predictions.numpy())
        result_df['predicted_class'] = pred_labels

        if true_labels is not None:
            result_df['true_class'] = true_labels

        # Get preprocessed features for probability computation
        X_processed = self._preprocess_data(X, is_training=False)
        if isinstance(X_processed, torch.Tensor):
            X_tensor = X_processed.clone().detach().to(self.device)
        else:
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)

        # Compute probabilities in batches
        batch_size = 128
        all_probabilities = []

        for i in range(0, len(X_tensor), batch_size):
            batch_end = min(i + batch_size, len(X_tensor))
            batch_X = X_tensor[i:batch_end]

            try:
                if self.model_type == "Histogram":
                    batch_probs, _ = self._compute_batch_posterior(batch_X)
                elif self.model_type == "Gaussian":
                    batch_probs, _ = self._compute_batch_posterior_std(batch_X)
                else:
                    raise ValueError(f"{self.model_type} is invalid")

                all_probabilities.append(batch_probs.cpu().numpy())

            except Exception as e:
                print("\033[K" +f"Error computing probabilities for batch {i}: {str(e)}")
                return None

        if all_probabilities:
            all_probabilities = np.vstack(all_probabilities)
        else:
            print("\033[K" +"No probabilities were computed successfully", end="\r", flush=True)
            return None

        # Ensure we're only using valid class indices
        valid_classes = self.label_encoder.classes_
        n_classes = len(valid_classes)

        # Verify probability array shape matches number of classes
        if all_probabilities.shape[1] != n_classes:
            print("\033[K" +f"Warning: Probability array shape ({all_probabilities.shape}) doesn't match number of classes ({n_classes})")
            # Adjust probabilities array if necessary
            if all_probabilities.shape[1] > n_classes:
                all_probabilities = all_probabilities[:, :n_classes]
            else:
                # Pad with zeros if needed
                pad_width = ((0, 0), (0, n_classes - all_probabilities.shape[1]))
                all_probabilities = np.pad(all_probabilities, pad_width, mode='constant')

        # Add probability columns for each valid class
        for i, class_name in enumerate(valid_classes):
            if i < all_probabilities.shape[1]:  # Safety check
                result_df[f'prob_{class_name}'] = all_probabilities[:, i]

        # Add maximum probability
        result_df['max_probability'] = all_probabilities.max(axis=1)

        # Create the output directory if it doesn't exist
        dataset_name = os.path.splitext(os.path.basename(self.dataset_name))[0]
        output_dir = os.path.join('data', dataset_name, 'Predictions')
        os.makedirs(output_dir, exist_ok=True)

        # Save the predictions file in the new directory
        output_path = os.path.join(output_dir, output_file)
        result_df.to_csv(output_path, index=False)
        print("\033[K" +f"Saved predictions to {output_path}", end="\r", flush=True)

        if true_labels is not None:
            # Verification analysis
            self.verify_classifications(X, true_labels, predictions)

        return result_df
#--------------------------------------------------------------------------------------------------------------

    def _save_model_components(self):
        """Save all model components to a pickle file"""
        components = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'likelihood_params': self.likelihood_params,
            'feature_pairs': self.feature_pairs,
            'categorical_encoders': self.categorical_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'target_classes': self.label_encoder.classes_,
            'target_mapping': dict(zip(self.label_encoder.classes_,
                                     range(len(self.label_encoder.classes_)))),
            'config': self.config,
            'high_cardinality_columns': getattr(self, 'high_cardinality_columns', []),
            'original_columns': getattr(self, 'original_columns', None),
            'best_error': self.best_error,  # Explicitly save best error
            'last_training_loss': getattr(self, 'last_training_loss', float('inf')),
            'weight_updater': self.weight_updater,
            'n_bins_per_dim': self.n_bins_per_dim,
            'bin_edges': self.bin_edges,  # Save bin_edges
            'gaussian_params': self.gaussian_params  # Save gaussian_params
        }

        # Get the filename using existing method
        components_file = self._get_model_components_filename()

        # Ensure directory exists
        os.makedirs(os.path.dirname(components_file), exist_ok=True)

        # Save components to file
        with open(components_file, 'wb') as f:
            pickle.dump(components, f)

        print("\033[K" +f"Saved model components to {components_file}")
        print("\033[K" +f"[DEBUG] File size after save: {os.path.getsize(components_file)} bytes", end="\r", flush=True)
        return True


    def _load_model_components(self):
        """Load all model components"""
        components_file = self._get_model_components_filename()
        if os.path.exists(components_file):
            print("\033[K" +f"[DEBUG] Loading model components from {components_file}", end="\r", flush=True)
            print("\033[K" +f"[DEBUG] File size: {os.path.getsize(components_file)} bytes", end="\r", flush=True)
            with open(components_file, 'rb') as f:
                components = pickle.load(f)
                self.label_encoder.classes_ = components['target_classes']
                self.scaler = components['scaler']
                self.label_encoder = components['label_encoder']
                self.likelihood_params = components['likelihood_params']
                self.feature_pairs = components['feature_pairs']
                self.feature_columns = components.get('feature_columns')
                self.categorical_encoders = components['categorical_encoders']
                self.high_cardinality_columns = components.get('high_cardinality_columns', [])
                self.weight_updater = components.get('weight_updater')
                self.n_bins_per_dim = components.get('n_bins_per_dim', 21)
                self.bin_edges = components.get('bin_edges')  # Load bin_edges
                self.gaussian_params = components.get('gaussian_params')  # Load gaussian_params
                print("\033[K" +f"Loaded model components from {components_file}", end="\r", flush=True)
                return True
        else:
            print("\033[K" +f"[DEBUG] Model components file not found: {components_file}", end="\r", flush=True)
        return False



    def predict_and_save(self, save_path=None, batch_size: int = 128):
        """Make predictions on data and save them using best model weights"""
        try:
            # First try to load existing model and components
            weights_loaded = os.path.exists(self._get_weights_filename())
            components_loaded = self._load_model_components()

            if not (weights_loaded and components_loaded):
                print("\033[K" +"Complete model not found. Training required.", end="\r", flush=True)
                results = self.fit_predict(batch_size=batch_size)
                return results

            # Load the model weights and encoders
            self._load_best_weights()
            self._load_categorical_encoders()

            # Explicitly use best weights for prediction
            if self.best_W is None:
                print("\033[K" +"No best weights found. Training required.", end="\r", flush=True)
                results = self.fit_predict(batch_size=batch_size)
                return results

            # Store current weights temporarily
            temp_W = self.current_W

            # Use best weights for prediction
            self.current_W = self.best_W.clone()

            try:
                # Load and preprocess input data
                X = self.data.drop(columns=[self.target_column])
                true_labels = self.data[self.target_column]

                # Preprocess the data using the existing method
                X_tensor = self._preprocess_data(X, is_training=False)

                # Make predictions
                print("\033[K" +f"{Colors.BLUE}Predctions for saving data{Colors.ENDC}", end="\r", flush=True)
                predictions = self.predict(X_tensor, batch_size=batch_size)

                # Save predictions and metrics
                if save_path:
                    self.save_predictions(X, predictions, save_path, true_labels)

                return predictions
            finally:
                # Restore current weights
                self.current_W = temp_W

        except Exception as e:
            print("\033[K" +f"Error during prediction process: {str(e)}")
            print("\033[K" +"Falling back to training pipeline...")
            history = self.adaptive_fit_predict(max_rounds=self.max_epochs, batch_size=batch_size)
            results = self.fit_predict(batch_size=batch_size)
            return results



#--------------------------------------------------Class Ends ----------------------------------------------------------

def run_gpu_benchmark(dataset_name: str, model=None, batch_size: int = 128):
    """Run benchmark using GPU-optimized implementation"""
    print("\033[K" +f"Running GPU benchmark on {Colors.highlight_dataset(dataset_name)} dataset...", end="\r", flush=True)

    if Train:
        # First run adaptive training if enabled
        if EnableAdaptive:
            history = model.adaptive_fit_predict(max_rounds=model.max_epochs, batch_size=batch_size)

        # Skip fit_predict and just do predictions
        results = model.predict_and_save(
            save_path=f"data/{dataset_name}/{dataset_name}_predictions.csv",
            batch_size=batch_size
        )

        plot_training_progress(results['error_rates'], dataset_name)
        plot_confusion_matrix(
            results['confusion_matrix'],
            model.label_encoder.classes_,
            dataset_name
        )

        print("\033[K" +f"{Colors.BOLD}Classification Report for {Colors.highlight_dataset(dataset_name)}:{Colors.ENDC}")
        print("\033[K" +results['classification_report'])

    return model, results



def plot_training_progress(error_rates: List[float], dataset_name: str):
    """Plot training error rates over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title(f'Training Progress - {dataset_name.capitalize()} Dataset')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(confusion_mat: np.ndarray, class_names: np.ndarray, dataset_name: str):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {dataset_name.capitalize()} Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def save_label_encoder(label_encoder, dataset_name):
    save_dir = os.path.join('Model', f'Best_{dataset_name}')
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = os.path.join(save_dir, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print("\033[K" +f"Label encoder saved to {encoder_path}", end="\r", flush=True)

def load_label_encoder(dataset_name):
    encoder_path = os.path.join('Model', f'Best_{dataset_name}', 'label_encoder.pkl')
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("\033[K" +f"Label encoder loaded from {encoder_path}", end="\r", flush=True)
        return label_encoder
    else:
        raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")


def generate_test_datasets():
    """Generate XOR and 3D XOR test datasets"""
    # Generate 2D XOR
    with open('xor.csv', 'w') as f:
        f.write('x1,x2,target\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')
        f.write('0,0,0\n')
        f.write('0,1,1\n')
        f.write('1,0,1\n')
        f.write('1,1,0\n')

    # Generate 3D XOR
    with open('xor3d.csv', 'w') as f:
        f.write('x1,x2,x3,target\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')
        f.write('0,0,0,0\n')
        f.write('0,0,1,1\n')
        f.write('0,1,0,1\n')
        f.write('0,1,1,1\n')
        f.write('1,0,0,1\n')
        f.write('1,0,1,1\n')
        f.write('1,1,0,1\n')
        f.write('1,1,1,0\n')


class DebugLogger:
    def __init__(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def log(self, msg, force=False):
        """Only print if enabled or forced"""
        if self.enabled or force:
            print(msg)

# Create single global instance
DEBUG = DebugLogger()

def configure_debug(config):
    """Configure debug state from config"""
    debug_enabled = config.get('training_params', {}).get('debug_enabled', False)
    if debug_enabled:
        DEBUG.enable()
    else:
        DEBUG.disable()


#-------------------------------------------------------unit test ----------------------------------
import os
import glob
import json
from typing import List, Tuple
import pandas as pd
from datetime import datetime

import json
import os

def load_or_create_config(config_path: str = 'adaptive_dbnn.conf') -> dict:
    """
    Load the configuration file if it exists, or create a default one if it doesn't.
    Update global variables based on the configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    default_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trials": 100,
        "cardinality_threshold": 0.9,
        "cardinality_tolerance": 4,
        "learning_rate": 0.1,
        "random_seed": 42,
        "epochs": 1000,
        "test_fraction": 0.2,
        "train": True,
        "train_only": False,
        "predict": True,
        "gen_samples": False,
        "enable_adaptive": True,
        "nokbd": False,
        "display": None
    }

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        config = default_config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Created default configuration file at {config_path}")

    # Update global variables based on the configuration file
    global Train_device, Trials, cardinality_threshold, cardinality_tolerance, LearningRate, TrainingRandomSeed, Epochs, TestFraction, Train, Train_only, Predict, Gen_Samples, EnableAdaptive, nokbd, display

    # Update Train_device based on the compute_device setting in the configuration file
    Train_device = config.get("compute_device", "cuda" if torch.cuda.is_available() else "cpu")
    Trials = config.get("trials", 100)
    cardinality_threshold = config.get("cardinality_threshold", 0.9)
    cardinality_tolerance = config.get("cardinality_tolerance", 4)
    LearningRate = config.get("learning_rate", 0.1)
    TrainingRandomSeed = config.get("random_seed", 42)
    Epochs = config.get("epochs", 1000)
    TestFraction = config.get("test_fraction", 0.2)
    Train = config.get("train", True)
    Train_only = config.get("train_only", False)
    Predict = config.get("predict", True)
    Gen_Samples = config.get("gen_samples", False)
    EnableAdaptive = config.get("enable_adaptive", True)
    nokbd = config.get("nokbd", False)
    display = config.get("display", None)

    return config

def find_dataset_pairs(data_dir: str = 'data') -> List[Tuple[str, str, str]]:
    """
    Recursively find all matching .conf and .csv files in the data directory structure.

    Args:
        data_dir: Root directory to search for datasets

    Returns:
        List of tuples (basename, conf_path, csv_path)
    """
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        print("\033[K" +f"No '{data_dir}' directory found. Creating one...", end="\r", flush=True)
        os.makedirs(data_dir)
        return []

    dataset_pairs = []
    processed_datasets = set()

    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_dir):
        # Process configuration files
        conf_files = [f for f in files if f.endswith('.conf') and f != 'adaptive_dbnn.conf']

        for conf_file in conf_files:
            basename = os.path.splitext(conf_file)[0]
            if basename in processed_datasets:
                continue

            conf_path = os.path.join(root, conf_file)

            # Look for matching CSV in same directory and dataset-specific subdirectory
            csv_file = f"{basename}.csv"
            csv_paths = [
                os.path.join(root, csv_file),
                os.path.join(root, basename, csv_file)
            ]
           # Load adaptive_dbnn.conf if it exists
            adaptive_conf = {}
            adaptive_conf_path =f'{data_dir}/{basename}/adaptive_dbnn.conf'
            if os.path.exists(adaptive_conf_path):
                try:
                    with open(adaptive_conf_path, 'r') as f:
                        adaptive_conf = json.load(f)
                    print("\033[K" +f"Loaded adaptive configuration from {adaptive_conf_path}", end="\r", flush=True)
                except Exception as e:
                    print(f"Warning: Could not load adaptive configuration from{adaptive_conf_path}: {str(e)}")
            else:
                print("\033[K" +f"No adaptive_dbnn.conf found in working directory {adaptive_conf_path}", end="\r", flush=True)

            csv_path = None
            for path in csv_paths:
                if os.path.exists(path):
                    csv_path = path
                    break

            if csv_path:
                # Update configuration with adaptive settings if available
                if adaptive_conf:
                    try:
                        with open(conf_path, 'r') as f:
                            dataset_conf = json.load(f)

                        # Update execution flags from adaptive configuration
                        if 'execution_flags' in adaptive_conf:
                            dataset_conf['execution_flags'] = adaptive_conf['execution_flags']

                        # Update training parameters
                        if 'training_params' in adaptive_conf:
                            dataset_conf['training_params'].update(adaptive_conf['training_params'])

                        # Save updated configuration
                        with open(conf_path, 'w') as f:
                            json.dump(dataset_conf, f, indent=4)
                        print("\033[K" +f"Updated configuration for {basename} with adaptive settings", end="\r", flush=True)

                    except Exception as e:
                        print(f"Warning: Could not update configuration for {basename}: {str(e)}")

                dataset_pairs.append((basename, conf_path, csv_path))
                processed_datasets.add(basename)
                print("\033[K" +f"Found dataset pair:", end="\r", flush=True)
                print("\033[K" +f"  Config: {conf_path}", end="\r", flush=True)
                print("\033[K" +f"  Data  : {csv_path}", end="\r", flush=True)
            else:
                print("\033[K" +f"Warning: Config file {conf_file} exists but no matching CSV found")
                print("\033[K" +f"Looked in:")
                for path in csv_paths:
                    print("\033[K" +f"  - {path}")

    if not dataset_pairs:
        print("\033[K" +"No matching .conf and .csv file pairs found.")
        print("\033[K" +"Each dataset should have both a .conf configuration file and a matching .csv data file.")
        print("\033[K" +"Example: 'dataset1.conf' and 'dataset1.csv'")

    return dataset_pairs

def process_datasets():
    """Main function to process all datasets"""
    # Find all dataset pairs
    dataset_pairs = find_dataset_pairs()

    if not dataset_pairs:
        print("\033[K" +"No matching .conf and .csv file pairs found in the current directory.")
        return

    print("\033[K" +f"Found {len(dataset_pairs)} dataset pair(s)")

    # Process each dataset
    for basename, conf_path, csv_path in dataset_pairs:
        print("\033[K" +f"{'='*60}")
        print("\033[K" +f"Dataset: {basename}")
        print("\033[K" +f"Config file: {conf_path}")
        print("\033[K" +f"Data file: {csv_path}")
        print("\033[K" +'='*60)

        # Validate dataset name
        if not basename or not isinstance(basename, str):
            print("\033[K" +f"Invalid dataset name: {basename}. Skipping...")
            continue

        # Print dataset information
        print_dataset_info(conf_path, csv_path)

        # Ask user if they want to process this dataset
        response = input("\nProcess this dataset? (y/n): ").lower()
        if response == 'n':
            print("\033[K" +f"Skipping dataset: {basename}")
            continue

        # Process dataset
        print("\033[K" +f"Processing dataset: {basename}")
        try:
            # Create DBNN instance with specific dataset name
            model = DBNN(dataset_name=basename)

            # Optionally create an invertible model
            if model.config.get('enable_invertible', False):
                invertible_model = model.create_invertible_model(
                    reconstruction_weight=model.config.get('reconstruction_weight', 0.5),
                    feedback_strength=model.config.get('feedback_strength', 0.3)
                )
                print("\033[K" +"Created invertible DBNN model")

            start_time = datetime.now()
            results = model.process_dataset(conf_path)
            end_time = datetime.now()

            # Print results
            print("\033[K" +"Processing complete!")
            print("\033[K" +f"Time taken: {(end_time - start_time).total_seconds():.1f} seconds")
            print("\033[K" +f"Results saved to: {results['results_path']}")
            print("\033[K" +f"Training log saved to: {results['log_path']}")
            print("\033[K" +f"Processed {results['n_samples']} samples with {results['n_features']} features")
            print("\033[K" +f"Excluded {results['n_excluded']} features")
            # Save the label encoder after training
            model = DBNN(dataset_name=basename)
            save_label_encoder(model.label_encoder, basename)

        except FileNotFoundError as e:
            print("\033[K" +f"Dataset file not found: {e}")
            print("\033[K" +"Searching UCI repository for the dataset...")
            dataset_processor = DatasetProcessor()
            datasets = dataset_processor.search_uci_repository(basename)
            if datasets:
                print("\033[K" +f"Found {len(datasets)} matching datasets in UCI repository:")
                for i, dataset in enumerate(datasets):
                    print("\033[K" +f"{i+1}. {dataset['name']} - {dataset['url']}")
                choice = input("\nEnter the number of the dataset to download (or 'q' to quit): ")
                if choice.lower() != 'q':
                    try:
                        selected_dataset = datasets[int(choice) - 1]
                        print("\033[K" +f"Downloading dataset: {selected_dataset['name']}")
                        dataset_path = dataset_processor.download_uci_dataset(selected_dataset['name'], selected_dataset['url'])
                        print("\033[K" +f"Dataset downloaded to: {dataset_path}")
                        # Process the downloaded dataset
                        model = DBNN(dataset_name=selected_dataset['name'])
                        results = model.process_dataset(dataset_path)
                        print("\033[K" +"Processing complete!")
                        print("\033[K" +f"Results saved to: {results['results_path']}")
                    except Exception as e:
                        print("\033[K" +f"Error downloading or processing dataset: {str(e)}")
            else:
                print("\033[K" +f"No matching datasets found in UCI repository for: {basename}")

        except Exception as e:
            print("\033[K" +f"Error processing dataset {basename}:")
            print("\033[K" +f"Error details: {str(e)}")
            traceback.print_exc()

    print("\033[K" +"All datasets processed")


def validate_config(conf_path: str) -> bool:
    """
    Validate configuration file and ensure proper folder structure

    Args:
        conf_path: Path to configuration file

    Returns:
        True if valid, False otherwise
    """
    try:
        with open(conf_path, 'r') as f:
            config = json.load(f)

        # Check required fields
        required_fields = ['file_path', 'column_names', 'target_column']
        for field in required_fields:
            if field not in config:
                print("\033[K" +f"Missing required field: {field}")
                return False

        # Validate dataset directory exists
        basename = os.path.splitext(os.path.basename(conf_path))[0]
        dataset_dir = os.path.join('data', basename)
        if not os.path.exists(dataset_dir):
            print("\033[K" +f"Dataset directory missing: {dataset_dir}")
            create_dir = input("Create directory? (y/n): ").lower() == 'y'
            if create_dir:
                os.makedirs(dataset_dir)
                print("\033[K" +f"Created directory: {dataset_dir}")
            else:
                return False

        return True
    except json.JSONDecodeError:
        print("\033[K" +f"Invalid JSON in configuration file: {conf_path}")
        return False
    except Exception as e:
        print("\033[K" +f"Error validating configuration: {str(e)}")
        return False

def print_dataset_info(conf_path: str, csv_path: str):
    """Print information about the dataset with robust error handling"""
    try:
        # Load configuration
        with open(conf_path, 'r') as f:
            config = json.load(f)

        # Get file sizes
        conf_size = os.path.getsize(conf_path)
        csv_size = os.path.getsize(csv_path)

        print("\033[K" +"Dataset Information:")
        print("\033[K" +f"Dataset name: {os.path.splitext(os.path.basename(conf_path))[0]}")
        print("\033[K" +f"Configuration file: {conf_path} ({conf_size/1024:.1f} KB)")
        print("\033[K" +f"Data file: {csv_path} ({csv_size/1024:.1f} KB)")
        print("\033[K" +f"Model type: {config.get('modelType', 'Not specified')}")

        # Safely access configuration values

        target_column = config.get('target_column', 'target')  # Default to 'target' if not specified
        print("\033[K" +f"Target column: {target_column}")

        # Safely handle column names
        column_names = config.get('column_names', [])
        if column_names:
            print("\033[K" +f"Number of columns: {len(column_names)}")

            # Count excluded features
            excluded = sum(1 for col in column_names if str(col).startswith('#'))
            print("\033[K" +f"Excluded features: {excluded}")

            # Show first few column names
            print("\033[K" +"Features:")
            for col in column_names[:5]:
                excluded = "  (excluded)" if str(col).startswith('#') else ""
                print("\033[K" +f"  {col}{excluded}")
            if len(column_names) > 5:
                print("\033[K" +f"  ... and {len(column_names)-5} more")
        else:
            # Try to get column info from CSV if no column names in config
            try:
                df = pd.read_csv(csv_path, nrows=0)
                columns = df.columns.tolist()
                print("\033[K" +f"Number of columns (from CSV): {len(columns)}")
                print("\033[K" +"Features (from CSV):")
                for col in columns[:5]:
                    print("\033[K" +f"  {col}")
                if len(columns) > 5:
                    print("\033[K" +f"  ... and {len(columns)-5} more")
            except Exception as e:
                print("\033[K" +"Could not read column information from CSV")

    except FileNotFoundError:
        print("\033[K" +f"Error: Could not find configuration file: {conf_path}")
    except json.JSONDecodeError:
        print("\033[K" +f"Error: Invalid JSON in configuration file: {conf_path}")
    except Exception as e:
        print("\033[K" +f"Error reading dataset info: {str(e)}")
        print("\033[K" +f"Traceback: {traceback.format_exc()}")


def main():
    parser = argparse.ArgumentParser(description='Process ML datasets')
    parser.add_argument("--file_path", nargs='?', help="Path to dataset file or folder")
    parser.add_argument('--mode', type=str, choices=['train', 'train_predict', 'invertDBNN'],
                        required=False, help="Mode to run the network: train, train_predict, or invertDBNN.")
    args = parser.parse_args()
    processor = DatasetProcessor()
    parser.print_help()
    if not args.file_path:
        parser.print_help()
        input("\nPress any key to search data folder for datasets (or Ctrl-C to exit)...")
        process_datasets()

    elif args.mode !="invertDBNN":
        processor.process_dataset(args.file_path)
        dataset_pairs = find_dataset_pairs()
        basename=args.file_path.split('/')[-1].split('.')[0]
        conf_path=os.path.join(f"data/{basename}/{basename}.conf")
        csv_path=os.path.join(f"data/{basename}/{basename}.csv")
        # Save the label encoder after training
       # Create DBNN instance with specific dataset name
        model = DBNN(dataset_name=basename)

        # Optionally create an invertible model
        if model.config.get('enable_invertible', False):
            invertible_model = model.create_invertible_model(
                reconstruction_weight=model.config.get('reconstruction_weight', 0.5),
                feedback_strength=model.config.get('feedback_strength', 0.3)
            )
            print("\033[K" +"Created invertible DBNN model")

        start_time = datetime.now()
        results = model.process_dataset(conf_path)
        end_time = datetime.now()

        # Print results
        print("\033[K" +"Processing complete!")
        print("\033[K" +f"Time taken: {(end_time - start_time).total_seconds():.1f} seconds")
        print("\033[K" +f"Results saved to: {results['results_path']}")
        print("\033[K" +f"Training log saved to: {results['log_path']}")
        print("\033[K" +f"Processed {results['n_samples']} samples with {results['n_features']} features")
        print("\033[K" +f"Excluded {results['n_excluded']} features")
        save_label_encoder(model.label_encoder, basename)

    elif args.mode =="invertDBNN":
        processor.process_dataset(args.file_path)
        dataset_pairs = find_dataset_pairs()
        basename=args.file_path.split('/')[-1].split('.')[0]
        conf_path=os.path.join(f"data/{basename}/{basename}.conf")
        csv_path=os.path.join(f"data/{basename}/{basename}.csv")
        # Invert DBNN mode
        model = DBNN(dataset_name=basename)
        model._load_model_components()
        # Load configuration
        with open(conf_path, 'r') as f:
            config_dict = json.load(f)

        print("\033[K" +"DEBUG: Inverse DBNN Settings:")
        for param in ['reconstruction_weight', 'feedback_strength', 'inverse_learning_rate']:
            value = config_dict.get('training_params', {}).get(param, 0.1)
            print("\033[K" +f"- {param}: {value}")

        print("\033[K" +"DEBUG: Initializing inverse model...")

        # Load the label encoder
        try:
            label_encoder = load_label_encoder(basename)
            model.label_encoder = label_encoder

            # Now you can safely access model.label_encoder.classes_
            #print("\033[K" +f"Classes in label encoder: {model.label_encoder.classes_}")

            # Proceed with inverse model initialization
            inverse_model = InvertibleDBNN(
                forward_model=model,
                feature_dims=model.data.shape[1] - 1,  # Exclude target column
                reconstruction_weight=config_dict['training_params'].get('reconstruction_weight', 0.5),
                feedback_strength=config_dict['training_params'].get('feedback_strength', 0.3)
            )

            # Reconstruct features
            X_test = model.data.drop(columns=[model.target_column])
            test_probs = model._get_test_probabilities(X_test)
            reconstruction_features = inverse_model.reconstruct_features(test_probs)

            # Save reconstructed features
            output_dir = os.path.join('data', basename, 'Predicted_features')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{basename}.csv')

            feature_columns = model.data.drop(columns=[model.target_column]).columns
            reconstructed_df = pd.DataFrame(reconstruction_features.cpu().numpy(), columns=feature_columns)
            reconstructed_df.to_csv(output_file, index=False)

            print("\033[K" +f"Reconstructed features saved to {output_file}")

        except FileNotFoundError as e:
            print("\033[K" +f"Error: {str(e)}")
            print("\033[K" +"Please ensure the model has been trained before using invertDBNN mode.")
            return

    else:
        print("\033[K" +"No datasets found in data folder")
if __name__ == "__main__":
    print("\033[K" +"DBNN Dataset Processor")
    print("\033[K" +"=" * 40)
    main()
