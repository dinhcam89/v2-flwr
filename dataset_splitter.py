import torch
import numpy as np
import random
import os
import json
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as transforms

class DatasetSplitter:
    """
    Class to handle loading a single dataset and splitting it across multiple clients
    for federated learning.
    """
    
    def __init__(self, dataset_type="california", num_clients=3, distribution="non-iid", seed=42):
        """
        Initialize the dataset splitter.
        
        Args:
            dataset_type: Type of dataset to use ('california', 'breast_cancer', 'diabetes', 'mnist', 'synthetic')
            num_clients: Number of clients to split the dataset for
            distribution: How to distribute the data ('iid' for random, 'non-iid' for stratified)
            seed: Random seed for reproducibility
        """
        self.dataset_type = dataset_type
        self.num_clients = num_clients
        self.distribution = distribution
        self.seed = seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Dictionary to store client datasets
        self.client_datasets = {}
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load the specified dataset and prepare it for splitting"""
        if self.dataset_type == "california":
            self._load_california()
        elif self.dataset_type == "breast_cancer":
            self._load_breast_cancer()
        elif self.dataset_type == "diabetes":
            self._load_diabetes()
        elif self.dataset_type == "mnist":
            self._load_mnist()
        elif self.dataset_type == "synthetic":
            self._create_synthetic()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
    def _load_california(self):
        """Load California Housing dataset"""
        X, y = fetch_california_housing(return_X_y=True)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Reshape y to match expected dimensions
        y = y.reshape(-1, 1)
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Store metadata
        self.input_dim = X.shape[1]
        self.output_dim = 1
        self.is_classification = False
        
    def _load_breast_cancer(self):
        """Load Breast Cancer dataset"""
        X, y = load_breast_cancer(return_X_y=True)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Reshape y to match expected dimensions
        y = y.reshape(-1, 1)
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Store metadata
        self.input_dim = X.shape[1]
        self.output_dim = 1
        self.is_classification = True
        
    def _load_diabetes(self):
        """Load Diabetes dataset"""
        X, y = load_diabetes(return_X_y=True)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Reshape y to match expected dimensions
        y = y.reshape(-1, 1)
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        # Store metadata
        self.input_dim = X.shape[1]
        self.output_dim = 1
        self.is_classification = False
        
    def _load_mnist(self):
        """Load MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST dataset
        mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Combine train and test for our own splitting
        self.mnist_train = mnist_train
        self.mnist_test = mnist_test
        
        # Store metadata
        self.input_dim = 28 * 28  # Flattened MNIST image
        self.output_dim = 10      # 10 classes
        self.is_classification = True
        
    def _create_synthetic(self):
        """Create synthetic dataset with non-linear relationships"""
        # Set dimensions
        num_samples = 3000  # Large enough to split among clients
        input_dim = 10
        output_dim = 1
        
        # Generate synthetic data with non-linear relationships
        X = torch.randn(num_samples, input_dim)
        
        # Create target with non-linear relationships
        y = torch.zeros(num_samples, output_dim)
        
        # Linear component
        w_linear = torch.randn(input_dim, output_dim)
        b_linear = torch.randn(output_dim)
        y += torch.matmul(X, w_linear) + b_linear
        
        # Add non-linear components
        # Quadratic terms
        y += 0.3 * X[:, 0].unsqueeze(1) * X[:, 1].unsqueeze(1)
        
        # Interaction terms
        y += 0.2 * torch.sin(X[:, 2]).unsqueeze(1) * torch.cos(X[:, 3]).unsqueeze(1)
        
        # Exponential terms
        y += 0.1 * torch.exp(torch.clamp(X[:, 4], -2, 2)).unsqueeze(1)
        
        # Add noise
        y += 0.2 * torch.randn(num_samples, output_dim)
        
        self.X = X
        self.y = y
        
        # Store metadata
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_classification = False
        
    def split_dataset(self):
        """
        Split the dataset across clients based on the specified distribution.
        
        Returns:
            Dictionary mapping client IDs to their dataset loaders
        """
        if self.dataset_type == "mnist":
            return self._split_mnist()
        else:
            return self._split_tabular()
            
    def _split_mnist(self):
        """Split MNIST dataset for multiple clients"""
        # Approach depends on distribution type
        if self.distribution == "iid":
            # Random split for IID distribution
            train_len = len(self.mnist_train)
            samples_per_client = train_len // self.num_clients
            
            # Create balanced splits
            train_splits = random_split(
                self.mnist_train,
                [samples_per_client] * (self.num_clients - 1) + [train_len - samples_per_client * (self.num_clients - 1)]
            )
            
            # Split test set similarly
            test_len = len(self.mnist_test)
            test_samples_per_client = test_len // self.num_clients
            test_splits = random_split(
                self.mnist_test,
                [test_samples_per_client] * (self.num_clients - 1) + [test_len - test_samples_per_client * (self.num_clients - 1)]
            )
            
            # Create dataloaders for each client
            for i in range(self.num_clients):
                train_loader = DataLoader(train_splits[i], batch_size=32, shuffle=True)
                test_loader = DataLoader(test_splits[i], batch_size=32, shuffle=False)
                
                self.client_datasets[f"client-{i+1}"] = {
                    "train_loader": train_loader,
                    "test_loader": test_loader,
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim
                }
                
        else:  # non-IID
            # Split by digit class
            targets = self.mnist_train.targets.numpy()
            
            # For 3 clients with 10 classes, distribute as follows:
            # Client 1: digits 0, 1, 2, 3
            # Client 2: digits 3, 4, 5, 6
            # Client 3: digits 6, 7, 8, 9
            # (Note the overlapping digits to simulate some shared knowledge)
            
            client_classes = [
                [0, 1, 2, 3],  # Client 1
                [3, 4, 5, 6],  # Client 2
                [6, 7, 8, 9]   # Client 3
            ]
            
            for i in range(self.num_clients):
                # Get indices for training data
                class_indices = np.where(np.isin(targets, client_classes[i]))[0]
                train_subset = Subset(self.mnist_train, class_indices)
                
                # Get indices for test data with the same classes
                test_targets = self.mnist_test.targets.numpy()
                test_indices = np.where(np.isin(test_targets, client_classes[i]))[0]
                test_subset = Subset(self.mnist_test, test_indices)
                
                # Create dataloaders
                train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
                
                self.client_datasets[f"client-{i+1}"] = {
                    "train_loader": train_loader,
                    "test_loader": test_loader,
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim,
                    "classes": client_classes[i]
                }
                
        # Return the created datasets
        return self.client_datasets
    
    def _split_tabular(self):
        """Split tabular dataset (California, Breast Cancer, Diabetes, Synthetic)"""
        # First, create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.seed
        )
        
        # Split based on distribution type
        if self.distribution == "iid":
            # Random, balanced split for IID
            train_len = len(X_train)
            samples_per_client = train_len // self.num_clients
            
            for i in range(self.num_clients):
                start_idx = i * samples_per_client
                end_idx = (i + 1) * samples_per_client if i < self.num_clients - 1 else train_len
                
                # Get client data
                client_X_train = X_train[start_idx:end_idx]
                client_y_train = y_train[start_idx:end_idx]
                
                # Get test data (also split for each client)
                test_len = len(X_test)
                test_samples_per_client = test_len // self.num_clients
                test_start = i * test_samples_per_client
                test_end = (i + 1) * test_samples_per_client if i < self.num_clients - 1 else test_len
                
                client_X_test = X_test[test_start:test_end]
                client_y_test = y_test[test_start:test_end]
                
                # Create datasets and loaders
                train_dataset = TensorDataset(client_X_train, client_y_train)
                test_dataset = TensorDataset(client_X_test, client_y_test)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                self.client_datasets[f"client-{i+1}"] = {
                    "train_loader": train_loader,
                    "test_loader": test_loader,
                    "input_dim": self.input_dim,
                    "output_dim": self.output_dim
                }
                
        else:  # non-IID
            # For regression tasks, sort by target value and split
            # For classification, stratify by class but with imbalanced distribution
            
            if self.is_classification:
                # Classification: Imbalanced class distribution
                unique_classes = torch.unique(y_train).numpy()
                
                # Distribute classes with some overlap
                if len(unique_classes) >= self.num_clients:
                    # Enough classes to distribute differently
                    classes_per_client = len(unique_classes) // self.num_clients + 1
                    
                    for i in range(self.num_clients):
                        # Select classes for this client (with overlap)
                        start_class = (i * (len(unique_classes) // self.num_clients))
                        client_classes = unique_classes[start_class:start_class + classes_per_client]
                        
                        # Get indices for these classes
                        train_indices = [j for j, y in enumerate(y_train) if y.item() in client_classes]
                        
                        # Create subset
                        client_X_train = X_train[train_indices]
                        client_y_train = y_train[train_indices]
                        
                        # Split test set similarly
                        test_indices = [j for j, y in enumerate(y_test) if y.item() in client_classes]
                        client_X_test = X_test[test_indices]
                        client_y_test = y_test[test_indices]
                        
                        # Create datasets and loaders
                        train_dataset = TensorDataset(client_X_train, client_y_train)
                        test_dataset = TensorDataset(client_X_test, client_y_test)
                        
                        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                        
                        self.client_datasets[f"client-{i+1}"] = {
                            "train_loader": train_loader,
                            "test_loader": test_loader,
                            "input_dim": self.input_dim,
                            "output_dim": self.output_dim,
                            "classes": client_classes.tolist()
                        }
                
                else:
                    # Too few classes, distribute with different class proportions
                    # This creates non-IID in terms of class balance rather than class distribution
                    for i in range(self.num_clients):
                        # Different sampling weights for each client
                        class_weights = np.random.dirichlet(np.ones(len(unique_classes)) * 0.5)
                        
                        # Sample according to these weights
                        train_indices = []
                        for class_idx, weight in enumerate(class_weights):
                            class_indices = [j for j, y in enumerate(y_train) if y.item() == unique_classes[class_idx]]
                            samples_to_take = int(len(class_indices) * weight * self.num_clients)
                            train_indices.extend(np.random.choice(class_indices, min(samples_to_take, len(class_indices)), replace=False))
                        
                        # Create subset
                        client_X_train = X_train[train_indices]
                        client_y_train = y_train[train_indices]
                        
                        # Use all test data for each client
                        client_X_test = X_test
                        client_y_test = y_test
                        
                        # Create datasets and loaders
                        train_dataset = TensorDataset(client_X_train, client_y_train)
                        test_dataset = TensorDataset(client_X_test, client_y_test)
                        
                        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                        
                        self.client_datasets[f"client-{i+1}"] = {
                            "train_loader": train_loader,
                            "test_loader": test_loader,
                            "input_dim": self.input_dim,
                            "output_dim": self.output_dim,
                            "class_weights": {str(unique_classes[idx]): w for idx, w in enumerate(class_weights)}
                        }
                        
            else:  # Regression
                # Sort by target value
                sorted_indices = torch.argsort(y_train.flatten())
                X_train_sorted = X_train[sorted_indices]
                y_train_sorted = y_train[sorted_indices]
                
                # Split into regions
                train_len = len(X_train_sorted)
                samples_per_client = train_len // self.num_clients
                
                for i in range(self.num_clients):
                    start_idx = i * samples_per_client
                    end_idx = (i + 1) * samples_per_client if i < self.num_clients - 1 else train_len
                    
                    # Get client data - contiguous regions of sorted targets
                    client_X_train = X_train_sorted[start_idx:end_idx]
                    client_y_train = y_train_sorted[start_idx:end_idx]
                    
                    # Use all test data for each client
                    client_X_test = X_test
                    client_y_test = y_test
                    
                    # Create datasets and loaders
                    train_dataset = TensorDataset(client_X_train, client_y_train)
                    test_dataset = TensorDataset(client_X_test, client_y_test)
                    
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                    
                    # Calculate statistics for this client's data
                    client_min = float(client_y_train.min())
                    client_max = float(client_y_train.max())
                    client_mean = float(client_y_train.mean())
                    
                    self.client_datasets[f"client-{i+1}"] = {
                        "train_loader": train_loader,
                        "test_loader": test_loader,
                        "input_dim": self.input_dim,
                        "output_dim": self.output_dim, 
                        "target_range": [client_min, client_max],
                        "target_mean": client_mean
                    }
                    
        return self.client_datasets
    
    def save_dataset_info(self, filepath="dataset_splits_info.json"):
        """Save information about the dataset splits for reference"""
        dataset_info = {
            "dataset_type": self.dataset_type,
            "distribution": self.distribution,
            "num_clients": self.num_clients,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "is_classification": self.is_classification,
            "clients": {}
        }
        
        # Add client-specific info (excluding actual data)
        for client_id, client_data in self.client_datasets.items():
            dataset_info["clients"][client_id] = {}
            
            # Add training size
            train_size = len(client_data["train_loader"].dataset)
            dataset_info["clients"][client_id]["train_size"] = train_size
            
            # Add test size
            test_size = len(client_data["test_loader"].dataset)
            dataset_info["clients"][client_id]["test_size"] = test_size
            
            # Add class distribution for classification tasks
            if self.is_classification:
                if "classes" in client_data:
                    dataset_info["clients"][client_id]["classes"] = client_data["classes"]
                if "class_weights" in client_data:
                    dataset_info["clients"][client_id]["class_weights"] = client_data["class_weights"]
            
            # Add target range for regression tasks
            if not self.is_classification and "target_range" in client_data:
                dataset_info["clients"][client_id]["target_range"] = client_data["target_range"]
                dataset_info["clients"][client_id]["target_mean"] = client_data["target_mean"]
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        return dataset_info