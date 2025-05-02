"""
Enhanced Federated Learning Client with GA-Stacking Ensemble optimization.
Extends the base client with local ensemble optimization capabilities.
"""

import os
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split


import random
import threading
from web3.exceptions import TransactionNotFound
from hexbytes import HexBytes
import pandas as pd

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import numpy as np

from ipfs_connector import IPFSConnector
from blockchain_connector import BlockchainConnector
from ga_stacking import GAStacking, EnsembleModel
from base_models import create_model_ensemble, get_ensemble_state_dict, load_ensemble_from_state_dict, create_model_ensemble_from_config, SklearnModelWrapper, MetaLearnerWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("FL-Client-Ensemble")


class GAStackingClient(fl.client.NumPyClient):
    """Federated Learning client with GA-Stacking ensemble optimization."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        ensemble_size: int = 5,
        ipfs_connector: Optional[IPFSConnector] = None,
        blockchain_connector: Optional[BlockchainConnector] = None,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        device: str = "cpu",
        client_id: str = None,
        ga_generations: int = 20,
        ga_population_size: int = 30
    ):
        """
        Initialize GA-Stacking client.
        
        Args:
            input_dim: Input dimension for models
            output_dim: Output dimension for models
            train_loader: Training data loader
            test_loader: Test data loader
            ensemble_size: Number of models in the ensemble
            ipfs_connector: IPFS connector
            blockchain_connector: Blockchain connector
            wallet_address: Client's wallet address
            private_key: Client's private key
            device: Device to use for computation
            client_id: Client identifier
            ga_generations: Number of GA generations to run
            ga_population_size: Size of GA population
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.ensemble_size = ensemble_size
        self.ipfs = ipfs_connector or IPFSConnector()
        self.blockchain = blockchain_connector
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.device = torch.device(device)
        self.client_id = client_id or f"client-{os.getpid()}"
        self.ga_generations = ga_generations
        self.ga_population_size = ga_population_size
        
        # Initialize ensemble models
        self.base_models, self.model_names = create_model_ensemble(
            input_dim=input_dim,
            output_dim=output_dim,
            ensemble_size=ensemble_size,
            device=device
        )
        
        # Initialize GA-Stacking optimizer
        self.ga_stacking = None
        self.ensemble_model = None
        
        # Metrics storage
        self.metrics_history = []
        
        logger.info(f"Initialized {self.client_id} with {ensemble_size} base models")
        logger.info(f"IPFS node: {self.ipfs.ipfs_api_url}")
        
        # Verify blockchain authentication if available
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if is_authorized:
                    logger.info(f"Client {self.wallet_address} is authorized on the blockchain ✅")
                else:
                    logger.warning(f"Client {self.wallet_address} is NOT authorized on the blockchain ❌")
                    logger.warning("The server may reject this client's contributions")
            except Exception as e:
                logger.error(f"Failed to verify client authorization: {e}")
    
    def split_train_val(self, validation_split: float = 0.2):
        """
        Split the training data into training and validation sets.
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Get the full dataset from the data loader
        train_dataset = self.train_loader.dataset
        
        # Calculate sizes
        val_size = int(len(train_dataset) * validation_split)
        train_size = len(train_dataset) - val_size
        
        # Split the dataset
        train_subset, val_subset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create new data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.train_loader.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.train_loader.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Get model parameters as a list of NumPy arrays.
        
        For ensemble models, we flatten the parameters of all models and their weights.
        The server will need to know how to reconstruct this.
        """
        if self.ensemble_model is None:
            # If ensemble not trained yet, use the first base model
            return [val.cpu().numpy() for _, val in self.base_models[0].state_dict().items()]
        
        # Get ensemble state including model weights and ensemble weights
        ensemble_state = get_ensemble_state_dict(self.ensemble_model)
        
        # Serialize ensemble state to JSON and convert to ndarray
        ensemble_bytes = json.dumps(ensemble_state).encode('utf-8')
        return [np.frombuffer(ensemble_bytes, dtype=np.uint8)]
    
    def set_parameters_from_ensemble_state(self, ensemble_state: Dict[str, Any]) -> None:
        """
        Set ensemble model parameters from an ensemble state dictionary.
        
        Args:
            ensemble_state: Ensemble state dictionary with weights and model states
        """
        try:
            # Check if model classes and params match current models
            if len(ensemble_state.get("model_state_dicts", [])) != len(self.base_models):
                logger.warning(
                    f"Ensemble state has {len(ensemble_state.get('model_state_dicts', []))} models, "
                    f"but we have {len(self.base_models)} base models. Using subset of models."
                )
            
            # Get model names
            model_names = ensemble_state.get("model_names", [f"model_{i}" for i in range(len(self.base_models))])
            
            # Identify base models and meta models
            base_indices = []
            meta_indices = []
            
            for i, name in enumerate(model_names):
                if i >= len(self.base_models):
                    break
                    
                if name == "meta_lr" or "MetaLearner" in name:
                    meta_indices.append(i)
                else:
                    base_indices.append(i)
            
            # Load weights for each model
            for i, model in enumerate(self.base_models):
                if i >= len(ensemble_state.get("model_state_dicts", [])):
                    continue
                    
                # Get state dict for this model
                state_dict = ensemble_state["model_state_dicts"][i]
                
                # Special handling for sklearn wrappers
                if hasattr(model, 'set_parameters'):
                    # For scikit-learn wrapped models
                    try:
                        # Make a copy of the state dict to avoid modifying the original
                        modified_dict = state_dict.copy()
                        
                        # If this is a meta learner, ensure input_dim is correct
                        if i in meta_indices:
                            modified_dict["input_dim"] = len(base_indices)
                            
                        model.set_parameters(modified_dict)
                    except Exception as e:
                        logger.error(f"Error setting parameters for model {i}: {e}")
                else:
                    # For PyTorch models
                    try:
                        # Convert state dict values to tensors
                        tensor_state_dict = {}
                        for key, value in state_dict.items():
                            if key not in ["model_type", "estimator", "input_dim", "output_dim"]:
                                try:
                                    # Handle different types of values
                                    if isinstance(value, (list, np.ndarray)):
                                        tensor_state_dict[key] = torch.tensor(value, device=self.device)
                                    elif isinstance(value, dict):
                                        # Handle nested dictionaries (unlikely but possible)
                                        tensor_state_dict[key] = {
                                            k: torch.tensor(v, device=self.device) if isinstance(v, (list, np.ndarray)) else v
                                            for k, v in value.items()
                                        }
                                    elif isinstance(value, str):
                                        # Skip string values (metadata)
                                        continue
                                    else:
                                        # Handle scalar values
                                        tensor_state_dict[key] = torch.tensor([value], device=self.device)
                                except Exception as tensor_e:
                                    logger.error(f"Error converting to tensor for key {key}: {tensor_e}")
                                    # Skip this key
                                    continue
                        
                        # Check if we have any tensor values
                        if tensor_state_dict:
                            # Update model parameters
                            model.load_state_dict(tensor_state_dict)
                    except Exception as e:
                        logger.error(f"Error loading state dict for model {i}: {e}")
            
            # Create ensemble model with loaded weights
            weights = ensemble_state.get("weights", [1.0/len(self.base_models)] * len(self.base_models))
            
            # Ensure weights match the number of models
            if len(weights) != len(self.base_models):
                logger.warning(f"Weight count mismatch. Using equal weights.")
                weights = [1.0/len(self.base_models)] * len(self.base_models)
            
            # Create the ensemble model
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=weights,
                model_names=model_names[:len(self.base_models)],
                device=self.device
            )
            
            logger.info(f"Ensemble model updated with {len(self.base_models)} models")
            
        except Exception as e:
            logger.error(f"Failed to set parameters from ensemble state: {e}")
            # No fallback needed here since we catch exceptions for each model individually
            weights = [1.0/len(self.base_models)] * len(self.base_models)
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=weights,
                model_names=self.model_names,
                device=self.device
            )
    
    def set_parameters_individual(self, parameters: List[np.ndarray]) -> None:
        """
        Set individual model parameters from a list of NumPy arrays.
        
        This is used as a fallback when ensemble state loading fails.
        
        Args:
            parameters: List of parameter arrays
        """
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            # Try to interpret as serialized ensemble state
            try:
                ensemble_bytes = parameters[0].tobytes()
                ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                self.set_parameters_from_ensemble_state(ensemble_state)
                return
            except Exception as e:
                logger.error(f"Failed to deserialize ensemble state: {e}")
                logger.warning("Falling back to individual model parameter loading")
        
        # Apply parameters to the first model only (fallback)
        try:
            params_dict = zip(self.base_models[0].state_dict().keys(), parameters)
            state_dict = {k: torch.Tensor(v) for k, v in params_dict}
            self.base_models[0].load_state_dict(state_dict, strict=True)
            logger.warning("Applied parameters to first base model only (fallback mode)")
        except Exception as e:
            logger.error(f"Failed to set individual parameters: {e}")
    
    def set_parameters_from_ipfs(self, ipfs_hash: str) -> None:
        """Set model parameters from IPFS."""
        try:
            # Get model data from IPFS
            model_data = self.ipfs.get_json(ipfs_hash)
            
            if model_data and "ensemble_state" in model_data:
                # Load ensemble state
                ensemble_state = model_data["ensemble_state"]
                
                # Check for model parameters in ensemble state
                if "model_parameters" in ensemble_state:
                    configs = ensemble_state["model_parameters"]
                    logger.info(f"Found {len(configs)} model configurations in ensemble state")
                    
                    # First identify which models are base models vs meta learners
                    base_configs = []
                    meta_configs = []
                    
                    for config in configs:
                        if config.get("estimator") == "meta_lr" or config.get("model_type") == "meta_lr":
                            meta_configs.append(config)
                        else:
                            base_configs.append(config)
                    
                    # Process base model configs first
                    for config in base_configs:
                        # Ensure each config has correct input/output dimensions
                        if "input_dim" not in config:
                            config["input_dim"] = self.input_dim
                        if "output_dim" not in config:
                            config["output_dim"] = self.output_dim
                    
                    # Then process meta learner configs
                    for config in meta_configs:
                        # Meta learner should have input dim equal to number of base models
                        config["input_dim"] = len(base_configs)
                        config["meta_input_dim"] = len(base_configs)  # For backward compatibility
                        if "output_dim" not in config:
                            config["output_dim"] = self.output_dim
                    
                    # Combine all configs again
                    all_configs = base_configs + meta_configs
                    
                    # Initialize models from configurations with enforced dimensions
                    self.base_models, self.model_names = create_model_ensemble_from_config(
                        all_configs,
                        input_dim=self.input_dim,  # This will be respected by base models
                        output_dim=self.output_dim,
                        device=self.device
                    )
                    
                    # Create ensemble model with weights
                    weights = ensemble_state.get("weights", [1.0/len(self.base_models)] * len(self.base_models))
                    
                    # Ensure weights match number of models
                    if len(weights) != len(self.base_models):
                        logger.warning(f"Weight count mismatch. Expected {len(self.base_models)}, got {len(weights)}")
                        weights = [1.0/len(self.base_models)] * len(self.base_models)
                    
                    self.ensemble_model = EnsembleModel(
                        models=self.base_models,
                        weights=weights,
                        model_names=self.model_names,
                        device=self.device
                    )
                    
                    logger.info(f"Ensemble model loaded from IPFS: {ipfs_hash}")
                    logger.info(f"Model dimensions - Input: {self.input_dim}, Output: {self.output_dim}")
                    
                elif "model_state_dicts" in ensemble_state:
                    # Legacy format with state dicts - handle similarly with special meta learner case
                    state_dicts = ensemble_state["model_state_dicts"]
                    model_names = ensemble_state.get("model_names", [])
                    
                    # Separate base models and meta learners
                    base_dicts = []
                    meta_dicts = []
                    base_names = []
                    meta_names = []
                    
                    for i, state_dict in enumerate(state_dicts):
                        name = model_names[i] if i < len(model_names) else f"model_{i}"
                        model_type = state_dict.get("estimator", state_dict.get("model_type", ""))
                        
                        if model_type == "meta_lr" or "MetaLearner" in name:
                            meta_dicts.append(state_dict)
                            meta_names.append(name)
                        else:
                            base_dicts.append(state_dict)
                            base_names.append(name)
                    
                    # Update meta learner dimensions
                    for state_dict in meta_dicts:
                        state_dict["input_dim"] = len(base_dicts)
                        state_dict["meta_input_dim"] = len(base_dicts)
                    
                    # Combine back
                    all_dicts = base_dicts + meta_dicts
                    all_names = base_names + meta_names
                    
                    # Create modified ensemble state
                    modified_state = {
                        "model_state_dicts": all_dicts,
                        "model_names": all_names,
                        "weights": ensemble_state.get("weights", [1.0/len(all_dicts)] * len(all_dicts))
                    }
                    
                    self.set_parameters_from_ensemble_state(modified_state)
                    logger.info(f"Ensemble model (legacy format) loaded from IPFS: {ipfs_hash}")
                
            elif model_data and "state_dict" in model_data:
                # Legacy format - single model
                state_dict = {
                    k: torch.tensor(v, device=self.device)
                    for k, v in model_data["state_dict"].items()
                }
                
                # Try to update the first base model
                try:
                    self.base_models[0].load_state_dict(state_dict)
                    logger.info(f"First base model loaded from IPFS: {ipfs_hash}")
                except Exception as e:
                    logger.warning(f"Failed to load state dict into base model: {e}")
                    
            else:
                logger.error(f"Invalid model data from IPFS: {ipfs_hash}")
                
        except Exception as e:
            logger.error(f"Failed to load model from IPFS: {e}")
            logger.error(f"Error details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def save_ensemble_to_ipfs(
        self, round_num: int, metrics: Dict[str, Scalar] = None
    ) -> str:
        """
        Save ensemble model to IPFS.
        
        Args:
            round_num: Current round number
            metrics: Optional metrics to include
            
        Returns:
            IPFS hash of the saved model
        """
        if self.ensemble_model is None:
            logger.warning("No ensemble model to save")
            return None
        
        # Get ensemble state
        ensemble_state = get_ensemble_state_dict(self.ensemble_model)
        
        # Create metadata
        model_metadata = {
            "ensemble_state": ensemble_state,
            "info": {
                "round": round_num,
                "client_id": self.client_id,
                "wallet_address": self.wallet_address if self.wallet_address else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ensemble_size": len(self.ensemble_model.models),
                "weights": ensemble_state["weights"]
            }
        }
        
        # Add metrics if provided
        if metrics:
            model_metadata["info"]["metrics"] = metrics
        
        # Store in IPFS
        ipfs_hash = self.ipfs.add_json(model_metadata)
        logger.info(f"Stored ensemble model in IPFS: {ipfs_hash}")
        
        return ipfs_hash
    
    def train_ensemble(self, validation_split: float = 0.2) -> EnsembleModel:
        """
        Train the ensemble using GA-Stacking.
        
        Args:
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Trained ensemble model
        """
        # Separate and handle meta learner models
        base_models = []
        meta_models = []
        base_model_names = []
        meta_model_names = []
        
        for i, model in enumerate(self.base_models):
            if isinstance(model, MetaLearnerWrapper) or self.model_names[i] == "meta_lr":
                meta_models.append(model)
                meta_model_names.append(self.model_names[i])
            else:
                base_models.append(model)
                base_model_names.append(self.model_names[i])
        
        # Split data into train and validation
        train_loader, val_loader = self.split_train_val(validation_split)
        
        # Train base models individually
        for i, model in enumerate(base_models):
            self._train_single_model(model, train_loader, epochs=5)
            logger.info(f"Base model {i+1}/{len(base_models)} ({base_model_names[i]}) trained")
        
        # Skip training meta learners for now
        for i, model in enumerate(meta_models):
            logger.info(f"Skipping training for {meta_model_names[i]} (already trained)")
        
        # Combine all models back together for GA-Stacking
        all_models = base_models + meta_models
        all_model_names = base_model_names + meta_model_names
        
        # Initialize GA-Stacking with all models
        self.ga_stacking = GAStacking(
            base_models=all_models,
            model_names=all_model_names,
            population_size=self.ga_population_size,
            generations=self.ga_generations,
            device=self.device
        )
        
        # Run GA optimization
        logger.info("Starting GA-Stacking optimization")
        best_weights = self.ga_stacking.optimize(train_loader, val_loader)
        
        # Get the optimized ensemble
        self.ensemble_model = self.ga_stacking.get_ensemble_model()
        
        # Log the final weights
        weight_str = ", ".join([
            f"{name}: {weight:.4f}" 
            for name, weight in zip(all_model_names, best_weights)
        ])
        logger.info(f"GA-Stacking complete. Ensemble weights: {weight_str}")
        
        return self.ensemble_model
    
    def _train_single_model(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        epochs: int = 5, 
        learning_rate: float = 0.01
    ) -> None:
        """
        Train a single model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            epochs: Number of epochs to train for
            learning_rate: Learning rate for optimization
        """
        # Skip training for scikit-learn model wrappers
        if isinstance(model, SklearnModelWrapper):
            logger.info(f"Skipping training for {model.model_type} (already trained)")
            return
            
        # Train PyTorch models
        model.train()
        criterion = nn.MSELoss()
        
        # Check if model has parameters before creating optimizer
        if sum(p.numel() for p in model.parameters() if p.requires_grad) == 0:
            logger.warning(f"Model has no trainable parameters, skipping training")
            return
            
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(data)
                epoch_samples += len(data)
            
            avg_epoch_loss = epoch_loss / epoch_samples
            if epoch % 2 == 0:  # Log every 2 epochs
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
    
    def initialize_models_from_config(
        self, 
        config: Dict[str, Any]
    ) -> Tuple[List[nn.Module], List[str]]:
        """
        Initialize models from configuration provided by the server.
        
        Args:
            config: Configuration from the server that may include initial_ensemble
            
        Returns:
            Tuple of (list of models, list of model names)
        """
        initial_ensemble = config.get("initial_ensemble", None)
        
        if initial_ensemble:
            logger.info(f"Initializing models from server-provided configuration ({len(initial_ensemble)} models)")
            models, model_names = create_model_ensemble_from_config(
                initial_ensemble,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                device=self.device
            )
            
            # Log the model types received
            model_types = ", ".join(model_names)
            logger.info(f"Initialized models: {model_types}")
            
            return models, model_names
        else:
            # Fall back to default model creation
            logger.info(f"No initial configuration provided, creating default ensemble with {self.ensemble_size} models")
            return create_model_ensemble(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                ensemble_size=self.ensemble_size,
                device=self.device
            )

    def fit(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train the model on the local dataset.
        
        For GA-Stacking, we train individual models and then optimize the ensemble.
        """
        # Check if client is authorized (if blockchain is available)
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if not is_authorized:
                    logger.warning(f"Client {self.wallet_address} is not authorized to participate")
                    # Return empty parameters with auth failure flag
                    return parameters_to_ndarrays(parameters), 0, {"error": "client_not_authorized"}
            except Exception as e:
                logger.error(f"Failed to check client authorization: {e}")
                # Continue anyway, server will check again
        
        # Get global model from IPFS if hash is provided
        ipfs_hash = config.get("ipfs_hash", None)
        round_num = config.get("server_round", 0)
        
        # If this is the first round, check for initial ensemble configuration
        if round_num == 1 and "initial_ensemble" in config:
            # Initialize models from configuration
            self.base_models, self.model_names = self.initialize_models_from_config(config)
        elif ipfs_hash:
            # Load model from IPFS with dimension fix
            try:
                self.set_parameters_from_ipfs(ipfs_hash)
                logger.info(f"Model loaded from IPFS: {ipfs_hash}")
            except Exception as e:
                logger.error(f"Failed to load model from IPFS: {str(e)}")
                # Continue with current models
        else:
            # Fallback to direct parameters
            params_arrays = parameters_to_ndarrays(parameters)
            self.set_parameters_individual(params_arrays)
        
        # Get training config
        local_epochs = int(config.get("local_epochs", 5))
        do_ga_stacking = bool(config.get("ga_stacking", True))
        validation_split = float(config.get("validation_split", 0.2))
        
        # Verify and fix models before training
        self._verify_and_fix_models()
        
        # Run GA-Stacking to optimize ensemble weights
        if do_ga_stacking:
            logger.info("Performing GA-Stacking optimization")
            try:
                self.train_ensemble(validation_split=validation_split)
            except Exception as e:
                logger.error(f"Error in GA-Stacking: {str(e)}")
                # Fall back to equal weights
                weights = np.ones(len(self.base_models)) / len(self.base_models)
                self.ensemble_model = EnsembleModel(
                    models=self.base_models,
                    weights=weights,
                    model_names=self.model_names,
                    device=self.device
                )
        else:
            logger.info("Skipping GA-Stacking, training individual models")
            # Train individual models
            for i, model in enumerate(self.base_models):
                try:
                    self._train_single_model(model, self.train_loader, epochs=local_epochs)
                    logger.info(f"Model {i+1}/{len(self.base_models)} trained")
                except Exception as e:
                    logger.error(f"Error training model {i+1}: {str(e)}")
            
            # Equal weighting if not using GA-Stacking
            weights = np.ones(len(self.base_models)) / len(self.base_models)
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=weights,
                model_names=self.model_names,
                device=self.device
            )
        
        # Ensure we have an ensemble model
        if self.ensemble_model is None:
            logger.warning("No ensemble model created, creating one with equal weights")
            weights = np.ones(len(self.base_models)) / len(self.base_models)
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=weights,
                model_names=self.model_names,
                device=self.device
            )
        
        # Evaluate the ensemble
        try:
            accuracy, loss = self._evaluate_ensemble(self.test_loader)
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            accuracy, loss = 0.0, float('inf')
        
        # Save ensemble to IPFS
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy)
        }
        
        try:
            client_ipfs_hash = self.save_ensemble_to_ipfs(round_num, metrics)
        except Exception as e:
            logger.error(f"Failed to save ensemble to IPFS: {str(e)}")
            client_ipfs_hash = None
        
        # Record contribution to blockchain if available
        if self.blockchain and self.wallet_address and client_ipfs_hash:
            try:
                tx_hash = self.blockchain.record_contribution(
                    client_address=self.wallet_address,
                    round_num=round_num,
                    ipfs_hash=client_ipfs_hash,
                    accuracy=accuracy
                )
                logger.info(f"Contribution recorded on blockchain, tx: {tx_hash}")
                metrics["blockchain_tx"] = tx_hash
            except Exception as e:
                logger.error(f"Failed to record contribution on blockchain: {e}")
        
        # Add to metrics history
        self.metrics_history.append({
            "round": round_num,
            "fit_loss": float(loss),
            "accuracy": float(accuracy),
            "ensemble_weights": self.ensemble_model.weights.cpu().numpy().tolist() if self.ensemble_model else [],
            "ipfs_hash": client_ipfs_hash,
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Include IPFS hash in metrics
        metrics["ipfs_hash"] = ipfs_hash  # Return the server's hash for tracking
        metrics["client_ipfs_hash"] = client_ipfs_hash
        metrics["wallet_address"] = self.wallet_address if self.wallet_address else "unknown"
        metrics["ensemble_size"] = len(self.base_models)
        
        # Get the ensemble parameters
        try:
            parameters_updated = self.get_parameters(config)
        except Exception as e:
            logger.error(f"Error getting parameters: {str(e)}")
            parameters_updated = parameters_to_ndarrays(parameters)  # Return original parameters
        
        # Get number of training samples
        num_samples = len(self.train_loader.dataset)
        
        # Make sure metrics only contains serializable values
        for key in list(metrics.keys()):
            if not isinstance(metrics[key], (int, float, str, bool, list)):
                metrics[key] = str(metrics[key])
        
        # Return the raw parameters (not wrapped in Flower Parameters)
        return parameters_updated, num_samples, metrics

    def _verify_and_fix_models(self):
        """Verify models have correct dimensions and fix if needed."""
        if not self.base_models:
            logger.warning("No base models to verify")
            return
            
        fixed_models = []
        
        # Check if we need to fix the meta_lr model's input dimension
        meta_model_idx = None
        num_base_models = 0
        
        # First pass: count base models and find meta learner
        for i, model_name in enumerate(self.model_names):
            if model_name == "meta_lr" or "meta" in model_name.lower():
                meta_model_idx = i
            else:
                num_base_models += 1
        
        # Second pass: fix or recreate models if needed
        for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
            if i == meta_model_idx:
                # Check meta learner's input dimension
                if hasattr(model, 'input_dim') and model.input_dim != num_base_models:
                    logger.warning(f"Fixing meta learner input dimension: {model.input_dim} -> {num_base_models}")
                    # Create a new meta learner with correct dimensions
                    from base_models import MetaLearnerWrapper
                    new_model = MetaLearnerWrapper(input_dim=num_base_models, output_dim=self.output_dim)
                    # Try to preserve learned coefficients if possible, but reshape if needed
                    if hasattr(model, 'coef_'):
                        # Initialize with equal weights
                        new_model.coef_ = np.ones((1, num_base_models)) / num_base_models
                        new_model.intercept_ = np.zeros(1) if hasattr(model, 'intercept_') else np.zeros(1)
                        new_model.is_initialized = True
                    fixed_models.append(new_model)
                    logger.info(f"Created new meta learner with input_dim={num_base_models}")
                else:
                    fixed_models.append(model)
            else:
                # Check base model dimensions
                if hasattr(model, 'input_dim') and model.input_dim != self.input_dim:
                    logger.warning(f"Input dimension mismatch for {name}: {model.input_dim} != {self.input_dim}")
                    # Recreate the model with correct dimensions
                    if name == "lr":
                        from base_models import LinearRegressionWrapper
                        new_model = LinearRegressionWrapper(input_dim=self.input_dim, output_dim=self.output_dim)
                        fixed_models.append(new_model)
                        logger.info(f"Created new {name} model with input_dim={self.input_dim}")
                    elif name == "svc":
                        from base_models import SVCWrapper
                        new_model = SVCWrapper(input_dim=self.input_dim, output_dim=self.output_dim)
                        fixed_models.append(new_model)
                        logger.info(f"Created new {name} model with input_dim={self.input_dim}")
                    elif name == "rf":
                        from base_models import RandomForestWrapper
                        new_model = RandomForestWrapper(input_dim=self.input_dim, output_dim=self.output_dim)
                        fixed_models.append(new_model)
                        logger.info(f"Created new {name} model with input_dim={self.input_dim}")
                    else:
                        # For unknown models, create a linear model
                        from base_models import LinearModel
                        new_model = LinearModel(input_dim=self.input_dim, output_dim=self.output_dim)
                        fixed_models.append(new_model)
                        logger.info(f"Created new linear model for {name} with input_dim={self.input_dim}")
                else:
                    fixed_models.append(model)
        
        # Update models if any were fixed
        if len(fixed_models) == len(self.base_models):
            self.base_models = fixed_models
            
        # Ensure all models are on the correct device
        for model in self.base_models:
            if hasattr(model, 'to'):
                model.to(self.device)
    
    def evaluate(
        self, parameters: Parameters, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the model on the local test dataset.
        """
        # Check if client is authorized (if blockchain is available)
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if not is_authorized:
                    logger.warning(f"Client {self.wallet_address} is not authorized to participate in evaluation")
                    # Return empty evaluation with auth failure flag
                    return 0.0, 0, {"error": "client_not_authorized"}
            except Exception as e:
                logger.error(f"Failed to check client authorization: {e}")
                # Continue anyway, server will check again
        
        # Get global model from IPFS if hash is provided
        ipfs_hash = config.get("ipfs_hash", None)
        round_num = config.get("server_round", 0)
        
        if ipfs_hash:
            self.set_parameters_from_ipfs(ipfs_hash)
        else:
            # Fallback to direct parameters
            params_arrays = parameters_to_ndarrays(parameters)
            self.set_parameters_individual(params_arrays)
        
        # Evaluate the ensemble
        accuracy, loss = self._evaluate_ensemble(self.test_loader)
        
        # Add to metrics history
        self.metrics_history.append({
            "round": round_num,
            "eval_loss": float(loss),
            "accuracy": float(accuracy),
            "eval_samples": len(self.test_loader.dataset),
            "ipfs_hash": ipfs_hash,
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        metrics = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "wallet_address": self.wallet_address if self.wallet_address else "unknown",
            "ensemble_size": len(self.base_models) if self.ensemble_model else 0
        }
        
        return float(loss), len(self.test_loader.dataset), metrics
    
    def _evaluate_ensemble(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the ensemble model.
        
        Args:
            data_loader: Data loader to evaluate on
            
        Returns:
            Tuple of (accuracy, loss)
        """
        if self.ensemble_model is None:
            logger.warning("No ensemble model for evaluation, using equal weights")
            # Create ensemble with equal weights if not available
            equal_weights = np.ones(len(self.base_models)) / len(self.base_models)
            self.ensemble_model = EnsembleModel(
                models=self.base_models,
                weights=equal_weights,
                model_names=self.model_names,
                device=self.device
            )
        
        self.ensemble_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.ensemble_model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item() * len(data)
                
                # For regression, we consider a prediction correct if it's within a threshold
                threshold = 0.5
                correct += torch.sum((torch.abs(output - target) < threshold)).item()
                total += target.size(0)
        
        # Calculate average loss and accuracy
        avg_loss = test_loss / total if total > 0 else float('inf')
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        logger.info(f"Ensemble evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return accuracy, avg_loss
    
    def _evaluate_single_model(self, model: nn.Module, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate a single model.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader to evaluate on
            
        Returns:
            Tuple of (accuracy, loss)
        """
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item() * len(data)
                
                # For regression, we consider a prediction correct if it's within a threshold
                threshold = 0.5
                correct += torch.sum((torch.abs(output - target) < threshold)).item()
                total += target.size(0)
        
        # Calculate average loss and accuracy
        avg_loss = test_loss / total if total > 0 else float('inf')
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return accuracy, avg_loss
    
    def save_metrics_history(self, filepath: str = "client_metrics.json"):
        """Save metrics history to a file."""
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved metrics history to {filepath}")


def start_client(
    server_address: str = "127.0.0.1:8080",
    ipfs_url: str = "http://127.0.0.1:5001",
    ganache_url: str = "http://127.0.0.1:7545",
    contract_address: Optional[str] = None,
    wallet_address: Optional[str] = None,
    private_key: Optional[str] = None,
    client_id: Optional[str] = None,
    input_dim: int = 10,
    output_dim: int = 1,
    ensemble_size: int = 5,
    device: str = "cpu",
    ga_generations: int = 20,
    ga_population_size: int = 30
) -> None:
    """
    Start a federated learning client with GA-Stacking ensemble optimization.
    
    Args:
        server_address: Server address (host:port)
        ipfs_url: IPFS API URL
        ganache_url: Ganache blockchain URL
        contract_address: Address of deployed EnhancedModelRegistry contract
        wallet_address: Client's Ethereum wallet address
        private_key: Client's private key (for signing transactions)
        client_id: Client identifier
        input_dim: Input dimension for the model
        output_dim: Output dimension for the model
        ensemble_size: Number of models in the ensemble
        device: Device to use for training ('cpu' or 'cuda')
        ga_generations: Number of GA generations to run
        ga_population_size: Size of GA population
    """
    # Create client ID if not provided
    if client_id is None:
        client_id = f"client-{os.getpid()}"
    
    # Create metrics directory
    os.makedirs(f"metrics/{client_id}", exist_ok=True)
    
    # Initialize IPFS connector
    ipfs_connector = IPFSConnector(ipfs_api_url=ipfs_url)
    logger.info(f"Initialized IPFS connector: {ipfs_url}")
    
    # Initialize blockchain connector if contract address is provided
    blockchain_connector = None
    if contract_address:
        try:
            blockchain_connector = BlockchainConnector(
                ganache_url=ganache_url,
                contract_address=contract_address,
                private_key=private_key
            )
            logger.info(f"Initialized blockchain connector: {ganache_url}")
            logger.info(f"Using contract at: {contract_address}")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connector: {e}")
            logger.warning("Continuing without blockchain features")
    
    # Load dataset from files
    def load_dataset_from_files(train_file, test_file):
        # Read train and test files
        train_df = pd.read_csv(train_file, comment='#')
        test_df = pd.read_csv(test_file, comment='#')
        
        # Extract features and targets
        feature_columns = [col for col in train_df.columns if col != 'target']
        
        # Convert to tensors
        train_x = torch.tensor(train_df[feature_columns].values, dtype=torch.float32)
        train_y = torch.tensor(train_df['target'].values.reshape(-1, 1), dtype=torch.float32)
        
        test_x = torch.tensor(test_df[feature_columns].values, dtype=torch.float32)
        test_y = torch.tensor(test_df['target'].values.reshape(-1, 1), dtype=torch.float32)
        
        # Update input and output dimensions based on the data
        nonlocal input_dim, output_dim
        input_dim = train_x.shape[1]
        output_dim = train_y.shape[1]
        
        logger.info(f"Dataset loaded - Input dim: {input_dim}, Output dim: {output_dim}")
        
        # Create data loaders
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        
        train_loader = DataLoader(
            SimpleDataset(train_x, train_y), batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            SimpleDataset(test_x, test_y), batch_size=32, shuffle=False
        )
        
        return train_loader, test_loader

    # Determine which dataset files to load based on client_id
    train_file = f"{client_id}_train.txt"
    test_file = f"{client_id}_test.txt"

    # Check if files exist
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        logger.warning(f"Dataset files not found for {client_id}, using default files")
        train_file = "client-1_train.txt"  # Fallback to client-1 if files not found
        test_file = "client-1_test.txt"

    # Load dataset
    train_loader, test_loader = load_dataset_from_files(train_file, test_file)
    
    # Create client
    client = GAStackingClient(
        input_dim=input_dim,
        output_dim=output_dim,
        train_loader=train_loader,
        test_loader=test_loader,
        ensemble_size=ensemble_size,
        ipfs_connector=ipfs_connector,
        blockchain_connector=blockchain_connector,
        wallet_address=wallet_address,
        private_key=private_key,
        device=device,
        client_id=client_id,
        ga_generations=ga_generations,
        ga_population_size=ga_population_size
    )
    
    # Start client
    fl.client.start_client(server_address=server_address, client=client)
    
    # Save metrics after client finishes
    client.save_metrics_history(filepath=f"metrics/{client_id}/metrics_history.json")
    
    logger.info(f"Client {client_id} completed federated learning with GA-Stacking")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start FL client with GA-Stacking ensemble optimization")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8088", help="Server address (host:port)")
    parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--ganache-url", type=str, default="http://192.168.1.146:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="Address of deployed EnhancedModelRegistry contract")
    parser.add_argument("--wallet-address", type=str, help="Client's Ethereum wallet address")
    parser.add_argument("--private-key", type=str, help="Client's private key (for signing transactions)")
    parser.add_argument("--client-id", type=str, help="Client identifier")
    parser.add_argument("--input-dim", type=int, default=10, help="Input dimension for the model")
    parser.add_argument("--output-dim", type=int, default=1, help="Output dimension for the model")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of models in the ensemble")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for training")
    parser.add_argument("--ga-generations", type=int, default=20, help="Number of GA generations to run")
    parser.add_argument("--ga-population-size", type=int, default=30, help="Size of GA population")
    parser.add_argument("--train-file", type=str, help="Path to training data file")
    parser.add_argument("--test-file", type=str, help="Path to test data file")
    args = parser.parse_args()
    
    # Check if contract address is stored in file
    if args.contract_address is None:
        try:
            with open("contract_address.txt", "r") as f:
                args.contract_address = f.read().strip()
                print(f"Loaded contract address from file: {args.contract_address}")
        except FileNotFoundError:
            print("No contract address provided or found in file")
    
    # Override train/test files if provided
    if args.train_file and args.test_file:
        train_file = args.train_file
        test_file = args.test_file
    
    start_client(
        server_address=args.server_address,
        ipfs_url=args.ipfs_url,
        ganache_url=args.ganache_url,
        contract_address=args.contract_address,
        wallet_address=args.wallet_address,
        private_key=args.private_key,
        client_id=args.client_id,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        ensemble_size=args.ensemble_size,
        device=args.device,
        ga_generations=args.ga_generations,
        ga_population_size=args.ga_population_size
        
    )