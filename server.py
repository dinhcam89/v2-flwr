"""
Enhanced Federated Learning Server with GA-Stacking support, IPFS and Blockchain integration.
Supports client authorization, contribution tracking, and ensemble model aggregation.
"""

import os
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from datetime import datetime, timezone
import logging
from pathlib import Path

import pytz

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
import numpy as np

from ipfs_connector import IPFSConnector
from blockchain_connector import BlockchainConnector
from ensemble_aggregation import EnsembleAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("FL-Server")

class EnhancedFedAvgWithGA(fl.server.strategy.FedAvg):
    """Enhanced Federated Averaging strategy with GA-Stacking, IPFS, and blockchain integration."""
    
    def __init__(
        self,
        *args,
        ipfs_connector: Optional[IPFSConnector] = None,
        blockchain_connector: Optional[BlockchainConnector] = None,
        version_prefix: str = "1.0",
        authorized_clients_only: bool = True,
        round_rewards: int = 1000,  # Reward points to distribute each round
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # Initialize IPFS connector
        self.ipfs = ipfs_connector or IPFSConnector()
        
        # Initialize blockchain connector
        self.blockchain = blockchain_connector
        
        # Set version prefix
        self.version_prefix = version_prefix
        
        # Flag to only allow authorized clients
        self.authorized_clients_only = authorized_clients_only
        
        # Rewards per round
        self.round_rewards = round_rewards
        
        # Set of authorized clients from blockchain
        self.authorized_clients: Set[str] = set()
        
        # Client contributions for current round
        self.current_round_contributions = {}
        
        # Metrics storage
        self.metrics_history = []
        
        # Ensemble aggregator
        self.ensemble_aggregator = EnsembleAggregator(device=device)
        
        # Load authorized clients from blockchain
        self._load_authorized_clients()
        
        logger.info(f"Initialized EnhancedFedAvgWithGA with IPFS node: {self.ipfs.ipfs_api_url}")
        if self.blockchain:
            logger.info(f"Blockchain integration enabled")
            if self.authorized_clients_only:
                logger.info(f"Only accepting contributions from authorized clients ({len(self.authorized_clients)} loaded)")
    
    def _load_authorized_clients(self):
        """Load authorized clients from blockchain."""
        if self.blockchain:
            try:
                clients = self.blockchain.get_all_authorized_clients()
                self.authorized_clients = set(clients)
                logger.info(f"Loaded {len(self.authorized_clients)} authorized clients from blockchain")
            except Exception as e:
                logger.error(f"Failed to load authorized clients: {e}")
    
    def is_client_authorized(self, wallet_address: str) -> bool:
        """Check if a client is authorized."""
        if not self.authorized_clients_only:
            return True
        
        if not wallet_address or wallet_address == "unknown":
            logger.warning(f"Client provided no wallet address")
            return False
        
        # Check local cache first
        if wallet_address in self.authorized_clients:
            return True
        
        # Check blockchain
        if self.blockchain:
            try:
                is_authorized = self.blockchain.is_client_authorized(wallet_address)
                # Update local cache
                if is_authorized:
                    self.authorized_clients.add(wallet_address)
                
                return is_authorized
            except Exception as e:
                logger.error(f"Failed to check client authorization: {e}")
                # Fall back to local cache
                return wallet_address in self.authorized_clients
        
        return False
    
    def get_version(self, round_num: int) -> str:
        """Generate a version string based on round number."""
        return f"{self.version_prefix}.{round_num}"

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        
        # Define the correct dimensions based on your configuration
        input_dim = 8  # Feature dimension (based on support vectors in your JSON)
        output_dim = 1  # Output dimension
        num_base_models = 3  # Number of base models
        
        # Define initial model ensemble configuration matching your JSON
        initial_ensemble_config = [
            {
                "estimator": "lr",
                "input_dim": input_dim,
                "output_dim": output_dim,
                "coef": [[0.1, 0.2, 0.1, 0.3, 0.15, 0.25, 0.1, 0.2]],  # 8 features
                "intercept": [0.0]
            },
            {
                "estimator": "svc",
                "input_dim": input_dim,
                "output_dim": output_dim,
                "dual_coef": [[0.5, -0.5]],
                "support_vectors": [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 8 features
                    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]   # 8 features
                ],
                "intercept": [0.0]
            },
            {
                "estimator": "rf",
                "input_dim": input_dim,
                "output_dim": output_dim,
                "n_estimators": 100,
                "feature_importances": [0.1, 0.15, 0.2, 0.1, 0.15, 0.1, 0.1, 0.1]  # 8 features
            },
            {
                "estimator": "meta_lr",
                "input_dim": num_base_models,
                "meta_input_dim": num_base_models,
                "output_dim": output_dim,
                "coef": [[0.33, 0.33, 0.34]],  # Equal weights for 3 base models
                "intercept": [0.0]
            }
        ]
        
        # Create ensemble state with initial configuration
        ensemble_state = {
            "model_parameters": initial_ensemble_config,
            "weights": [0.33, 0.33, 0.34, 0.0],  # Initial weights prioritizing base models
            "model_names": ["lr", "svc", "rf", "meta_lr"]
        }
        
        # Serialize ensemble state
        ensemble_bytes = json.dumps(ensemble_state).encode('utf-8')
        parameters = ndarrays_to_parameters([np.frombuffer(ensemble_bytes, dtype=np.uint8)])
        
        logger.info(f"Initialized server with custom ensemble configuration: input_dim={input_dim}, meta_input_dim={num_base_models}")
        
        return parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        # Reset round contributions
        self.current_round_contributions = {}
        
        # Get parameters as ndarrays
        params_ndarrays = parameters_to_ndarrays(parameters)
        
        # Check if we have an ensemble or a regular model
        is_ensemble = len(params_ndarrays) == 1 and params_ndarrays[0].dtype == np.uint8
        
        # Store in IPFS
        if is_ensemble:
            # Handle ensemble model
            try:
                # Deserialize ensemble
                ensemble_bytes = params_ndarrays[0].tobytes()
                ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                
                # Create metadata with ensemble state
                model_metadata = {
                    "ensemble_state": ensemble_state,
                    "info": {
                        "round": server_round,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "version": self.get_version(server_round),
                        "is_ensemble": True,
                        "num_models": len(ensemble_state["model_names"]),
                        "model_names": ensemble_state["model_names"]
                    }
                }
                
                # Store in IPFS
                ipfs_hash = self.ipfs.add_json(model_metadata)
                logger.info(f"Stored ensemble model in IPFS: {ipfs_hash}")
                
            except Exception as e:
                logger.error(f"Failed to process ensemble model: {e}")
                # Fall back to storing raw parameters
                ipfs_hash = self._store_raw_parameters_in_ipfs(params_ndarrays, server_round)
        else:
            # Handle regular model
            ipfs_hash = self._store_raw_parameters_in_ipfs(params_ndarrays, server_round)
        
        # Register in blockchain if available
        if self.blockchain:
            try:
                # Use register_or_update_model instead of register_model
                tx_hash = self.blockchain.register_or_update_model(
                    ipfs_hash=ipfs_hash,
                    round_num=server_round,
                    version=self.get_version(server_round),
                    participating_clients=0  # Will be updated after fit
                )
                logger.info(f"Registered model in blockchain, tx: {tx_hash}")
            except Exception as e:
                logger.error(f"Failed to register model in blockchain: {e}")
        
        # Configure fit instructions for clients with is_ensemble flag
        config = {
            "ipfs_hash": ipfs_hash, 
            "server_round": server_round,
            "ga_stacking": True,  # Enable GA-Stacking on clients
            "local_epochs": 5,
            "validation_split": 0.2,
            "is_ensemble": is_ensemble  # Add this flag to indicate ensemble model
        }
        
        fit_ins = FitIns(parameters, config)
        
        # Sample clients for this round
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, 
            min_num_clients=self.min_available_clients
        )
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    

    def _store_raw_parameters_in_ipfs(self, params_ndarrays: List[np.ndarray], server_round: int) -> str:
        """Store raw parameters in IPFS."""
        # Create state dict from weights
        state_dict = {}
        layer_names = ["linear.weight", "linear.bias"]  # Adjust based on your model
        
        for i, name in enumerate(layer_names):
            if i < len(params_ndarrays):
                state_dict[name] = params_ndarrays[i].tolist()
        
        # Create metadata
        model_metadata = {
            "state_dict": state_dict,
            "info": {
                "round": server_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": self.get_version(server_round),
                "is_ensemble": False
            }
        }
        
        # Store in IPFS
        ipfs_hash = self.ipfs.add_json(model_metadata)
        logger.info(f"Stored global model in IPFS: {ipfs_hash}")
        
        return ipfs_hash
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients."""
        
        # Filter out unauthorized clients
        authorized_results = []
        unauthorized_clients = []
        
        for client, fit_res in results:
            wallet_address = fit_res.metrics.get("wallet_address", "unknown")
            
            # Check for auth error flag
            if fit_res.metrics.get("error") == "client_not_authorized":
                logger.warning(f"Client {wallet_address} reported as unauthorized")
                unauthorized_clients.append((client, wallet_address))
                continue
            
            # Verify client authorization
            if self.is_client_authorized(wallet_address):
                authorized_results.append((client, fit_res))
                
                # Record contribution metrics
                client_ipfs_hash = fit_res.metrics.get("client_ipfs_hash")
                accuracy = fit_res.metrics.get("accuracy", 0.0)
                
                if client_ipfs_hash and wallet_address != "unknown":
                    self.current_round_contributions[wallet_address] = {
                        "ipfs_hash": client_ipfs_hash,
                        "accuracy": accuracy
                    }
            else:
                logger.warning(f"Rejecting contribution from unauthorized client: {wallet_address}")
                unauthorized_clients.append((client, wallet_address))
        
        # Check if enough clients returned results
        if not authorized_results:
            if unauthorized_clients:
                logger.error(f"All {len(unauthorized_clients)} clients were unauthorized. No aggregation possible.")
            else:
                logger.error("No clients returned results. No aggregation possible.")
            return None, {"error": "no_authorized_clients"}
        
        # Calculate the total number of examples used for training
        num_examples_total = sum([fit_res.num_examples for _, fit_res in authorized_results])
        
        # Create weights for weighted average of client models
        if num_examples_total > 0:
            weights = [fit_res.num_examples / num_examples_total for _, fit_res in authorized_results]
        else:
            weights = [1.0 / len(authorized_results) for _ in authorized_results]
        
        # Check if we need to aggregate ensembles or regular models
        any_ensemble = False
        for _, fit_res in authorized_results:
            params = parameters_to_ndarrays(fit_res.parameters)
            if len(params) == 1 and params[0].dtype == np.uint8:
                any_ensemble = True
                break
        
        # Aggregate the updates
        if any_ensemble:
            # Use ensemble aggregation
            logger.info("Aggregating ensemble models")
            parameters_aggregated, agg_metrics = self.ensemble_aggregator.aggregate_fit_results(
                authorized_results, weights
            )
        else:
            # Fall back to standard FedAvg
            logger.info("Aggregating standard models")
            parameters_aggregated, metrics = super().aggregate_fit(server_round, authorized_results, failures)
            agg_metrics = metrics
        
        if parameters_aggregated is not None:
            # Add metrics about client participation
            agg_metrics["total_clients"] = len(results)
            agg_metrics["authorized_clients"] = len(authorized_results)
            agg_metrics["unauthorized_clients"] = len(unauthorized_clients)
            
            # Store metrics for history
            self.metrics_history.append({
                "round": server_round,
                "metrics": agg_metrics,
                "num_clients": len(authorized_results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Record contributions to blockchain
            if self.blockchain and self.current_round_contributions:
                logger.info(f"Recording {len(self.current_round_contributions)} client contributions to blockchain")
                
                for wallet_address, contribution in self.current_round_contributions.items():
                    try:
                        tx_hash = self.blockchain.record_contribution(
                            client_address=wallet_address,
                            round_num=server_round,
                            ipfs_hash=contribution["ipfs_hash"],
                            accuracy=contribution["accuracy"]
                        )
                        logger.info(f"Recorded contribution for {wallet_address}, tx: {tx_hash}")
                    except Exception as e:
                        logger.error(f"Failed to record contribution for {wallet_address}: {e}")
                
                # Allocate rewards for this round
                try:
                    tx_hash = self.blockchain.allocate_rewards_for_round(
                        round_num=server_round,
                        total_reward=self.round_rewards
                    )
                    logger.info(f"Allocated rewards for round {server_round}, tx: {tx_hash}")
                except Exception as e:
                    logger.error(f"Failed to allocate rewards for round {server_round}: {e}")
            
            # Update participating clients in blockchain if available
            if self.blockchain:
                try:
                    # Get the global model hash from the first client's config
                    # Assumes all clients received the same model
                    ipfs_hash = authorized_results[0][1].metrics.get("ipfs_hash", None)
                    
                    if ipfs_hash:
                        # Update model in blockchain with actual client count
                        tx_hash = self.blockchain.register_or_update_model(
                            ipfs_hash=ipfs_hash,
                            round_num=server_round,
                            version=self.get_version(server_round),
                            participating_clients=len(authorized_results)
                        )
                        logger.info(f"Updated model in blockchain with {len(authorized_results)} clients, tx: {tx_hash}")
                except Exception as e:
                    logger.error(f"Failed to update model in blockchain: {e}")
        
        return parameters_aggregated, agg_metrics
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the evaluation round."""
        
        # Get parameters as ndarrays
        params_ndarrays = parameters_to_ndarrays(parameters)
        
        # Check if we have an ensemble or a regular model
        is_ensemble = len(params_ndarrays) == 1 and params_ndarrays[0].dtype == np.uint8
        
        # Store in IPFS with evaluation flag
        if is_ensemble:
            # Handle ensemble model
            try:
                # Deserialize ensemble
                ensemble_bytes = params_ndarrays[0].tobytes()
                ensemble_state = json.loads(ensemble_bytes.decode('utf-8'))
                
                # Create metadata with ensemble state and evaluation flag
                model_metadata = {
                    "ensemble_state": ensemble_state,
                    "info": {
                        "round": server_round,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "version": self.get_version(server_round),
                        "is_ensemble": True,
                        "evaluation": True,
                        "num_models": len(ensemble_state["model_names"]),
                        "model_names": ensemble_state["model_names"]
                    }
                }
                
                # Store in IPFS
                ipfs_hash = self.ipfs.add_json(model_metadata)
                logger.info(f"Stored evaluation ensemble model in IPFS: {ipfs_hash}")
                
            except Exception as e:
                logger.error(f"Failed to process ensemble model for evaluation: {e}")
                # Fall back to storing raw parameters
                ipfs_hash = self._store_raw_parameters_in_ipfs_for_eval(params_ndarrays, server_round)
        else:
            # Handle regular model
            ipfs_hash = self._store_raw_parameters_in_ipfs_for_eval(params_ndarrays, server_round)
        
        # Include IPFS hash in config
        config = {"ipfs_hash": ipfs_hash, "server_round": server_round}
        
        # Configure evaluation instructions for clients
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Sample clients for evaluation
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients, 
            min_num_clients=self.min_available_clients
        )
        
        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    
    def _store_raw_parameters_in_ipfs_for_eval(self, params_ndarrays: List[np.ndarray], server_round: int) -> str:
        """Store raw parameters in IPFS for evaluation."""
        # Create state dict from weights
        state_dict = {}
        layer_names = ["linear.weight", "linear.bias"]  # Adjust based on your model
        
        for i, name in enumerate(layer_names):
            if i < len(params_ndarrays):
                state_dict[name] = params_ndarrays[i].tolist()
        
        # Create metadata with evaluation flag
        model_metadata = {
            "state_dict": state_dict,
            "info": {
                "round": server_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": self.get_version(server_round),
                "is_ensemble": False,
                "evaluation": True
            }
        }
        
        # Store in IPFS
        ipfs_hash = self.ipfs.add_json(model_metadata)
        logger.info(f"Stored evaluation model in IPFS: {ipfs_hash}")
        
        return ipfs_hash
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        
        # Filter out unauthorized clients
        authorized_results = []
        unauthorized_clients = []
        
        for client, eval_res in results:
            wallet_address = eval_res.metrics.get("wallet_address", "unknown")
            
            # Check for auth error flag
            if eval_res.metrics.get("error") == "client_not_authorized":
                logger.warning(f"Client {wallet_address} reported as unauthorized")
                unauthorized_clients.append((client, wallet_address))
                continue
            
            # Verify client authorization
            if self.is_client_authorized(wallet_address):
                authorized_results.append((client, eval_res))
            else:
                logger.warning(f"Rejecting evaluation from unauthorized client: {wallet_address}")
                unauthorized_clients.append((client, wallet_address))
        
        if not authorized_results:
            if unauthorized_clients:
                logger.error(f"All {len(unauthorized_clients)} clients were unauthorized. No evaluation aggregation possible.")
            else:
                logger.error("No clients returned evaluation results.")
            return None, {"error": "no_authorized_clients"}
        
        # Check if any client has returned ensemble metrics
        has_ensemble_metrics = False
        for _, eval_res in authorized_results:
            if eval_res.metrics.get("ensemble_size", 0) > 1:
                has_ensemble_metrics = True
                break
        
        # Calculate the total number of examples
        num_examples_total = sum([eval_res.num_examples for _, eval_res in authorized_results])
        
        # Create weights for weighted average
        if num_examples_total > 0:
            weights = [eval_res.num_examples / num_examples_total for _, eval_res in authorized_results]
        else:
            weights = [1.0 / len(authorized_results) for _ in authorized_results]
        
        # Aggregate evaluation results
        if has_ensemble_metrics:
            # Use ensemble evaluation aggregation
            loss_aggregated, metrics = self.ensemble_aggregator.aggregate_evaluate_results(
                authorized_results, weights
            )
        else:
            # Use standard aggregation
            loss_aggregated, metrics = super().aggregate_evaluate(server_round, authorized_results, failures)
        
        # Add metrics about client participation
        metrics["total_clients"] = len(results)
        metrics["authorized_clients"] = len(authorized_results)
        metrics["unauthorized_clients"] = len(unauthorized_clients)
        
        # Calculate average accuracy
        accuracies = [res.metrics.get("accuracy", 0.0) for _, res in authorized_results]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            metrics["avg_accuracy"] = avg_accuracy
        
        # Add evaluation metrics to history
        if loss_aggregated is not None:
            eval_metrics = {
                "round": server_round,
                "eval_loss": loss_aggregated,
                "eval_metrics": metrics,
                "num_clients": len(authorized_results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Save evaluation metrics
            self.metrics_history.append(eval_metrics)
            
            # Log the evaluation results
            logger.info(f"Round {server_round} evaluation: Loss={loss_aggregated:.4f}, Metrics={metrics}")
        
        return loss_aggregated, metrics
    
    def save_metrics_history(self, filepath: str = "metrics/metrics_history.json"):
        """Save metrics history to a file."""
        # Save combined metrics history
        with open(filepath, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved metrics history to {filepath}")
        
        # Save individual round metrics to separate files
        metrics_dir = Path(filepath).parent
        for round_metrics in self.metrics_history:
            round_num = round_metrics.get("round", 0)
            round_file = metrics_dir / f"round_{round_num}_metrics.json"
            with open(round_file, "w") as f:
                json.dump(round_metrics, f, indent=2)
            logger.info(f"Saved round {round_num} metrics to {round_file}")
    
    def save_client_stats(self, filepath: str = "metrics/client_stats.json"):
        """Save client contribution statistics to a file."""
        if not self.blockchain:
            logger.warning("Blockchain connector not available. Cannot save client stats.")
            return
        
        client_stats = {}
        metrics_dir = Path(filepath).parent
        
        try:
            # Get all authorized clients
            clients = self.authorized_clients
            
            for client in clients:
                try:
                    # Get contribution details
                    details = self.blockchain.get_client_contribution_details(client)
                    
                    # Get contribution records
                    records = self.blockchain.get_client_contribution_records(client)
                    
                    # Store in stats
                    client_stats[client] = {
                        "details": details,
                        "records": records
                    }
                    
                    # Save individual client stats to separate files
                    client_file = metrics_dir / f"client_{client[-8:]}_stats.json"
                    with open(client_file, "w") as f:
                        json.dump(client_stats[client], f, indent=2)
                    logger.info(f"Saved client {client[-8:]} stats to {client_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to get stats for client {client}: {e}")
            
            # Save combined stats to file
            with open(filepath, "w") as f:
                json.dump(client_stats, f, indent=2)
                
            logger.info(f"Saved combined client stats to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save client stats: {e}")
            
    def save_model_history(self, filepath: str = "metrics/model_history.json"):
        """Save model history from blockchain to a file."""
        if not self.blockchain:
            logger.warning("Blockchain connector not available. Cannot save model history.")
            return
        
        try:
            # Get model history from blockchain
            models = self.blockchain.get_all_models(self.version_prefix)
            
            metrics_dir = Path(filepath).parent
            
            # Save combined model history
            with open(filepath, "w") as f:
                json.dump(models, f, indent=2)
            
            logger.info(f"Saved model history to {filepath}")
            
            # Save individual model data to separate files
            for model in models:
                round_num = model.get("round", 0)
                model_file = metrics_dir / f"model_round_{round_num}.json"
                
                # Try to get the model data from IPFS
                try:
                    ipfs_hash = model.get("ipfs_hash")
                    if ipfs_hash:
                        model_data = self.ipfs.get_json(ipfs_hash)
                        if model_data:
                            # Save the complete model data including weights
                            with open(model_file, "w") as f:
                                json.dump(model_data, f, indent=2)
                            logger.info(f"Saved round {round_num} model data to {model_file}")
                            
                            # Save a lightweight model info file (without weights)
                            info_file = metrics_dir / f"model_round_{round_num}_info.json"
                            model_info = {**model}
                            if model_data and "info" in model_data:
                                model_info["model_info"] = model_data["info"]
                            with open(info_file, "w") as f:
                                json.dump(model_info, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to get model data for round {round_num}: {e}")
                    # Save just the model metadata if we couldn't get the full data
                    model_file = metrics_dir / f"model_round_{round_num}_metadata.json"
                    with open(model_file, "w") as f:
                        json.dump(model, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save model history: {e}")

def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 3,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    fraction_fit: float = 1.0,
    ipfs_url: str = "http://127.0.0.1:5001",
    ganache_url: str = "http://127.0.0.1:7545",
    contract_address: Optional[str] = None,
    private_key: Optional[str] = None,
    deploy_contract: bool = False,
    version_prefix: str = "1.0",
    authorized_clients_only: bool = True,
    authorized_clients: Optional[List[str]] = None,
    round_rewards: int = 1000,
    device: str = "cpu"
) -> None:
    """
    Start the enhanced federated learning server with GA-Stacking, IPFS and blockchain integration.
    
    Args:
        server_address: Server address (host:port)
        num_rounds: Number of federated learning rounds
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        fraction_fit: Fraction of clients to use for training
        ipfs_url: IPFS API URL
        ganache_url: Ganache blockchain URL
        contract_address: Federation contract address (if already deployed)
        private_key: Private key for blockchain transactions
        deploy_contract: Whether to deploy a new contract if address not provided
        version_prefix: Version prefix for model versioning
        authorized_clients_only: Whether to only accept contributions from authorized clients
        authorized_clients: List of client addresses to authorize (if not already authorized)
        round_rewards: Reward points to distribute each round
        device: Device to use for computation
    """
    # Initialize IPFS connector
    ipfs_connector = IPFSConnector(ipfs_api_url=ipfs_url)
    logger.info(f"Initialized IPFS connector: {ipfs_url}")
    
    # Initialize blockchain connector
    blockchain_connector = None
    if ganache_url:
        try:
            blockchain_connector = BlockchainConnector(
                ganache_url=ganache_url,
                contract_address=contract_address,
                private_key=private_key
            )
            
            # Deploy contract if needed
            if contract_address is None and deploy_contract:
                contract_address = blockchain_connector.deploy_contract()
                logger.info(f"Deployed new contract at: {contract_address}")
                
                # Save contract address to file for future use
                with open("contract_address.txt", "w") as f:
                    f.write(contract_address)
                
                # Initialize the blockchain connector with the new contract
                blockchain_connector = BlockchainConnector(
                    ganache_url=ganache_url,
                    contract_address=contract_address,
                    private_key=private_key
                )
            elif contract_address is None:
                logger.warning("No contract address provided and deploy_contract=False. Blockchain features disabled.")
                blockchain_connector = None
                
            # Authorize clients if provided
            if blockchain_connector and authorized_clients:
                # Check which clients are not already authorized
                to_authorize = []
                for client in authorized_clients:
                    if not blockchain_connector.is_client_authorized(client):
                        to_authorize.append(client)
                
                if to_authorize:
                    logger.info(f"Authorizing {len(to_authorize)} new clients")
                    blockchain_connector.authorize_clients(to_authorize)
                else:
                    logger.info("All provided clients are already authorized")
                
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connector: {e}")
            logger.warning("Continuing without blockchain integration")
            blockchain_connector = None
    
    # Configure strategy with GA-Stacking support
    strategy = EnhancedFedAvgWithGA(
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_fit_clients,
        ipfs_connector=ipfs_connector,
        blockchain_connector=blockchain_connector,
        version_prefix=version_prefix,
        authorized_clients_only=authorized_clients_only,
        round_rewards=round_rewards,
        device=device
    )
    
    
    # Create metrics directory with timestamp to keep each training run separate
    vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
    local_time = datetime.now(vn_timezone)
    timestamp = local_time.strftime("%Y-%m-%d_%H-%M-%S")
    metrics_dir = Path(f"metrics/run_{timestamp}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Start server
    server = fl.server.Server(client_manager=fl.server.SimpleClientManager(), strategy=strategy)
    
    # Run server
    fl.server.start_server(
        server_address=server_address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )
    
    # Save metrics history (both combined and per-round)
    strategy.save_metrics_history(filepath=str(metrics_dir / "metrics_history.json"))
    
    # Save client stats
    strategy.save_client_stats(filepath=str(metrics_dir / "client_stats.json"))
    
    # Save model history
    strategy.save_model_history(filepath=str(metrics_dir / "model_history.json"))
    
    # Create a summary file with key information
    summary = {
        "timestamp": timestamp,
        "num_rounds": num_rounds,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "authorized_clients_only": authorized_clients_only,
        "version_prefix": version_prefix,
        "contract_address": contract_address,
        "final_metrics": strategy.metrics_history[-1] if strategy.metrics_history else None
    }
    
    with open(metrics_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Server completed {num_rounds} rounds of federated learning with GA-Stacking")
    logger.info(f"All metrics saved to {metrics_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start enhanced FL server with GA-Stacking, IPFS and blockchain integration")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8088", help="Server address (host:port)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum number of clients for training")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum number of clients for evaluation")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients to use for training")
    parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--ganache-url", type=str, default="http://192.168.1.146:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="Federation contract address")
    parser.add_argument("--private-key", type=str, help="Private key for blockchain transactions")
    parser.add_argument("--deploy-contract", action="store_true", help="Deploy a new contract if address not provided")
    parser.add_argument("--version-prefix", type=str, default="1.0", help="Version prefix for model versioning")
    parser.add_argument("--authorized-clients-only", action="store_true", help="Only accept contributions from authorized clients")
    parser.add_argument("--authorize-clients", nargs="+", help="List of client addresses to authorize")
    parser.add_argument("--round-rewards", type=int, default=1000, help="Reward points to distribute each round")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for computation")
    
    args = parser.parse_args()
    
    # Check if contract address is stored in file
    if args.contract_address is None and not args.deploy_contract:
        try:
            with open("contract_address.txt", "r") as f:
                args.contract_address = f.read().strip()
                logger.info(f"Loaded contract address from file: {args.contract_address}")
        except FileNotFoundError:
            logger.warning("No contract address provided or found in file")
    
    start_server(
        server_address=args.server_address,
        num_rounds=args.rounds,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        fraction_fit=args.fraction_fit,
        ipfs_url=args.ipfs_url,
        ganache_url=args.ganache_url,
        contract_address=args.contract_address,
        private_key=args.private_key,
        deploy_contract=args.deploy_contract,
        version_prefix=args.version_prefix,
        authorized_clients_only=args.authorized_clients_only,
        authorized_clients=args.authorize_clients,
        round_rewards=args.round_rewards,
        device=args.device
    )