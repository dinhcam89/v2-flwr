# """
# Enhanced Federated Learning Client with IPFS integration and blockchain authentication.
# """

# import os
# import json
# import pickle
# import time
# from typing import Dict, List, Optional, Tuple, Union, Any
# import logging
# from pathlib import Path
# from datetime import datetime, timezone

# import flwr as fl
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from flwr.common import (
#     Parameters,
#     Scalar,
#     FitIns,
#     FitRes,
#     EvaluateIns,
#     EvaluateRes,
#     ndarrays_to_parameters,
#     parameters_to_ndarrays,
# )
# import numpy as np

# from ipfs_connector import IPFSConnector
# from blockchain_connector import BlockchainConnector

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# )
# logger = logging.getLogger("FL-Client")


# class SimpleLinearModel(nn.Module):
#     """Simple linear model for demonstration purposes."""
    
#     def __init__(self, input_dim=10, output_dim=1):
#         super(SimpleLinearModel, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
        
#     def forward(self, x):
#         return self.linear(x)


# class EnhancedClient(fl.client.NumPyClient):
#     """Federated Learning client with IPFS integration and blockchain authentication."""
    
#     def __init__(
#         self,
#         model: nn.Module,
#         train_loader: DataLoader,
#         test_loader: DataLoader,
#         ipfs_connector: Optional[IPFSConnector] = None,
#         blockchain_connector: Optional[BlockchainConnector] = None,
#         wallet_address: Optional[str] = None,
#         private_key: Optional[str] = None,
#         device: str = "cpu",
#         client_id: str = None,
#     ):
#         self.model = model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.ipfs = ipfs_connector or IPFSConnector()
#         self.blockchain = blockchain_connector
#         self.wallet_address = wallet_address
#         self.private_key = private_key
#         self.device = torch.device(device)
#         self.model.to(self.device)
#         self.client_id = client_id or f"client-{os.getpid()}"
        
#         # Metrics storage
#         self.metrics_history = []
        
#         logger.info(f"Initialized {self.client_id} with IPFS node: {self.ipfs.ipfs_api_url}")
        
#         # Verify blockchain authentication if available
#         if self.blockchain and self.wallet_address:
#             try:
#                 is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
#                 if is_authorized:
#                     logger.info(f"Client {self.wallet_address} is authorized on the blockchain ✅")
#                 else:
#                     logger.warning(f"Client {self.wallet_address} is NOT authorized on the blockchain ❌")
#                     logger.warning("The server may reject this client's contributions")
#             except Exception as e:
#                 logger.error(f"Failed to verify client authorization: {e}")
    
#     def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
#         """Get model parameters as a list of NumPy arrays."""
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
#     def set_parameters(self, parameters: List[np.ndarray]) -> None:
#         """Set model parameters from a list of NumPy arrays."""
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = {k: torch.Tensor(v) for k, v in params_dict}
#         self.model.load_state_dict(state_dict, strict=True)
    
#     def set_parameters_from_ipfs(self, ipfs_hash: str) -> None:
#         """Set model parameters from IPFS."""
#         try:
#             # Get model data from IPFS
#             model_data = self.ipfs.get_json(ipfs_hash)
            
#             if model_data and "state_dict" in model_data:
#                 # Load state dict
#                 state_dict = {
#                     k: torch.tensor(v, device=self.device)
#                     for k, v in model_data["state_dict"].items()
#                 }
                
#                 # Update model
#                 self.model.load_state_dict(state_dict)
#                 logger.info(f"Model loaded from IPFS: {ipfs_hash}")
                
#                 # Log metadata
#                 if "info" in model_data:
#                     logger.info(f"Model info: {model_data['info']}")
#             else:
#                 logger.error(f"Invalid model data from IPFS: {ipfs_hash}")
                
#         except Exception as e:
#             logger.error(f"Failed to load model from IPFS: {e}")
    
#     def save_parameters_to_ipfs(
#         self, parameters: List[np.ndarray], round_num: int, metrics: Dict[str, Scalar] = None
#     ) -> str:
#         """Save model parameters to IPFS."""
#         # Create state dict from parameters
#         state_dict = {}
#         for i, key in enumerate(self.model.state_dict().keys()):
#             if i < len(parameters):
#                 state_dict[key] = parameters[i].tolist()
        
#         # Create metadata
#         model_metadata = {
#             "state_dict": state_dict,
#             "info": {
#                 "round": round_num,
#                 "client_id": self.client_id,
#                 "wallet_address": self.wallet_address if self.wallet_address else "unknown",
#                 "timestamp": datetime.now(timezone.utc).isoformat()
#             }
#         }
        
#         # Add metrics if provided
#         if metrics:
#             model_metadata["info"]["metrics"] = metrics
        
#         # Store in IPFS
#         ipfs_hash = self.ipfs.add_json(model_metadata)
#         logger.info(f"Stored model in IPFS: {ipfs_hash}")
        
#         return ipfs_hash
    
    
#     def fit(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[Parameters, int, Dict[str, Scalar]]:
#         """Train the model on the local dataset, with support for CID-only mode."""
#         # Check if client is authorized (if blockchain is available)
#         if self.blockchain and self.wallet_address:
#             try:
#                 is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
#                 if not is_authorized:
#                     logger.warning(f"Client {self.wallet_address} is not authorized to participate")
#                     # Return empty parameters with auth failure flag
#                     return parameters, 0, {"error": "client_not_authorized"}
#             except Exception as e:
#                 logger.error(f"Failed to check client authorization: {e}")
#                 # Continue anyway, server will check again
        
#         # Get global model from IPFS if hash is provided
#         ipfs_hash = config.get("ipfs_hash", None)
#         round_num = config.get("server_round", 0)
#         cid_only_mode = config.get("cid_only", False)
        
#         if ipfs_hash:
#             # Always use IPFS in CID-only mode, or optionally in hybrid mode
#             self.set_parameters_from_ipfs(ipfs_hash)
#             logger.info(f"Set parameters from IPFS with hash: {ipfs_hash}")
#         elif not cid_only_mode:
#             # Fallback to direct parameters only if not in CID-only mode
#             self.set_parameters(parameters_to_ndarrays(parameters))
#             logger.info("Set parameters directly from server")
#         else:
#             # This shouldn't happen in CID-only mode
#             logger.error("No IPFS hash provided in CID-only mode")
#             return parameters, 0, {"error": "missing_ipfs_hash"}
        
#         # Get training config
#         epochs = int(config.get("epochs", 1))
#         learning_rate = float(config.get("learning_rate", 0.01))
        
#         # Define loss function and optimizer
#         criterion = nn.MSELoss()
#         optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
#         # Training loop
#         self.model.train()
#         train_loss = 0.0
#         num_samples = 0
        
#         for epoch in range(epochs):
#             epoch_loss = 0.0
#             epoch_samples = 0
            
#             for batch_idx, (data, target) in enumerate(self.train_loader):
#                 data, target = data.to(self.device), target.to(self.device)
                
#                 optimizer.zero_grad()
#                 output = self.model(data)
#                 loss = criterion(output, target)
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item() * len(data)
#                 epoch_samples += len(data)
            
#             avg_epoch_loss = epoch_loss / epoch_samples
#             logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
#             train_loss += epoch_loss
#             num_samples += epoch_samples
        
#         # Calculate average loss
#         avg_loss = train_loss / num_samples if num_samples > 0 else None
        
#         # Get updated parameters
#         parameters_updated = self.get_parameters(config={})
        
#         # Save model to IPFS
#         accuracy = self._calculate_accuracy()
#         metrics = {
#             "loss": float(avg_loss) if avg_loss is not None else 0.0,
#             "accuracy": float(accuracy)
#         }
#         client_ipfs_hash = self.save_parameters_to_ipfs(parameters_updated, round_num, metrics)
        
#         # Record contribution to blockchain if available
#         if self.blockchain and self.wallet_address:
#             try:
#                 tx_hash = self.blockchain.record_contribution(
#                     client_address=self.wallet_address,
#                     round_num=round_num,
#                     ipfs_hash=client_ipfs_hash,
#                     accuracy=accuracy
#                 )
#                 logger.info(f"Contribution recorded on blockchain, tx: {tx_hash}")
#                 metrics["blockchain_tx"] = tx_hash
#             except Exception as e:
#                 logger.error(f"Failed to record contribution on blockchain: {e}")
        
#         # Add to metrics history
#         self.metrics_history.append({
#             "round": round_num,
#             "fit_loss": float(avg_loss) if avg_loss is not None else 0.0,
#             "accuracy": float(accuracy),
#             "train_samples": num_samples,
#             "ipfs_hash": client_ipfs_hash,
#             "wallet_address": self.wallet_address if self.wallet_address else "unknown",
#             "timestamp": datetime.now(timezone.utc).isoformat()
#         })
        
#         # Include IPFS hash in metrics
#         metrics["ipfs_hash"] = ipfs_hash  # Return the server's hash for tracking
#         metrics["client_ipfs_hash"] = client_ipfs_hash
#         metrics["wallet_address"] = self.wallet_address if self.wallet_address else "unknown"
        
#         # If in CID-only mode, return minimal parameters
#         if cid_only_mode:
#             logger.info(f"Operating in CID-only mode, returning minimal parameters with CID: {client_ipfs_hash}")
#             # Create minimal parameters (a single value)
#             minimal_params = [np.array([0.0], dtype=np.float32)]
#             metrics["cid_only"] = True
            
#             # Return the numpy arrays directly, NOT the Parameters proto object
#             return minimal_params, num_samples, metrics
#         else:
#             # Convert parameters to Flower format
#             parameters_proto = ndarrays_to_parameters(parameters_updated)

#         # Make sure metrics only contains serializable values
#         for key in list(metrics.keys()):
#             if not isinstance(metrics[key], (int, float, str, bool)):
#                 metrics[key] = str(metrics[key])

#         # Return the exact expected format
#         return parameters_proto, num_samples, metrics
    
#     def evaluate(
#         self, parameters: Parameters, config: Dict[str, Scalar]
#     ) -> Tuple[float, int, Dict[str, Scalar]]:
#         """Evaluate the model on the local test dataset."""
#         # Check if client is authorized (if blockchain is available)
#         if self.blockchain and self.wallet_address:
#             try:
#                 is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
#                 if not is_authorized:
#                     logger.warning(f"Client {self.wallet_address} is not authorized to participate in evaluation")
#                     # Return empty evaluation with auth failure flag
#                     return 0.0, 0, {"error": "client_not_authorized"}
#             except Exception as e:
#                 logger.error(f"Failed to check client authorization: {e}")
#                 # Continue anyway, server will check again
        
#         # Get global model from IPFS if hash is provided
#         ipfs_hash = config.get("ipfs_hash", None)
#         round_num = config.get("server_round", 0)
        
#         if ipfs_hash:
#             self.set_parameters_from_ipfs(ipfs_hash)
#         else:
#             # Fallback to direct parameters
#             self.set_parameters(parameters_to_ndarrays(parameters))
        
#         # Evaluation loop
#         self.model.eval()
#         test_loss = 0.0
#         num_samples = 0
#         criterion = nn.MSELoss()
        
#         with torch.no_grad():
#             for data, target in self.test_loader:
#                 data, target = data.to(self.device), target.to(self.device)
                
#                 output = self.model(data)
#                 loss = criterion(output, target)
                
#                 test_loss += loss.item() * len(data)
#                 num_samples += len(data)
        
#         # Calculate average loss
#         avg_loss = test_loss / num_samples if num_samples > 0 else None
        
#         # Calculate accuracy
#         accuracy = self._calculate_accuracy()
        
#         # Add to metrics history
#         self.metrics_history.append({
#             "round": round_num,
#             "eval_loss": float(avg_loss) if avg_loss is not None else 0.0,
#             "accuracy": float(accuracy),
#             "eval_samples": num_samples,
#             "ipfs_hash": ipfs_hash,
#             "wallet_address": self.wallet_address if self.wallet_address else "unknown",
#             "timestamp": datetime.now(timezone.utc).isoformat()
#         })
        
#         metrics = {
#             "loss": float(avg_loss) if avg_loss is not None else 0.0,
#             "accuracy": float(accuracy),
#             "wallet_address": self.wallet_address if self.wallet_address else "unknown"
#         }
        
#         return float(avg_loss) if avg_loss is not None else 0.0, num_samples, metrics
    
#     def _calculate_accuracy(self) -> float:
#         """Calculate accuracy on the test dataset."""
#         self.model.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for data, target in self.test_loader:
#                 data, target = data.to(self.device), target.to(self.device)
                
#                 outputs = self.model(data)
                
#                 # For regression, we consider a prediction correct if it's within a threshold
#                 threshold = 0.5
#                 correct += torch.sum((torch.abs(outputs - target) < threshold)).item()
#                 total += target.size(0)
        
#         accuracy = 100.0 * correct / total if total > 0 else 0.0
#         logger.info(f"Accuracy: {accuracy:.2f}%")
        
#         return accuracy
    
#     def save_metrics_history(self, filepath: str = "client_metrics.json"):
#         """Save metrics history to a file."""
#         with open(filepath, "w") as f:
#             json.dump(self.metrics_history, f, indent=2)
#         logger.info(f"Saved metrics history to {filepath}")