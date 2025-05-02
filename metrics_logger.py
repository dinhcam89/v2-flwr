import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

class FederationMetricsLogger:
    """A class for logging metrics during federated learning."""
    
    def __init__(self, log_dir: str = "metrics", prefix: str = ""):
        """Initialize the metrics logger.
        
        Args:
            log_dir: Directory to store the metrics logs
            prefix: Optional prefix for the log files
        """
        self.log_dir = log_dir
        self.prefix = prefix
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_history = {
            "rounds": [],
            "global": {
                "loss": [],
                "accuracy": []
            },
            "client_metrics": {},
            "blockchain_metrics": {
                "transactions": [],
                "gas_used": []
            },
            "ipfs_metrics": {
                "model_hashes": []
            },
            "timestamps": [],
            "system_metrics": {
                "round_durations": []
            }
        }
        self.current_round = 0
        self.round_start_time = None
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("FL-Metrics")
        self.logger.setLevel(logging.INFO)
        
        # Ensure we don't add duplicate handlers
        if not self.logger.handlers:
            # Create file handler
            file_handler = logging.FileHandler(os.path.join(
                log_dir, f"{prefix}federation_metrics_{self.timestamp}.log"))
            file_handler.setLevel(logging.INFO)
            
            # Create formatter and add it to the handler
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(file_handler)
    
    def start_round(self, round_num: int) -> None:
        """Mark the start of a new federated learning round.
        
        Args:
            round_num: The current round number
        """
        self.current_round = round_num
        self.round_start_time = time.time()
        self.metrics_history["rounds"].append(round_num)
        self.metrics_history["timestamps"].append(datetime.now().isoformat())
        
        self.logger.info(f"Starting round {round_num}")
    
    def log_global_metrics(self, loss: float, metrics: Dict[str, Any]) -> None:
        """Log global metrics after a round completes.
        
        Args:
            loss: The global loss value
            metrics: Dictionary of additional metrics
        """
        self.metrics_history["global"]["loss"].append((self.current_round, loss))
        
        if "avg_accuracy" in metrics:
            self.metrics_history["global"]["accuracy"].append((self.current_round, metrics["avg_accuracy"]))
        
        # Log all metrics from this round
        round_metrics = {f"round_{self.current_round}": {"loss": loss, **metrics}}
        self._save_round_metrics(round_metrics)
        
        self.logger.info(f"Round {self.current_round} metrics - Loss: {loss}, Metrics: {metrics}")
    
    def log_client_contribution(self, client_id: str, metrics: Dict[str, Any] = None) -> None:
        """Log an individual client's contribution.
        
        Args:
            client_id: The client's identifier
            metrics: Optional metrics from the client
        """
        if client_id not in self.metrics_history["client_metrics"]:
            self.metrics_history["client_metrics"][client_id] = {"rounds_participated": []}
        
        self.metrics_history["client_metrics"][client_id]["rounds_participated"].append(self.current_round)
        
        if metrics:
            if "metrics" not in self.metrics_history["client_metrics"][client_id]:
                self.metrics_history["client_metrics"][client_id]["metrics"] = {}
            
            self.metrics_history["client_metrics"][client_id]["metrics"][f"round_{self.current_round}"] = metrics
        
        self.logger.info(f"Client {client_id} contributed to round {self.current_round}")
    
    def log_blockchain_transaction(self, tx_hash: str, operation: str, gas_used: int = None) -> None:
        """Log a blockchain transaction.
        
        Args:
            tx_hash: The transaction hash
            operation: The operation performed (e.g., 'register_model', 'update_model')
            gas_used: Optional gas used for the transaction
        """
        tx_data = {
            "round": self.current_round,
            "tx_hash": tx_hash,
            "operation": operation,
            "timestamp": datetime.now().isoformat()
        }
        
        if gas_used is not None:
            tx_data["gas_used"] = gas_used
            self.metrics_history["blockchain_metrics"]["gas_used"].append((self.current_round, gas_used))
        
        self.metrics_history["blockchain_metrics"]["transactions"].append(tx_data)
        
        self.logger.info(f"Blockchain transaction {tx_hash} for {operation} in round {self.current_round}")
    
    def log_ipfs_model(self, model_hash: str, model_type: str = "global") -> None:
        """Log an IPFS model hash.
        
        Args:
            model_hash: The IPFS hash of the model
            model_type: The type of model ('global' or 'evaluation')
        """
        model_data = {
            "round": self.current_round,
            "model_hash": model_hash,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.metrics_history["ipfs_metrics"]["model_hashes"].append(model_data)
        
        self.logger.info(f"IPFS model stored with hash {model_hash} for round {self.current_round}")
    
    def end_round(self) -> None:
        """Mark the end of the current federated learning round."""
        if self.round_start_time is not None:
            duration = time.time() - self.round_start_time
            self.metrics_history["system_metrics"]["round_durations"].append((self.current_round, duration))
            
            self.logger.info(f"Round {self.current_round} completed in {duration:.2f} seconds")
    
    def _save_round_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics for the current round to a dedicated file.
        
        Args:
            metrics: The metrics to save
        """
        round_file = os.path.join(
            self.log_dir, 
            f"{self.prefix}round_{self.current_round}_metrics_{self.timestamp}.json"
        )
        
        with open(round_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_all_metrics(self) -> str:
        """Save all collected metrics to a file.
        
        Returns:
            The path to the saved metrics file
        """
        metrics_file = os.path.join(
            self.log_dir, 
            f"{self.prefix}federation_metrics_history_{self.timestamp}.json"
        )
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"All metrics saved to {metrics_file}")
        
        return metrics_file
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the metrics.
        
        Returns:
            A dictionary with metrics summary
        """
        num_rounds = len(self.metrics_history["rounds"])
        num_clients = len(self.metrics_history["client_metrics"])
        
        summary = {
            "total_rounds": num_rounds,
            "total_clients": num_clients,
            "final_loss": self.metrics_history["global"]["loss"][-1][1] if self.metrics_history["global"]["loss"] else None,
            "final_accuracy": self.metrics_history["global"]["accuracy"][-1][1] if self.metrics_history["global"]["accuracy"] else None,
            "round_durations": {
                "average": sum([d[1] for d in self.metrics_history["system_metrics"]["round_durations"]]) / num_rounds if num_rounds > 0 else 0,
                "min": min([d[1] for d in self.metrics_history["system_metrics"]["round_durations"]]) if self.metrics_history["system_metrics"]["round_durations"] else 0,
                "max": max([d[1] for d in self.metrics_history["system_metrics"]["round_durations"]]) if self.metrics_history["system_metrics"]["round_durations"] else 0
            }
        }
        
        return summary
    
    def save_summary(self) -> str:
        """Generate and save a summary of the metrics.
        
        Returns:
            The path to the saved summary file
        """
        summary = self.generate_summary()
        
        summary_file = os.path.join(
            self.log_dir, 
            f"{self.prefix}federation_summary_{self.timestamp}.json"
        )
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Metrics summary saved to {summary_file}")
        
        return summary_file