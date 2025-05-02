"""
Enhanced base model implementations for GA-Stacking in federated learning.
Provides different model architectures including scikit-learn compatible models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from collections import OrderedDict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("BaseModels")


class LinearModel(nn.Module):
    """Simple linear model."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 1, hidden_dim: int = 64):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPModel(nn.Module):
    """Multi-layer perceptron model with configurable hidden layers."""
    
    def __init__(
        self, 
        input_dim: int = 10, 
        output_dim: int = 1, 
        hidden_dims: List[int] = [64, 32],
        activation: nn.Module = nn.ReLU()
    ):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Scikit-learn compatible wrappers
class SklearnModelWrapper(nn.Module):
    """Base wrapper for scikit-learn models to be used in PyTorch-based GA-Stacking."""
    
    def __init__(self, model_type: str = "lr", input_dim: int = 10, output_dim: int = 1):
        super(SklearnModelWrapper, self).__init__()
        self.model_type = model_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.is_initialized = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the wrapped model."""
        # Convert PyTorch tensor to numpy if needed
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        # Reshape if needed
        if len(x_np.shape) == 3 and x_np.shape[1] == 1:
            # Handle case where input is [batch_size, 1, features]
            x_np = x_np.squeeze(1)
            
        # Make prediction
        if not self.is_initialized:
            # Return zeros if model is not initialized
            if isinstance(x, torch.Tensor):
                return torch.zeros(x.shape[0], self.output_dim, device=x.device)
            else:
                return np.zeros((x_np.shape[0], self.output_dim))
            
        try:
            # Make prediction
            if hasattr(self.model, "predict"):
                y_pred = self.model.predict(x_np)
            else:
                y_pred = self.model.decision_function(x_np)
                
            # Reshape if needed
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)
                
            # Convert back to torch tensor if input was tensor
            if isinstance(x, torch.Tensor):
                y_pred = torch.tensor(y_pred, dtype=torch.float32, device=x.device)
                
            return y_pred
            
        except Exception as e:
            logger.error(f"Error in forward pass for {self.model_type}: {e}")
            # Return zeros as fallback
            if isinstance(x, torch.Tensor):
                return torch.zeros(x.shape[0], self.output_dim, device=x.device)
            else:
                return np.zeros((x_np.shape[0], self.output_dim))

class LinearRegressionWrapper(SklearnModelWrapper):
    """Wrapper for Linear Regression model."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 1):
        super(LinearRegressionWrapper, self).__init__(
            model_type="lr", input_dim=input_dim, output_dim=output_dim
        )
        
        # Initialize with dummy model that can be updated
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.coef_ = np.zeros((output_dim, input_dim))
        self.intercept_ = np.zeros(output_dim)
        self.is_initialized = False
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set model parameters from dictionary."""
        if "coef" in params and "intercept" in params:
            coef = np.array(params["coef"])
            intercept = np.array(params["intercept"])
            
            # Reshape coefficients if needed
            if len(coef.shape) == 1:
                coef = coef.reshape(1, -1)
                
            # Ensure dimensions match
            if coef.shape != (self.output_dim, self.input_dim):
                logger.warning(f"Coefficient shape mismatch. Expected {(self.output_dim, self.input_dim)}, got {coef.shape}")
                coef = np.zeros((self.output_dim, self.input_dim))
            
            if len(intercept) != self.output_dim:
                logger.warning(f"Intercept length mismatch. Expected {self.output_dim}, got {len(intercept)}")
                intercept = np.zeros(self.output_dim)
                
            self.coef_ = coef
            self.intercept_ = intercept
            
            # Update model attributes
            self.model.coef_ = coef
            self.model.intercept_ = intercept
            
            self.is_initialized = True
            
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters as dictionary."""
        return {
            "estimator": "lr",
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_.tolist(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }


class SVCWrapper(SklearnModelWrapper):
    """Wrapper for Support Vector Classifier."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 1):
        super(SVCWrapper, self).__init__(
            model_type="svc", input_dim=input_dim, output_dim=output_dim
        )
        
        # Initialize with dummy SVC
        from sklearn.svm import SVC
        self.model = SVC(kernel='linear')
        self.dual_coef_ = np.zeros((1, 2))
        self.support_vectors_ = np.zeros((2, input_dim))
        self.intercept_ = np.zeros(1)
        self.n_support_ = np.array([1, 1])
        self.support_ = np.array([0, 1])
        self.is_initialized = False
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set model parameters from dictionary."""
        if all(k in params for k in ["dual_coef", "support_vectors", "intercept"]):
            try:
                dual_coef = np.array(params["dual_coef"])
                support_vectors = np.array(params["support_vectors"])
                intercept = np.array(params["intercept"])
                
                # Check dimensions
                if support_vectors.shape[1] != self.input_dim:
                    logger.warning(f"Support vectors dimension mismatch. Expected {self.input_dim}, got {support_vectors.shape[1]}")
                    support_vectors = np.zeros((2, self.input_dim))
                    dual_coef = np.array([[1.0, -1.0]])
                
                # Create a new SVC model instead of modifying the existing one
                from sklearn.svm import SVC
                self.model = SVC(kernel='linear')
                
                # Store the values as attributes on the wrapper, not directly on the model
                self.dual_coef_ = dual_coef
                self.support_vectors_ = support_vectors
                self.intercept_ = intercept
                self.n_support_ = np.array([support_vectors.shape[0]])
                self.support_ = np.arange(support_vectors.shape[0])
                
                # Override predict and decision_function to use our stored values
                self._original_predict = self.model.predict
                self._original_decision_function = self.model.decision_function
                
                def custom_predict(X):
                    # Simple implementation using our stored parameters
                    decision_values = np.dot(X, self.support_vectors_.T)
                    decision_values = np.dot(decision_values, self.dual_coef_.T) + self.intercept_
                    return np.sign(decision_values)
                    
                def custom_decision_function(X):
                    decision_values = np.dot(X, self.support_vectors_.T)
                    return np.dot(decision_values, self.dual_coef_.T) + self.intercept_
                
                # Monkey patch the methods
                self.model.predict = custom_predict
                self.model.decision_function = custom_decision_function
                
                self.is_initialized = True
                
            except Exception as e:
                logger.error(f"Error setting SVC parameters: {e}")
            
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters as dictionary."""
        return {
            "estimator": "svc",
            "dual_coef": self.dual_coef_.tolist(),
            "support_vectors": self.support_vectors_.tolist(),
            "intercept": self.intercept_.tolist(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }


class RandomForestWrapper(SklearnModelWrapper):
    """Wrapper for Random Forest model."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 1):
        super(RandomForestWrapper, self).__init__(
            model_type="rf", input_dim=input_dim, output_dim=output_dim
        )
        
        # Initialize with dummy Random Forest
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=10)
        self.n_estimators = 10
        self.feature_importances_ = np.ones(input_dim) / input_dim
        self.is_initialized = False
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set model parameters from dictionary."""
        if "n_estimators" in params and "feature_importances" in params:
            try:
                self.n_estimators = params["n_estimators"]
                feature_importances = np.array(params["feature_importances"])
                
                # Check dimensions
                if len(feature_importances) != self.input_dim:
                    logger.warning(f"Feature importances dimension mismatch. Expected {self.input_dim}, got {len(feature_importances)}")
                    feature_importances = np.ones(self.input_dim) / self.input_dim
                
                # Create a new model with the specified number of estimators
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=self.n_estimators)
                
                # Store feature importances on the wrapper, not directly on the model
                self.feature_importances_ = feature_importances
                
                # Override predict to use our feature importances
                self._original_predict = self.model.predict
                
                def custom_predict(X):
                    # Simple weighted prediction using feature importances
                    weighted_X = X * self.feature_importances_
                    return np.sum(weighted_X, axis=1).reshape(-1, 1)
                    
                # Monkey patch the method
                self.model.predict = custom_predict
                
                self.is_initialized = True
                
            except Exception as e:
                logger.error(f"Error setting RF parameters: {e}")
            
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters as dictionary."""
        return {
            "estimator": "rf",
            "n_estimators": self.n_estimators,
            "feature_importances": self.feature_importances_.tolist(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }


class MetaLearnerWrapper(SklearnModelWrapper):
    """Wrapper for meta learner model (usually a simple linear model)."""
    
    def __init__(self, input_dim: int = 3, output_dim: int = 1):
        super(MetaLearnerWrapper, self).__init__(
            model_type="meta_lr", input_dim=input_dim, output_dim=output_dim
        )
        
        # Initialize with dummy model
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.coef_ = np.ones((output_dim, input_dim)) / input_dim
        self.intercept_ = np.zeros(output_dim)
        self.is_initialized = False
        
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set model parameters from dictionary."""
        if "coef" in params and "intercept" in params:
            coef = np.array(params["coef"])
            intercept = np.array(params["intercept"])
            
            # Reshape coefficients if needed
            if len(coef.shape) == 1:
                coef = coef.reshape(1, -1)
                
            # Ensure dimensions match
            if coef.shape != (self.output_dim, self.input_dim):
                logger.warning(f"Coefficient shape mismatch. Expected {(self.output_dim, self.input_dim)}, got {coef.shape}")
                coef = np.ones((self.output_dim, self.input_dim)) / self.input_dim
            
            if len(intercept) != self.output_dim:
                logger.warning(f"Intercept length mismatch. Expected {self.output_dim}, got {len(intercept)}")
                intercept = np.zeros(self.output_dim)
                
            self.coef_ = coef
            self.intercept_ = intercept
            
            # Update model attributes
            self.model.coef_ = coef
            self.model.intercept_ = intercept
            
            self.is_initialized = True
            
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters as dictionary."""
        return {
            "estimator": "meta_lr",
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_.tolist(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }


def create_model_from_config(config: Dict[str, Any], input_dim: int = 10, output_dim: int = 1, device: str = "cpu") -> nn.Module:
    """
    Create a model from configuration parameters.
    
    Args:
        config: Model configuration dictionary
        input_dim: Input dimension (MUST be respected)
        output_dim: Output dimension (MUST be respected)
        device: Device to use for the model
        
    Returns:
        Initialized model
    """
    model_type = config.get("estimator", "")
    
    # Extract dimensions from config if provided, otherwise use provided parameters
    config_input_dim = config.get("input_dim", input_dim)
    config_output_dim = config.get("output_dim", output_dim)
    
    # Use the input_dim and output_dim parameters as overrides
    actual_input_dim = input_dim
    actual_output_dim = output_dim
    
    # Special case for meta learner - it needs to match the number of base models
    if model_type == "meta_lr":
        # Meta learner should have input_dim equal to the number of base models
        # If config specifies a different input_dim for meta_lr, use it
        actual_input_dim = config.get("meta_input_dim", config.get("input_dim", 3))
    
    try:
        if model_type == "lr":
            model = LinearRegressionWrapper(actual_input_dim, actual_output_dim)
            # Set parameters only after initialization with correct dimensions
            if any(k in config for k in ["coef", "intercept"]):
                # Create a modified config with correct dimensions
                adjusted_config = config.copy()
                if "coef" in adjusted_config:
                    coef = np.array(adjusted_config["coef"])
                    # Ensure coefficients match the dimensions
                    if coef.shape != (actual_output_dim, actual_input_dim):
                        # If dimensions don't match, initialize with zeros
                        logger.warning(f"Coefficient shape mismatch for lr. Expected {(actual_output_dim, actual_input_dim)}, got {coef.shape}")
                        adjusted_config["coef"] = np.zeros((actual_output_dim, actual_input_dim)).tolist()
                if "intercept" in adjusted_config:
                    intercept = np.array(adjusted_config["intercept"])
                    if len(intercept) != actual_output_dim:
                        logger.warning(f"Intercept length mismatch for lr. Expected {actual_output_dim}, got {len(intercept)}")
                        adjusted_config["intercept"] = np.zeros(actual_output_dim).tolist()
                model.set_parameters(adjusted_config)
            
        elif model_type == "svc":
            model = SVCWrapper(actual_input_dim, actual_output_dim)
            if any(k in config for k in ["dual_coef", "support_vectors", "intercept"]):
                # Check and adjust dimensions
                adjusted_config = config.copy()
                if "support_vectors" in adjusted_config:
                    support_vectors = np.array(adjusted_config["support_vectors"])
                    if support_vectors.shape[1] != actual_input_dim:
                        logger.warning(f"Support vectors dimension mismatch. Expected {actual_input_dim}, got {support_vectors.shape[1]}")
                        # Initialize with minimal support vectors
                        adjusted_config["support_vectors"] = np.zeros((2, actual_input_dim)).tolist()
                        adjusted_config["dual_coef"] = np.array([[1.0, -1.0]]).tolist()
                model.set_parameters(adjusted_config)
            
        elif model_type == "rf":
            model = RandomForestWrapper(actual_input_dim, actual_output_dim)
            if "feature_importances" in config:
                adjusted_config = config.copy()
                feature_importances = np.array(adjusted_config["feature_importances"])
                if len(feature_importances) != actual_input_dim:
                    logger.warning(f"Feature importances dimension mismatch. Expected {actual_input_dim}, got {len(feature_importances)}")
                    # Initialize with equal importances
                    adjusted_config["feature_importances"] = (np.ones(actual_input_dim) / actual_input_dim).tolist()
                model.set_parameters(adjusted_config)
            
        elif model_type == "meta_lr":
            model = MetaLearnerWrapper(actual_input_dim, actual_output_dim)
            if "coef" in config:
                adjusted_config = config.copy()
                coef = np.array(adjusted_config["coef"])
                if coef.shape != (actual_output_dim, actual_input_dim):
                    logger.warning(f"Coefficient shape mismatch for meta_lr. Expected {(actual_output_dim, actual_input_dim)}, got {coef.shape}")
                    # Initialize with equal weights
                    adjusted_config["coef"] = (np.ones((actual_output_dim, actual_input_dim)) / actual_input_dim).tolist()
                if "intercept" in adjusted_config:
                    intercept = np.array(adjusted_config["intercept"])
                    if len(intercept) != actual_output_dim:
                        logger.warning(f"Intercept length mismatch for meta_lr. Expected {actual_output_dim}, got {len(intercept)}")
                        adjusted_config["intercept"] = np.zeros(actual_output_dim).tolist()
                model.set_parameters(adjusted_config)
            
        elif model_type == "linear":
            model = LinearModel(actual_input_dim, actual_output_dim)
            if "weights" in config:
                weights = config["weights"]
                if isinstance(weights, dict) and "weight" in weights:
                    weight_tensor = torch.tensor(weights["weight"], device=device)
                    if weight_tensor.shape != (actual_output_dim, actual_input_dim):
                        logger.warning(f"Weight shape mismatch for linear model. Expected {(actual_output_dim, actual_input_dim)}, got {weight_tensor.shape}")
                        # Initialize with random weights
                        weight_tensor = torch.randn(actual_output_dim, actual_input_dim, device=device)
                    state_dict = OrderedDict([
                        ('linear.weight', weight_tensor),
                        ('linear.bias', torch.tensor(weights.get("bias", torch.zeros(actual_output_dim)), device=device))
                    ])
                    model.load_state_dict(state_dict)
            
        elif model_type == "mlp":
            hidden_dims = config.get("hidden_dims", [64, 32])
            model = MLPModel(actual_input_dim, actual_output_dim, hidden_dims)
            if "weights" in config:
                # For MLP, we need to handle multiple layers
                # This is more complex and requires proper weight shape matching
                pass  # Implement if needed
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Created {model_type} model with input_dim={actual_input_dim}, output_dim={actual_output_dim}")
        return model.to(device)
        
    except Exception as e:
        logger.error(f"Error creating model from config: {e}")
        # Return a default model with correct dimensions as fallback
        if model_type in ["lr", "linear"]:
            return LinearRegressionWrapper(actual_input_dim, actual_output_dim).to(device)
        elif model_type == "meta_lr":
            return MetaLearnerWrapper(actual_input_dim, actual_output_dim).to(device)
        else:
            return LinearModel(actual_input_dim, actual_output_dim).to(device)

def create_model_ensemble_from_config(
    configs: List[Dict[str, Any]],
    input_dim: int = 10,
    output_dim: int = 1,
    device: str = "cpu"
) -> Tuple[List[nn.Module], List[str]]:
    """
    Create an ensemble of models from configurations.
    
    Args:
        configs: List of model configurations
        input_dim: Input dimension
        output_dim: Output dimension
        device: Device to use for models
        
    Returns:
        Tuple of (list of models, list of model names)
    """
    models = []
    model_names = []
    
    for i, config in enumerate(configs):
        try:
            # Make sure config is a dictionary
            if not isinstance(config, dict):
                logger.error(f"Config at index {i} is not a dictionary: {type(config)}")
                config = {} if config is None else {"estimator": "linear"}
            
            model_type = config.get("estimator", "linear")
            
            # Special handling for meta learner input dimension
            if model_type == "meta_lr":
                # Meta learner should have input_dim equal to the number of base models
                meta_input_dim = len([c for c in configs if isinstance(c, dict) and c.get("estimator") != "meta_lr"])
                config["meta_input_dim"] = meta_input_dim
            
            model = create_model_from_config(config, input_dim, output_dim, device)
            models.append(model)
            model_names.append(model_type)
            logger.info(f"Created model {i+1}/{len(configs)}: {model_type}")
        except Exception as e:
            logger.error(f"Error creating model from config: {e}")
            # Create a default model as fallback
            fallback_model = LinearModel(input_dim, output_dim).to(device)
            models.append(fallback_model)
            model_names.append("linear")
    
    return models, model_names

def create_model_ensemble(
    input_dim: int, 
    output_dim: int,
    ensemble_size: int = 5,
    device: str = "cpu"
) -> Tuple[List[nn.Module], List[str]]:
    """
    Create a diverse ensemble of models with different architectures.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        ensemble_size: Number of models to create
        device: Device to use ("cpu" or "cuda")
        
    Returns:
        Tuple of (list of models, list of model names)
    """
    models = []
    model_names = []
    
    # Define model configurations
    model_configs = [
        # Simple linear model
        {
            "class": LinearModel,
            "name": "Linear",
            "params": {"input_dim": input_dim, "output_dim": output_dim}
        },
        # Wide MLP
        {
            "class": MLPModel,
            "name": "WideMLP",
            "params": {"input_dim": input_dim, "output_dim": output_dim, "hidden_dims": [128, 64]}
        },
        # Deep MLP
        {
            "class": MLPModel,
            "name": "DeepMLP",
            "params": {"input_dim": input_dim, "output_dim": output_dim, "hidden_dims": [32, 32, 32, 32]}
        },
        # Linear Regression
        {
            "class": LinearRegressionWrapper,
            "name": "LinearRegression",
            "params": {"input_dim": input_dim, "output_dim": output_dim}
        },
        # SVC
        {
            "class": SVCWrapper,
            "name": "SVC",
            "params": {"input_dim": input_dim, "output_dim": output_dim}
        },
        # Random Forest
        {
            "class": RandomForestWrapper,
            "name": "RandomForest",
            "params": {"input_dim": input_dim, "output_dim": output_dim}
        },
        # Meta Learner
        {
            "class": MetaLearnerWrapper,
            "name": "MetaLearner",
            "params": {"input_dim": input_dim, "output_dim": output_dim}
        }
    ]
    
    # Use all defined models if ensemble_size is large enough
    if ensemble_size <= len(model_configs):
        model_configs = model_configs[:ensemble_size]
    else:
        # For large ensembles, add variations of existing models
        original_count = len(model_configs)
        for i in range(ensemble_size - original_count):
            # Pick a random model config to create a variation
            base_config = model_configs[i % original_count].copy()
            
            # Create a variation
            if base_config["class"] == MLPModel:
                # Vary hidden layer dimensions
                hidden_dims = base_config["params"]["hidden_dims"].copy()
                factor = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2
                hidden_dims = [int(dim * factor) for dim in hidden_dims]
                
                base_config["params"]["hidden_dims"] = hidden_dims
                base_config["name"] = f"{base_config['name']}_var_{i+1}"
            
            model_configs.append(base_config)
    
    # Create the models
    for config in model_configs:
        model = config["class"](**config["params"])
        model.to(device)
        models.append(model)
        model_names.append(config["name"])
    
    return models, model_names

def get_ensemble_state_dict(
    ensemble_model: nn.Module
) -> Dict[str, Dict[str, Any]]:
    """
    Get the state dict for an ensemble model including weights and architecture.
    
    Args:
        ensemble_model: Ensemble model to get state dict for
        
    Returns:
        Dictionary with ensemble state
    """
    if not hasattr(ensemble_model, 'models') or not hasattr(ensemble_model, 'weights'):
        raise ValueError("Expected an EnsembleModel with models and weights attributes")
    
    # Get weights
    weights = ensemble_model.weights.cpu().numpy().tolist()
    
    # Get model state dicts
    model_state_dicts = []
    for model in ensemble_model.models:
        if hasattr(model, 'get_parameters'):
            # For scikit-learn wrapped models
            state_dict = model.get_parameters()
            model_state_dicts.append(state_dict)
        else:
            # For PyTorch models
            state_dict = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
            state_dict["model_type"] = "pytorch"  # Add model type for later identification
            model_state_dicts.append(state_dict)
    
    # Get model names if available
    model_names = getattr(ensemble_model, 'model_names', [f"model_{i}" for i in range(len(ensemble_model.models))])
    
    # Create ensemble state dict
    ensemble_state = {
        "weights": weights,
        "model_names": model_names,
        "model_state_dicts": model_state_dicts
    }
    
    return ensemble_state


def load_ensemble_from_state_dict(
    ensemble_state: Dict[str, Any],
    model_classes: List[nn.Module] = None,
    model_params: List[Dict[str, Any]] = None,
    device: str = "cpu"
) -> nn.Module:
    """
    Load an ensemble model from a state dict.
    
    Args:
        ensemble_state: Ensemble state dict
        model_classes: List of model classes to instantiate
        model_params: Parameters for model initialization
        device: Device to load the models on
        
    Returns:
        Loaded ensemble model
    """
    from ga_stacking import EnsembleModel
    
    # Initialize lists for models and names
    models = []
    model_names = ensemble_state["model_names"]
    
    # If model_classes not provided, infer from state dicts
    if model_classes is None:
        model_classes = []
        for state_dict in ensemble_state["model_state_dicts"]:
            if "estimator" in state_dict:
                model_type = state_dict["estimator"]
                if model_type == "lr":
                    model_classes.append(LinearRegressionWrapper)
                elif model_type == "svc":
                    model_classes.append(SVCWrapper)
                elif model_type == "rf":
                    model_classes.append(RandomForestWrapper)
                elif model_type == "meta_lr":
                    model_classes.append(MetaLearnerWrapper)
                else:
                    model_classes.append(LinearModel)  # Default fallback
            elif "model_type" in state_dict and state_dict["model_type"] == "pytorch":
                model_classes.append(LinearModel)  # For PyTorch models
            else:
                model_classes.append(LinearModel)  # Default fallback
    
    # If model_params not provided, use empty dicts
    if model_params is None:
        model_params = []
        for state_dict in ensemble_state["model_state_dicts"]:
            input_dim = state_dict.get("input_dim", 10)
            output_dim = state_dict.get("output_dim", 1)
            model_params.append({"input_dim": input_dim, "output_dim": output_dim})
    
    # Load individual models
    for i, (model_class, params, state_dict) in enumerate(
        zip(model_classes, model_params, ensemble_state["model_state_dicts"])
    ):
        try:
            # Initialize model
            model = model_class(**params)
            
            # Check if it's a scikit-learn wrapper
            if hasattr(model, 'set_parameters'):
                # For scikit-learn wrapped models
                model.set_parameters(state_dict)
            else:
                # For PyTorch models
                # Filter out non-tensor keys
                tensor_state_dict = {}
                for key, value in state_dict.items():
                    if key not in ["model_type", "estimator", "input_dim", "output_dim"]:
                        tensor_state_dict[key] = torch.tensor(value, device=device)
                
                # Load weights
                model.load_state_dict(tensor_state_dict)
            
            model.to(device)
            models.append(model)
            
        except Exception as e:
            logger.error(f"Error loading model {i}: {e}")
            # Create a dummy model as fallback
            fallback_params = {"input_dim": params.get("input_dim", 10), "output_dim": params.get("output_dim", 1)}
            
            if isinstance(model_class, LinearRegressionWrapper) or state_dict.get("estimator") == "lr":
                model = LinearRegressionWrapper(**fallback_params)
            elif isinstance(model_class, SVCWrapper) or state_dict.get("estimator") == "svc":
                model = SVCWrapper(**fallback_params)
            elif isinstance(model_class, RandomForestWrapper) or state_dict.get("estimator") == "rf":
                model = RandomForestWrapper(**fallback_params)
            elif isinstance(model_class, MetaLearnerWrapper) or state_dict.get("estimator") == "meta_lr":
                model = MetaLearnerWrapper(**fallback_params)
            else:
                model = LinearModel(**fallback_params)
            
            model.to(device)
            models.append(model)
    
    # Create ensemble model
    ensemble = EnsembleModel(
        models=models,
        weights=ensemble_state["weights"],
        model_names=model_names,
        device=device
    )
    
    return ensemble