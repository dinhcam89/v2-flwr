"""
GA-Stacking implementation for federated learning.
Provides genetic algorithm-based ensemble stacking optimization.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Callable, Any, Optional, Union
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("GA-Stacking")

class GAStacking:
    """Genetic Algorithm based Stacking Ensemble optimization."""
    
    def __init__(
        self,
        base_models: List[nn.Module],
        model_names: List[str],
        population_size: int = 30,
        generations: int = 20,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.7,
        elite_size: int = 2,
        device: str = "cpu"
    ):
        """
        Initialize the GA-Stacking optimizer.
        
        Args:
            base_models: List of PyTorch models for base learners
            model_names: Names of the base models (for logging/tracking)
            population_size: Size of the GA population
            generations: Number of GA generations to run
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of top individuals to keep without modification
            device: Device to use for computation ("cpu" or "cuda")
        """
        self.base_models = base_models
        self.model_names = model_names
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.device = torch.device(device)
        
        self.num_models = len(base_models)
        self.best_weights = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        # Ensure all models are on the correct device
        for model in self.base_models:
            model.to(self.device)
            model.eval()  # Set to evaluation mode
    
    def initialize_population(self) -> List[np.ndarray]:
        """
        Initialize the GA population with random weights.
        
        Returns:
            List of weight arrays for the population
        """
        population = []
        for _ in range(self.population_size):
            # Generate random weights and normalize
            weights = np.random.random(self.num_models)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
            population.append(weights)
        
        return population
    
    def fitness_function(
        self, weights: np.ndarray, val_data: torch.utils.data.DataLoader
    ) -> float:
        """
        Calculate fitness (validation accuracy) for a set of weights.
        
        Args:
            weights: Array of weights for the ensemble
            val_data: Validation data loader
            
        Returns:
            Fitness score (accuracy)
        """
        correct = 0
        total = 0
        total_loss = 0.0
        
        # Use eval context
        with torch.no_grad():
            for data, targets in val_data:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # First, collect predictions from each model (excluding meta_lr)
                model_outputs = []
                base_models = []
                meta_model = None
                
                for i, model in enumerate(self.base_models):
                    if self.model_names[i] == "meta_lr" or "MetaLearner" in self.model_names[i]:
                        meta_model = model
                        continue
                        
                    base_models.append(model)
                    try:
                        outputs = model(data)
                        model_outputs.append(outputs)
                    except Exception as e:
                        # Log error and continue with a dummy output
                        print(f"Error in model forward pass for {self.model_names[i]}: {e}")
                        dummy_output = torch.zeros_like(targets)
                        model_outputs.append(dummy_output)
                
                # Calculate weighted sum of base model outputs (excluding meta learner)
                num_base_models = len(model_outputs)
                if num_base_models == 0:
                    # No base models to use, return negative infinity
                    return float('-inf')
                    
                # Normalize weights for base models (excluding meta learner weight)
                base_weights = np.array([w for i, w in enumerate(weights) 
                                    if self.model_names[i] != "meta_lr" and "MetaLearner" not in self.model_names[i]])
                if len(base_weights) > 0:
                    base_weights = base_weights / (np.sum(base_weights) + 1e-10)  # Avoid division by zero
                
                # Calculate weighted prediction from base models
                weighted_sum = torch.zeros_like(targets)
                for i, output in enumerate(model_outputs):
                    if i < len(base_weights):
                        weighted_sum += output * base_weights[i]
                
                # For regression: calculate MSE
                if targets.dim() == 1 or targets.size(1) == 1:  # Regression case
                    mse = torch.mean((weighted_sum - targets) ** 2).item()
                    total_loss += mse * targets.size(0)
                    # Consider prediction correct if within a threshold
                    threshold = 0.5
                    correct += torch.sum((torch.abs(weighted_sum - targets) < threshold)).item()
                # For classification: calculate accuracy
                else:
                    _, predicted = torch.max(weighted_sum, 1)
                    _, target_classes = torch.max(targets, 1)
                    correct += (predicted == target_classes).sum().item()
                    
                total += targets.size(0)
            
            avg_loss = total_loss / total if total > 0 else float('inf')
            accuracy = correct / total if total > 0 else 0
            
            # For regression, we want to minimize loss, so we return negative loss
            return 1 / avg_loss
    
    def select_parents(
        self, population: List[np.ndarray], fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """
        Select parents using tournament selection.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for the population
            
        Returns:
            Selected parents
        """
        parents = []
        for _ in range(self.population_size):
            # Tournament selection with size 3
            tournament_indices = random.sample(range(self.population_size), 3)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent weights
            parent2: Second parent weights
            
        Returns:
            Two offspring weight arrays
        """
        if random.random() < self.crossover_rate:
            # Single point crossover
            point = random.randint(1, self.num_models - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            
            # Normalize weights to sum to 1
            child1 = child1 / np.sum(child1)
            child2 = child2 / np.sum(child2)
            
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply mutation to an individual.
        
        Args:
            individual: Weight array to mutate
            
        Returns:
            Mutated weight array
        """
        for i in range(self.num_models):
            if random.random() < self.mutation_rate:
                # Add Gaussian noise
                individual[i] += np.random.normal(0, 0.1)
                
                # Ensure no negative weights
                individual[i] = max(0, individual[i])
        
        # Re-normalize weights to sum to 1
        individual = individual / np.sum(individual)
        
        return individual
    
    def evolve_population(
        self, population: List[np.ndarray], fitness_scores: List[float]
    ) -> List[np.ndarray]:
        """
        Evolve the population to the next generation.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for the population
            
        Returns:
            New population
        """
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        
        # Keep elite individuals
        new_population = sorted_population[:self.elite_size]
        
        # Selection
        parents = self.select_parents(population, fitness_scores)
        
        # Crossover and mutation
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population
    
    def optimize(
        self, 
        train_data: torch.utils.data.DataLoader, 
        val_data: torch.utils.data.DataLoader
    ) -> np.ndarray:
        """
        Run the GA optimization process.
        
        Args:
            train_data: Training data loader
            val_data: Validation data loader
            
        Returns:
            Best weights found for the ensemble
        """
        # Initialize population
        population = self.initialize_population()
        
        # Run GA for specified number of generations
        for generation in range(self.generations):
            # Calculate fitness for each individual
            fitness_scores = []
            for individual in population:
                fitness = self.fitness_function(individual, val_data)
                fitness_scores.append(fitness)
            
            # Track best individual
            max_fitness_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[max_fitness_idx]
            current_best_weights = population[max_fitness_idx]
            
            # Update global best if needed
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_weights = current_best_weights.copy()
            
            # Add to history
            self.fitness_history.append({
                "generation": generation,
                "best_fitness": current_best_fitness,
                "avg_fitness": np.mean(fitness_scores),
                "best_weights": current_best_weights.tolist()
            })
            
            # Log progress
            logger.info(f"Generation {generation+1}/{self.generations}, "
                       f"Best Fitness: {current_best_fitness:.4f}, "
                       f"Avg Fitness: {np.mean(fitness_scores):.4f}")
            
            # Evolve population
            if generation < self.generations - 1:
                population = self.evolve_population(population, fitness_scores)
        
        # Log final weights
        weight_str = ", ".join([f"{self.model_names[i]}: {self.best_weights[i]:.4f}" 
                               for i in range(self.num_models)])
        logger.info(f"Optimization complete. Best weights: {weight_str}")
        
        return self.best_weights
    
    def get_ensemble_model(self) -> 'EnsembleModel':
        """
        Create an ensemble model with the optimized weights.
        
        Returns:
            Weighted ensemble model
        """
        if self.best_weights is None:
            raise ValueError("Must run optimize() before getting the ensemble model")
        
        return EnsembleModel(
            models=self.base_models,
            weights=self.best_weights,
            model_names=self.model_names,
            device=self.device
        )


class EnsembleModel(nn.Module):
    """Weighted ensemble model that combines multiple base models."""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: np.ndarray,
        model_names: List[str],
        device: str = "cpu"
    ):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of base models
            weights: Array of weights for the models
            model_names: Names of the base models
            device: Device to use for computation
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(weights, device=device)
        self.model_names = model_names
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted output from the ensemble
        """
        # Separate base models from meta learner
        base_models = []
        base_model_indices = []
        meta_model = None
        meta_model_index = None
        
        for i, model in enumerate(self.models):
            model_name = self.model_names[i] if i < len(self.model_names) else ""
            if model_name == "meta_lr" or "meta" in model_name.lower():
                meta_model = model
                meta_model_index = i
            else:
                base_models.append(model)
                base_model_indices.append(i)
        
        # If no meta model, just use weighted average of base models
        if meta_model is None:
            outputs = []
            for model in self.models:
                try:
                    output = model(x)
                    outputs.append(output)
                except Exception as e:
                    print(f"Error in model forward pass: {str(e)}")
                    dummy_output = torch.zeros((x.size(0), 1), device=self.device)
                    outputs.append(dummy_output)
            
            # Weight the outputs
            weighted_sum = torch.zeros_like(outputs[0]) if outputs else torch.zeros((x.size(0), 1), device=self.device)
            for i, output in enumerate(outputs):
                if i < len(self.weights):
                    weighted_sum += output * self.weights[i]
            
            return weighted_sum
        
        # If we have a meta model, run base models first
        base_outputs = []
        for model in base_models:
            try:
                output = model(x)
                base_outputs.append(output)
            except Exception as e:
                print(f"Error in base model forward pass: {str(e)}")
                dummy_output = torch.zeros((x.size(0), 1), device=self.device)
                base_outputs.append(dummy_output)
        
        # If we have base outputs, try to run meta model
        if base_outputs:
            try:
                # Stack outputs if there are multiple
                if len(base_outputs) > 1:
                    stacked_base_outputs = torch.cat(base_outputs, dim=1)
                else:
                    stacked_base_outputs = base_outputs[0]
                
                # Run meta model
                meta_output = meta_model(stacked_base_outputs)
                
                # Weight meta output with other base outputs
                meta_weight = self.weights[meta_model_index]
                base_weight_sum = sum(self.weights[i] for i in base_model_indices)
                
                if base_weight_sum > 0:
                    # Weighted combination of meta and base outputs
                    base_combined = torch.zeros_like(base_outputs[0])
                    for i, idx in enumerate(base_model_indices):
                        base_combined += base_outputs[i] * (self.weights[idx] / base_weight_sum)
                    
                    result = meta_output * meta_weight + base_combined * (1 - meta_weight)
                    return result
                else:
                    # Just use meta output
                    return meta_output
                
            except Exception as e:
                print(f"Error in meta model forward pass: {str(e)}")
                # Fall back to base models
        
        # If we get here, either meta model failed or we have no base outputs
        # Fall back to weighted base outputs
        weighted_sum = torch.zeros_like(base_outputs[0]) if base_outputs else torch.zeros((x.size(0), 1), device=self.device)
        total_weight = 0.0
        
        for i, idx in enumerate(base_model_indices):
            if i < len(base_outputs):
                weight = self.weights[idx]
                weighted_sum += base_outputs[i] * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_sum = weighted_sum / total_weight
        
        return weighted_sum
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get the model weights as a dictionary.
        
        Returns:
            Dictionary mapping model names to weights
        """
        return {name: float(weight) for name, weight in zip(self.model_names, self.weights)}