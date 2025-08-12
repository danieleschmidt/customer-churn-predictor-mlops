"""
Neural Architecture Search (NAS) for Automated Model Discovery.

This module implements advanced neural architecture search algorithms to automatically
discover optimal neural network architectures for customer churn prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import json
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import random
import hashlib
import copy
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import itertools

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .config import load_config

logger = get_logger(__name__)


class ActivationFunction(Enum):
    """Neural network activation functions."""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    GELU = "gelu"


class LayerType(Enum):
    """Neural network layer types."""
    DENSE = "dense"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    RESIDUAL = "residual"
    ATTENTION = "attention"


class OptimizerType(Enum):
    """Optimizer types for neural networks."""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"


@dataclass
class NeuralLayer:
    """Represents a neural network layer configuration."""
    layer_type: LayerType
    units: Optional[int] = None
    activation: Optional[ActivationFunction] = None
    dropout_rate: Optional[float] = None
    use_bias: bool = True
    layer_id: str = field(default_factory=lambda: hashlib.md5(f"{random.random()}".encode()).hexdigest()[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert layer to dictionary representation."""
        return {
            'layer_type': self.layer_type.value if self.layer_type else None,
            'units': self.units,
            'activation': self.activation.value if self.activation else None,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'layer_id': self.layer_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralLayer':
        """Create layer from dictionary representation."""
        return cls(
            layer_type=LayerType(data['layer_type']) if data['layer_type'] else None,
            units=data.get('units'),
            activation=ActivationFunction(data['activation']) if data.get('activation') else None,
            dropout_rate=data.get('dropout_rate'),
            use_bias=data.get('use_bias', True),
            layer_id=data.get('layer_id', '')
        )


@dataclass
class NeuralArchitecture:
    """Represents a complete neural network architecture."""
    architecture_id: str
    layers: List[NeuralLayer]
    optimizer: OptimizerType
    learning_rate: float
    batch_size: int
    l2_regularization: float
    early_stopping_patience: int
    max_epochs: int
    fitness_score: float = 0.0
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    complexity_score: float = 0.0
    
    def __post_init__(self):
        """Calculate architecture complexity."""
        self.complexity_score = self._calculate_complexity()
    
    def _calculate_complexity(self) -> float:
        """Calculate architecture complexity score."""
        complexity = 0.0
        
        # Layer count penalty
        complexity += len(self.layers) * 0.1
        
        # Parameter count estimation
        total_params = 0
        prev_units = None
        
        for layer in self.layers:
            if layer.layer_type == LayerType.DENSE and layer.units:
                if prev_units is not None:
                    total_params += prev_units * layer.units
                prev_units = layer.units
            elif layer.layer_type == LayerType.DROPOUT:
                # Dropout doesn't add parameters but adds complexity
                complexity += 0.05
        
        # Normalize parameter count
        complexity += np.log10(max(total_params, 1)) * 0.1
        
        return complexity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary."""
        return {
            'architecture_id': self.architecture_id,
            'layers': [layer.to_dict() for layer in self.layers],
            'optimizer': self.optimizer.value,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'l2_regularization': self.l2_regularization,
            'early_stopping_patience': self.early_stopping_patience,
            'max_epochs': self.max_epochs,
            'fitness_score': self.fitness_score,
            'complexity_score': self.complexity_score,
            'training_history': self.training_history
        }
    
    def mutate(self, mutation_rate: float = 0.3, mutation_strength: float = 1.0) -> 'NeuralArchitecture':
        """Apply mutations to architecture."""
        mutated = copy.deepcopy(self)
        mutated.architecture_id = self._generate_id()
        
        # Layer mutations
        if random.random() < mutation_rate:
            mutation_type = random.choice(['add_layer', 'remove_layer', 'modify_layer'])
            
            if mutation_type == 'add_layer' and len(mutated.layers) < 10:
                new_layer = self._generate_random_layer()
                insert_pos = random.randint(0, len(mutated.layers))
                mutated.layers.insert(insert_pos, new_layer)
            
            elif mutation_type == 'remove_layer' and len(mutated.layers) > 2:
                remove_idx = random.randint(0, len(mutated.layers) - 1)
                mutated.layers.pop(remove_idx)
            
            elif mutation_type == 'modify_layer' and mutated.layers:
                layer_idx = random.randint(0, len(mutated.layers) - 1)
                mutated.layers[layer_idx] = self._mutate_layer(mutated.layers[layer_idx])
        
        # Hyperparameter mutations
        if random.random() < mutation_rate:
            mutated.learning_rate *= np.random.lognormal(0, 0.3)
            mutated.learning_rate = np.clip(mutated.learning_rate, 1e-5, 1.0)
        
        if random.random() < mutation_rate:
            mutated.batch_size = random.choice([16, 32, 64, 128, 256])
        
        if random.random() < mutation_rate:
            mutated.l2_regularization *= np.random.lognormal(0, 0.5)
            mutated.l2_regularization = np.clip(mutated.l2_regularization, 1e-6, 1e-1)
        
        if random.random() < mutation_rate:
            mutated.optimizer = random.choice(list(OptimizerType))
        
        mutated._post_init()
        return mutated
    
    def crossover(self, other: 'NeuralArchitecture') -> Tuple['NeuralArchitecture', 'NeuralArchitecture']:
        """Perform crossover with another architecture."""
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        child1.architecture_id = self._generate_id()
        child2.architecture_id = self._generate_id()
        
        # Layer crossover - exchange segments
        if len(self.layers) > 1 and len(other.layers) > 1:
            crossover_point1 = random.randint(1, len(self.layers) - 1)
            crossover_point2 = random.randint(1, len(other.layers) - 1)
            
            # Exchange layer segments
            child1.layers = self.layers[:crossover_point1] + other.layers[crossover_point2:]
            child2.layers = other.layers[:crossover_point2] + self.layers[crossover_point1:]
        
        # Hyperparameter crossover
        if random.random() < 0.5:
            child1.learning_rate, child2.learning_rate = child2.learning_rate, child1.learning_rate
        
        if random.random() < 0.5:
            child1.batch_size, child2.batch_size = child2.batch_size, child1.batch_size
        
        if random.random() < 0.5:
            child1.optimizer, child2.optimizer = child2.optimizer, child1.optimizer
        
        child1.__post_init__()
        child2.__post_init__()
        
        return child1, child2
    
    def _generate_random_layer(self) -> NeuralLayer:
        """Generate a random neural layer."""
        layer_type = random.choice([LayerType.DENSE, LayerType.DROPOUT])
        
        if layer_type == LayerType.DENSE:
            return NeuralLayer(
                layer_type=LayerType.DENSE,
                units=random.choice([32, 64, 128, 256, 512]),
                activation=random.choice(list(ActivationFunction)),
                use_bias=random.choice([True, False])
            )
        elif layer_type == LayerType.DROPOUT:
            return NeuralLayer(
                layer_type=LayerType.DROPOUT,
                dropout_rate=random.uniform(0.1, 0.5)
            )
    
    def _mutate_layer(self, layer: NeuralLayer) -> NeuralLayer:
        """Mutate a single layer."""
        mutated_layer = copy.deepcopy(layer)
        
        if layer.layer_type == LayerType.DENSE:
            if random.random() < 0.5 and layer.units:
                # Mutate units
                factor = np.random.lognormal(0, 0.3)
                new_units = int(layer.units * factor)
                mutated_layer.units = max(16, min(512, new_units))
            
            if random.random() < 0.3:
                mutated_layer.activation = random.choice(list(ActivationFunction))
        
        elif layer.layer_type == LayerType.DROPOUT:
            if layer.dropout_rate:
                noise = np.random.normal(0, 0.1)
                mutated_layer.dropout_rate = np.clip(layer.dropout_rate + noise, 0.05, 0.8)
        
        return mutated_layer
    
    def _generate_id(self) -> str:
        """Generate unique architecture ID."""
        return hashlib.md5(f"{datetime.utcnow().isoformat()}_{random.random()}".encode()).hexdigest()[:12]


class NeuralArchitectureSearcher:
    """Main Neural Architecture Search system."""
    
    def __init__(self, 
                 population_size: int = 25,
                 elite_size: int = 5,
                 max_generations: int = 20,
                 mutation_rate: float = 0.4,
                 crossover_rate: float = 0.6):
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population: List[NeuralArchitecture] = []
        self.search_history = []
        self.best_architecture = None
        self.generation_counter = 0
        
        # Architecture constraints
        self.min_layers = 2
        self.max_layers = 8
        self.min_units = 16
        self.max_units = 512
        
    async def search_architectures(self, 
                                 X_train: pd.DataFrame, 
                                 y_train: pd.Series,
                                 X_val: Optional[pd.DataFrame] = None,
                                 y_val: Optional[pd.Series] = None,
                                 time_budget_minutes: int = 60) -> Dict[str, Any]:
        """
        Search for optimal neural architectures.
        
        Args:
            X_train: Training features
            y_train: Training labels  
            X_val: Validation features
            y_val: Validation labels
            time_budget_minutes: Time budget for search
            
        Returns:
            Search results with best architecture
        """
        logger.info(f"Starting Neural Architecture Search with {self.population_size} architectures")
        start_time = time.time()
        time_budget_seconds = time_budget_minutes * 60
        
        # Split validation data if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Data preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Initialize population
        await self._initialize_architecture_population(X_train.shape[1])
        
        # Evolution loop with time budget
        generation = 0
        while generation < self.max_generations and (time.time() - start_time) < time_budget_seconds:
            self.generation_counter = generation
            logger.info(f"NAS Generation {generation + 1}/{self.max_generations}")
            
            # Evaluate architectures
            await self._evaluate_architecture_population(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # Record generation statistics
            self._record_search_generation_stats(generation)
            
            # Early stopping if time budget exceeded
            if (time.time() - start_time) > time_budget_seconds * 0.9:
                logger.info(f"Approaching time budget, stopping search")
                break
            
            # Create next generation
            if generation < self.max_generations - 1:
                await self._create_next_architecture_generation()
            
            generation += 1
        
        # Compile final results
        search_duration = time.time() - start_time
        results = self._compile_search_results(search_duration)
        
        logger.info(f"NAS complete. Best architecture fitness: {self.best_architecture.fitness_score:.4f}")
        
        return results
    
    async def _initialize_architecture_population(self, input_features: int):
        """Initialize population of random architectures."""
        self.population = []
        
        for i in range(self.population_size):
            architecture = self._generate_random_architecture(input_features)
            architecture.architecture_id = f"arch_{i:03d}_gen_0"
            self.population.append(architecture)
        
        logger.info(f"Initialized {len(self.population)} random architectures")
    
    def _generate_random_architecture(self, input_features: int) -> NeuralArchitecture:
        """Generate a random neural architecture."""
        num_layers = random.randint(self.min_layers, self.max_layers)
        layers = []
        
        # First hidden layer
        layers.append(NeuralLayer(
            layer_type=LayerType.DENSE,
            units=random.randint(self.min_units, self.max_units),
            activation=random.choice(list(ActivationFunction)),
            use_bias=True
        ))
        
        # Hidden layers with optional dropout
        for i in range(1, num_layers - 1):
            # Dense layer
            layers.append(NeuralLayer(
                layer_type=LayerType.DENSE,
                units=random.randint(self.min_units, self.max_units),
                activation=random.choice(list(ActivationFunction)),
                use_bias=random.choice([True, False])
            ))
            
            # Optional dropout
            if random.random() < 0.4:
                layers.append(NeuralLayer(
                    layer_type=LayerType.DROPOUT,
                    dropout_rate=random.uniform(0.1, 0.5)
                ))
        
        # Output layer
        layers.append(NeuralLayer(
            layer_type=LayerType.DENSE,
            units=1,  # Binary classification
            activation=ActivationFunction.SIGMOID,
            use_bias=True
        ))
        
        architecture = NeuralArchitecture(
            architecture_id=self._generate_id(),
            layers=layers,
            optimizer=random.choice(list(OptimizerType)),
            learning_rate=np.random.lognormal(-3, 1),  # log-normal around 0.05
            batch_size=random.choice([32, 64, 128]),
            l2_regularization=np.random.lognormal(-5, 2),  # log-normal around 0.007
            early_stopping_patience=random.randint(5, 15),
            max_epochs=random.randint(50, 200)
        )
        
        return architecture
    
    async def _evaluate_architecture_population(self, X_train: np.ndarray, y_train: pd.Series, 
                                              X_val: np.ndarray, y_val: pd.Series):
        """Evaluate all architectures in population."""
        
        async def evaluate_single_architecture(architecture: NeuralArchitecture) -> NeuralArchitecture:
            """Evaluate a single architecture."""
            try:
                # Convert architecture to MLPClassifier
                model = self._architecture_to_mlp(architecture, X_train.shape[1])
                
                # Train with time limit
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
                
                # Multi-objective fitness: performance vs complexity
                performance_score = 0.5 * accuracy + 0.3 * f1 + 0.2 * precision
                complexity_penalty = architecture.complexity_score * 0.1
                training_time_penalty = min(0.1, training_time / 300)  # Penalty for >5min training
                
                fitness = performance_score - complexity_penalty - training_time_penalty
                architecture.fitness_score = max(0.0, fitness)
                
                # Record training history
                training_record = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'generation': self.generation_counter,
                    'fitness': architecture.fitness_score,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'training_time': training_time,
                    'complexity_score': architecture.complexity_score,
                    'performance_score': performance_score
                }
                architecture.training_history.append(training_record)
                
                return architecture
                
            except Exception as e:
                logger.warning(f"Architecture evaluation failed: {e}")
                architecture.fitness_score = 0.0
                return architecture
        
        # Evaluate all architectures
        tasks = [evaluate_single_architecture(arch) for arch in self.population]
        evaluated_architectures = await asyncio.gather(*tasks)
        
        self.population = evaluated_architectures
        
        # Update best architecture
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        if self.best_architecture is None or self.population[0].fitness_score > self.best_architecture.fitness_score:
            self.best_architecture = copy.deepcopy(self.population[0])
    
    def _architecture_to_mlp(self, architecture: NeuralArchitecture, input_features: int) -> MLPClassifier:
        """Convert architecture to scikit-learn MLPClassifier."""
        
        # Extract hidden layer sizes
        hidden_layers = []
        for layer in architecture.layers:
            if layer.layer_type == LayerType.DENSE and layer.units and layer.units > 1:
                hidden_layers.append(layer.units)
        
        # Remove output layer (handled by MLPClassifier)
        if hidden_layers and hidden_layers[-1] == 1:
            hidden_layers = hidden_layers[:-1]
        
        if not hidden_layers:
            hidden_layers = [64]  # Default fallback
        
        # Map activation function
        activation_map = {
            ActivationFunction.RELU: 'relu',
            ActivationFunction.TANH: 'tanh',
            ActivationFunction.SIGMOID: 'logistic'
        }
        
        # Get primary activation (most common in architecture)
        activations = [layer.activation for layer in architecture.layers 
                      if layer.activation and layer.activation in activation_map]
        primary_activation = max(set(activations), key=activations.count) if activations else ActivationFunction.RELU
        
        # Map optimizer
        solver_map = {
            OptimizerType.ADAM: 'adam',
            OptimizerType.SGD: 'sgd',
            OptimizerType.ADAGRAD: 'adam'  # Fallback to adam
        }
        
        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation=activation_map.get(primary_activation, 'relu'),
            solver=solver_map.get(architecture.optimizer, 'adam'),
            alpha=architecture.l2_regularization,
            learning_rate_init=architecture.learning_rate,
            max_iter=min(architecture.max_epochs, 300),  # Cap for performance
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=architecture.early_stopping_patience,
            random_state=42,
            batch_size=min(architecture.batch_size, 200)
        )
        
        return mlp
    
    async def _create_next_architecture_generation(self):
        """Create next generation of architectures."""
        # Elite selection
        elite = self.population[:self.elite_size]
        next_generation = [copy.deepcopy(arch) for arch in elite]
        
        # Tournament selection for breeding
        breeding_pool = self._tournament_selection_nas(self.population_size - self.elite_size)
        
        # Generate offspring
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.sample(breeding_pool, 2)
            
            if random.random() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
                children = [child1, child2]
            else:
                children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
            
            # Apply mutations
            for child in children:
                if len(next_generation) < self.population_size:
                    if random.random() < self.mutation_rate:
                        child = child.mutate(self.mutation_rate)
                    
                    child.architecture_id = f"arch_{len(next_generation):03d}_gen_{self.generation_counter + 1}"
                    next_generation.append(child)
        
        self.population = next_generation[:self.population_size]
    
    def _tournament_selection_nas(self, num_individuals: int, tournament_size: int = 3) -> List[NeuralArchitecture]:
        """Tournament selection for architecture breeding."""
        selected = []
        
        for _ in range(num_individuals):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness_score)
            selected.append(winner)
        
        return selected
    
    def _record_search_generation_stats(self, generation: int):
        """Record NAS generation statistics."""
        fitness_scores = [arch.fitness_score for arch in self.population]
        complexity_scores = [arch.complexity_score for arch in self.population]
        
        stats = {
            'generation': generation,
            'timestamp': datetime.utcnow().isoformat(),
            'best_fitness': max(fitness_scores),
            'average_fitness': np.mean(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'best_complexity': min(complexity_scores),
            'average_complexity': np.mean(complexity_scores),
            'architecture_diversity': self._calculate_architecture_diversity(),
            'layer_count_distribution': self._get_layer_count_distribution(),
            'optimizer_distribution': self._get_optimizer_distribution()
        }
        
        self.search_history.append(stats)
        
        if generation % 5 == 0:
            logger.info(f"NAS Gen {generation}: Best={stats['best_fitness']:.4f}, "
                       f"Avg={stats['average_fitness']:.4f}, "
                       f"Diversity={stats['architecture_diversity']:.4f}")
    
    def _calculate_architecture_diversity(self) -> float:
        """Calculate population diversity for architectures."""
        if len(self.population) < 2:
            return 0.0
        
        diversity_sum = 0.0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                arch1 = self.population[i]
                arch2 = self.population[j]
                
                # Architecture distance based on layer structure
                distance = self._calculate_architecture_distance(arch1, arch2)
                diversity_sum += distance
                comparisons += 1
        
        return diversity_sum / comparisons if comparisons > 0 else 0.0
    
    def _calculate_architecture_distance(self, arch1: NeuralArchitecture, arch2: NeuralArchitecture) -> float:
        """Calculate distance between two architectures."""
        distance = 0.0
        
        # Layer count difference
        distance += abs(len(arch1.layers) - len(arch2.layers)) * 0.2
        
        # Layer structure comparison
        max_layers = max(len(arch1.layers), len(arch2.layers))
        for i in range(max_layers):
            layer1 = arch1.layers[i] if i < len(arch1.layers) else None
            layer2 = arch2.layers[i] if i < len(arch2.layers) else None
            
            if layer1 is None or layer2 is None:
                distance += 0.5  # Missing layer penalty
            else:
                # Layer type difference
                if layer1.layer_type != layer2.layer_type:
                    distance += 0.3
                
                # Units difference (for dense layers)
                if layer1.layer_type == LayerType.DENSE and layer2.layer_type == LayerType.DENSE:
                    if layer1.units and layer2.units:
                        max_units = max(layer1.units, layer2.units)
                        distance += abs(layer1.units - layer2.units) / max_units * 0.2
                
                # Activation difference
                if layer1.activation != layer2.activation:
                    distance += 0.1
        
        # Hyperparameter differences
        distance += abs(arch1.learning_rate - arch2.learning_rate) / max(arch1.learning_rate, arch2.learning_rate, 1e-6)
        distance += abs(arch1.batch_size - arch2.batch_size) / max(arch1.batch_size, arch2.batch_size)
        
        if arch1.optimizer != arch2.optimizer:
            distance += 0.2
        
        return min(distance, 2.0)  # Cap maximum distance
    
    def _get_layer_count_distribution(self) -> Dict[int, int]:
        """Get distribution of layer counts in population."""
        distribution = {}
        for arch in self.population:
            layer_count = len([l for l in arch.layers if l.layer_type == LayerType.DENSE])
            distribution[layer_count] = distribution.get(layer_count, 0) + 1
        return distribution
    
    def _get_optimizer_distribution(self) -> Dict[str, int]:
        """Get distribution of optimizers in population."""
        distribution = {}
        for arch in self.population:
            optimizer = arch.optimizer.value
            distribution[optimizer] = distribution.get(optimizer, 0) + 1
        return distribution
    
    def _compile_search_results(self, duration: float) -> Dict[str, Any]:
        """Compile final search results."""
        best_performance = self.best_architecture.training_history[-1] if self.best_architecture else {}
        
        results = {
            'search_id': hashlib.md5(f"{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8],
            'timestamp': datetime.utcnow().isoformat(),
            'duration_seconds': duration,
            'generations_completed': self.generation_counter + 1,
            'population_size': self.population_size,
            'best_architecture': {
                'id': self.best_architecture.architecture_id if self.best_architecture else None,
                'fitness_score': self.best_architecture.fitness_score if self.best_architecture else 0.0,
                'complexity_score': self.best_architecture.complexity_score if self.best_architecture else 0.0,
                'layers': [layer.to_dict() for layer in self.best_architecture.layers] if self.best_architecture else [],
                'hyperparameters': {
                    'optimizer': self.best_architecture.optimizer.value if self.best_architecture else None,
                    'learning_rate': self.best_architecture.learning_rate if self.best_architecture else None,
                    'batch_size': self.best_architecture.batch_size if self.best_architecture else None,
                    'l2_regularization': self.best_architecture.l2_regularization if self.best_architecture else None
                },
                'performance': best_performance
            },
            'search_history': self.search_history,
            'final_population_diversity': self._calculate_architecture_diversity(),
            'layer_count_distribution': self._get_layer_count_distribution(),
            'optimizer_distribution': self._get_optimizer_distribution()
        }
        
        return results
    
    def get_best_model(self, input_features: int) -> Optional[MLPClassifier]:
        """Get best discovered model."""
        if self.best_architecture is None:
            return None
        
        return self._architecture_to_mlp(self.best_architecture, input_features)
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        return hashlib.md5(f"{datetime.utcnow().isoformat()}_{random.random()}".encode()).hexdigest()[:12]


async def run_nas_experiment(data_path: str, time_budget_minutes: int = 30) -> Dict[str, Any]:
    """
    Run Neural Architecture Search experiment.
    
    Args:
        data_path: Path to dataset CSV
        time_budget_minutes: Time budget for search in minutes
        
    Returns:
        NAS experiment results
    """
    logger.info(f"Starting NAS experiment with {time_budget_minutes} minute budget")
    
    try:
        # Load data
        data = pd.read_csv(data_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Initialize NAS
        nas = NeuralArchitectureSearcher(
            population_size=20,
            max_generations=15,
            mutation_rate=0.35,
            crossover_rate=0.65
        )
        
        # Run search
        results = await nas.search_architectures(
            X, y, time_budget_minutes=time_budget_minutes
        )
        
        # Get best model
        best_model = nas.get_best_model(X.shape[1])
        
        if best_model:
            logger.info(f"NAS successful. Best architecture found with fitness: {results['best_architecture']['fitness_score']:.4f}")
        
        # Record metrics
        metrics_collector = get_metrics_collector()
        metrics_collector.record_research_experiment(
            experiment_type='neural_architecture_search',
            duration=results['duration_seconds'],
            accuracy=results['best_architecture']['performance'].get('accuracy', 0.0),
            success=True
        )
        
        return results
        
    except Exception as e:
        logger.error(f"NAS experiment failed: {e}")
        
        # Record error
        metrics_collector = get_metrics_collector()
        metrics_collector.record_research_experiment(
            experiment_type='neural_architecture_search',
            duration=0.0,
            accuracy=0.0,
            success=False
        )
        
        return {
            'error': str(e),
            'success': False,
            'timestamp': datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    async def main():
        results = await run_nas_experiment(
            "data/processed/processed_features.csv",
            time_budget_minutes=15
        )
        
        print(f"NAS Results:")
        print(f"Best Fitness: {results.get('best_architecture', {}).get('fitness_score', 'N/A')}")
        print(f"Generations: {results.get('generations_completed', 'N/A')}")
        print(f"Duration: {results.get('duration_seconds', 'N/A')}s")
    
    asyncio.run(main())