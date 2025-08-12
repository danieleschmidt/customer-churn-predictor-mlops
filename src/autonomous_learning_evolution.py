"""
Autonomous Learning Evolution System for Self-Improving ML Models.

This module implements evolutionary algorithms and self-adaptive learning systems
that continuously evolve model architectures and hyperparameters without human intervention.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import json
import logging
import asyncio
from dataclasses import dataclass, field, asdict
from enum import Enum
import random
import hashlib
import pickle
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib

from .logging_config import get_logger
from .metrics import get_metrics_collector
from .config import load_config

logger = get_logger(__name__)


class EvolutionStrategy(Enum):
    """Evolution strategies for model improvement."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    HYBRID_EVOLUTION = "hybrid_evolution"


class MutationType(Enum):
    """Types of mutations for model evolution."""
    HYPERPARAMETER_TWEAK = "hyperparameter_tweak"
    ARCHITECTURE_CHANGE = "architecture_change"
    FEATURE_SELECTION = "feature_selection"
    ENSEMBLE_MODIFICATION = "ensemble_modification"
    LEARNING_RATE_ADAPTATION = "learning_rate_adaptation"
    REGULARIZATION_CHANGE = "regularization_change"


@dataclass
class ModelGene:
    """Represents a genetic component of a machine learning model."""
    gene_type: str
    value: Any
    mutation_rate: float = 0.1
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_impact: float = 0.0
    
    def mutate(self, mutation_strength: float = 1.0) -> 'ModelGene':
        """Apply mutation to gene with adaptive strength."""
        mutated_gene = copy.deepcopy(self)
        
        if random.random() < self.mutation_rate * mutation_strength:
            if self.gene_type == "numeric":
                if isinstance(self.value, (int, float)):
                    # Gaussian mutation with adaptive variance
                    std = abs(self.value * 0.2) if self.value != 0 else 0.1
                    mutation = np.random.normal(0, std)
                    mutated_gene.value = max(0.001, self.value + mutation)
            
            elif self.gene_type == "categorical":
                if isinstance(self.value, list):
                    # Random choice from categories
                    mutated_gene.value = random.choice(self.value)
            
            elif self.gene_type == "boolean":
                mutated_gene.value = not self.value
            
            # Record mutation
            mutated_gene.adaptation_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'mutation_type': 'value_change',
                'old_value': self.value,
                'new_value': mutated_gene.value,
                'mutation_strength': mutation_strength
            })
        
        return mutated_gene
    
    def crossover(self, other: 'ModelGene') -> Tuple['ModelGene', 'ModelGene']:
        """Perform crossover with another gene."""
        if self.gene_type != other.gene_type:
            return self, other
        
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        if self.gene_type == "numeric":
            # Arithmetic crossover
            alpha = random.random()
            child1.value = alpha * self.value + (1 - alpha) * other.value
            child2.value = (1 - alpha) * self.value + alpha * other.value
        
        elif self.gene_type == "categorical":
            # Uniform crossover
            if random.random() < 0.5:
                child1.value, child2.value = other.value, self.value
        
        return child1, child2


@dataclass
class ModelChromosome:
    """Represents a complete model configuration as a chromosome."""
    chromosome_id: str
    genes: Dict[str, ModelGene]
    model_type: str
    fitness_score: float = 0.0
    age: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def mutate(self, mutation_rate: float = 0.1, adaptive_strength: bool = True) -> 'ModelChromosome':
        """Mutate chromosome with adaptive mutation strategy."""
        mutated_chromosome = copy.deepcopy(self)
        mutated_chromosome.chromosome_id = self._generate_id()
        mutated_chromosome.parent_ids = [self.chromosome_id]
        
        # Adaptive mutation strength based on performance
        if adaptive_strength and self.performance_history:
            recent_performance = [h['fitness'] for h in self.performance_history[-3:]]
            if len(recent_performance) > 1:
                performance_trend = np.mean(np.diff(recent_performance))
                # Increase mutation if performance is declining
                mutation_strength = 1.5 if performance_trend < 0 else 0.8
            else:
                mutation_strength = 1.0
        else:
            mutation_strength = 1.0
        
        # Mutate genes
        mutated_genes = {}
        for gene_name, gene in self.genes.items():
            if random.random() < mutation_rate:
                mutated_genes[gene_name] = gene.mutate(mutation_strength)
            else:
                mutated_genes[gene_name] = copy.deepcopy(gene)
        
        mutated_chromosome.genes = mutated_genes
        return mutated_chromosome
    
    def crossover(self, other: 'ModelChromosome') -> Tuple['ModelChromosome', 'ModelChromosome']:
        """Perform crossover with another chromosome."""
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        child1.chromosome_id = self._generate_id()
        child2.chromosome_id = self._generate_id()
        child1.parent_ids = [self.chromosome_id, other.chromosome_id]
        child2.parent_ids = [self.chromosome_id, other.chromosome_id]
        
        # Gene-wise crossover
        for gene_name in self.genes.keys():
            if gene_name in other.genes:
                gene1, gene2 = self.genes[gene_name].crossover(other.genes[gene_name])
                child1.genes[gene_name] = gene1
                child2.genes[gene_name] = gene2
        
        return child1, child2
    
    def _generate_id(self) -> str:
        """Generate unique chromosome ID."""
        timestamp = datetime.utcnow().isoformat()
        content = f"{self.model_type}_{timestamp}_{random.randint(1000, 9999)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_model_params(self) -> Dict[str, Any]:
        """Convert chromosome to model parameters."""
        params = {}
        for gene_name, gene in self.genes.items():
            params[gene_name] = gene.value
        return params


class EvolutionaryModelFactory:
    """Factory for creating and evolving machine learning models."""
    
    def __init__(self):
        self.model_templates = {
            'logistic_regression': self._create_logistic_regression_template,
            'random_forest': self._create_random_forest_template,
            'gradient_boosting': self._create_gradient_boosting_template,
            'svm': self._create_svm_template,
            'mlp': self._create_mlp_template
        }
    
    def create_random_chromosome(self, model_type: str) -> ModelChromosome:
        """Create random model chromosome."""
        if model_type not in self.model_templates:
            raise ValueError(f"Unknown model type: {model_type}")
        
        template_func = self.model_templates[model_type]
        genes = template_func()
        
        chromosome = ModelChromosome(
            chromosome_id=self._generate_id(),
            genes=genes,
            model_type=model_type,
            generation=0
        )
        
        return chromosome
    
    def chromosome_to_model(self, chromosome: ModelChromosome) -> BaseEstimator:
        """Convert chromosome to actual scikit-learn model."""
        params = chromosome.to_model_params()
        
        if chromosome.model_type == 'logistic_regression':
            return LogisticRegression(**params, random_state=42)
        elif chromosome.model_type == 'random_forest':
            return RandomForestClassifier(**params, random_state=42)
        elif chromosome.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**params, random_state=42)
        elif chromosome.model_type == 'svm':
            return SVC(**params, random_state=42)
        elif chromosome.model_type == 'mlp':
            return MLPClassifier(**params, random_state=42, max_iter=300)
        else:
            raise ValueError(f"Unknown model type: {chromosome.model_type}")
    
    def _create_logistic_regression_template(self) -> Dict[str, ModelGene]:
        """Create logistic regression gene template."""
        return {
            'C': ModelGene('numeric', np.random.uniform(0.01, 100), 0.2),
            'penalty': ModelGene('categorical', random.choice(['l1', 'l2', 'elasticnet']), 0.1),
            'solver': ModelGene('categorical', random.choice(['liblinear', 'saga', 'lbfgs']), 0.1)
        }
    
    def _create_random_forest_template(self) -> Dict[str, ModelGene]:
        """Create random forest gene template."""
        return {
            'n_estimators': ModelGene('numeric', random.randint(10, 200), 0.15),
            'max_depth': ModelGene('numeric', random.randint(3, 20), 0.15),
            'min_samples_split': ModelGene('numeric', random.randint(2, 20), 0.1),
            'min_samples_leaf': ModelGene('numeric', random.randint(1, 10), 0.1),
            'max_features': ModelGene('categorical', random.choice(['auto', 'sqrt', 'log2']), 0.1)
        }
    
    def _create_gradient_boosting_template(self) -> Dict[str, ModelGene]:
        """Create gradient boosting gene template."""
        return {
            'n_estimators': ModelGene('numeric', random.randint(50, 300), 0.15),
            'learning_rate': ModelGene('numeric', np.random.uniform(0.01, 0.3), 0.2),
            'max_depth': ModelGene('numeric', random.randint(3, 10), 0.15),
            'min_samples_split': ModelGene('numeric', random.randint(2, 20), 0.1),
            'subsample': ModelGene('numeric', np.random.uniform(0.6, 1.0), 0.1)
        }
    
    def _create_svm_template(self) -> Dict[str, ModelGene]:
        """Create SVM gene template."""
        return {
            'C': ModelGene('numeric', np.random.uniform(0.1, 100), 0.2),
            'kernel': ModelGene('categorical', random.choice(['rbf', 'linear', 'poly']), 0.1),
            'gamma': ModelGene('categorical', random.choice(['scale', 'auto']), 0.1)
        }
    
    def _create_mlp_template(self) -> Dict[str, ModelGene]:
        """Create MLP gene template."""
        hidden_layer_size = random.randint(50, 200)
        return {
            'hidden_layer_sizes': ModelGene('numeric', (hidden_layer_size,), 0.15),
            'alpha': ModelGene('numeric', np.random.uniform(0.0001, 0.01), 0.2),
            'learning_rate': ModelGene('categorical', random.choice(['constant', 'adaptive']), 0.1),
            'activation': ModelGene('categorical', random.choice(['relu', 'tanh', 'logistic']), 0.1)
        }
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        return hashlib.md5(f"{datetime.utcnow().isoformat()}_{random.randint(1000, 9999)}".encode()).hexdigest()[:12]


class AutonomousLearningEvolution:
    """Main evolutionary learning system for autonomous model improvement."""
    
    def __init__(self, 
                 population_size: int = 20,
                 elite_size: int = 5,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 max_generations: int = 50):
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        
        self.model_factory = EvolutionaryModelFactory()
        self.population: List[ModelChromosome] = []
        self.evolution_history = []
        self.best_chromosome = None
        self.generation_counter = 0
        
        self.performance_targets = {
            'accuracy': 0.85,
            'f1_score': 0.80,
            'roc_auc': 0.85
        }
        
    async def evolve_models(self, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           X_val: Optional[pd.DataFrame] = None,
                           y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Run autonomous evolution to find optimal model configurations.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dict containing evolution results and best model
        """
        logger.info(f"Starting autonomous model evolution with {self.population_size} individuals")
        start_time = time.time()
        
        # Split validation data if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Initialize population
        await self._initialize_population()
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation_counter = generation
            logger.info(f"Generation {generation + 1}/{self.max_generations}")
            
            # Evaluate population fitness
            await self._evaluate_population_fitness(X_train, y_train, X_val, y_val)
            
            # Record generation statistics
            self._record_generation_stats(generation)
            
            # Check if evolution targets are met
            if self._check_evolution_targets():
                logger.info(f"Evolution targets achieved at generation {generation + 1}")
                break
            
            # Create next generation
            await self._create_next_generation()
            
            # Adaptive parameter adjustment
            self._adapt_evolution_parameters(generation)
        
        # Final evaluation and results
        evolution_duration = time.time() - start_time
        results = self._compile_evolution_results(evolution_duration)
        
        logger.info(f"Evolution complete. Best fitness: {self.best_chromosome.fitness_score:.4f}")
        
        return results
    
    async def _initialize_population(self):
        """Initialize random population of model chromosomes."""
        self.population = []
        model_types = list(self.model_factory.model_templates.keys())
        
        # Create diverse population
        for i in range(self.population_size):
            model_type = random.choice(model_types)
            chromosome = self.model_factory.create_random_chromosome(model_type)
            chromosome.generation = 0
            self.population.append(chromosome)
        
        logger.info(f"Initialized population with {len(model_types)} model types")
    
    async def _evaluate_population_fitness(self, X_train: pd.DataFrame, y_train: pd.Series,
                                         X_val: pd.DataFrame, y_val: pd.Series):
        """Evaluate fitness of entire population using parallel processing."""
        
        async def evaluate_single_chromosome(chromosome: ModelChromosome) -> ModelChromosome:
            """Evaluate single chromosome fitness."""
            try:
                # Create model from chromosome
                model = self.model_factory.chromosome_to_model(chromosome)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Calculate multiple metrics
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                
                try:
                    roc_auc = roc_auc_score(y_val, y_prob)
                except:
                    roc_auc = accuracy  # Fallback if ROC AUC fails
                
                # Composite fitness score
                fitness = 0.4 * accuracy + 0.3 * f1 + 0.3 * roc_auc
                
                # Apply age penalty (promote diversity)
                age_penalty = min(0.1, chromosome.age * 0.01)
                fitness -= age_penalty
                
                chromosome.fitness_score = fitness
                chromosome.age += 1
                
                # Record performance
                performance_record = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'generation': self.generation_counter,
                    'fitness': fitness,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'age_penalty': age_penalty
                }
                chromosome.performance_history.append(performance_record)
                
                return chromosome
                
            except Exception as e:
                logger.warning(f"Chromosome evaluation failed: {e}")
                chromosome.fitness_score = 0.0
                return chromosome
        
        # Evaluate all chromosomes in parallel
        tasks = [evaluate_single_chromosome(chrom) for chrom in self.population]
        evaluated_chromosomes = await asyncio.gather(*tasks)
        
        self.population = evaluated_chromosomes
        
        # Update best chromosome
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        if self.best_chromosome is None or self.population[0].fitness_score > self.best_chromosome.fitness_score:
            self.best_chromosome = copy.deepcopy(self.population[0])
    
    async def _create_next_generation(self):
        """Create next generation using selection, crossover, and mutation."""
        
        # Elite selection - keep best performers
        elite = self.population[:self.elite_size]
        next_generation = [copy.deepcopy(chrom) for chrom in elite]
        
        # Tournament selection for breeding
        breeding_pool = self._tournament_selection(self.population_size - self.elite_size)
        
        # Crossover and mutation
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.sample(breeding_pool, 2)
            
            if random.random() < self.crossover_rate:
                # Crossover
                child1, child2 = parent1.crossover(parent2)
                children = [child1, child2]
            else:
                # Direct copy
                children = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
            
            # Mutation
            for child in children:
                if len(next_generation) < self.population_size:
                    if random.random() < self.mutation_rate:
                        child = child.mutate(self.mutation_rate, adaptive_strength=True)
                    
                    child.generation = self.generation_counter + 1
                    next_generation.append(child)
        
        self.population = next_generation[:self.population_size]
    
    def _tournament_selection(self, num_individuals: int, tournament_size: int = 3) -> List[ModelChromosome]:
        """Tournament selection for breeding."""
        selected = []
        
        for _ in range(num_individuals):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness_score)
            selected.append(winner)
        
        return selected
    
    def _record_generation_stats(self, generation: int):
        """Record statistics for current generation."""
        fitness_scores = [chrom.fitness_score for chrom in self.population]
        
        stats = {
            'generation': generation,
            'timestamp': datetime.utcnow().isoformat(),
            'best_fitness': max(fitness_scores),
            'average_fitness': np.mean(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'population_diversity': self._calculate_population_diversity(),
            'model_type_distribution': self._get_model_type_distribution(),
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate
        }
        
        self.evolution_history.append(stats)
        
        if generation % 10 == 0:
            logger.info(f"Gen {generation}: Best={stats['best_fitness']:.4f}, "
                       f"Avg={stats['average_fitness']:.4f}, "
                       f"Diversity={stats['population_diversity']:.4f}")
    
    def _check_evolution_targets(self) -> bool:
        """Check if evolution targets have been achieved."""
        if self.best_chromosome is None:
            return False
        
        latest_performance = self.best_chromosome.performance_history[-1]
        
        targets_met = (
            latest_performance['accuracy'] >= self.performance_targets['accuracy'] and
            latest_performance['f1_score'] >= self.performance_targets['f1_score'] and
            latest_performance['roc_auc'] >= self.performance_targets['roc_auc']
        )
        
        return targets_met
    
    def _adapt_evolution_parameters(self, generation: int):
        """Adapt evolution parameters based on progress."""
        if len(self.evolution_history) < 5:
            return
        
        # Calculate progress over last 5 generations
        recent_best = [stats['best_fitness'] for stats in self.evolution_history[-5:]]
        progress = np.mean(np.diff(recent_best))
        
        # Adapt mutation rate based on progress
        if progress < 0.001:  # Low progress
            self.mutation_rate = min(0.4, self.mutation_rate * 1.2)  # Increase exploration
        elif progress > 0.01:  # High progress
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)  # Decrease exploration
        
        # Adapt crossover rate
        diversity = self.evolution_history[-1]['population_diversity']
        if diversity < 0.1:  # Low diversity
            self.crossover_rate = min(0.9, self.crossover_rate * 1.1)
        elif diversity > 0.8:  # High diversity
            self.crossover_rate = max(0.3, self.crossover_rate * 0.95)
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity metric."""
        if len(self.population) < 2:
            return 0.0
        
        # Compare chromosomes based on their gene values
        diversity_sum = 0.0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                chromosome1 = self.population[i]
                chromosome2 = self.population[j]
                
                # Calculate genetic distance
                distance = self._calculate_chromosome_distance(chromosome1, chromosome2)
                diversity_sum += distance
                comparisons += 1
        
        return diversity_sum / comparisons if comparisons > 0 else 0.0
    
    def _calculate_chromosome_distance(self, chrom1: ModelChromosome, chrom2: ModelChromosome) -> float:
        """Calculate distance between two chromosomes."""
        if chrom1.model_type != chrom2.model_type:
            return 1.0  # Maximum distance for different model types
        
        distance = 0.0
        common_genes = set(chrom1.genes.keys()) & set(chrom2.genes.keys())
        
        if len(common_genes) == 0:
            return 1.0
        
        for gene_name in common_genes:
            gene1 = chrom1.genes[gene_name]
            gene2 = chrom2.genes[gene_name]
            
            if gene1.gene_type == 'numeric':
                # Normalized numeric distance
                val1, val2 = gene1.value, gene2.value
                if isinstance(val1, tuple):
                    val1 = val1[0] if len(val1) > 0 else 0
                if isinstance(val2, tuple):
                    val2 = val2[0] if len(val2) > 0 else 0
                
                max_val = max(abs(val1), abs(val2), 1)
                distance += abs(val1 - val2) / max_val
            
            elif gene1.gene_type == 'categorical':
                # Binary distance for categorical
                distance += 0.0 if gene1.value == gene2.value else 1.0
            
            elif gene1.gene_type == 'boolean':
                distance += 0.0 if gene1.value == gene2.value else 1.0
        
        return distance / len(common_genes)
    
    def _get_model_type_distribution(self) -> Dict[str, int]:
        """Get distribution of model types in population."""
        distribution = {}
        for chromosome in self.population:
            model_type = chromosome.model_type
            distribution[model_type] = distribution.get(model_type, 0) + 1
        return distribution
    
    def _compile_evolution_results(self, duration: float) -> Dict[str, Any]:
        """Compile final evolution results."""
        best_performance = self.best_chromosome.performance_history[-1] if self.best_chromosome else {}
        
        results = {
            'evolution_id': hashlib.md5(f"{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8],
            'timestamp': datetime.utcnow().isoformat(),
            'duration_seconds': duration,
            'generations_completed': self.generation_counter + 1,
            'population_size': self.population_size,
            'best_chromosome': {
                'id': self.best_chromosome.chromosome_id if self.best_chromosome else None,
                'model_type': self.best_chromosome.model_type if self.best_chromosome else None,
                'fitness_score': self.best_chromosome.fitness_score if self.best_chromosome else 0.0,
                'parameters': self.best_chromosome.to_model_params() if self.best_chromosome else {},
                'performance': best_performance
            },
            'evolution_history': self.evolution_history,
            'final_population_diversity': self._calculate_population_diversity(),
            'model_type_distribution': self._get_model_type_distribution(),
            'performance_targets': self.performance_targets,
            'targets_achieved': self._check_evolution_targets()
        }
        
        return results
    
    def get_best_model(self) -> Optional[BaseEstimator]:
        """Get the best evolved model."""
        if self.best_chromosome is None:
            return None
        
        return self.model_factory.chromosome_to_model(self.best_chromosome)
    
    def save_evolution_state(self, filepath: str):
        """Save evolution state for resuming."""
        state = {
            'population': [asdict(chrom) for chrom in self.population],
            'best_chromosome': asdict(self.best_chromosome) if self.best_chromosome else None,
            'evolution_history': self.evolution_history,
            'generation_counter': self.generation_counter,
            'parameters': {
                'population_size': self.population_size,
                'elite_size': self.elite_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_generations': self.max_generations
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Evolution state saved to {filepath}")


async def run_autonomous_evolution_experiment(data_path: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run autonomous evolution experiment for model optimization.
    
    Args:
        data_path: Path to dataset CSV
        config_path: Optional configuration file path
        
    Returns:
        Dict with evolution results
    """
    logger.info("Starting autonomous evolution experiment")
    
    try:
        # Load data
        data = pd.read_csv(data_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Initialize evolution system
        evolution_system = AutonomousLearningEvolution(
            population_size=15,
            max_generations=30,
            mutation_rate=0.15,
            crossover_rate=0.75
        )
        
        # Run evolution
        results = await evolution_system.evolve_models(X, y)
        
        # Get best model
        best_model = evolution_system.get_best_model()
        
        if best_model:
            logger.info(f"Evolution successful. Best model: {type(best_model).__name__}")
        
        # Record experiment metrics
        metrics_collector = get_metrics_collector()
        metrics_collector.record_research_experiment(
            experiment_type='autonomous_evolution',
            duration=results['duration_seconds'],
            accuracy=results['best_chromosome']['performance'].get('accuracy', 0.0),
            success=True
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Autonomous evolution experiment failed: {e}")
        
        # Record error
        metrics_collector = get_metrics_collector()
        metrics_collector.record_research_experiment(
            experiment_type='autonomous_evolution',
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
        results = await run_autonomous_evolution_experiment(
            "data/processed/processed_features.csv"
        )
        
        print(f"Evolution Results:")
        print(f"Best Fitness: {results.get('best_chromosome', {}).get('fitness_score', 'N/A')}")
        print(f"Generations: {results.get('generations_completed', 'N/A')}")
        print(f"Duration: {results.get('duration_seconds', 'N/A')}s")
    
    asyncio.run(main())