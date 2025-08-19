"""
Enhanced Research API for Advanced ML Experiments.

This module extends the main API with research-focused endpoints that expose
the novel machine learning frameworks for academic and industrial research.

Key Features:
- Causal discovery and causal-aware prediction endpoints
- Temporal graph neural network experimentation
- Multi-modal fusion with text and behavioral data
- Uncertainty-aware ensemble predictions with confidence intervals
- Comprehensive experimental frameworks for publication-ready research
"""

import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import mlflow

from .logging_config import get_logger
from .data_validation import validate_customer_data
from .causal_discovery_framework import (
    CausalGraphNeuralNetwork, 
    CausalDiscoveryConfig, 
    run_causal_discovery_experiment
)
from .temporal_graph_networks import (
    TemporalGraphNeuralNetwork,
    TemporalGraphConfig,
    run_temporal_graph_experiment
)
from .multimodal_fusion_framework import (
    MultiModalFusionNetwork,
    MultiModalConfig,
    run_multimodal_fusion_experiment,
    create_synthetic_multimodal_data
)
from .uncertainty_aware_ensembles import (
    UncertaintyAwareEnsemble,
    UncertaintyConfig,
    run_uncertainty_experiment
)

logger = get_logger(__name__)

# Pydantic models for research API requests/responses

class CausalPredictionRequest(BaseModel):
    """Request model for causal-aware predictions."""
    customer_data: Dict[str, Union[str, int, float]] = Field(
        ...,
        description="Customer features for causal analysis",
        example={
            "tenure": 12,
            "MonthlyCharges": 70.0,
            "TotalCharges": 840.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "gender": "Female",
            "SeniorCitizen": 0
        }
    )
    causal_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Causal discovery configuration parameters"
    )

class CausalPredictionResponse(BaseModel):
    """Response model for causal-aware predictions."""
    prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    probability: float = Field(..., description="Churn probability")
    causal_importance: Dict[str, float] = Field(..., description="Causal feature importance scores")
    causal_edges: int = Field(..., description="Number of causal relationships discovered")
    timestamp: str = Field(..., description="Prediction timestamp")

class TemporalPredictionRequest(BaseModel):
    """Request model for temporal graph predictions."""
    customer_data: List[Dict[str, Union[str, int, float]]] = Field(
        ...,
        description="Temporal sequence of customer data",
        example=[
            {"customer_id": "C001", "timestamp": "2023-01-01", "MonthlyCharges": 50.0, "tenure": 6},
            {"customer_id": "C001", "timestamp": "2023-02-01", "MonthlyCharges": 55.0, "tenure": 7},
            {"customer_id": "C001", "timestamp": "2023-03-01", "MonthlyCharges": 60.0, "tenure": 8}
        ]
    )
    temporal_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Temporal graph configuration parameters"
    )

class TemporalPredictionResponse(BaseModel):
    """Response model for temporal graph predictions."""
    prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    probability: float = Field(..., description="Churn probability")
    temporal_attention: Dict[str, float] = Field(..., description="Temporal attention weights")
    sequence_length: int = Field(..., description="Length of temporal sequence used")
    timestamp: str = Field(..., description="Prediction timestamp")

class MultiModalPredictionRequest(BaseModel):
    """Request model for multi-modal fusion predictions."""
    customer_data: Dict[str, Union[str, int, float]] = Field(
        ...,
        description="Tabular customer features"
    )
    text_data: Optional[str] = Field(
        None,
        description="Customer text data (reviews, support tickets, etc.)",
        example="The service quality has been declining recently, considering alternatives"
    )
    behavioral_data: Optional[List[str]] = Field(
        None,
        description="Customer behavioral sequence",
        example=["login", "view_billing", "support_contact", "view_product", "logout"]
    )
    multimodal_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Multi-modal fusion configuration parameters"
    )

class MultiModalPredictionResponse(BaseModel):
    """Response model for multi-modal fusion predictions."""
    prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    probability: float = Field(..., description="Churn probability")
    modality_importance: Dict[str, float] = Field(..., description="Importance scores for each modality")
    text_features: int = Field(..., description="Number of text features extracted")
    behavioral_features: int = Field(..., description="Number of behavioral features extracted")
    timestamp: str = Field(..., description="Prediction timestamp")

class UncertaintyPredictionRequest(BaseModel):
    """Request model for uncertainty-aware predictions."""
    customer_data: Dict[str, Union[str, int, float]] = Field(
        ...,
        description="Customer features for uncertainty analysis"
    )
    uncertainty_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Uncertainty estimation configuration parameters"
    )

class UncertaintyPredictionResponse(BaseModel):
    """Response model for uncertainty-aware predictions."""
    prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    probability: float = Field(..., description="Ensemble probability")
    calibrated_probability: float = Field(..., description="Calibrated probability")
    epistemic_uncertainty: float = Field(..., description="Model uncertainty (epistemic)")
    aleatoric_uncertainty: float = Field(..., description="Data uncertainty (aleatoric)")
    total_uncertainty: float = Field(..., description="Total uncertainty")
    confidence_interval: List[float] = Field(..., description="Prediction confidence interval [lower, upper]")
    high_confidence: bool = Field(..., description="Whether prediction is high confidence")
    timestamp: str = Field(..., description="Prediction timestamp")

class ExperimentRequest(BaseModel):
    """Request model for research experiments."""
    experiment_type: str = Field(
        ...,
        description="Type of experiment to run",
        regex="^(causal|temporal|multimodal|uncertainty)$"
    )
    dataset_path: Optional[str] = Field(
        None,
        description="Path to experimental dataset (optional, uses default if not provided)"
    )
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Experiment configuration parameters"
    )
    n_samples: Optional[int] = Field(
        1000,
        description="Number of samples to generate for synthetic experiments"
    )

class ExperimentResponse(BaseModel):
    """Response model for research experiments."""
    experiment_type: str = Field(..., description="Type of experiment conducted")
    experiment_id: str = Field(..., description="Unique experiment identifier")
    performance: Dict[str, Any] = Field(..., description="Performance metrics and comparisons")
    statistical_significance: Dict[str, Any] = Field(..., description="Statistical test results")
    experiment_config: Dict[str, Any] = Field(..., description="Configuration used")
    runtime_seconds: float = Field(..., description="Total experiment runtime")
    timestamp: str = Field(..., description="Experiment completion timestamp")

# Initialize research models (lazy loading)
_research_models = {
    'causal': None,
    'temporal': None, 
    'multimodal': None,
    'uncertainty': None
}

def get_causal_model() -> CausalGraphNeuralNetwork:
    """Get or create causal discovery model."""
    if _research_models['causal'] is None:
        _research_models['causal'] = CausalGraphNeuralNetwork()
    return _research_models['causal']

def get_temporal_model() -> TemporalGraphNeuralNetwork:
    """Get or create temporal graph model."""
    if _research_models['temporal'] is None:
        _research_models['temporal'] = TemporalGraphNeuralNetwork()
    return _research_models['temporal']

def get_multimodal_model() -> MultiModalFusionNetwork:
    """Get or create multi-modal fusion model."""
    if _research_models['multimodal'] is None:
        _research_models['multimodal'] = MultiModalFusionNetwork()
    return _research_models['multimodal']

def get_uncertainty_model() -> UncertaintyAwareEnsemble:
    """Get or create uncertainty-aware ensemble."""
    if _research_models['uncertainty'] is None:
        _research_models['uncertainty'] = UncertaintyAwareEnsemble()
    return _research_models['uncertainty']

def load_sample_dataset(n_samples: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    """Load or generate sample dataset for experiments."""
    try:
        # Try to load real dataset first
        data_path = "/root/repo/data/processed/processed_features.csv"
        target_path = "/root/repo/data/processed/processed_target.csv"
        
        if os.path.exists(data_path) and os.path.exists(target_path):
            X = pd.read_csv(data_path)
            y = pd.read_csv(target_path).iloc[:, 0]  # Assume first column is target
            
            # Sample if too large
            if len(X) > n_samples:
                indices = np.random.choice(len(X), n_samples, replace=False)
                X = X.iloc[indices].reset_index(drop=True)
                y = y.iloc[indices].reset_index(drop=True)
                
            return X, y
        else:
            # Generate synthetic dataset
            logger.info("Real dataset not found, generating synthetic data")
            return generate_synthetic_dataset(n_samples)
            
    except Exception as e:
        logger.warning(f"Error loading dataset: {e}, generating synthetic data")
        return generate_synthetic_dataset(n_samples)

def generate_synthetic_dataset(n_samples: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic customer dataset for experiments."""
    np.random.seed(42)
    
    # Generate synthetic features
    data = {
        'tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 100, n_samples),
        'TotalCharges': np.random.uniform(100, 5000, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'customer_id': [f'C{i:06d}' for i in range(n_samples)]
    }
    
    # Add timestamp for temporal experiments
    start_date = pd.Timestamp('2023-01-01')
    data['timestamp'] = [start_date + pd.Timedelta(days=np.random.randint(0, 365)) 
                        for _ in range(n_samples)]
    
    X = pd.DataFrame(data)
    
    # Generate target with some logic
    churn_prob = (
        0.3 * (X['tenure'] < 12).astype(int) +
        0.2 * (X['MonthlyCharges'] > 70).astype(int) +
        0.2 * (X['Contract'] == 'Month-to-month').astype(int) +
        0.1 * np.random.random(n_samples)
    )
    
    y = pd.Series((churn_prob > 0.5).astype(int), name='churn')
    
    return X, y

# API endpoints for research functionality

def create_research_endpoints(app: FastAPI):
    """Create research API endpoints."""
    
    @app.post("/research/causal/predict", 
              response_model=CausalPredictionResponse,
              tags=["Research - Causal Discovery"])
    async def predict_with_causal_model(request: CausalPredictionRequest) -> CausalPredictionResponse:
        """
        Make predictions using causal-aware machine learning.
        
        This endpoint uses causal discovery to identify causal relationships
        between customer features and churn, providing more interpretable
        and robust predictions than correlation-based approaches.
        """
        try:
            start_time = time.time()
            
            # Convert request to DataFrame
            customer_df = pd.DataFrame([request.customer_data])
            
            # Get causal model
            causal_model = get_causal_model()
            
            # Check if model is fitted, if not, fit with sample data
            if not causal_model.is_fitted:
                logger.info("Causal model not fitted, training with sample data")
                X_sample, y_sample = load_sample_dataset(500)
                causal_model.fit(X_sample, y_sample)
            
            # Make prediction
            prediction = causal_model.predict(customer_df)[0]
            probability = causal_model.predict_proba(customer_df)[0, 1]
            
            # Get causal importance
            causal_importance = causal_model.get_causal_importance()
            
            # Count causal edges
            causal_edges = len(causal_model.causal_graph.edges) if causal_model.causal_graph else 0
            
            response = CausalPredictionResponse(
                prediction=int(prediction),
                probability=float(probability),
                causal_importance=causal_importance,
                causal_edges=causal_edges,
                timestamp=datetime.now().isoformat()
            )
            
            prediction_time = time.time() - start_time
            logger.info(f"Causal prediction completed in {prediction_time:.3f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in causal prediction: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Causal prediction failed: {str(e)}"
            )
    
    @app.post("/research/temporal/predict",
              response_model=TemporalPredictionResponse,
              tags=["Research - Temporal Graphs"])
    async def predict_with_temporal_model(request: TemporalPredictionRequest) -> TemporalPredictionResponse:
        """
        Make predictions using temporal graph neural networks.
        
        This endpoint models customer journeys as temporal graphs to capture
        sequential dependencies and interaction patterns for improved churn prediction.
        """
        try:
            start_time = time.time()
            
            # Convert request to DataFrame
            temporal_df = pd.DataFrame(request.customer_data)
            
            # Ensure required columns
            if 'customer_id' not in temporal_df.columns:
                temporal_df['customer_id'] = 'temp_customer'
            
            # Get temporal model
            temporal_model = get_temporal_model()
            
            # Check if model is fitted
            if not temporal_model.is_fitted:
                logger.info("Temporal model not fitted, training with sample data")
                X_sample, y_sample = load_sample_dataset(500)
                temporal_model.fit(X_sample, y_sample)
            
            # For single prediction, use the last row
            prediction_row = temporal_df.iloc[[-1]]
            prediction = temporal_model.predict(prediction_row)[0]
            probability = temporal_model.predict_proba(prediction_row)[0, 1]
            
            # Get temporal attention (if available)
            customer_id = prediction_row['customer_id'].iloc[0]
            temporal_attention = temporal_model.get_temporal_attention_weights(str(customer_id))
            
            response = TemporalPredictionResponse(
                prediction=int(prediction),
                probability=float(probability),
                temporal_attention=temporal_attention,
                sequence_length=len(temporal_df),
                timestamp=datetime.now().isoformat()
            )
            
            prediction_time = time.time() - start_time
            logger.info(f"Temporal prediction completed in {prediction_time:.3f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in temporal prediction: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Temporal prediction failed: {str(e)}"
            )
    
    @app.post("/research/multimodal/predict",
              response_model=MultiModalPredictionResponse,
              tags=["Research - Multi-Modal Fusion"])
    async def predict_with_multimodal_model(request: MultiModalPredictionRequest) -> MultiModalPredictionResponse:
        """
        Make predictions using multi-modal fusion.
        
        This endpoint combines tabular customer data with text and behavioral
        information to achieve superior prediction performance through
        holistic customer understanding.
        """
        try:
            start_time = time.time()
            
            # Convert request to DataFrame
            customer_df = pd.DataFrame([request.customer_data])
            
            # Prepare multi-modal data
            text_data = [request.text_data] if request.text_data else None
            behavioral_data = [request.behavioral_data] if request.behavioral_data else None
            
            # Get multi-modal model
            multimodal_model = get_multimodal_model()
            
            # Check if model is fitted
            if not multimodal_model.is_fitted:
                logger.info("Multi-modal model not fitted, training with sample data")
                X_sample, y_sample = load_sample_dataset(500)
                
                # Generate synthetic multi-modal data
                text_sample, behavioral_sample = create_synthetic_multimodal_data(X_sample, y_sample)
                multimodal_model.fit(X_sample, y_sample, text_sample, behavioral_sample)
            
            # Make prediction
            prediction = multimodal_model.predict(customer_df, text_data, behavioral_data)[0]
            probability = multimodal_model.predict_proba(customer_df, text_data, behavioral_data)[0, 1]
            
            # Get modality importance
            modality_importance = multimodal_model.get_modality_importance()
            
            # Count features
            text_features = multimodal_model.text_encoder.config.text_embedding_dim if text_data else 0
            behavioral_features = multimodal_model.behavioral_encoder.config.behavior_embedding_dim * 3 if behavioral_data else 0
            
            response = MultiModalPredictionResponse(
                prediction=int(prediction),
                probability=float(probability),
                modality_importance=modality_importance,
                text_features=text_features,
                behavioral_features=behavioral_features,
                timestamp=datetime.now().isoformat()
            )
            
            prediction_time = time.time() - start_time
            logger.info(f"Multi-modal prediction completed in {prediction_time:.3f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in multi-modal prediction: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Multi-modal prediction failed: {str(e)}"
            )
    
    @app.post("/research/uncertainty/predict",
              response_model=UncertaintyPredictionResponse,
              tags=["Research - Uncertainty Quantification"])
    async def predict_with_uncertainty_model(request: UncertaintyPredictionRequest) -> UncertaintyPredictionResponse:
        """
        Make predictions with uncertainty quantification.
        
        This endpoint provides predictions with comprehensive uncertainty estimates,
        including epistemic (model) and aleatoric (data) uncertainty, enabling
        risk-sensitive business decision making.
        """
        try:
            start_time = time.time()
            
            # Convert request to DataFrame
            customer_df = pd.DataFrame([request.customer_data])
            
            # Get uncertainty model
            uncertainty_model = get_uncertainty_model()
            
            # Check if model is fitted
            if not uncertainty_model.is_fitted:
                logger.info("Uncertainty model not fitted, training with sample data")
                X_sample, y_sample = load_sample_dataset(500)
                uncertainty_model.fit(X_sample, y_sample)
            
            # Make prediction with uncertainty
            uncertainty_results = uncertainty_model.predict_with_uncertainty(customer_df)
            
            prediction = uncertainty_results['predictions'][0]
            probability = uncertainty_results['probabilities'][0]
            calibrated_probability = uncertainty_results['calibrated_probabilities'][0]
            epistemic_uncertainty = uncertainty_results['epistemic_uncertainty'][0]
            aleatoric_uncertainty = uncertainty_results['aleatoric_uncertainty'][0]
            total_uncertainty = uncertainty_results['total_uncertainty'][0]
            confidence_interval = uncertainty_results['confidence_intervals'][0].tolist()
            
            # Check if prediction is high confidence
            high_confidence_results = uncertainty_model.get_high_confidence_predictions(customer_df)
            high_confidence = len(high_confidence_results['confident_indices']) > 0
            
            response = UncertaintyPredictionResponse(
                prediction=int(prediction),
                probability=float(probability),
                calibrated_probability=float(calibrated_probability),
                epistemic_uncertainty=float(epistemic_uncertainty),
                aleatoric_uncertainty=float(aleatoric_uncertainty),
                total_uncertainty=float(total_uncertainty),
                confidence_interval=confidence_interval,
                high_confidence=high_confidence,
                timestamp=datetime.now().isoformat()
            )
            
            prediction_time = time.time() - start_time
            logger.info(f"Uncertainty prediction completed in {prediction_time:.3f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in uncertainty prediction: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Uncertainty prediction failed: {str(e)}"
            )
    
    @app.post("/research/experiment",
              response_model=ExperimentResponse,
              tags=["Research - Experiments"])
    async def run_research_experiment(
        request: ExperimentRequest,
        background_tasks: BackgroundTasks
    ) -> ExperimentResponse:
        """
        Run comprehensive research experiments.
        
        This endpoint executes publication-ready experimental frameworks
        for evaluating novel machine learning approaches against baselines
        with statistical significance testing.
        """
        try:
            start_time = time.time()
            experiment_id = f"exp_{int(time.time())}_{request.experiment_type}"
            
            logger.info(f"Starting {request.experiment_type} experiment {experiment_id}")
            
            # Load dataset
            if request.dataset_path and os.path.exists(request.dataset_path):
                X = pd.read_csv(request.dataset_path)
                y = pd.read_csv(request.dataset_path.replace('features', 'target')).iloc[:, 0]
            else:
                X, y = load_sample_dataset(request.n_samples or 1000)
            
            # Start MLflow run for experiment tracking
            mlflow.start_run(run_name=experiment_id)
            mlflow.log_param("experiment_type", request.experiment_type)
            mlflow.log_param("n_samples", len(X))
            
            # Run appropriate experiment
            if request.experiment_type == "causal":
                config = CausalDiscoveryConfig(**request.config) if request.config else CausalDiscoveryConfig()
                results = run_causal_discovery_experiment(X, y, config)
                
            elif request.experiment_type == "temporal":
                config = TemporalGraphConfig(**request.config) if request.config else TemporalGraphConfig()
                results = run_temporal_graph_experiment(X, y, config)
                
            elif request.experiment_type == "multimodal":
                config = MultiModalConfig(**request.config) if request.config else MultiModalConfig()
                text_data, behavioral_data = create_synthetic_multimodal_data(X, y)
                results = run_multimodal_fusion_experiment(X, y, text_data, behavioral_data, config)
                
            elif request.experiment_type == "uncertainty":
                config = UncertaintyConfig(**request.config) if request.config else UncertaintyConfig()
                results = run_uncertainty_experiment(X, y, config)
                
            else:
                raise ValueError(f"Unknown experiment type: {request.experiment_type}")
            
            # Log results to MLflow
            performance = results.get('performance', {})
            for metric_group, metrics in performance.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{metric_group}_{metric_name}", value)
            
            mlflow.end_run()
            
            runtime = time.time() - start_time
            
            response = ExperimentResponse(
                experiment_type=request.experiment_type,
                experiment_id=experiment_id,
                performance=results.get('performance', {}),
                statistical_significance=results.get('performance', {}).get('improvement', {}),
                experiment_config=results.get('config', {}),
                runtime_seconds=runtime,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Experiment {experiment_id} completed in {runtime:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in research experiment: {str(e)}")
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Research experiment failed: {str(e)}"
            )
    
    @app.get("/research/models/status",
             tags=["Research - Model Management"])
    async def get_research_models_status() -> Dict[str, Any]:
        """
        Get status of all research models.
        
        Returns information about which research models are loaded,
        their configuration, and readiness for predictions.
        """
        status_info = {}
        
        for model_name, model in _research_models.items():
            if model is not None:
                status_info[model_name] = {
                    "loaded": True,
                    "fitted": getattr(model, 'is_fitted', False),
                    "model_class": model.__class__.__name__,
                    "config": getattr(model, 'config', None)
                }
            else:
                status_info[model_name] = {
                    "loaded": False,
                    "fitted": False
                }
        
        return {
            "models": status_info,
            "total_loaded": sum(1 for m in _research_models.values() if m is not None),
            "timestamp": datetime.now().isoformat()
        }
    
    return app

# Export main functions
__all__ = [
    'create_research_endpoints',
    'CausalPredictionRequest',
    'CausalPredictionResponse',
    'TemporalPredictionRequest', 
    'TemporalPredictionResponse',
    'MultiModalPredictionRequest',
    'MultiModalPredictionResponse',
    'UncertaintyPredictionRequest',
    'UncertaintyPredictionResponse',
    'ExperimentRequest',
    'ExperimentResponse'
]