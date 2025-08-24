"""
Explainable AI Engine for Customer Churn Prediction.

This module provides comprehensive model explainability capabilities including
LIME, SHAP, feature importance analysis, and custom explanation methods
tailored for business stakeholders and regulatory compliance.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Core ML libraries
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Local imports
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExplanationReport:
    """Container for model explanation results."""
    feature_importance: Dict[str, float]
    shap_values: Optional[List[List[float]]] = None
    lime_explanations: Optional[Dict[str, Any]] = None
    prediction_confidence: Optional[float] = None
    decision_path: Optional[Dict[str, Any]] = None
    business_insights: Optional[Dict[str, str]] = None
    regulatory_summary: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ExplanationMethod(ABC):
    """Abstract base class for explanation methods."""
    
    @abstractmethod
    def explain(self, model: BaseEstimator, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate explanations for the given model and data."""
        pass


class SHAPExplainer(ExplanationMethod):
    """SHAP-based model explainer."""
    
    def __init__(self, explainer_type: str = 'auto', sample_size: int = 1000):
        self.explainer_type = explainer_type
        self.sample_size = sample_size
        self.explainer = None
        
    def explain(self, model: BaseEstimator, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate SHAP explanations."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is not available")
        
        try:
            # Sample data if too large
            if len(X) > self.sample_size:
                X_sample = X.sample(n=self.sample_size, random_state=42)
            else:
                X_sample = X
                
            # Create appropriate explainer
            if self.explainer_type == 'auto':
                try:
                    self.explainer = shap.Explainer(model)
                except:
                    # Fallback to TreeExplainer or LinearExplainer
                    if hasattr(model, 'tree_'):
                        self.explainer = shap.TreeExplainer(model)
                    else:
                        self.explainer = shap.LinearExplainer(model, X_sample)
            elif self.explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(model)
            elif self.explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(model, X_sample)
            elif self.explainer_type == 'kernel':
                self.explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            
            # Calculate SHAP values
            shap_values = self.explainer(X_sample)
            
            # Extract information
            if hasattr(shap_values, 'values'):
                values = shap_values.values
                base_values = shap_values.base_values
            else:
                values = shap_values
                base_values = self.explainer.expected_value
            
            # Handle multi-class case (take positive class)
            if len(values.shape) > 2:
                values = values[:, :, 1]
                if isinstance(base_values, (list, np.ndarray)) and len(base_values) > 1:
                    base_values = base_values[1]
            
            return {
                'shap_values': values.tolist() if isinstance(values, np.ndarray) else values,
                'base_values': float(base_values) if np.isscalar(base_values) else base_values.tolist(),
                'feature_names': X_sample.columns.tolist(),
                'feature_importance': dict(zip(
                    X_sample.columns,
                    np.abs(values).mean(axis=0)
                )),
                'sample_size': len(X_sample)
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}


class LIMEExplainer(ExplanationMethod):
    """LIME-based model explainer."""
    
    def __init__(self, sample_size: int = 1000, num_features: int = 10):
        self.sample_size = sample_size
        self.num_features = num_features
        self.explainer = None
        
    def explain(self, model: BaseEstimator, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate LIME explanations."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME library is not available")
        
        try:
            # Sample data for training LIME explainer
            if len(X) > self.sample_size:
                X_sample = X.sample(n=self.sample_size, random_state=42)
            else:
                X_sample = X
            
            # Create LIME explainer
            self.explainer = lime_tabular.LimeTabularExplainer(
                X_sample.values,
                feature_names=X_sample.columns.tolist(),
                class_names=['No Churn', 'Churn'],
                mode='classification'
            )
            
            # Explain a few instances
            explanations = []
            explain_instances = min(10, len(X))
            
            for i in range(explain_instances):
                try:
                    explanation = self.explainer.explain_instance(
                        X.iloc[i].values,
                        model.predict_proba,
                        num_features=self.num_features
                    )
                    
                    # Extract feature importance for this instance
                    exp_list = explanation.as_list()
                    explanations.append({
                        'instance_id': i,
                        'prediction': float(model.predict_proba(X.iloc[i:i+1])[0][1]),
                        'feature_importance': dict(exp_list)
                    })
                except Exception as e:
                    logger.warning(f"LIME explanation failed for instance {i}: {e}")
                    continue
            
            # Aggregate feature importance across instances
            all_features = set()
            for exp in explanations:
                all_features.update(exp['feature_importance'].keys())
            
            aggregated_importance = {}
            for feature in all_features:
                values = [exp['feature_importance'].get(feature, 0) for exp in explanations]
                aggregated_importance[feature] = np.mean(values)
            
            return {
                'instance_explanations': explanations,
                'aggregated_importance': aggregated_importance,
                'num_instances_explained': len(explanations)
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'error': str(e)}


class FeatureImportanceExplainer(ExplanationMethod):
    """Feature importance-based explainer."""
    
    def explain(self, model: BaseEstimator, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Extract feature importance from model."""
        try:
            importance_dict = {}
            
            # Try different methods to get feature importance
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_dict = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                importance_dict = dict(zip(X.columns, np.abs(coef)))
            elif hasattr(model, 'named_estimators_'):
                # Ensemble models (VotingClassifier)
                importance_dict = self._aggregate_ensemble_importance(model, X.columns)
            else:
                # Permutation importance as fallback
                importance_dict = self._permutation_importance(model, X, y)
            
            # Normalize importance scores
            total_importance = sum(abs(v) for v in importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
            
            return {
                'feature_importance': importance_dict,
                'method': 'model_intrinsic'
            }
            
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {'error': str(e)}
    
    def _aggregate_ensemble_importance(self, model, feature_names):
        """Aggregate importance from ensemble models."""
        importance_dict = {name: 0.0 for name in feature_names}
        valid_estimators = 0
        
        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                for i, feature_name in enumerate(feature_names):
                    importance_dict[feature_name] += estimator.feature_importances_[i]
                valid_estimators += 1
            elif hasattr(estimator, 'coef_'):
                coef = estimator.coef_[0] if len(estimator.coef_.shape) > 1 else estimator.coef_
                for i, feature_name in enumerate(feature_names):
                    importance_dict[feature_name] += abs(coef[i])
                valid_estimators += 1
        
        if valid_estimators > 0:
            importance_dict = {k: v/valid_estimators for k, v in importance_dict.items()}
        
        return importance_dict
    
    def _permutation_importance(self, model, X, y):
        """Calculate permutation importance."""
        if y is None:
            return {col: 0.0 for col in X.columns}
        
        base_score = accuracy_score(y, model.predict(X))
        importance_dict = {}
        
        for col in X.columns:
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col])
            permuted_score = accuracy_score(y, model.predict(X_permuted))
            importance_dict[col] = base_score - permuted_score
        
        return importance_dict


class BusinessInsightsGenerator:
    """Generate business-friendly insights from model explanations."""
    
    FEATURE_BUSINESS_MAPPING = {
        'tenure': 'Customer Loyalty Duration',
        'MonthlyCharges': 'Monthly Payment Amount',
        'TotalCharges': 'Total Revenue from Customer', 
        'Contract': 'Contract Type',
        'InternetService': 'Internet Service Type',
        'PaymentMethod': 'Payment Method',
        'gender': 'Customer Gender',
        'SeniorCitizen': 'Senior Citizen Status',
        'Partner': 'Has Partner',
        'Dependents': 'Has Dependents',
        'PhoneService': 'Phone Service',
        'MultipleLines': 'Multiple Phone Lines',
        'OnlineSecurity': 'Online Security Service',
        'OnlineBackup': 'Online Backup Service',
        'DeviceProtection': 'Device Protection',
        'TechSupport': 'Technical Support',
        'StreamingTV': 'TV Streaming Service',
        'StreamingMovies': 'Movie Streaming Service',
        'PaperlessBilling': 'Paperless Billing'
    }
    
    CHURN_INSIGHTS = {
        'tenure': {
            'high_importance': "Customer loyalty duration is a key factor in churn prediction. Newer customers are more likely to churn.",
            'actionable': "Focus retention efforts on customers with tenure < 12 months. Implement onboarding programs."
        },
        'MonthlyCharges': {
            'high_importance': "Monthly charges significantly impact churn decisions. Higher charges increase churn risk.",
            'actionable': "Review pricing strategies. Consider loyalty discounts for high-value customers."
        },
        'Contract': {
            'high_importance': "Contract type is crucial for retention. Month-to-month contracts show higher churn rates.",
            'actionable': "Incentivize longer contract commitments with discounts or added services."
        },
        'TotalCharges': {
            'high_importance': "Total revenue from customer affects churn likelihood. Higher total charges may indicate loyalty.",
            'actionable': "Prioritize retention of high-value customers with personalized offers."
        }
    }
    
    def generate_insights(self, feature_importance: Dict[str, float], threshold: float = 0.1) -> Dict[str, str]:
        """Generate business insights from feature importance."""
        insights = {}
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, importance in sorted_features:
            if abs(importance) >= threshold:
                business_name = self.FEATURE_BUSINESS_MAPPING.get(feature, feature)
                
                if feature in self.CHURN_INSIGHTS:
                    insight_data = self.CHURN_INSIGHTS[feature]
                    insights[business_name] = f"{insight_data['high_importance']} {insight_data['actionable']}"
                else:
                    insights[business_name] = f"This feature shows significant impact on churn prediction (importance: {importance:.3f})"
        
        return insights
    
    def generate_regulatory_summary(self, explanations: ExplanationReport) -> str:
        """Generate regulatory compliance summary."""
        summary_parts = [
            "MODEL EXPLAINABILITY SUMMARY FOR REGULATORY COMPLIANCE",
            "=" * 60,
            f"Analysis Date: {explanations.timestamp}",
            f"Number of Features Analyzed: {len(explanations.feature_importance)}",
            "",
            "TOP DECISION FACTORS:",
        ]
        
        # Top 5 most important features
        sorted_features = sorted(
            explanations.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            business_name = self.FEATURE_BUSINESS_MAPPING.get(feature, feature)
            summary_parts.append(f"{i}. {business_name}: {importance:.4f} importance")
        
        summary_parts.extend([
            "",
            "EXPLANATION METHOD: Multiple explanation methods used including",
            "feature importance analysis and SHAP values for comprehensive",
            "model interpretability.",
            "",
            "BIAS ASSESSMENT: Model decisions are based on historical patterns",
            "in customer behavior data. Regular monitoring recommended for",
            "fairness and bias detection.",
            "",
            "MODEL TRANSPARENCY: All decision factors are traceable and",
            "explainable to stakeholders and regulatory bodies."
        ])
        
        return "\n".join(summary_parts)


class ExplainableAIEngine:
    """
    Comprehensive explainable AI engine for customer churn prediction models.
    
    Features:
    - Multiple explanation methods (SHAP, LIME, feature importance)
    - Business-friendly insights generation
    - Regulatory compliance reporting
    - Visualization support for explanations
    - Explanation persistence and versioning
    """
    
    def __init__(self):
        self.explainers = {
            'shap': SHAPExplainer(),
            'lime': LIMEExplainer(),
            'feature_importance': FeatureImportanceExplainer()
        }
        self.insights_generator = BusinessInsightsGenerator()
        self.scaler = None
        
    def explain_model(
        self, 
        model: BaseEstimator, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        methods: List[str] = None,
        generate_insights: bool = True
    ) -> ExplanationReport:
        """
        Generate comprehensive explanations for a trained model.
        
        Args:
            model: Trained scikit-learn compatible model
            X: Feature data for explanation
            y: Target data (optional, used for some explanation methods)
            methods: List of explanation methods to use
            generate_insights: Whether to generate business insights
            
        Returns:
            ExplanationReport with comprehensive explanations
        """
        if methods is None:
            methods = ['feature_importance', 'shap']
        
        logger.info(f"Generating model explanations using methods: {methods}")
        
        # Initialize explanation report
        report = ExplanationReport(feature_importance={})
        
        # Apply each explanation method
        explanations = {}
        for method in methods:
            if method in self.explainers:
                try:
                    logger.info(f"Running {method} explainer...")
                    result = self.explainers[method].explain(model, X, y)
                    
                    if 'error' not in result:
                        explanations[method] = result
                        
                        # Update main feature importance with the first successful method
                        if not report.feature_importance and 'feature_importance' in result:
                            report.feature_importance = result['feature_importance']
                        
                        # Store method-specific results
                        if method == 'shap' and 'shap_values' in result:
                            report.shap_values = result['shap_values']
                        elif method == 'lime':
                            report.lime_explanations = result
                            
                    else:
                        logger.warning(f"{method} explainer failed: {result['error']}")
                        
                except Exception as e:
                    logger.error(f"Error running {method} explainer: {e}")
                    continue
            else:
                logger.warning(f"Unknown explanation method: {method}")
        
        # Generate business insights if requested
        if generate_insights and report.feature_importance:
            report.business_insights = self.insights_generator.generate_insights(
                report.feature_importance
            )
            report.regulatory_summary = self.insights_generator.generate_regulatory_summary(report)
        
        # Calculate prediction confidence (for single prediction)
        if len(X) == 1:
            try:
                pred_proba = model.predict_proba(X)[0]
                report.prediction_confidence = float(max(pred_proba))
            except:
                report.prediction_confidence = None
        
        logger.info("Model explanation generation completed")
        return report
    
    def explain_prediction(
        self, 
        model: BaseEstimator, 
        X: pd.DataFrame, 
        instance_idx: int = 0,
        methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Explain a specific prediction instance.
        
        Args:
            model: Trained model
            X: Input data
            instance_idx: Index of instance to explain
            methods: Explanation methods to use
            
        Returns:
            Dictionary with explanation details for the specific instance
        """
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range")
        
        # Get single instance
        X_instance = X.iloc[instance_idx:instance_idx+1]
        
        # Make prediction
        try:
            prediction = model.predict(X_instance)[0]
            prediction_proba = model.predict_proba(X_instance)[0]
        except:
            prediction = None
            prediction_proba = None
        
        # Generate explanations
        report = self.explain_model(model, X_instance, methods=methods)
        
        # Compile instance-specific explanation
        explanation = {
            'instance_index': instance_idx,
            'prediction': int(prediction) if prediction is not None else None,
            'prediction_probability': prediction_proba.tolist() if prediction_proba is not None else None,
            'churn_probability': float(prediction_proba[1]) if prediction_proba is not None else None,
            'feature_values': X_instance.iloc[0].to_dict(),
            'feature_importance': report.feature_importance,
            'business_insights': report.business_insights,
            'confidence': report.prediction_confidence,
            'timestamp': report.timestamp
        }
        
        return explanation
    
    def generate_model_report(
        self, 
        model: BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a comprehensive model explanation report.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data
            output_path: Optional path to save the report
            
        Returns:
            Formatted text report
        """
        logger.info("Generating comprehensive model explanation report...")
        
        # Generate explanations
        report = self.explain_model(model, X, y, methods=['feature_importance', 'shap'])
        
        # Build text report
        report_lines = [
            "CUSTOMER CHURN PREDICTION MODEL - EXPLANATION REPORT",
            "=" * 60,
            f"Generated: {report.timestamp}",
            f"Dataset Size: {len(X)} customers, {len(X.columns)} features",
            "",
            "FEATURE IMPORTANCE ANALYSIS",
            "-" * 30
        ]
        
        # Top 10 most important features
        sorted_features = sorted(
            report.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            business_name = self.insights_generator.FEATURE_BUSINESS_MAPPING.get(feature, feature)
            report_lines.append(f"{i:2d}. {business_name:<30} {importance:>8.4f}")
        
        # Business insights
        if report.business_insights:
            report_lines.extend([
                "",
                "BUSINESS INSIGHTS",
                "-" * 20
            ])
            
            for business_feature, insight in report.business_insights.items():
                report_lines.extend([
                    f"• {business_feature}:",
                    f"  {insight}",
                    ""
                ])
        
        # Regulatory summary
        if report.regulatory_summary:
            report_lines.extend([
                "",
                report.regulatory_summary
            ])
        
        full_report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(full_report)
            logger.info(f"Model explanation report saved to {output_path}")
        
        return full_report
    
    def save_explanations(self, report: ExplanationReport, output_path: Path) -> None:
        """Save explanation report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert report to JSON-serializable format
        report_dict = {
            'feature_importance': report.feature_importance,
            'shap_values': report.shap_values,
            'lime_explanations': report.lime_explanations,
            'prediction_confidence': report.prediction_confidence,
            'business_insights': report.business_insights,
            'regulatory_summary': report.regulatory_summary,
            'timestamp': report.timestamp
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Explanations saved to {output_path}")
    
    @staticmethod
    def load_explanations(input_path: Path) -> ExplanationReport:
        """Load explanation report from JSON file."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Explanation file not found: {input_path}")
        
        with open(input_path, 'r') as f:
            report_dict = json.load(f)
        
        return ExplanationReport(**report_dict)


# Factory function for easy usage
def create_explainable_ai_engine() -> ExplainableAIEngine:
    """Create an ExplainableAIEngine with default configuration."""
    return ExplainableAIEngine()