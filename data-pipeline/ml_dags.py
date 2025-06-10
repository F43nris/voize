#!/usr/bin/env python3
"""
ML Pipeline DAGs
================
Concrete DAG definitions for the medical text classification pipeline.
Integrates with existing data-pipeline components to demonstrate
workflow orchestration.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add the data-pipeline directory to Python path
sys.path.append(str(Path(__file__).parent))

from dag_scheduler import DAG, Task, DAGScheduler
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# TASK FUNCTIONS (Wrapper functions for existing pipeline components)
# ==============================================================================

def check_data_quality(**context):
    """Data quality monitoring task"""
    logger.info("🔍 Checking data quality...")
    
    try:
        # Check if raw data exists
        data_path = Path("data/medical_transcripts.csv")
        if not data_path.exists():
            return {
                'status': 'no_data',
                'message': 'No raw data file found',
                'data_count': 0
            }
        
        # Load and analyze data
        df = pd.read_csv(data_path)
        
        quality_metrics = {
            'total_records': len(df),
            'missing_transcriptions': df['transcription'].isna().sum() if 'transcription' in df.columns else 0,
            'missing_specialties': df['medical_specialty'].isna().sum() if 'medical_specialty' in df.columns else 0,
            'avg_text_length': df['transcription'].str.len().mean() if 'transcription' in df.columns else 0,
            'unique_specialties': df['medical_specialty'].nunique() if 'medical_specialty' in df.columns else 0,
            'check_time': datetime.now().isoformat()
        }
        
        # Data quality rules
        issues = []
        if quality_metrics['missing_transcriptions'] > len(df) * 0.05:
            issues.append(f"High missing transcriptions: {quality_metrics['missing_transcriptions']}")
        
        if quality_metrics['avg_text_length'] < 50:
            issues.append(f"Low average text length: {quality_metrics['avg_text_length']:.1f}")
        
        status = 'pass' if not issues else 'warning'
        
        logger.info(f"✅ Data quality check complete: {status}")
        logger.info(f"   Records: {quality_metrics['total_records']:,}")
        logger.info(f"   Specialties: {quality_metrics['unique_specialties']}")
        
        return {
            'status': status,
            'metrics': quality_metrics,
            'issues': issues
        }
        
    except Exception as e:
        logger.error(f"❌ Data quality check failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

def clean_data(**context):
    """Data cleaning task - wrapper for clean.py"""
    logger.info("🧹 Starting data cleaning...")
    
    try:
        # Import and run the cleaning function
        from clean import create_clean_dataset
        
        clean_df = create_clean_dataset(
            outlier_method='cap',
            max_text_length=8000,
            min_text_length=50,
            n_classes=15
        )
        
        # Save cleaned data
        output_path = Path("data/cleaned_data.csv")
        clean_df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Data cleaning complete: {len(clean_df):,} records")
        
        return {
            'status': 'success',
            'output_path': str(output_path),
            'record_count': len(clean_df),
            'class_count': clean_df['medical_specialty'].nunique() if 'medical_specialty' in clean_df.columns else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Data cleaning failed: {e}")
        raise

def feature_engineering(**context):
    """Feature engineering task - wrapper for preprocess.py"""
    logger.info("🔧 Starting feature engineering...")
    
    try:
        # Import and run preprocessing
        from preprocess import create_features
        
        X, y, feature_engine = create_features(
            max_features=5000,
            conservative_cleaning=True
        )
        
        logger.info(f"✅ Feature engineering complete:")
        logger.info(f"   Feature matrix: {X.shape}")
        logger.info(f"   Classes: {len(np.unique(y))}")
        
        return {
            'status': 'success',
            'feature_count': X.shape[1],
            'sample_count': X.shape[0],
            'class_count': len(np.unique(y))
        }
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        raise

def train_model(**context):
    """Model training task - wrapper for optimize_multinomial_nb.py (not train_optimized_model.py)"""
    logger.info("🤖 Starting Multinomial NB model training...")
    
    try:
        from optimize_multinomial_nb import MultinomialNBOptimizer
        
        optimizer = MultinomialNBOptimizer()
        results = optimizer.run_full_optimization()
        
        logger.info(f"✅ Multinomial NB training complete:")
        logger.info(f"   Test accuracy: {results['test_accuracy']:.3f}")
        logger.info(f"   Best variant: {results['best_variant']}")
        
        return {
            'status': 'success',
            'test_accuracy': results['test_accuracy'],
            'cv_accuracy': results['hyperparameter_score'],  # Use hyperparameter CV score
            'best_variant': results['best_variant'],
            'variant_score': results['variant_score'],
            'model_path': results['model_path']
        }
        
    except Exception as e:
        logger.error(f"❌ Multinomial NB training failed: {e}")
        raise

def validate_model(**context):
    """Model validation task"""
    logger.info("✅ Starting model validation...")
    
    try:
        # ADD MONITORING: Check what files exist in data directory
        import os
        data_files = []
        if os.path.exists("data"):
            data_files = [f for f in os.listdir("data") if f.endswith('.pkl')]
            logger.info(f"🔍 MONITORING: Files in data directory: {data_files}")
        else:
            logger.error("🔍 MONITORING: data directory doesn't exist!")
        
        # Get training results to find the actual model path
        train_output = context.get('train_model_output', {})
        actual_model_path = train_output.get('model_path')
        
        logger.info(f"🔍 MONITORING: Training output: {train_output}")
        logger.info(f"🔍 MONITORING: Expected model path from training: {actual_model_path}")
        
        # Try multiple possible model paths
        possible_paths = [
            Path("data/optimized_multinomial_nb.pkl"),  # What training actually saves
            Path("data/optimized_model.pkl"),          # What we were looking for before
            Path(actual_model_path) if actual_model_path else None  # Path from training output
        ]
        
        model_path = None
        for path in possible_paths:
            if path and path.exists():
                model_path = path
                logger.info(f"✅ MONITORING: Found model at: {model_path}")
                break
        
        if not model_path:
            logger.error(f"❌ MONITORING: Model not found in any of these locations:")
            for path in possible_paths:
                if path:
                    logger.error(f"   - {path.absolute()} (exists: {path.exists()})")
            raise FileNotFoundError("Trained model not found in any expected location")
        
        # Load and basic validation
        logger.info(f"🔍 MONITORING: Loading model from {model_path}")
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"🔍 MONITORING: Model type: {type(model)}")
        
        # Check model has required methods
        required_methods = ['predict', 'predict_proba']
        for method in required_methods:
            if not hasattr(model, method):
                raise ValueError(f"Model missing required method: {method}")
            logger.info(f"✅ MONITORING: Model has {method} method")
        
        # Get training results for validation
        test_accuracy = train_output.get('test_accuracy', 0)
        logger.info(f"🔍 MONITORING: Test accuracy from training: {test_accuracy}")
        
        # Validation rules
        validation_status = 'pass'
        issues = []
        
        if test_accuracy < 0.4:  # 40% minimum accuracy
            issues.append(f"Low test accuracy: {test_accuracy:.3f}")
            validation_status = 'fail'
        
        if test_accuracy < 0.6:  # 60% good accuracy
            issues.append(f"Below target accuracy: {test_accuracy:.3f}")
            validation_status = 'warning'
        
        logger.info(f"✅ Model validation complete: {validation_status}")
        logger.info(f"🔍 MONITORING: Validation issues: {issues}")
        
        return {
            'status': validation_status,
            'test_accuracy': test_accuracy,
            'issues': issues,
            'model_path': str(model_path),
            'available_files': data_files  # Include for monitoring
        }
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")
        logger.error(f"🔍 MONITORING: Exception details:", exc_info=True)
        raise

def deploy_model(**context):
    """Model deployment task (mock deployment)"""
    logger.info("🚀 Starting model deployment...")
    
    try:
        # ADD MONITORING: Get validation results to find actual model path
        validate_output = context.get('validate_model_output', {})
        actual_model_path = validate_output.get('model_path')
        
        logger.info(f"🔍 MONITORING: Validation output: {validate_output}")
        logger.info(f"🔍 MONITORING: Actual model path: {actual_model_path}")
        
        # Mock deployment - copy model to "production" location
        import shutil
        
        # Use the actual model path from validation
        if actual_model_path and Path(actual_model_path).exists():
            model_path = Path(actual_model_path)
        else:
            # Fallback to looking for the model
            model_path = Path("data/optimized_multinomial_nb.pkl")
            if not model_path.exists():
                model_path = Path("data/optimized_model.pkl")
        
        deploy_path = Path("data/production_model.pkl")
        
        logger.info(f"🔍 MONITORING: Copying {model_path} → {deploy_path}")
        
        if model_path.exists():
            shutil.copy2(model_path, deploy_path)
            logger.info(f"✅ MONITORING: Model copied successfully")
        else:
            raise FileNotFoundError(f"Source model not found: {model_path}")
            
        # Mock deployment metadata
        deployment_info = {
            'deployment_time': datetime.now().isoformat(),
            'source_model_path': str(model_path),
            'deployed_model_path': str(deploy_path),
            'status': 'deployed',
            'version': f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
        }
        
        # Save deployment info
        import json
        with open("data/deployment_info.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"✅ Model deployment complete: {deployment_info['version']}")
        logger.info(f"🔍 MONITORING: Deployment info saved to data/deployment_info.json")
        
        return deployment_info
        
    except Exception as e:
        logger.error(f"❌ Model deployment failed: {e}")
        logger.error(f"🔍 MONITORING: Exception details:", exc_info=True)
        raise

def send_notification(**context):
    """Send notification about pipeline completion"""
    logger.info("📧 Sending pipeline completion notification...")
    
    try:
        # Get results from upstream tasks
        train_output = context.get('train_model_output', {})
        validate_output = context.get('validate_model_output', {})
        deploy_output = context.get('deploy_model_output', {})
        
        # Create notification message
        message = {
            'pipeline': context.get('dag_id', 'unknown'),
            'run_date': context.get('run_date', datetime.now()).isoformat(),
            'status': 'success',
            'results': {
                'test_accuracy': train_output.get('test_accuracy', 'unknown'),
                'validation_status': validate_output.get('status', 'unknown'),
                'deployment_version': deploy_output.get('version', 'unknown')
            }
        }
        
        # In a real system, this would send email/Slack/etc.
        # For demo, just log and save to file
        logger.info("📧 Pipeline notification:")
        logger.info(f"   Accuracy: {message['results']['test_accuracy']}")
        logger.info(f"   Validation: {message['results']['validation_status']}")
        logger.info(f"   Version: {message['results']['deployment_version']}")
        
        # Save notification
        with open("data/last_notification.json", 'w') as f:
            import json
            json.dump(message, f, indent=2)
        
        return message
        
    except Exception as e:
        logger.error(f"❌ Notification failed: {e}")
        # Don't fail the entire pipeline for notification issues
        return {'status': 'error', 'error': str(e)}

# ==============================================================================
# DAG DEFINITIONS
# ==============================================================================

def create_model_retraining_dag() -> DAG:
    """Create the main model retraining DAG"""
    
    dag = DAG(
        dag_id="model_retraining_pipeline",
        description="Complete ML pipeline for retraining the medical text classifier",
        schedule="weekly"  # Run weekly
    )
    
    # Define tasks
    tasks = [
        Task(
            task_id="check_data_quality",
            func=check_data_quality,
            dependencies=[],
            max_retries=1
        ),
        Task(
            task_id="clean_data",
            func=clean_data,
            dependencies=["check_data_quality"],
            max_retries=2
        ),
        Task(
            task_id="feature_engineering",
            func=feature_engineering,
            dependencies=["clean_data"],
            max_retries=2
        ),
        Task(
            task_id="train_model",
            func=train_model,
            dependencies=["feature_engineering"],
            max_retries=1,
            timeout=7200  # 2 hours for training
        ),
        Task(
            task_id="validate_model",
            func=validate_model,
            dependencies=["train_model"],
            max_retries=1
        ),
        Task(
            task_id="deploy_model",
            func=deploy_model,
            dependencies=["validate_model"],
            max_retries=2
        ),
        Task(
            task_id="send_notification",
            func=send_notification,
            dependencies=["deploy_model"],
            max_retries=3
        )
    ]
    
    # Add tasks to DAG
    for task in tasks:
        dag.add_task(task)
    
    return dag

def create_data_monitoring_dag() -> DAG:
    """Create a simple data monitoring DAG"""
    
    dag = DAG(
        dag_id="data_quality_monitoring",
        description="Daily data quality monitoring",
        schedule="daily"
    )
    
    # Simple monitoring task
    dag.add_task(Task(
        task_id="daily_data_check",
        func=check_data_quality,
        dependencies=[],
        max_retries=2
    ))
    
    return dag

def create_model_performance_dag() -> DAG:
    """Create a model performance monitoring DAG"""
    
    def check_model_performance(**context):
        """Check deployed model performance"""
        logger.info("📊 Checking model performance...")
        
        try:
            # Mock performance check
            deploy_info_path = Path("data/deployment_info.json")
            if deploy_info_path.exists():
                import json
                with open(deploy_info_path) as f:
                    deploy_info = json.load(f)
                
                # Mock performance metrics
                performance = {
                    'accuracy': 0.75 + (hash(str(datetime.now().date())) % 100) / 1000,  # Mock daily variation
                    'precision': 0.73 + (hash(str(datetime.now().date())) % 80) / 1000,
                    'recall': 0.77 + (hash(str(datetime.now().date())) % 90) / 1000,
                    'check_date': datetime.now().isoformat(),
                    'model_version': deploy_info.get('version', 'unknown')
                }
                
                logger.info(f"📊 Model performance: {performance['accuracy']:.3f} accuracy")
                return performance
            else:
                return {'status': 'no_model', 'message': 'No deployed model found'}
                
        except Exception as e:
            logger.error(f"❌ Performance check failed: {e}")
            raise
    
    dag = DAG(
        dag_id="model_performance_monitoring",
        description="Monitor deployed model performance",
        schedule="every 6 hours"
    )
    
    dag.add_task(Task(
        task_id="check_performance",
        func=check_model_performance,
        dependencies=[],
        max_retries=2
    ))
    
    return dag

# ==============================================================================
# MAIN SCHEDULER SETUP
# ==============================================================================

def setup_ml_scheduler() -> DAGScheduler:
    """Set up the ML pipeline scheduler with all DAGs"""
    
    scheduler = DAGScheduler()
    
    # Register all DAGs
    dags = [
        create_model_retraining_dag(),
        create_data_monitoring_dag(),
        create_model_performance_dag()
    ]
    
    for dag in dags:
        scheduler.register_dag(dag)
        logger.info(f"📋 Registered DAG: {dag.dag_id}")
    
    return scheduler

if __name__ == "__main__":
    # Demo: Run the main retraining pipeline once
    print("🎯 ML Pipeline DAG Demo")
    print("=" * 50)
    
    # Create and run the retraining DAG
    dag = create_model_retraining_dag()
    
    print(f"📋 DAG: {dag.dag_id}")
    print(f"📝 Description: {dag.description}")
    print(f"🔗 Tasks: {list(dag.tasks.keys())}")
    
    # Run the DAG
    try:
        result = dag.run()
        print(f"\n🎉 DAG completed: {result['status']}")
        print(f"⏱️  Duration: {result['duration_seconds']:.1f}s")
        print(f"✅ Success: {result['success_count']}")
        print(f"❌ Failed: {result['failed_count']}")
    except Exception as e:
        print(f"\n💥 DAG failed: {e}") 