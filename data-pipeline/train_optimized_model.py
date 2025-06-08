#!/usr/bin/env python3
"""
Medical Text Classification - Optimized Model Training
======================================================
Based on feature selection testing results, optimize the winning approach:
- F-score feature selection (150-250 features)
- Logistic Regression with proper settings
- Fix convergence issues and improve performance
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# Import feature engine class
try:
    from preprocess import MedicalTextFeatureEngine
except ImportError:
    print("Warning: Could not import MedicalTextFeatureEngine")

class OptimizedMedicalClassifier:
    """Optimized classifier based on testing results"""
    
    def __init__(self):
        self.best_pipeline = None
        self.feature_selector = None
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.selected_features = None
        
    def load_data(self, features_path='data/X_features.npy', 
                  target_path='data/y_target.npy',
                  feature_engine_path='data/feature_engine.pkl'):
        """Load and prepare the data"""
        print("📊 LOADING DATA FOR OPTIMIZED TRAINING")
        print("=" * 45)
        
        # Load features and target
        self.X = np.load(features_path, allow_pickle=True)
        self.y = np.load(target_path, allow_pickle=True)
        
        # Load feature names
        with open(feature_engine_path, 'rb') as f:
            feature_engine = pickle.load(f)
        self.feature_names = feature_engine.feature_names
        self.label_encoder = feature_engine.label_encoder
        
        # Convert to numeric if needed
        if self.X.dtype == 'object':
            print("Converting to numeric...")
            X_numeric = pd.DataFrame(self.X).apply(pd.to_numeric, errors='coerce').fillna(0).values
            self.X = X_numeric.astype(float)
        
        print(f"✅ Data loaded:")
        print(f"  Samples: {self.X.shape[0]:,}")
        print(f"  Features: {self.X.shape[1]:,}")
        print(f"  Classes: {len(np.unique(self.y))}")
        
        return self.X, self.y
    
    def fine_tune_feature_count(self, feature_range=[150, 175, 200, 225, 250]):
        """Fine-tune the optimal number of features"""
        print(f"\n🎯 FINE-TUNING FEATURE COUNT")
        print("=" * 35)
        
        results = {}
        
        for n_features in feature_range:
            print(f"Testing {n_features} features...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k=n_features)),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    max_iter=2000,  # Increased iterations
                    random_state=42,
                    C=1.0,
                    penalty='l2'
                ))
            ])
            
            # Cross-validation
            scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='accuracy')
            
            results[n_features] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores
            }
            
            print(f"  Accuracy: {scores.mean():.3f} (±{scores.std():.3f})")
        
        # Find best
        best_n_features = max(results.keys(), key=lambda k: results[k]['mean_accuracy'])
        best_accuracy = results[best_n_features]['mean_accuracy']
        
        print(f"\n🏆 Best feature count: {best_n_features} ({best_accuracy:.3f} accuracy)")
        
        return best_n_features, results
    
    def optimize_logistic_regression(self, n_features=200):
        """Optimize logistic regression hyperparameters"""
        print(f"\n🔧 OPTIMIZING LOGISTIC REGRESSION")
        print("=" * 40)
        
        # Create base pipeline
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=n_features)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=2000, random_state=42))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'classifier__C': [0.1, 0.5, 1.0, 2.0, 5.0],  # Regularization strength
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],  # Regularization type
            'classifier__solver': ['liblinear', 'saga'],  # Solvers that support L1
        }
        
        # Handle elasticnet specific parameters
        param_grid_elasticnet = {
            'classifier__C': [0.1, 0.5, 1.0, 2.0],
            'classifier__penalty': ['elasticnet'],
            'classifier__solver': ['saga'],
            'classifier__l1_ratio': [0.1, 0.5, 0.7, 0.9]  # Mix of L1 and L2
        }
        
        print("Testing regularization parameters...")
        
        # Grid search for L1/L2
        grid_search = GridSearchCV(
            pipeline, param_grid, 
            cv=3, scoring='accuracy', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X, self.y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.3f}")
        
        # Test elasticnet separately if saga solver works
        try:
            pipeline_elastic = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k=n_features)),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=2000, random_state=42))
            ])
            
            grid_elastic = GridSearchCV(
                pipeline_elastic, param_grid_elasticnet,
                cv=3, scoring='accuracy', n_jobs=-1, verbose=0
            )
            
            grid_elastic.fit(self.X, self.y)
            
            if grid_elastic.best_score_ > grid_search.best_score_:
                print(f"ElasticNet better: {grid_elastic.best_params_}")
                print(f"ElasticNet accuracy: {grid_elastic.best_score_:.3f}")
                return grid_elastic.best_estimator_, grid_elastic.best_params_, grid_elastic.best_score_
                
        except Exception as e:
            print(f"ElasticNet failed: {str(e)[:50]}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def train_final_model(self, best_pipeline):
        """Train the final optimized model"""
        print(f"\n🚀 TRAINING FINAL MODEL")
        print("=" * 30)
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Train the model
        print("Training on 80% of data...")
        best_pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = best_pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Test accuracy: {test_accuracy:.3f}")
        
        # Get selected features
        feature_selector = best_pipeline.named_steps['feature_selection']
        selected_indices = feature_selector.get_support()
        selected_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_indices[i]]
        
        print(f"\nTop 10 selected features:")
        feature_scores = feature_selector.scores_
        top_features = [(self.feature_names[i], feature_scores[i]) for i in range(len(self.feature_names)) if selected_indices[i]]
        top_features.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(top_features[:10]):
            print(f"  {i+1:2}. {feature[:50]:50} | F-score: {score:.2f}")
        
        # Classification report
        print(f"\n📊 DETAILED PERFORMANCE:")
        class_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        
        # Store results
        self.best_pipeline = best_pipeline
        self.selected_features = selected_feature_names
        self.test_accuracy = test_accuracy
        
        return best_pipeline, test_accuracy, selected_feature_names
    
    def save_optimized_model(self, output_dir='data/'):
        """Save the optimized model and metadata"""
        print(f"\n💾 SAVING OPTIMIZED MODEL")
        print("=" * 30)
        
        # Save the pipeline
        model_path = f'{output_dir}optimized_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_pipeline, f)
        
        # Save metadata
        metadata = {
            'test_accuracy': self.test_accuracy,
            'selected_features': self.selected_features,
            'n_features': len(self.selected_features),
            'feature_names': self.feature_names,
        }
        
        metadata_path = f'{output_dir}model_metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Metadata saved to: {metadata_path}")
        
        return model_path, metadata_path
    
    def run_optimization_pipeline(self):
        """Run the complete optimization pipeline"""
        print("🎯 MEDICAL TEXT CLASSIFIER OPTIMIZATION")
        print("=" * 50)
        
        start_time = time.time()
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Fine-tune feature count
        best_n_features, feature_results = self.fine_tune_feature_count()
        
        # Step 3: Optimize logistic regression
        best_pipeline, best_params, best_cv_score = self.optimize_logistic_regression(best_n_features)
        
        # Step 4: Train final model
        final_pipeline, test_accuracy, selected_features = self.train_final_model(best_pipeline)
        
        # Step 5: Save model
        model_path, metadata_path = self.save_optimized_model()
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print(f"\n🎉 OPTIMIZATION COMPLETE!")
        print("=" * 30)
        print(f"  Best feature count: {best_n_features}")
        print(f"  Best parameters: {best_params}")
        print(f"  CV accuracy: {best_cv_score:.3f}")
        print(f"  Test accuracy: {test_accuracy:.3f}")
        print(f"  Total time: {elapsed_time:.1f} seconds")
        print(f"  Model saved: {model_path}")
        
        return {
            'best_n_features': best_n_features,
            'best_params': best_params,
            'cv_accuracy': best_cv_score,
            'test_accuracy': test_accuracy,
            'selected_features': selected_features,
            'model_path': model_path
        }

def main():
    """Run the optimization pipeline"""
    optimizer = OptimizedMedicalClassifier()
    results = optimizer.run_optimization_pipeline()
    
    print(f"\n✅ Optimization complete! Check results above.")
    return optimizer, results

if __name__ == "__main__":
    optimizer, results = main() 