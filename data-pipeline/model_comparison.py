#!/usr/bin/env python3
"""
Medical Text Classification - Comprehensive Model Comparison
============================================================
Test multiple algorithms to see if we can beat 37% accuracy:
- Logistic Regression (baseline)
- Random Forest 
- XGBoost/LightGBM
- SVM (linear & RBF)
- Naive Bayes
- Neural Networks
- Ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Models to test
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Advanced models
# try:
#     import lightgbm as lgb
#     HAVE_LGB = True
# except ImportError:
#     HAVE_LGB = False
#     print("LightGBM not available - install with: pip install lightgbm")

import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# Import feature engine class
try:
    from preprocess import MedicalTextFeatureEngine
except ImportError:
    print("Warning: Could not import MedicalTextFeatureEngine")

class ModelComparison:
    """Compare multiple models systematically"""
    
    def __init__(self):
        self.results = {}
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_data(self, features_path='data/X_features.npy', 
                  target_path='data/y_target.npy',
                  feature_engine_path='data/feature_engine.pkl'):
        """Load and prepare the data"""
        print("📊 LOADING DATA FOR MODEL COMPARISON")
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
        print(f"  Random baseline: {1/len(np.unique(self.y)):.1%}")
        
        return self.X, self.y
    
    def get_model_configs(self):
        """Define all models to test"""
        models = {
            # Linear Models
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=2000, random_state=42),
                'needs_scaling': True,
                'n_features': 200
            },
            
            'Linear SVM': {
                'model': LinearSVC(max_iter=2000, random_state=42),
                'needs_scaling': True,
                'n_features': 200
            },
            
            # Tree-based Models (don't need scaling)
            'Random Forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'needs_scaling': False,
                'n_features': 500  # Can handle more features
            },
            
            'Extra Trees': {
                'model': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'needs_scaling': False,
                'n_features': 500
            },
            
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'needs_scaling': False,
                'n_features': 200  # Slower, use fewer features
            },
            
            # Naive Bayes (good for text)
            'Multinomial NB': {
                'model': MultinomialNB(),
                'needs_scaling': False,
                'n_features': 500,
                'needs_positive': True  # Requires positive features
            },
            
            'Complement NB': {
                'model': ComplementNB(),
                'needs_scaling': False,
                'n_features': 500,
                'needs_positive': True
            },
            
            # Non-linear SVM
            'RBF SVM': {
                'model': SVC(kernel='rbf', random_state=42),
                'needs_scaling': True,
                'n_features': 100  # Slow with many features
            },
            
            # Neural Network
            'Neural Network': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
                'needs_scaling': True,
                'n_features': 200
            },
            
            # k-NN
            'k-Nearest Neighbors': {
                'model': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
                'needs_scaling': True,
                'n_features': 100  # Curse of dimensionality
            }
        }
        
        # Add LightGBM if available
        # if HAVE_LGB:
        #     models['LightGBM'] = {
        #         'model': lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
        #         'needs_scaling': False,
        #         'n_features': 300
        #     }
        
        return models
    
    def test_model(self, name, config):
        """Test a single model configuration"""
        print(f"Testing {name}...")
        
        try:
            # Build pipeline components
            pipeline_steps = []
            
            # Feature selection
            n_features = config['n_features']
            pipeline_steps.append(('feature_selection', SelectKBest(f_classif, k=n_features)))
            
            # Handle positive features for Naive Bayes
            if config.get('needs_positive', False):
                from sklearn.preprocessing import MinMaxScaler
                pipeline_steps.append(('positive_transform', MinMaxScaler()))
            
            # Scaling if needed
            if config['needs_scaling']:
                pipeline_steps.append(('scaler', StandardScaler()))
            
            # Model
            pipeline_steps.append(('classifier', config['model']))
            
            # Create pipeline
            pipeline = Pipeline(pipeline_steps)
            
            # Cross-validation
            start_time = time.time()
            scores = cross_val_score(pipeline, self.X, self.y, cv=3, scoring='accuracy', n_jobs=1)
            elapsed_time = time.time() - start_time
            
            result = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores,
                'n_features': n_features,
                'time_seconds': elapsed_time,
                'status': 'success'
            }
            
            print(f"  ✅ {scores.mean():.3f} (±{scores.std():.3f}) | {elapsed_time:.1f}s | {n_features} features")
            
        except Exception as e:
            result = {
                'mean_accuracy': 0,
                'std_accuracy': 0,
                'scores': [0],
                'n_features': config['n_features'],
                'time_seconds': 0,
                'status': f'failed: {str(e)[:50]}',
                'error': str(e)
            }
            print(f"  ❌ Failed: {str(e)[:50]}")
        
        return result
    
    def run_comparison(self):
        """Run comprehensive model comparison"""
        print("🏆 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 50)
        print("Testing multiple algorithms to beat 37% accuracy...")
        
        models = self.get_model_configs()
        
        print(f"\n🔍 TESTING {len(models)} MODELS")
        print("-" * 30)
        
        start_time = time.time()
        
        # Test each model
        for name, config in models.items():
            self.results[name] = self.test_model(name, config)
        
        total_time = time.time() - start_time
        
        # Analyze results
        self.analyze_results(total_time)
        
        return self.results
    
    def analyze_results(self, total_time):
        """Analyze and rank results"""
        print(f"\n📊 RESULTS ANALYSIS")
        print("=" * 25)
        
        # Filter successful results
        successful_results = {name: result for name, result in self.results.items() 
                            if result['status'] == 'success'}
        
        if not successful_results:
            print("❌ No models succeeded!")
            return
        
        # Sort by accuracy
        sorted_results = sorted(successful_results.items(), 
                              key=lambda x: x[1]['mean_accuracy'], 
                              reverse=True)
        
        print(f"🏆 MODEL RANKINGS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<20} {'Accuracy':<12} {'±Std':<8} {'Features':<8} {'Time':<8}")
        print("-" * 80)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            acc = result['mean_accuracy']
            std = result['std_accuracy']
            features = result['n_features']
            time_taken = result['time_seconds']
            
            # Highlight top performers
            marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            print(f"{marker} {i:<2} {name:<20} {acc:.3f} ({acc:.1%}){std:>6.3f} {features:>8} {time_taken:>6.1f}s")
        
        # Best model analysis
        best_name, best_result = sorted_results[0]
        best_acc = best_result['mean_accuracy']
        
        print(f"\n🎯 BEST MODEL: {best_name}")
        print(f"   Accuracy: {best_acc:.1%} (vs 37% baseline)")
        print(f"   Improvement: {((best_acc - 0.37) / 0.37 * 100):+.1f}% over baseline")
        print(f"   vs Random: {best_acc / (1/16):.1f}x better than random")
        
        # Failed models
        failed_models = [name for name, result in self.results.items() 
                        if result['status'] != 'success']
        
        if failed_models:
            print(f"\n❌ FAILED MODELS: {', '.join(failed_models)}")
        
        print(f"\n⏱️  Total time: {total_time:.1f} seconds")
        
        # Recommendations
        self.make_recommendations(sorted_results)
    
    def make_recommendations(self, sorted_results):
        """Make recommendations based on results"""
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 20)
        
        best_name, best_result = sorted_results[0]
        best_acc = best_result['mean_accuracy']
        
        if best_acc > 0.45:  # >45%
            print("🎉 EXCELLENT! Found models with >45% accuracy")
            print("   → This is strong for 16-class medical text classification")
            print("   → Ready for hyperparameter tuning and production")
        
        elif best_acc > 0.40:  # 40-45%
            print("👍 GOOD! Found models with >40% accuracy")
            print("   → Significant improvement over baseline")
            print("   → Consider ensemble methods for further gains")
        
        elif best_acc > 0.37:  # 37-40%
            print("📈 MODEST improvement over 37% baseline")
            print("   → Try different feature engineering approaches")
            print("   → Consider deep learning or ensemble methods")
        
        else:
            print("😐 No significant improvement found")
            print("   → Problem may be inherently difficult")
            print("   → Consider collecting more/better data")
            print("   → Try advanced feature engineering")
        
        # Specific recommendations
        print(f"\n🔧 NEXT STEPS:")
        
        # Top 3 models
        top_3 = sorted_results[:3]
        print(f"1. Focus on top performers: {', '.join([name for name, _ in top_3])}")
        
        # Fast models
        fast_models = [(name, result) for name, result in sorted_results 
                      if result['time_seconds'] < 10 and result['mean_accuracy'] > 0.35]
        if fast_models:
            print(f"2. Fast models for iteration: {', '.join([name for name, _ in fast_models[:2]])}")
        
        # Ensemble potential
        diverse_models = [name for name, result in sorted_results[:5] 
                         if 'Forest' in name or 'XG' in name or 'LightGBM' in name or 'Logistic' in name]
        if len(diverse_models) >= 2:
            print(f"3. Try ensemble of: {', '.join(diverse_models[:3])}")
        
        print(f"4. Hyperparameter tune the best model: {best_name}")

def main():
    """Run comprehensive model comparison"""
    comparator = ModelComparison()
    comparator.load_data()
    results = comparator.run_comparison()
    
    print(f"\n✅ Model comparison complete!")
    return comparator, results

if __name__ == "__main__":
    comparator, results = main() 