#!/usr/bin/env python3
"""
Medical Text Classification - Feature Selection Testing
=======================================================
Test different feature selection strategies to find optimal approach:
1. Remove zero-variance features
2. Test different numbers of selected features (50, 100, 200, 500)
3. Compare different selection methods (F-score, MI, Random Forest)
4. Evaluate model performance on each approach
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, 
    mutual_info_classif, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# Import feature engine class
try:
    from preprocess import MedicalTextFeatureEngine
except ImportError:
    print("Warning: Could not import MedicalTextFeatureEngine")

class FeatureSelectionTester:
    """Test different feature selection approaches systematically"""
    
    def __init__(self):
        self.results = {}
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_data(self, features_path='data/X_features.npy', 
                  target_path='data/y_target.npy',
                  feature_engine_path='data/feature_engine.pkl'):
        """Load the preprocessed data"""
        print("📊 LOADING DATA FOR FEATURE SELECTION TESTING")
        print("=" * 50)
        
        # Load features and target
        self.X = np.load(features_path, allow_pickle=True)
        self.y = np.load(target_path, allow_pickle=True)
        
        # Load feature names
        with open(feature_engine_path, 'rb') as f:
            feature_engine = pickle.load(f)
        self.feature_names = feature_engine.feature_names
        
        # Convert to numeric if needed
        if self.X.dtype == 'object':
            print("Converting object array to numeric...")
            X_numeric = pd.DataFrame(self.X).apply(pd.to_numeric, errors='coerce').fillna(0).values
            self.X = X_numeric.astype(float)
        
        print(f"✅ Loaded data:")
        print(f"  Samples: {self.X.shape[0]:,}")
        print(f"  Features: {self.X.shape[1]:,}")
        print(f"  Classes: {len(np.unique(self.y))}")
        
        return self.X, self.y
    
    def remove_zero_variance_features(self):
        """Step 1: Remove zero-variance features"""
        print(f"\n🧹 STEP 1: Removing Zero-Variance Features")
        print("=" * 45)
        
        # Remove features with exactly zero variance
        zero_var_selector = VarianceThreshold(threshold=0.0)
        X_filtered = zero_var_selector.fit_transform(self.X)
        
        # Get indices of kept features
        kept_indices = zero_var_selector.get_support()
        kept_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if kept_indices[i]]
        
        removed_count = self.X.shape[1] - X_filtered.shape[1]
        
        print(f"  Original features: {self.X.shape[1]:,}")
        print(f"  Zero-variance features removed: {removed_count:,}")
        print(f"  Remaining features: {X_filtered.shape[1]:,}")
        
        return X_filtered, kept_feature_names, zero_var_selector
    
    def test_feature_selection_methods(self, X_base, feature_names, test_sizes=[50, 100, 200, 500]):
        """Test different feature selection methods and sizes"""
        print(f"\n🎯 STEP 2: Testing Feature Selection Methods")
        print("=" * 45)
        
        methods = {
            'f_score': f_classif,
            'mutual_info': mutual_info_classif
        }
        
        results = {}
        
        for method_name, score_func in methods.items():
            print(f"\n📊 Testing {method_name.upper()} method:")
            results[method_name] = {}
            
            for k in test_sizes:
                if k > X_base.shape[1]:
                    print(f"  Skipping k={k} (more than available features: {X_base.shape[1]})")
                    continue
                    
                print(f"  Testing with {k} features...")
                
                # Select features
                selector = SelectKBest(score_func=score_func, k=k)
                X_selected = selector.fit_transform(X_base, self.y)
                
                # Get selected feature names
                selected_indices = selector.get_support()
                selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_indices[i]]
                
                # Test model performance
                performance = self.evaluate_model_performance(X_selected, self.y)
                
                results[method_name][k] = {
                    'performance': performance,
                    'selected_features': selected_features[:10],  # Store top 10 for inspection
                    'selector': selector
                }
                
                print(f"    Accuracy: {performance['accuracy']:.3f} (±{performance['accuracy_std']:.3f})")
        
        return results
    
    def test_random_forest_selection(self, X_base, feature_names, test_sizes=[50, 100, 200, 500]):
        """Test Random Forest-based feature selection"""
        print(f"\n🌳 STEP 3: Testing Random Forest Feature Selection")
        print("=" * 50)
        
        print("  Training Random Forest for feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_base, self.y)
        
        # Get feature importances
        feature_importances = rf.feature_importances_
        
        results = {}
        
        for k in test_sizes:
            if k > X_base.shape[1]:
                continue
                
            print(f"  Testing with top {k} RF features...")
            
            # Select top k features by importance
            top_indices = np.argsort(feature_importances)[-k:][::-1]
            X_selected = X_base[:, top_indices]
            
            # Get selected feature names
            selected_features = [feature_names[i] for i in top_indices]
            
            # Test performance
            performance = self.evaluate_model_performance(X_selected, self.y)
            
            results[k] = {
                'performance': performance,
                'selected_features': selected_features[:10],
                'feature_importances': feature_importances[top_indices][:10]
            }
            
            print(f"    Accuracy: {performance['accuracy']:.3f} (±{performance['accuracy_std']:.3f})")
        
        return results
    
    def evaluate_model_performance(self, X, y, cv_folds=3):
        """Evaluate model performance with cross-validation"""
        
        # Test both Logistic Regression and Random Forest
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42, 
                                         C=1.0, penalty='l2'),  # L2 regularization
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for model_name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                results[f'{model_name}_accuracy'] = scores.mean()
                results[f'{model_name}_std'] = scores.std()
            except Exception as e:
                print(f"    Error with {model_name}: {str(e)[:50]}")
                results[f'{model_name}_accuracy'] = 0.0
                results[f'{model_name}_std'] = 0.0
        
        # Return the better performing model's results
        if results['logistic_accuracy'] >= results['random_forest_accuracy']:
            results['accuracy'] = results['logistic_accuracy']
            results['accuracy_std'] = results['logistic_std']
            results['best_model'] = 'logistic'
        else:
            results['accuracy'] = results['random_forest_accuracy']
            results['accuracy_std'] = results['random_forest_std']
            results['best_model'] = 'random_forest'
        
        return results
    
    def summarize_results(self, selection_results, rf_results):
        """Summarize and compare all results"""
        print(f"\n📊 SUMMARY OF RESULTS")
        print("=" * 40)
        
        print(f"{'Method':<15} {'Features':<10} {'Accuracy':<10} {'Best Model':<15}")
        print("-" * 55)
        
        all_results = []
        
        # Process statistical methods
        for method_name, method_results in selection_results.items():
            for k, result in method_results.items():
                perf = result['performance']
                row = {
                    'method': f"{method_name}",
                    'features': k,
                    'accuracy': perf['accuracy'],
                    'std': perf['accuracy_std'],
                    'best_model': perf['best_model']
                }
                all_results.append(row)
                print(f"{method_name:<15} {k:<10} {perf['accuracy']:.3f}±{perf['accuracy_std']:.3f} {perf['best_model']:<15}")
        
        # Process Random Forest results
        for k, result in rf_results.items():
            perf = result['performance']
            row = {
                'method': 'random_forest',
                'features': k,
                'accuracy': perf['accuracy'],
                'std': perf['accuracy_std'],
                'best_model': perf['best_model']
            }
            all_results.append(row)
            print(f"{'random_forest':<15} {k:<10} {perf['accuracy']:.3f}±{perf['accuracy_std']:.3f} {perf['best_model']:<15}")
        
        # Find best result
        best_result = max(all_results, key=lambda x: x['accuracy'])
        
        print(f"\n🏆 BEST RESULT:")
        print(f"  Method: {best_result['method']}")
        print(f"  Features: {best_result['features']}")
        print(f"  Accuracy: {best_result['accuracy']:.3f} (±{best_result['std']:.3f})")
        print(f"  Best Model: {best_result['best_model']}")
        
        return best_result, all_results
    
    def run_complete_test(self, test_sizes=[50, 100, 200, 500]):
        """Run the complete feature selection testing pipeline"""
        print("🧪 FEATURE SELECTION TESTING PIPELINE")
        print("=" * 50)
        
        start_time = time.time()
        
        # Load data
        self.load_data()
        
        # Step 1: Remove zero-variance features
        X_filtered, kept_feature_names, zero_var_selector = self.remove_zero_variance_features()
        
        # Adjust test sizes based on available features
        max_features = X_filtered.shape[1]
        test_sizes = [k for k in test_sizes if k <= max_features]
        
        if not test_sizes:
            print(f"⚠️  Warning: No test sizes possible with {max_features} features after filtering")
            test_sizes = [min(50, max_features)]
        
        print(f"Testing with sizes: {test_sizes}")
        
        # Step 2: Test statistical methods
        selection_results = self.test_feature_selection_methods(X_filtered, kept_feature_names, test_sizes)
        
        # Step 3: Test Random Forest selection
        rf_results = self.test_random_forest_selection(X_filtered, kept_feature_names, test_sizes)
        
        # Step 4: Summarize results
        best_result, all_results = self.summarize_results(selection_results, rf_results)
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed_time:.1f} seconds")
        
        # Store results
        self.results = {
            'selection_results': selection_results,
            'rf_results': rf_results,
            'best_result': best_result,
            'all_results': all_results,
            'zero_var_selector': zero_var_selector
        }
        
        return self.results

def main():
    """Run feature selection testing"""
    tester = FeatureSelectionTester()
    results = tester.run_complete_test()
    
    print(f"\n✅ Testing complete! Results available in tester.results")
    return tester, results

if __name__ == "__main__":
    tester, results = main() 