#!/usr/bin/env python3
"""
Medical Text Classification - Feature Analysis & Model Selection
==============================================================
Analyze features before training to understand:
1. Feature importance and relevance
2. Feature correlations and redundancy
3. Dimensionality considerations
4. Model selection recommendations
5. Data characteristics for ML algorithm choice
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the feature engine class so pickle can deserialize it
try:
    from preprocess import MedicalTextFeatureEngine
except ImportError:
    print("Warning: Could not import MedicalTextFeatureEngine from preprocess.py")
    print("This might cause issues when loading the pickled feature engine.")
    pass

class MedicalTextFeatureAnalyzer:
    """Comprehensive feature analysis for medical text classification"""
    
    def __init__(self):
        self.feature_importance_results = {}
        self.correlation_analysis = {}
        self.dimensionality_analysis = {}
        self.model_recommendations = {}
        
    def load_features(self, features_path='data/X_features.npy', 
                     target_path='data/y_target.npy',
                     feature_engine_path='data/feature_engine.pkl'):
        """Load preprocessed features and metadata"""
        print("📊 LOADING FEATURES FOR ANALYSIS")
        print("=" * 40)
        
        # Load feature matrix and target
        self.X = np.load(features_path, allow_pickle=True)
        self.y = np.load(target_path, allow_pickle=True)
        
        # Load feature engine for feature names
        with open(feature_engine_path, 'rb') as f:
            self.feature_engine = pickle.load(f)
        
        self.feature_names = self.feature_engine.feature_names
        
        print(f"✅ Loaded features:")
        print(f"  Feature matrix: {self.X.shape}")
        print(f"  Target vector: {self.y.shape}")
        print(f"  Feature names: {len(self.feature_names)}")
        print(f"  Classes: {len(np.unique(self.y))}")
        
        # Basic feature matrix statistics
        print(f"\n📈 Feature Matrix Statistics:")
        print(f"  Data type: {self.X.dtype}")
        
        # Convert to numeric if needed
        if self.X.dtype == 'object':
            print("  Converting object array to numeric...")
            # Try to convert to float, filling non-numeric with 0
            X_numeric = pd.DataFrame(self.X).apply(pd.to_numeric, errors='coerce').fillna(0).values
            self.X = X_numeric.astype(float)
        
        print(f"  Sparsity: {np.count_nonzero(self.X == 0) / self.X.size * 100:.1f}%")
        print(f"  Max value: {self.X.max():.4f}")
        print(f"  Min value: {self.X.min():.4f}")
        print(f"  Mean: {self.X.mean():.4f}")
        
        return self.X, self.y, self.feature_names
    
    def analyze_feature_variance(self, variance_threshold=0.01):
        """Step 1: Analyze feature variance to identify low-variance features"""
        print("\n📊 STEP 1: Feature Variance Analysis")
        print("=" * 40)
        
        # Calculate feature variances
        feature_variances = np.var(self.X, axis=0)
        
        # Identify low variance features
        low_variance_mask = feature_variances < variance_threshold
        low_variance_count = np.sum(low_variance_mask)
        
        print(f"Feature variance analysis:")
        print(f"  Total features: {len(feature_variances):,}")
        print(f"  Low variance features (<{variance_threshold}): {low_variance_count:,} ({low_variance_count/len(feature_variances)*100:.1f}%)")
        print(f"  High variance features: {len(feature_variances) - low_variance_count:,}")
        
        # Show variance distribution
        print(f"\nVariance distribution:")
        print(f"  Min variance: {feature_variances.min():.6f}")
        print(f"  Max variance: {feature_variances.max():.6f}")
        print(f"  Mean variance: {feature_variances.mean():.6f}")
        print(f"  Median variance: {np.median(feature_variances):.6f}")
        
        # Identify TF-IDF vs other features variance
        tfidf_features = [i for i, name in enumerate(self.feature_names) if name.startswith('tfidf_')]
        other_features = [i for i, name in enumerate(self.feature_names) if not name.startswith('tfidf_')]
        
        if tfidf_features and other_features:
            tfidf_var_mean = feature_variances[tfidf_features].mean()
            other_var_mean = feature_variances[other_features].mean()
            
            print(f"\nFeature type variance comparison:")
            print(f"  TF-IDF features mean variance: {tfidf_var_mean:.6f}")
            print(f"  Other features mean variance: {other_var_mean:.6f}")
        
        self.feature_importance_results['variance_analysis'] = {
            'variances': feature_variances,
            'low_variance_mask': low_variance_mask,
            'low_variance_count': low_variance_count
        }
        
        return feature_variances, low_variance_mask
    
    def analyze_feature_importance_univariate(self, k_best=100):
        """Step 2: Univariate feature importance analysis"""
        print(f"\n🎯 STEP 2: Univariate Feature Importance (Top {k_best})")
        print("=" * 50)
        
        # F-score based selection
        f_selector = SelectKBest(score_func=f_classif, k=k_best)
        f_scores = f_classif(self.X, self.y)[0]
        
        # Mutual information based selection  
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        # Get top features for each method
        top_f_indices = np.argsort(f_scores)[-k_best:][::-1]
        top_mi_indices = np.argsort(mi_scores)[-k_best:][::-1]
        
        print(f"Top 10 features by F-score:")
        for i, idx in enumerate(top_f_indices[:10]):
            feature_name = self.feature_names[idx][:50]
            print(f"  {i+1:2}. {feature_name:50} | F-score: {f_scores[idx]:.4f}")
        
        print(f"\nTop 10 features by Mutual Information:")
        for i, idx in enumerate(top_mi_indices[:10]):
            feature_name = self.feature_names[idx][:50]
            print(f"  {i+1:2}. {feature_name:50} | MI: {mi_scores[idx]:.4f}")
        
        # Analyze overlap between methods
        overlap = set(top_f_indices) & set(top_mi_indices)
        overlap_pct = len(overlap) / k_best * 100
        
        print(f"\nFeature selection method agreement:")
        print(f"  Overlap between F-score and MI: {len(overlap)}/{k_best} ({overlap_pct:.1f}%)")
        
        self.feature_importance_results['univariate'] = {
            'f_scores': f_scores,
            'mi_scores': mi_scores,
            'top_f_indices': top_f_indices,
            'top_mi_indices': top_mi_indices,
            'overlap': overlap
        }
        
        return f_scores, mi_scores, top_f_indices, top_mi_indices
    
    def analyze_feature_importance_model_based(self, max_features=500):
        """Step 3: Model-based feature importance"""
        print(f"\n🌳 STEP 3: Model-Based Feature Importance")
        print("=" * 45)
        
        # Use Random Forest for feature importance (handles multiclass well)
        print("Training Random Forest for feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X, self.y)
        
        feature_importances = rf.feature_importances_
        
        # Get top features
        top_indices = np.argsort(feature_importances)[-max_features:][::-1]
        
        print(f"Top 15 features by Random Forest importance:")
        for i, idx in enumerate(top_indices[:15]):
            feature_name = self.feature_names[idx][:50]
            importance = feature_importances[idx]
            print(f"  {i+1:2}. {feature_name:50} | Importance: {importance:.6f}")
        
        # Analyze feature type importance
        tfidf_indices = [i for i in range(len(self.feature_names)) if self.feature_names[i].startswith('tfidf_')]
        other_indices = [i for i in range(len(self.feature_names)) if not self.feature_names[i].startswith('tfidf_')]
        
        if tfidf_indices and other_indices:
            tfidf_importance_sum = feature_importances[tfidf_indices].sum()
            other_importance_sum = feature_importances[other_indices].sum()
            
            print(f"\nFeature type importance:")
            print(f"  TF-IDF features total importance: {tfidf_importance_sum:.4f}")
            print(f"  Other features total importance: {other_importance_sum:.4f}")
            print(f"  TF-IDF features average importance: {feature_importances[tfidf_indices].mean():.6f}")
            print(f"  Other features average importance: {feature_importances[other_indices].mean():.6f}")
        
        self.feature_importance_results['model_based'] = {
            'feature_importances': feature_importances,
            'top_indices': top_indices,
            'rf_model': rf
        }
        
        return feature_importances, top_indices
    
    def analyze_feature_correlations(self, sample_size=1000, correlation_threshold=0.8):
        """Step 4: Feature correlation analysis"""
        print(f"\n🔗 STEP 4: Feature Correlation Analysis")
        print("=" * 40)
        
        print(f"Analyzing correlations (sampling {sample_size} features for efficiency)...")
        
        # Sample features for correlation analysis (too many for full correlation)
        if self.X.shape[1] > sample_size:
            # Sample top features from previous analysis
            if 'model_based' in self.feature_importance_results:
                top_indices = self.feature_importance_results['model_based']['top_indices'][:sample_size]
            else:
                # Random sampling if no previous analysis
                top_indices = np.random.choice(self.X.shape[1], sample_size, replace=False)
            
            X_sample = self.X[:, top_indices]
            feature_names_sample = [self.feature_names[i] for i in top_indices]
        else:
            X_sample = self.X
            feature_names_sample = self.feature_names
            top_indices = np.arange(len(self.feature_names))
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_sample.T)
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > correlation_threshold:
                    high_corr_pairs.append((
                        top_indices[i], top_indices[j], 
                        feature_names_sample[i][:40], feature_names_sample[j][:40],
                        corr_matrix[i, j]
                    ))
        
        print(f"High correlation analysis (>{correlation_threshold}):")
        print(f"  Features analyzed: {len(feature_names_sample):,}")
        print(f"  Highly correlated pairs: {len(high_corr_pairs)}")
        
        if high_corr_pairs:
            print(f"\nTop 10 most correlated feature pairs:")
            sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[4]), reverse=True)
            for i, (idx1, idx2, name1, name2, corr) in enumerate(sorted_pairs[:10]):
                print(f"  {i+1:2}. {name1:35} <-> {name2:35} | r = {corr:6.3f}")
        
        # Overall correlation statistics
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        print(f"\nCorrelation statistics:")
        print(f"  Mean absolute correlation: {np.abs(upper_triangle).mean():.4f}")
        print(f"  Max correlation: {np.abs(upper_triangle).max():.4f}")
        print(f"  Correlations > 0.5: {np.sum(np.abs(upper_triangle) > 0.5)} ({np.sum(np.abs(upper_triangle) > 0.5)/len(upper_triangle)*100:.1f}%)")
        print(f"  Correlations > 0.8: {np.sum(np.abs(upper_triangle) > 0.8)} ({np.sum(np.abs(upper_triangle) > 0.8)/len(upper_triangle)*100:.1f}%)")
        
        self.correlation_analysis = {
            'correlation_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'sampled_indices': top_indices,
            'correlation_stats': {
                'mean_abs_corr': np.abs(upper_triangle).mean(),
                'max_corr': np.abs(upper_triangle).max(),
                'high_corr_count': len(high_corr_pairs)
            }
        }
        
        return corr_matrix, high_corr_pairs
    
    def analyze_dimensionality_and_pca(self, n_components=100):
        """Step 5: Dimensionality analysis and PCA"""
        print(f"\n📐 STEP 5: Dimensionality Analysis")
        print("=" * 40)
        
        n_samples, n_features = self.X.shape
        
        print(f"Dimensionality overview:")
        print(f"  Samples: {n_samples:,}")
        print(f"  Features: {n_features:,}")
        print(f"  Samples/Features ratio: {n_samples/n_features:.2f}")
        print(f"  Classes: {len(np.unique(self.y))}")
        
        # Dimensionality recommendations
        if n_features > n_samples:
            print(f"  ⚠️  HIGH DIMENSIONAL: More features than samples!")
            print(f"      Risk of overfitting - consider dimensionality reduction")
        elif n_features > n_samples * 0.5:
            print(f"  ⚠️  MEDIUM DIMENSIONAL: Many features relative to samples")
            print(f"      Consider feature selection or regularization")
        else:
            print(f"  ✅ REASONABLE DIMENSIONAL: Good samples/features ratio")
        
        # PCA Analysis
        print(f"\nPCA Analysis (top {n_components} components):")
        
        # Standardize features for PCA (important for TF-IDF + other features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        pca = PCA(n_components=min(n_components, min(n_samples, n_features)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Analyze explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"  PCA Results:")
        print(f"    First component explains: {explained_variance_ratio[0]*100:.1f}% of variance")
        print(f"    Top 10 components explain: {cumulative_variance[9]*100:.1f}% of variance")
        print(f"    Top 50 components explain: {cumulative_variance[49]*100:.1f}% of variance")
        if len(cumulative_variance) > 99:
            print(f"    Top 100 components explain: {cumulative_variance[99]*100:.1f}% of variance")
        
        # Find components needed for different variance thresholds
        for threshold in [0.8, 0.9, 0.95]:
            components_needed = np.argmax(cumulative_variance >= threshold) + 1
            print(f"    Components for {threshold*100:.0f}% variance: {components_needed}")
        
        self.dimensionality_analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'ratio': n_samples/n_features,
            'pca': pca,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'scaler': scaler
        }
        
        return pca, explained_variance_ratio, cumulative_variance
    
    def model_selection_analysis(self, sample_size=2000):
        """Step 6: Model selection recommendations based on data characteristics"""
        print(f"\n🤖 STEP 6: Model Selection Analysis")
        print("=" * 40)
        
        n_samples, n_features = self.X.shape
        n_classes = len(np.unique(self.y))
        
        print(f"Data characteristics for model selection:")
        print(f"  Samples: {n_samples:,}")
        print(f"  Features: {n_features:,}")
        print(f"  Classes: {n_classes}")
        print(f"  Feature density: {np.count_nonzero(self.X) / self.X.size * 100:.1f}%")
        
        # Sample data if too large
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = self.X[indices]
            y_sample = self.y[indices]
            print(f"  Sampling {sample_size} samples for model comparison")
        else:
            X_sample = self.X
            y_sample = self.y
        
        # Quick model comparison
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        }
        
        model_results = {}
        
        print(f"\nQuick model comparison (3-fold CV):")
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_sample, y_sample, cv=3, scoring='accuracy', n_jobs=-1)
                model_results[name] = {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std(),
                    'scores': scores
                }
                print(f"  {name:20}: {scores.mean():.3f} (±{scores.std():.3f})")
            except Exception as e:
                print(f"  {name:20}: Error - {str(e)[:50]}")
                model_results[name] = {'error': str(e)}
        
        # Model recommendations based on data characteristics
        recommendations = []
        
        if n_features > n_samples:
            recommendations.append("🔴 HIGH DIMENSIONAL DATA:")
            recommendations.append("   - Logistic Regression with L1/L2 regularization")
            recommendations.append("   - Linear SVM with regularization")
            recommendations.append("   - Feature selection before training")
            recommendations.append("   - Consider PCA/TruncatedSVD")
        
        if n_features > 1000:
            recommendations.append("🟡 MANY FEATURES:")
            recommendations.append("   - Regularized linear models (good starting point)")
            recommendations.append("   - Gradient boosting (XGBoost, LightGBM)")
            recommendations.append("   - Avoid k-NN (curse of dimensionality)")
        
        if n_classes > 10:
            recommendations.append("🟡 MULTICLASS PROBLEM:")
            recommendations.append("   - Tree-based models handle multiclass naturally")
            recommendations.append("   - Logistic regression with 'ovr' or 'multinomial'")
            recommendations.append("   - Neural networks for complex patterns")
        
        if self.X.max() <= 1 and np.all(self.X >= 0):  # TF-IDF characteristics
            recommendations.append("✅ TF-IDF FEATURES DETECTED:")
            recommendations.append("   - Linear models work well (Logistic Regression, SVM)")
            recommendations.append("   - Naive Bayes is good for text")
            recommendations.append("   - Consider feature scaling for tree-based models")
        
        print(f"\n🎯 MODEL RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")
        
        # Specific recommendations based on your data size
        if n_samples < 5000:
            print(f"\n📊 DATASET SIZE RECOMMENDATIONS:")
            print(f"  Your dataset ({n_samples:,} samples) is MEDIUM-SIZED")
            print(f"  ✅ Good for: Logistic Regression, SVM, Random Forest")
            print(f"  ⚠️  Avoid: Deep learning (insufficient data)")
            print(f"  💡 Consider: Ensemble methods, cross-validation")
        
        self.model_recommendations = {
            'data_characteristics': {
                'n_samples': n_samples,
                'n_features': n_features, 
                'n_classes': n_classes,
                'is_high_dimensional': n_features > n_samples,
                'is_sparse': np.count_nonzero(self.X) / self.X.size < 0.1
            },
            'model_results': model_results,
            'recommendations': recommendations
        }
        
        return model_results, recommendations
    
    def generate_feature_selection_recommendations(self, target_features=500):
        """Step 7: Generate actionable feature selection recommendations"""
        print(f"\n🎯 STEP 7: Feature Selection Recommendations")
        print("=" * 50)
        
        n_features = self.X.shape[1]
        reduction_ratio = target_features / n_features
        
        print(f"Feature reduction strategy:")
        print(f"  Current features: {n_features:,}")
        print(f"  Target features: {target_features:,}")
        print(f"  Reduction ratio: {reduction_ratio:.1%}")
        
        strategies = []
        
        # Strategy 1: Variance-based filtering
        if 'variance_analysis' in self.feature_importance_results:
            low_var_count = self.feature_importance_results['variance_analysis']['low_variance_count']
            if low_var_count > 0:
                strategies.append(f"1. Remove {low_var_count:,} low-variance features")
        
        # Strategy 2: Correlation-based filtering
        if 'correlation_stats' in self.correlation_analysis:
            high_corr_count = self.correlation_analysis['correlation_stats']['high_corr_count']
            if high_corr_count > 0:
                strategies.append(f"2. Remove {high_corr_count//2:,} highly correlated features")
        
        # Strategy 3: Univariate selection
        if 'univariate' in self.feature_importance_results:
            strategies.append(f"3. Keep top {target_features} features by F-score/MI")
        
        # Strategy 4: Model-based selection
        if 'model_based' in self.feature_importance_results:
            strategies.append(f"4. Keep top {target_features} features by Random Forest importance")
        
        # Strategy 5: PCA
        if 'pca' in self.dimensionality_analysis:
            pca = self.dimensionality_analysis['pca']
            cum_var = self.dimensionality_analysis['cumulative_variance']
            components_90 = np.argmax(cum_var >= 0.9) + 1
            strategies.append(f"5. Use PCA: {components_90} components for 90% variance")
        
        print(f"\nRecommended strategies (in order of preference):")
        for strategy in strategies:
            print(f"  {strategy}")
        
        # Provide implementation code
        print(f"\n💻 Implementation example:")
        print(f"  # Remove low variance features")
        print(f"  from sklearn.feature_selection import VarianceThreshold")
        print(f"  selector = VarianceThreshold(threshold=0.01)")
        print(f"  X_filtered = selector.fit_transform(X)")
        print(f"  ")
        print(f"  # Select top K features")
        print(f"  from sklearn.feature_selection import SelectKBest, f_classif")
        print(f"  selector = SelectKBest(f_classif, k={target_features})")
        print(f"  X_selected = selector.fit_transform(X, y)")
        
        return strategies
    
    def run_complete_analysis(self, features_path='data/X_features.npy',
                            target_path='data/y_target.npy',
                            feature_engine_path='data/feature_engine.pkl'):
        """Run complete feature analysis pipeline"""
        print("🔍 MEDICAL TEXT FEATURE ANALYSIS")
        print("=" * 45)
        
        # Load features
        self.load_features(features_path, target_path, feature_engine_path)
        
        # Run all analysis steps
        self.analyze_feature_variance()
        self.analyze_feature_importance_univariate()
        self.analyze_feature_importance_model_based()
        self.analyze_feature_correlations()
        self.analyze_dimensionality_and_pca()
        self.model_selection_analysis()
        self.generate_feature_selection_recommendations()
        
        return {
            'feature_importance': self.feature_importance_results,
            'correlations': self.correlation_analysis,
            'dimensionality': self.dimensionality_analysis,
            'model_recommendations': self.model_recommendations
        }

def main():
    """Run feature analysis"""
    analyzer = MedicalTextFeatureAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"Results stored in analyzer object for further investigation.")
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main() 