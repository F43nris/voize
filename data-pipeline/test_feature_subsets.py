#!/usr/bin/env python3
"""
Medical Text Classification - Feature Subset Testing
====================================================
Test Multinomial NB with different subsets of original engineered features:
- All features vs No keywords vs TF-IDF only vs Readability only etc.
- Understand which feature types help/hurt performance
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# Import feature engine class
try:
    from preprocess import MedicalTextFeatureEngine
except ImportError:
    print("Warning: Could not import MedicalTextFeatureEngine")

class FeatureSubsetTester:
    """Test Multinomial NB with different feature subsets"""
    
    def __init__(self):
        self.results = {}
        self.feature_names = None
        self.feature_types = {}
        
    def load_data(self, features_path='data/X_features.npy', 
                  target_path='data/y_target.npy',
                  feature_engine_path='data/feature_engine.pkl'):
        """Load data and identify feature types"""
        print("📊 LOADING DATA FOR FEATURE SUBSET TESTING")
        print("=" * 50)
        
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
        
        # Identify feature types
        self.identify_feature_types()
        
        return self.X, self.y
    
    def identify_feature_types(self):
        """Identify different types of features from names"""
        print(f"\n🔍 IDENTIFYING FEATURE TYPES")
        print("=" * 35)
        
        self.feature_types = {
            'tfidf': [],
            'keywords': [],
            'readability': [],
            'length': [],
            'medical_entities': [],
            'other': []
        }
        
        for i, name in enumerate(self.feature_names):
            name_lower = name.lower()
            
            # TF-IDF features (typically word features)
            if any(word in name_lower for word in ['tfidf_', 'word_', 'token_']) or \
               (len(name.split('_')) == 1 and name.isalpha() and len(name) > 2):
                self.feature_types['tfidf'].append(i)
            
            # Medical keywords
            elif any(word in name_lower for word in ['keyword_', 'medical_keyword', 'specialty_keyword']):
                self.feature_types['keywords'].append(i)
            
            # Readability metrics
            elif any(word in name_lower for word in [
                'flesch', 'kincaid', 'coleman', 'automated_readability', 
                'gunning_fog', 'smog', 'syllable', 'sentence_count'
            ]):
                self.feature_types['readability'].append(i)
            
            # Length features
            elif any(word in name_lower for word in [
                'length', 'word_count', 'char_count', 'line_count',
                'avg_word_length', 'avg_sentence_length'
            ]):
                self.feature_types['length'].append(i)
            
            # Medical entities
            elif any(word in name_lower for word in [
                'entity_', 'medical_entity', 'ner_', 'drug_', 'disease_', 'symptom_'
            ]):
                self.feature_types['medical_entities'].append(i)
            
            # Everything else
            else:
                self.feature_types['other'].append(i)
        
        # Print feature type counts
        for feature_type, indices in self.feature_types.items():
            print(f"  {feature_type.capitalize():<15}: {len(indices):>4} features")
            if len(indices) > 0 and len(indices) <= 5:  # Show examples if few features
                example_names = [self.feature_names[i] for i in indices[:3]]
                print(f"    Examples: {', '.join(example_names)}")
    
    def create_feature_subsets(self):
        """Create different feature subsets to test"""
        print(f"\n🎨 CREATING FEATURE SUBSETS")
        print("=" * 35)
        
        subsets = {}
        
        # 1. All features (baseline)
        all_indices = list(range(len(self.feature_names)))
        subsets['all_features'] = {
            'indices': all_indices,
            'description': 'All original features'
        }
        
        # 2. No keywords
        no_keywords_indices = [i for i in all_indices if i not in self.feature_types['keywords']]
        subsets['no_keywords'] = {
            'indices': no_keywords_indices,
            'description': 'All features except medical keywords'
        }
        
        # 3. TF-IDF only
        if self.feature_types['tfidf']:
            subsets['tfidf_only'] = {
                'indices': self.feature_types['tfidf'],
                'description': 'TF-IDF features only'
            }
        
        # 4. Keywords only
        if self.feature_types['keywords']:
            subsets['keywords_only'] = {
                'indices': self.feature_types['keywords'],
                'description': 'Medical keywords only'
            }
        
        # 5. Readability + Length (meta features)
        meta_indices = self.feature_types['readability'] + self.feature_types['length']
        if meta_indices:
            subsets['meta_features'] = {
                'indices': meta_indices,
                'description': 'Readability + Length features'
            }
        
        # 6. TF-IDF + Meta (no keywords, no entities)
        tfidf_meta_indices = self.feature_types['tfidf'] + meta_indices
        if tfidf_meta_indices:
            subsets['tfidf_plus_meta'] = {
                'indices': tfidf_meta_indices,
                'description': 'TF-IDF + Readability + Length'
            }
        
        # 7. Everything except TF-IDF (engineered features only)
        non_tfidf_indices = [i for i in all_indices if i not in self.feature_types['tfidf']]
        if non_tfidf_indices:
            subsets['no_tfidf'] = {
                'indices': non_tfidf_indices,
                'description': 'All engineered features (no TF-IDF)'
            }
        
        # 8. Medical entities only (if available)
        if self.feature_types['medical_entities']:
            subsets['entities_only'] = {
                'indices': self.feature_types['medical_entities'],
                'description': 'Medical entities only'
            }
        
        # 9. High-level combinations
        core_indices = self.feature_types['tfidf'] + self.feature_types['readability'] + self.feature_types['length']
        if core_indices:
            subsets['core_features'] = {
                'indices': core_indices,
                'description': 'Core features (TF-IDF + Meta, no keywords/entities)'
            }
        
        print(f"Created {len(subsets)} feature subsets:")
        for name, info in subsets.items():
            print(f"  {name:<20}: {len(info['indices']):>4} features - {info['description']}")
        
        return subsets
    
    def test_subset(self, name, subset_info):
        """Test Multinomial NB on a specific feature subset"""
        indices = subset_info['indices']
        
        if len(indices) == 0:
            return {
                'mean_accuracy': 0,
                'std_accuracy': 0,
                'error': 'No features in subset'
            }
        
        try:
            # Extract subset of features
            X_subset = self.X[:, indices]
            
            # Create pipeline - adjust feature selection based on subset size
            k_features = min(500, len(indices), int(len(indices) * 0.8))
            
            pipeline = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k=k_features)),
                ('scaler', MinMaxScaler()),
                ('classifier', MultinomialNB(alpha=1.0, fit_prior=True))  # Use good defaults
            ])
            
            # Cross-validation
            start_time = time.time()
            scores = cross_val_score(pipeline, X_subset, self.y, cv=3, scoring='accuracy')
            elapsed_time = time.time() - start_time
            
            result = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'scores': scores,
                'n_features_total': len(indices),
                'n_features_selected': k_features,
                'time_seconds': elapsed_time
            }
            
            print(f"  ✅ {scores.mean():.3f} (±{scores.std():.3f}) | {len(indices)}/{k_features} features | {elapsed_time:.1f}s")
            
        except Exception as e:
            result = {
                'mean_accuracy': 0,
                'std_accuracy': 0,
                'error': str(e),
                'n_features_total': len(indices)
            }
            print(f"  ❌ Failed: {str(e)[:50]}")
        
        return result
    
    def test_all_subsets(self, subsets):
        """Test all feature subsets"""
        print(f"\n🧪 TESTING FEATURE SUBSETS")
        print("=" * 35)
        
        results = {}
        
        for name, subset_info in subsets.items():
            print(f"Testing {name}...")
            results[name] = self.test_subset(name, subset_info)
            results[name]['description'] = subset_info['description']
        
        return results
    
    def analyze_results(self, results):
        """Analyze and rank subset results"""
        print(f"\n📊 FEATURE SUBSET ANALYSIS")
        print("=" * 35)
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            print("❌ No subsets succeeded!")
            return
        
        # Sort by accuracy
        sorted_results = sorted(successful_results.items(), 
                              key=lambda x: x[1]['mean_accuracy'], 
                              reverse=True)
        
        print(f"🏆 FEATURE SUBSET RANKINGS:")
        print("-" * 90)
        print(f"{'Rank':<4} {'Subset':<20} {'Accuracy':<12} {'±Std':<8} {'Features':<10} {'Description':<25}")
        print("-" * 90)
        
        baseline_acc = None
        
        for i, (name, result) in enumerate(sorted_results, 1):
            acc = result['mean_accuracy']
            std = result['std_accuracy']
            n_features = result.get('n_features_selected', result.get('n_features_total', 0))
            description = result.get('description', '')
            
            # Track baseline (all features)
            if name == 'all_features':
                baseline_acc = acc
            
            # Highlight top performers
            marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            
            print(f"{marker} {i:<2} {name:<20} {acc:.3f} ({acc:.1%}){std:>6.3f} {n_features:>8} {description[:25]}")
        
        # Analysis insights
        print(f"\n💡 KEY INSIGHTS:")
        
        # Find best and worst
        best_name, best_result = sorted_results[0]
        worst_name, worst_result = sorted_results[-1]
        
        print(f"🏆 Best subset: {best_name} ({best_result['mean_accuracy']:.1%})")
        print(f"💔 Worst subset: {worst_name} ({worst_result['mean_accuracy']:.1%})")
        
        if baseline_acc:
            # Compare to baseline
            for name, result in sorted_results:
                if name != 'all_features':
                    acc = result['mean_accuracy']
                    change = ((acc - baseline_acc) / baseline_acc) * 100
                    if abs(change) > 1:  # Only show significant changes
                        direction = "📈" if change > 0 else "📉"
                        print(f"{direction} {name}: {change:+.1f}% vs all features")
        
        # Keyword impact analysis
        if 'all_features' in successful_results and 'no_keywords' in successful_results:
            all_acc = successful_results['all_features']['mean_accuracy']
            no_kw_acc = successful_results['no_keywords']['mean_accuracy']
            keyword_impact = ((no_kw_acc - all_acc) / all_acc) * 100
            
            print(f"\n🎯 KEYWORD IMPACT:")
            if keyword_impact > 1:
                print(f"✅ Removing keywords HELPS: +{keyword_impact:.1f}% improvement")
            elif keyword_impact < -1:
                print(f"❌ Removing keywords HURTS: {keyword_impact:.1f}% decrease")
            else:
                print(f"😐 Keywords have minimal impact: {keyword_impact:+.1f}%")
        
        # Feature efficiency analysis
        print(f"\n⚡ EFFICIENCY ANALYSIS:")
        efficiency_results = [(name, result['mean_accuracy'] / result.get('n_features_selected', 1)) 
                            for name, result in successful_results.items()]
        efficiency_results.sort(key=lambda x: x[1], reverse=True)
        
        for name, efficiency in efficiency_results[:3]:
            print(f"  {name}: {efficiency:.6f} accuracy per feature")
        
        return sorted_results
    
    def run_subset_testing(self):
        """Run complete feature subset testing"""
        print("🎯 FEATURE SUBSET TESTING FOR MULTINOMIAL NB")
        print("=" * 55)
        
        start_time = time.time()
        
        # Step 1: Load data and identify feature types
        self.load_data()
        
        # Step 2: Create feature subsets
        subsets = self.create_feature_subsets()
        
        # Step 3: Test all subsets
        results = self.test_all_subsets(subsets)
        
        # Step 4: Analyze results
        sorted_results = self.analyze_results(results)
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print(f"\n🎉 SUBSET TESTING COMPLETE!")
        print("=" * 35)
        print(f"  Tested {len(subsets)} feature subsets")
        print(f"  Total time: {elapsed_time:.1f} seconds")
        
        # Store results
        self.results = results
        
        return results, sorted_results

def main():
    """Run the feature subset testing"""
    tester = FeatureSubsetTester()
    results, sorted_results = tester.run_subset_testing()
    
    print(f"\n✅ Feature subset testing complete!")
    return tester, results

if __name__ == "__main__":
    tester, results = main() 