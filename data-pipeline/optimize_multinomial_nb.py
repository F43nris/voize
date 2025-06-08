#!/usr/bin/env python3
"""
Medical Text Classification - Multinomial Naive Bayes Optimization
==================================================================
Comprehensive optimization of our winning model from both angles:
1. Hyperparameter tuning (alpha, priors, etc.)
2. Feature engineering (n-grams, TF-IDF params, feature selection)
3. Data preprocessing variations
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import time
import warnings
import re
warnings.filterwarnings('ignore')

# Import feature engine class
try:
    from preprocess import MedicalTextFeatureEngine
except ImportError:
    print("Warning: Could not import MedicalTextFeatureEngine")

class MultinomialNBOptimizer:
    """Comprehensive Multinomial NB optimization"""
    
    def __init__(self):
        self.best_pipeline = None
        self.best_params = None
        self.best_score = 0
        self.optimization_results = {}
        
    def load_data(self, features_path='data/X_features.npy', 
                  target_path='data/y_target.npy',
                  feature_engine_path='data/feature_engine.pkl',
                  raw_text_path='data/medical_transcripts.csv'):
        """Load both processed features and raw text"""
        print("📊 LOADING DATA FOR MULTINOMIAL NB OPTIMIZATION")
        print("=" * 55)
        
        # Load processed features
        self.X_processed = np.load(features_path, allow_pickle=True)
        self.y = np.load(target_path, allow_pickle=True)
        
        # Load feature names and label encoder
        with open(feature_engine_path, 'rb') as f:
            feature_engine = pickle.load(f)
        self.feature_names = feature_engine.feature_names
        self.label_encoder = feature_engine.label_encoder
        
        # Convert processed features to numeric
        if self.X_processed.dtype == 'object':
            print("Converting processed features to numeric...")
            X_numeric = pd.DataFrame(self.X_processed).apply(pd.to_numeric, errors='coerce').fillna(0).values
            self.X_processed = X_numeric.astype(float)
        
        # Try to load raw text for re-engineering
        try:
            df = pd.read_csv(raw_text_path)
            if 'transcription' in df.columns and 'medical_specialty' in df.columns:
                self.raw_texts = df['transcription'].values
                self.raw_labels = df['medical_specialty'].values
                print("✅ Raw text loaded for feature re-engineering")
            else:
                self.raw_texts = None
                print("⚠️  Raw text columns not found, using processed features only")
        except:
            self.raw_texts = None
            print("⚠️  Raw text file not found, using processed features only")
        
        print(f"✅ Data loaded:")
        print(f"  Samples: {len(self.y):,}")
        print(f"  Processed features: {self.X_processed.shape[1]:,}")
        print(f"  Classes: {len(np.unique(self.y))}")
        print(f"  Raw text available: {'Yes' if self.raw_texts is not None else 'No'}")
        
        return self.X_processed, self.y
    
    def test_hyperparameters(self, X, y):
        """Optimize Multinomial NB hyperparameters"""
        print(f"\n🔧 HYPERPARAMETER OPTIMIZATION")
        print("=" * 40)
        
        # Create base pipeline with current best feature count (500)
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(f_classif, k=500)),
            ('scaler', MinMaxScaler()),  # Ensure positive values
            ('classifier', MultinomialNB())
        ])
        
        # Comprehensive parameter grid
        param_grid = {
            # Feature selection
            'feature_selection__k': [300, 400, 500, 600, 750],
            'feature_selection__score_func': [f_classif, chi2],
            
            # Multinomial NB parameters
            'classifier__alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],  # Smoothing
            'classifier__fit_prior': [True, False],  # Learn class priors or use uniform
        }
        
        print("Testing parameter combinations...")
        print(f"Grid size: {len(param_grid['feature_selection__k']) * len(param_grid['feature_selection__score_func']) * len(param_grid['classifier__alpha']) * len(param_grid['classifier__fit_prior'])} combinations")
        
        # Grid search with 3-fold CV for speed
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=3, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X, y)
        elapsed_time = time.time() - start_time
        
        print(f"\n🏆 BEST HYPERPARAMETERS:")
        print(f"  Parameters: {grid_search.best_params_}")
        print(f"  CV Score: {grid_search.best_score_:.3f} ({grid_search.best_score_:.1%})")
        print(f"  Time: {elapsed_time:.1f} seconds")
        
        self.optimization_results['hyperparameters'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'time_seconds': elapsed_time
        }
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def create_feature_variants(self):
        """Create different feature engineering variants"""
        print(f"\n🎨 TESTING FEATURE ENGINEERING VARIANTS")
        print("=" * 45)
        
        if self.raw_texts is None:
            print("⚠️  No raw text available, using processed features only")
            return {'processed_features': self.X_processed}
        
        variants = {}
        
        # Preprocess text function
        def preprocess_text(text):
            if pd.isna(text):
                return ""
            # Basic cleaning
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            return text.strip()
        
        print("Creating feature variants...")
        
        # Variant 1: TF-IDF with different n-grams
        print("1. TF-IDF Unigrams + Bigrams...")
        tfidf_1_2 = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english',
            preprocessor=preprocess_text
        )
        
        try:
            X_tfidf_1_2 = tfidf_1_2.fit_transform(self.raw_texts).toarray()
            variants['tfidf_1_2_grams'] = X_tfidf_1_2
            print(f"   ✅ Shape: {X_tfidf_1_2.shape}")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:50]}")
        
        # Variant 2: TF-IDF with trigrams
        print("2. TF-IDF Unigrams + Bigrams + Trigrams...")
        tfidf_1_3 = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9,
            stop_words='english',
            preprocessor=preprocess_text
        )
        
        try:
            X_tfidf_1_3 = tfidf_1_3.fit_transform(self.raw_texts).toarray()
            variants['tfidf_1_3_grams'] = X_tfidf_1_3
            print(f"   ✅ Shape: {X_tfidf_1_3.shape}")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:50]}")
        
        # Variant 3: Count-based features
        print("3. Count Vectorizer...")
        count_vec = CountVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english',
            preprocessor=preprocess_text
        )
        
        try:
            X_count = count_vec.fit_transform(self.raw_texts).toarray()
            variants['count_features'] = X_count
            print(f"   ✅ Shape: {X_count.shape}")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:50]}")
        
        # Variant 4: Character-level n-grams
        print("4. Character n-grams...")
        char_tfidf = TfidfVectorizer(
            max_features=1500,
            analyzer='char',
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            preprocessor=preprocess_text
        )
        
        try:
            X_char = char_tfidf.fit_transform(self.raw_texts).toarray()
            variants['char_ngrams'] = X_char
            print(f"   ✅ Shape: {X_char.shape}")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:50]}")
        
        # Always include processed features
        variants['processed_features'] = self.X_processed
        
        print(f"\n✅ Created {len(variants)} feature variants")
        return variants
    
    def test_feature_variants(self, variants):
        """Test different feature engineering approaches"""
        print(f"\n🧪 TESTING FEATURE VARIANTS")
        print("=" * 35)
        
        results = {}
        
        # Use the best hyperparameters found earlier if available
        best_alpha = 1.0
        best_fit_prior = True
        if 'hyperparameters' in self.optimization_results:
            best_params = self.optimization_results['hyperparameters']['best_params']
            best_alpha = best_params.get('classifier__alpha', 1.0)
            best_fit_prior = best_params.get('classifier__fit_prior', True)
        
        for name, X_variant in variants.items():
            print(f"Testing {name}...")
            
            try:
                # Create pipeline for this variant
                pipeline = Pipeline([
                    ('feature_selection', SelectKBest(f_classif, k=min(500, X_variant.shape[1]))),
                    ('scaler', MinMaxScaler()),
                    ('classifier', MultinomialNB(alpha=best_alpha, fit_prior=best_fit_prior))
                ])
                
                # Cross-validation
                scores = cross_val_score(pipeline, X_variant, self.y, cv=3, scoring='accuracy')
                
                results[name] = {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std(),
                    'scores': scores,
                    'n_features': X_variant.shape[1]
                }
                
                print(f"  ✅ {scores.mean():.3f} (±{scores.std():.3f}) | {X_variant.shape[1]} features")
                
            except Exception as e:
                results[name] = {
                    'mean_accuracy': 0,
                    'std_accuracy': 0,
                    'error': str(e)
                }
                print(f"  ❌ Failed: {str(e)[:50]}")
        
        # Find best variant
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        if successful_results:
            best_variant = max(successful_results.keys(), key=lambda k: successful_results[k]['mean_accuracy'])
            best_score = successful_results[best_variant]['mean_accuracy']
            
            print(f"\n🏆 BEST FEATURE VARIANT: {best_variant}")
            print(f"   Accuracy: {best_score:.1%}")
            
            self.optimization_results['feature_variants'] = results
            return best_variant, variants[best_variant], best_score
        
        return None, None, 0
    
    def train_final_optimized_model(self, best_X, best_params):
        """Train final model with best features and parameters"""
        print(f"\n🚀 TRAINING FINAL OPTIMIZED MODEL")
        print("=" * 40)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            best_X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Create final pipeline
        final_pipeline = Pipeline([
            ('feature_selection', SelectKBest(
                score_func=best_params.get('feature_selection__score_func', f_classif),
                k=best_params.get('feature_selection__k', 500)
            )),
            ('scaler', MinMaxScaler()),
            ('classifier', MultinomialNB(
                alpha=best_params.get('classifier__alpha', 1.0),
                fit_prior=best_params.get('classifier__fit_prior', True)
            ))
        ])
        
        # Train
        final_pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = final_pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Final test accuracy: {test_accuracy:.3f} ({test_accuracy:.1%})")
        
        # Detailed evaluation
        print(f"\n📊 DETAILED PERFORMANCE:")
        class_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        
        # Store results
        self.best_pipeline = final_pipeline
        self.best_score = test_accuracy
        
        return final_pipeline, test_accuracy
    
    def save_optimized_model(self, output_dir='data/'):
        """Save the final optimized model"""
        print(f"\n💾 SAVING OPTIMIZED MULTINOMIAL NB")
        print("=" * 40)
        
        # Save pipeline
        model_path = f'{output_dir}optimized_multinomial_nb.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_pipeline, f)
        
        # Save optimization results
        results_path = f'{output_dir}optimization_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(self.optimization_results, f)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Results saved: {results_path}")
        
        return model_path, results_path
    
    def run_full_optimization(self):
        """Run the complete optimization pipeline"""
        print("🎯 MULTINOMIAL NAIVE BAYES OPTIMIZATION")
        print("=" * 50)
        
        start_time = time.time()
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Optimize hyperparameters on processed features
        best_pipeline, best_params, best_cv_score = self.test_hyperparameters(self.X_processed, self.y)
        
        # Step 3: Test feature engineering variants
        feature_variants = self.create_feature_variants()
        best_variant_name, best_X, best_variant_score = self.test_feature_variants(feature_variants)
        
        # Step 4: Train final model
        if best_X is not None:
            final_pipeline, test_accuracy = self.train_final_optimized_model(best_X, best_params)
        else:
            final_pipeline, test_accuracy = self.train_final_optimized_model(self.X_processed, best_params)
        
        # Step 5: Save model
        model_path, results_path = self.save_optimized_model()
        
        elapsed_time = time.time() - start_time
        
        # Summary
        print(f"\n🎉 OPTIMIZATION COMPLETE!")
        print("=" * 30)
        print(f"  Hyperparameter CV: {best_cv_score:.3f}")
        print(f"  Best feature variant: {best_variant_name}")
        if best_variant_score > 0:
            print(f"  Variant CV score: {best_variant_score:.3f}")
        print(f"  Final test accuracy: {test_accuracy:.3f} ({test_accuracy:.1%})")
        print(f"  Improvement over baseline (38.6%): {((test_accuracy - 0.386) / 0.386 * 100):+.1f}%")
        print(f"  Total time: {elapsed_time:.1f} seconds")
        
        return {
            'hyperparameter_score': best_cv_score,
            'variant_score': best_variant_score,
            'test_accuracy': test_accuracy,
            'best_variant': best_variant_name,
            'model_path': model_path
        }

def main():
    """Run the optimization"""
    optimizer = MultinomialNBOptimizer()
    results = optimizer.run_full_optimization()
    
    print(f"\n✅ Multinomial NB optimization complete!")
    return optimizer, results

if __name__ == "__main__":
    optimizer, results = main() 