#!/usr/bin/env python3
"""
Medical Text Classification - Feature Engineering
================================================
Transform cleaned data into ML-ready features based on EDA insights:
1. Text preprocessing (transcription + description)
2. TF-IDF vectorization 
3. Text length features
4. Categorical features (sample_name)
5. Feature combination and selection
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import string

class MedicalTextFeatureEngine:
    """Configurable feature engineering for medical text classification"""
    
    def __init__(self, target_col='medical_specialty'):
        self.target_col = target_col
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.sample_name_encoder = None
        self.feature_names = []
        
    def analyze_text_artifacts(self, df, text_col='transcription', sample_size=100):
        """Step 0: Analyze what artifacts are actually in the text"""
        print("🔍 STEP 0: Text Artifacts Analysis")
        print("=" * 40)
        
        if text_col not in df.columns:
            print(f"❌ Column '{text_col}' not found!")
            return {}
        
        # Sample texts for analysis
        sample_texts = df[text_col].dropna().head(sample_size).astype(str)
        
        # Define patterns to look for
        patterns = {
            'dates': r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b|\b\w+\s+\d{1,2},?\s+\d{4}\b',
            'times': r'\b\d{1,2}:\d{2}\b',
            'doctor_refs': r'\b(dr|doctor|physician|md|m\.d\.)\b',
            'patient_refs': r'\bpatient\b',
            'hospital_refs': r'\bhospital\b',
            'date_labels': r'\b(date|time)\s*[:\-]',
            'phone_numbers': r'\b\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4}\b',
            'medical_ids': r'\b[A-Z0-9]{3,}\b'  # Potential IDs
        }
        
        findings = {}
        
        print("Analyzing artifacts in sample texts:")
        for pattern_name, pattern in patterns.items():
            matches = []
            count = 0
            
            for text in sample_texts:
                found = re.findall(pattern, text.lower(), re.IGNORECASE)
                if found:
                    matches.extend(found[:3])  # Keep first 3 matches per text
                    count += len(found)
            
            findings[pattern_name] = {
                'count': count,
                'examples': list(set(matches))[:5]  # Top 5 unique examples
            }
            
            if count > 0:
                print(f"  {pattern_name}: {count} matches")
                if matches:
                    examples = ', '.join(matches[:3])
                    print(f"    Examples: {examples}")
            else:
                print(f"  {pattern_name}: No matches found")
        
        return findings

    def preprocess_text(self, df, text_cols=['transcription', 'description'], 
                       conservative_cleaning=True):
        """Step 1: Data-driven text preprocessing"""
        print("\n📝 STEP 1: Text Preprocessing")
        print("=" * 35)
        
        # First analyze what artifacts exist
        if 'transcription' in df.columns:
            artifacts = self.analyze_text_artifacts(df, 'transcription')
        else:
            artifacts = {}
        
        df_processed = df.copy()
        
        def clean_medical_text(text):
            if not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            if conservative_cleaning:
                # Only remove artifacts we actually found
                
                # Remove explicit date labels (based on findings)
                if artifacts.get('date_labels', {}).get('count', 0) > 0:
                    text = re.sub(r'\b(date|time)\s*[:\-]\s*[^\s,]+', '', text)
                
                # Remove phone numbers if found
                if artifacts.get('phone_numbers', {}).get('count', 0) > 0:
                    text = re.sub(r'\b\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4}\b', '', text)
                
                # Normalize common medical abbreviations (these are definitely in medical text)
                medical_abbrevs = {
                    r'\bp\.?o\.?\b': 'oral',
                    r'\bi\.?v\.?\b': 'intravenous', 
                    r'\bq\.?i\.?d\.?\b': 'daily',
                    r'\bb\.?i\.?d\.?\b': 'twice daily',
                    r'\bt\.?i\.?d\.?\b': 'three times daily',
                    r'\bprn\b': 'as needed',
                    r'\bstat\b': 'immediately',
                }
                
                for abbrev, full in medical_abbrevs.items():
                    text = re.sub(abbrev, full, text)
            
            else:
                # More aggressive cleaning (original approach)
                text = re.sub(r'\b(date|time|patient|dr|md|hospital)\s*[:\-]?\s*\S+', '', text)
            
            # Remove punctuation but keep medical decimals (e.g., "2.5 mg")
            text = re.sub(r'[^\w\s\.]', ' ', text)
            text = re.sub(r'\.(?!\d)', ' ', text)  # Remove periods not followed by digits
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        
        # Process each text column
        for col in text_cols:
            if col in df_processed.columns:
                print(f"Processing {col}...")
                df_processed[f'{col}_cleaned'] = df_processed[col].apply(clean_medical_text)
        
        # Create combined text AFTER processing all columns
        if ('transcription_cleaned' in df_processed.columns and 
            'description_cleaned' in df_processed.columns):
            print("Creating combined text...")
            df_processed['combined_text'] = (
                df_processed['description_cleaned'].fillna('') + ' ' + 
                df_processed['transcription_cleaned'].fillna('')
            ).str.strip()
        
        print(f"✅ Text preprocessing complete")
        return df_processed
    
    def create_tfidf_features(self, df, text_col='transcription_cleaned', 
                             max_features=5000, min_df=2, max_df=0.95):
        """Step 2: Create TF-IDF features"""
        print("\n🔤 STEP 2: TF-IDF Feature Creation")
        print("=" * 40)
        
        if text_col not in df.columns:
            print(f"❌ Column '{text_col}' not found!")
            return df, np.array([])
        
        # Initialize TF-IDF vectorizer with medical text optimizations
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,           # Ignore terms in less than 2 documents
            max_df=max_df,           # Ignore terms in more than 95% of documents  
            stop_words='english',    # Remove common English stop words
            ngram_range=(1, 2),      # Include unigrams and bigrams
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Keep alphanumeric tokens
        )
        
        # Fit and transform the text
        texts = df[text_col].fillna('')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        print(f"TF-IDF features created:")
        print(f"  Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_):,}")
        print(f"  Feature matrix shape: {tfidf_matrix.shape}")
        print(f"  Sparsity: {(1 - tfidf_matrix.nnz / tfidf_matrix.size) * 100:.1f}%")
        
        # Show top features by average TF-IDF score
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        top_features = [(feature_names[i], mean_scores[i]) 
                       for i in mean_scores.argsort()[-10:][::-1]]
        
        print(f"\nTop 10 TF-IDF features:")
        for feature, score in top_features:
            print(f"  {feature}: {score:.4f}")
        
        return df, tfidf_matrix
    
    def create_length_features(self, df):
        """Step 3: Create text length features"""
        print("\n📏 STEP 3: Text Length Features")
        print("=" * 35)
        
        df_features = df.copy()
        
        # Character length features
        if 'transcription' in df.columns:
            df_features['transcription_char_length'] = df['transcription'].str.len().fillna(0)
            df_features['transcription_word_count'] = df['transcription'].str.split().str.len().fillna(0)
        
        if 'description' in df.columns:
            df_features['description_char_length'] = df['description'].str.len().fillna(0)
            df_features['description_word_count'] = df['description'].str.split().str.len().fillna(0)
        
        # Ratio features (if both exist)
        if 'transcription' in df.columns and 'description' in df.columns:
            df_features['text_length_ratio'] = (
                df_features['transcription_char_length'] / 
                (df_features['description_char_length'] + 1)  # +1 to avoid division by zero
            )
        
        # Length category features (based on EDA insights)
        if 'transcription_char_length' in df_features.columns:
            # Create length categories based on our EDA findings
            df_features['text_length_category'] = pd.cut(
                df_features['transcription_char_length'],
                bins=[0, 1608, 4011, 8000, float('inf')],  # Based on EDA quartiles
                labels=['short', 'medium', 'long', 'very_long']
            )
        
        length_cols = [col for col in df_features.columns if 'length' in col or 'count' in col]
        print(f"✅ Created {len(length_cols)} length features")
        
        return df_features
    
    def create_categorical_features(self, df):
        """Step 4: Create categorical features"""
        print("\n🏷️  STEP 4: Categorical Features")
        print("=" * 35)
        
        df_features = df.copy()
        
        # Encode sample_name if available
        if 'sample_name' in df.columns:
            self.sample_name_encoder = LabelEncoder()
            df_features['sample_name_encoded'] = self.sample_name_encoder.fit_transform(
                df['sample_name'].fillna('unknown')
            )
            
            n_categories = len(self.sample_name_encoder.classes_)
            print(f"✅ Encoded sample_name: {n_categories} categories")
        
        # Medical report structure features (based on EDA findings)
        if 'transcription' in df.columns:
            medical_sections = ['subjective', 'objective', 'assessment', 'plan']
            
            for section in medical_sections:
                df_features[f'has_{section}'] = (
                    df['transcription'].str.lower().str.contains(section, na=False).astype(int)
                )
            
            print(f"✅ Created medical structure features: {medical_sections}")
        
        return df_features
    
    def prepare_features_and_target(self, df, tfidf_matrix):
        """Step 5: Combine all features and prepare target"""
        print("\n🎯 STEP 5: Feature Combination & Target Preparation")
        print("=" * 50)
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df[self.target_col])
        
        print(f"Target encoding:")
        print(f"  Classes: {len(self.label_encoder.classes_)}")
        print(f"  Class distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for i, (cls, count) in enumerate(zip(self.label_encoder.classes_, counts)):
            if i < 5:  # Show top 5
                print(f"    {cls}: {count}")
        
        # Collect numerical features
        numerical_features = []
        feature_names = []
        
        # Length features
        length_cols = [col for col in df.columns if 
                      any(x in col for x in ['length', 'count', 'ratio'])]
        if length_cols:
            numerical_features.append(df[length_cols].values)
            feature_names.extend(length_cols)
            print(f"✅ Added {len(length_cols)} length features")
        
        # Categorical features  
        categorical_cols = [col for col in df.columns if 
                          any(x in col for x in ['encoded', 'has_'])]
        if categorical_cols:
            numerical_features.append(df[categorical_cols].values)
            feature_names.extend(categorical_cols)
            print(f"✅ Added {len(categorical_cols)} categorical features")
        
        # Combine TF-IDF with other features
        if len(numerical_features) > 0:
            other_features = np.hstack(numerical_features)
            X = np.hstack([tfidf_matrix.toarray(), other_features])
            
            # Add TF-IDF feature names
            tfidf_names = [f'tfidf_{name}' for name in self.tfidf_vectorizer.get_feature_names_out()]
            all_feature_names = tfidf_names + feature_names
        else:
            X = tfidf_matrix.toarray()
            all_feature_names = [f'tfidf_{name}' for name in self.tfidf_vectorizer.get_feature_names_out()]
        
        self.feature_names = all_feature_names
        
        print(f"\n📊 Final feature matrix:")
        print(f"  Shape: {X.shape}")
        print(f"  TF-IDF features: {tfidf_matrix.shape[1]:,}")
        print(f"  Other features: {X.shape[1] - tfidf_matrix.shape[1]}")
        
        return X, y
    
    def process_pipeline(self, df, max_features=5000, conservative_cleaning=True):
        """Run complete feature engineering pipeline"""
        print("🔧 MEDICAL TEXT FEATURE ENGINEERING")
        print("=" * 45)
        
        # Step 1: Preprocess text (now with artifact analysis)
        df_processed = self.preprocess_text(df, conservative_cleaning=conservative_cleaning)
        
        # Step 2: Create TF-IDF features
        df_processed, tfidf_matrix = self.create_tfidf_features(
            df_processed, max_features=max_features
        )
        
        # Step 3: Create length features
        df_processed = self.create_length_features(df_processed)
        
        # Step 4: Create categorical features
        df_processed = self.create_categorical_features(df_processed)
        
        # Step 5: Combine features and prepare target
        X, y = self.prepare_features_and_target(df_processed, tfidf_matrix)
        
        return X, y, df_processed

def create_features(input_file='data/mtsamples_clean.csv',
                   output_dir='data/',
                   max_features=5000,
                   conservative_cleaning=True):
    """Main feature engineering function"""
    print("🔬 MEDICAL TEXT CLASSIFICATION - FEATURE ENGINEERING")
    print("=" * 55)
    
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples")
    
    # Initialize feature engine
    feature_engine = MedicalTextFeatureEngine()
    
    # Process features
    X, y, df_processed = feature_engine.process_pipeline(
        df, 
        max_features=max_features,
        conservative_cleaning=conservative_cleaning
    )
    
    # Save processed features
    np.save(f'{output_dir}X_features.npy', X)
    np.save(f'{output_dir}y_target.npy', y)
    
    # Save feature names and encoders for later use
    import pickle
    with open(f'{output_dir}feature_engine.pkl', 'wb') as f:
        pickle.dump(feature_engine, f)
    
    print(f"\n✅ Features saved:")
    print(f"  Features: {output_dir}X_features.npy")
    print(f"  Target: {output_dir}y_target.npy") 
    print(f"  Feature engine: {output_dir}feature_engine.pkl")
    
    return X, y, feature_engine

if __name__ == "__main__":
    # Run feature engineering with conservative cleaning
    X, y, feature_engine = create_features(
        max_features=5000,
        conservative_cleaning=True  # Only clean artifacts we actually find
    )
    
    print(f"\n🎯 Ready for model training!")
    print(f"Feature matrix: {X.shape}")
    print(f"Target classes: {len(np.unique(y))}") 