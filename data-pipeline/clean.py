#!/usr/bin/env python3
"""
Medical Text Classification - Data Cleaning
==========================================
Handle data quality issues identified in EDA:
1. Missing transcriptions (0.7%)
2. Missing keywords (21.4%) 
3. Class imbalance (40 classes)
4. Text length outliers (3% outliers, max 18k chars)
"""

import pandas as pd
import numpy as np

def clean_missing_data(df):
    """Step 1: Handle missing data"""
    print("🧹 STEP 1: Cleaning Missing Data")
    print("=" * 40)
    
    original_shape = df.shape
    print(f"Original dataset: {original_shape}")
    
    # Check missing data before cleaning
    print(f"\nMissing data before cleaning:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            pct = (missing / len(df)) * 100
            print(f"  {col}: {missing} ({pct:.1f}%)")
    
    # Strategy 1: Drop rows with missing transcription (our main feature)
    if 'transcription' in df.columns:
        before_drop = len(df)
        df_clean = df.dropna(subset=['transcription']).copy()
        dropped = before_drop - len(df_clean)
        print(f"\n✅ Dropped {dropped} rows with missing transcription")
    else:
        df_clean = df.copy()
    
    # Strategy 2: Drop keywords column (21.4% missing - too much)
    if 'keywords' in df_clean.columns:
        df_clean = df_clean.drop('keywords', axis=1)
        print(f"✅ Dropped 'keywords' column (too many missing values)")
    
    # Strategy 3: Drop index column if exists
    if 'Unnamed: 0' in df_clean.columns:
        df_clean = df_clean.drop('Unnamed: 0', axis=1)
        print(f"✅ Dropped index column")
    
    print(f"Clean dataset: {df_clean.shape}")
    print(f"Rows removed: {original_shape[0] - df_clean.shape[0]}")
    
    return df_clean

def handle_text_outliers(df, text_col='transcription', method='cap', max_length=8000):
    """Step 2: Handle text length outliers"""
    print(f"\n📏 STEP 2: Handling Text Outliers")
    print("=" * 40)
    
    if text_col not in df.columns:
        print(f"❌ Column '{text_col}' not found!")
        return df
    
    texts = df[text_col].dropna()
    char_lengths = texts.str.len()
    
    # Calculate outliers using IQR method
    q1 = char_lengths.quantile(0.25)
    q3 = char_lengths.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    outliers_high = char_lengths[char_lengths > upper_bound]
    
    print(f"Text length analysis:")
    print(f"  IQR upper bound: {upper_bound:.0f} chars")
    print(f"  Outliers found: {len(outliers_high)} ({len(outliers_high)/len(texts)*100:.1f}%)")
    print(f"  Max length: {char_lengths.max():.0f} chars")
    
    df_clean = df.copy()
    
    if method == 'cap':
        # Cap very long texts at max_length
        before_cap = (df_clean[text_col].str.len() > max_length).sum()
        df_clean[text_col] = df_clean[text_col].apply(
            lambda x: x[:max_length] if isinstance(x, str) and len(x) > max_length else x
        )
        print(f"✅ Capped {before_cap} texts at {max_length} characters")
        
    elif method == 'remove':
        # Remove extremely long texts
        before_remove = len(df_clean)
        df_clean = df_clean[df_clean[text_col].str.len() <= max_length]
        removed = before_remove - len(df_clean)
        print(f"✅ Removed {removed} texts longer than {max_length} characters")
        
    elif method == 'keep':
        # Keep all texts as-is
        print(f"✅ Keeping all text lengths as-is")
    
    return df_clean

def reduce_class_complexity(df, strategy='top_n', n_classes=15):
    """Step 3: Handle class imbalance"""
    print(f"\n🎯 STEP 3: Reducing Class Complexity")
    print("=" * 40)
    
    target_col = 'medical_specialty'
    original_classes = df[target_col].nunique()
    
    print(f"Original classes: {original_classes}")
    print(f"Strategy: Keep top {n_classes} classes + 'Other'")
    
    # Keep top N classes, group rest as 'Other'
    top_classes = df[target_col].value_counts().head(n_classes).index
    
    df_reduced = df.copy()
    df_reduced[target_col] = df_reduced[target_col].apply(
        lambda x: x if x in top_classes else 'Other'
    )
    
    # Show results
    new_distribution = df_reduced[target_col].value_counts()
    print(f"\nNew class distribution:")
    for i, (specialty, count) in enumerate(new_distribution.head(10).items(), 1):
        pct = (count / len(df_reduced)) * 100
        print(f"  {i:2}. {specialty[:30]:30} | {count:3} ({pct:4.1f}%)")
    
    if len(new_distribution) > 10:
        others = len(new_distribution) - 10
        print(f"  ... and {others} more classes")
    
    print(f"\nFinal classes: {df_reduced[target_col].nunique()}")
    
    return df_reduced

def clean_text_quality(df, text_col='transcription', min_length=50):
    """Step 1.5: Clean text quality issues identified in EDA"""
    print(f"\n🧼 STEP 1.5: Text Quality Cleaning")
    print("=" * 40)
    
    if text_col not in df.columns:
        print(f"❌ Column '{text_col}' not found!")
        return df
    
    df_clean = df.copy()
    original_count = len(df_clean)
    
    # 1. Filter very short texts (EDA found 23 texts <50 chars)
    before_filter = len(df_clean)
    df_clean = df_clean[df_clean[text_col].str.len() >= min_length]
    short_removed = before_filter - len(df_clean)
    print(f"✅ Removed {short_removed} very short texts (< {min_length} chars)")
    
    # 2. Handle non-ASCII characters (EDA found 2.5% with non-ASCII)
    # Replace common medical non-ASCII characters, remove others
    def clean_non_ascii(text):
        if not isinstance(text, str):
            return text
        
        # Replace common medical symbols
        replacements = {
            '°': ' degrees ',
            '±': ' plus/minus ',
            '×': ' x ',
            '→': ' to ',
            '≤': ' less than or equal to ',
            '≥': ' greater than or equal to ',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove remaining non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
    
    non_ascii_before = df_clean[text_col].str.contains(r'[^\x00-\x7F]', regex=True).sum()
    df_clean[text_col] = df_clean[text_col].apply(clean_non_ascii)
    non_ascii_after = df_clean[text_col].str.contains(r'[^\x00-\x7F]', regex=True).sum()
    
    print(f"✅ Cleaned non-ASCII characters: {non_ascii_before} → {non_ascii_after} texts affected")
    
    # 3. Basic text normalization
    def normalize_text(text):
        if not isinstance(text, str):
            return text
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove multiple punctuation (e.g., "..." -> ".")
        import re
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text
    
    df_clean[text_col] = df_clean[text_col].apply(normalize_text)
    print(f"✅ Applied text normalization (whitespace, punctuation)")
    
    total_removed = original_count - len(df_clean)
    print(f"Total texts removed: {total_removed} ({total_removed/original_count*100:.1f}%)")
    
    return df_clean

def create_clean_dataset(input_file='data/mtsamples.csv', 
                        output_file='data/mtsamples_clean.csv',
                        n_classes=15,
                        outlier_method='cap',
                        max_text_length=8000,
                        min_text_length=50):
    """Main cleaning pipeline with text quality handling"""
    print("🔬 MEDICAL TEXT CLASSIFICATION - DATA CLEANING")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file)
    
    # Clean missing data
    df_clean = clean_missing_data(df)
    
    # Clean text quality issues (NEW)
    df_clean = clean_text_quality(df_clean, min_length=min_text_length)
    
    # Handle text outliers
    df_clean = handle_text_outliers(df_clean, 
                                   method=outlier_method, 
                                   max_length=max_text_length)
    
    # Reduce class complexity  
    df_final = reduce_class_complexity(df_clean, n_classes=n_classes)
    
    # Save cleaned dataset
    df_final.to_csv(output_file, index=False)
    print(f"\n✅ Clean dataset saved to: {output_file}")
    
    # Enhanced summary with quality metrics
    print(f"\n📋 CLEANING SUMMARY:")
    print(f"  Original: {df.shape[0]} rows, {df['medical_specialty'].nunique()} classes")
    print(f"  Final:    {df_final.shape[0]} rows, {df_final['medical_specialty'].nunique()} classes")
    print(f"  Removed:  {df.shape[0] - df_final.shape[0]} rows ({(df.shape[0] - df_final.shape[0])/df.shape[0]*100:.1f}%)")
    
    # Quality insights based on EDA
    print(f"\n📊 QUALITY INSIGHTS:")
    print(f"  Text length range: {df_final['transcription'].str.len().min()}-{df_final['transcription'].str.len().max()} chars")
    print(f"  Average text length: {df_final['transcription'].str.len().mean():.0f} chars")
    print(f"  Classes retained: {', '.join(df_final['medical_specialty'].value_counts().head(5).index)}")
    
    return df_final

if __name__ == "__main__":
    # Run with enhanced cleaning based on EDA findings
    clean_df = create_clean_dataset(
        outlier_method='cap',       # Cap long texts
        max_text_length=8000,       # Based on IQR analysis  
        min_text_length=50,         # Filter very short texts (EDA found 23)
        n_classes=15                # Top 15 = 84.1% coverage
    )
    
    print(f"\n🎯 Ready for preprocessing and model training!") 