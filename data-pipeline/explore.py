#!/usr/bin/env python3
"""
Medical Text Classification - Exploratory Data Analysis
=====================================================

This script performs comprehensive EDA on the MTSamples dataset to understand:
1. Dataset structure and basic statistics
2. Target variable (sample_name) distribution
3. Text characteristics and patterns
4. Data quality assessment
5. Feature engineering insights

Author: MLOps Engineer
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(file_path):
    """Load and perform initial data inspection"""
    print("📊 Loading MTSamples Dataset...")
    df = pd.read_csv(file_path)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    return df

def basic_statistics(df):
    """Generate basic dataset statistics"""
    print("\n" + "="*50)
    print("📈 BASIC DATASET STATISTICS")
    print("="*50)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn Information:")
    print("-" * 30)
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        print(f"{col:20} | Non-null: {non_null:5} | Null: {null_count:5}")
    
    print("\nFirst few rows:")
    print("-" * 30)
    display_cols = ['sample_name', 'medical_specialty', 'description']
    for col in display_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].head(3).tolist())

def analyze_target_variable(df):
    """Analyze the target variable (sample_name) distribution"""
    print("\n" + "="*50)
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("="*50)
    
    target_col = 'sample_name'
    
    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found!")
        return
    
    # Basic target statistics
    print(f"Unique sample names: {df[target_col].nunique()}")
    print(f"Total samples: {len(df)}")
    
    # Top 15 most common sample names
    print("\nTop 15 Most Common Sample Names:")
    print("-" * 40)
    top_samples = df[target_col].value_counts().head(15)
    for name, count in top_samples.items():
        percentage = (count / len(df)) * 100
        print(f"{name[:35]:35} | {count:4} ({percentage:5.1f}%)")
    
    # Plot target distribution
    plt.figure(figsize=(15, 8))
    
    # Top 20 sample names
    top_20 = df[target_col].value_counts().head(20)
    
    plt.barh(range(len(top_20)), top_20.values)
    plt.yticks(range(len(top_20)), [name[:30] + '...' if len(name) > 30 else name 
                                    for name in top_20.index])
    plt.xlabel('Number of Samples')
    plt.title('Top 20 Sample Names Distribution')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('eda_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Class imbalance analysis
    print(f"\nClass Imbalance Analysis:")
    print("-" * 25)
    print(f"Most common class: {top_samples.iloc[0]} samples ({top_samples.iloc[0]/len(df)*100:.1f}%)")
    print(f"Least common class: {df[target_col].value_counts().iloc[-1]} samples")
    print(f"Balance ratio (max/min): {top_samples.iloc[0]/df[target_col].value_counts().iloc[-1]:.1f}")
    
    return top_samples

def analyze_text_features(df):
    """Analyze text characteristics"""
    print("\n" + "="*50)
    print("📝 TEXT CHARACTERISTICS ANALYSIS")
    print("="*50)
    
    text_cols = ['description', 'transcription']
    
    for col in text_cols:
        if col not in df.columns:
            continue
            
        print(f"\n{col.upper()} Analysis:")
        print("-" * 30)
        
        # Remove null values for analysis
        texts = df[col].dropna()
        
        if len(texts) == 0:
            print(f"❌ No valid text data in {col}")
            continue
        
        # Text length statistics
        text_lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        print(f"Text Length (characters):")
        print(f"  Mean: {text_lengths.mean():.0f}")
        print(f"  Median: {text_lengths.median():.0f}")
        print(f"  Std: {text_lengths.std():.0f}")
        print(f"  Min: {text_lengths.min()}")
        print(f"  Max: {text_lengths.max()}")
        
        print(f"\nWord Count:")
        print(f"  Mean: {word_counts.mean():.0f}")
        print(f"  Median: {word_counts.median():.0f}")
        print(f"  Std: {word_counts.std():.0f}")
        print(f"  Min: {word_counts.min()}")
        print(f"  Max: {word_counts.max()}")
        
        # Plot distributions
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.title(f'{col} - Text Length Distribution')
        
        plt.subplot(1, 2, 2)
        plt.hist(word_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title(f'{col} - Word Count Distribution')
        
        plt.tight_layout()
        plt.savefig(f'eda_{col}_lengths.png', dpi=300, bbox_inches='tight')
        plt.show()

def analyze_medical_specialties(df):
    """Analyze medical specialties"""
    print("\n" + "="*50)
    print("🏥 MEDICAL SPECIALTIES ANALYSIS")
    print("="*50)
    
    specialty_col = 'medical_specialty'
    
    if specialty_col not in df.columns:
        print(f"❌ Column '{specialty_col}' not found!")
        return
    
    print(f"Unique medical specialties: {df[specialty_col].nunique()}")
    
    # Top specialties
    print("\nTop 15 Medical Specialties:")
    print("-" * 35)
    top_specialties = df[specialty_col].value_counts().head(15)
    for specialty, count in top_specialties.items():
        percentage = (count / len(df)) * 100
        print(f"{specialty[:30]:30} | {count:4} ({percentage:5.1f}%)")
    
    # Plot specialty distribution
    plt.figure(figsize=(12, 8))
    top_specialties.plot(kind='barh')
    plt.xlabel('Number of Samples')
    plt.title('Top Medical Specialties Distribution')
    plt.tight_layout()
    plt.savefig('eda_specialties_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def data_quality_assessment(df):
    """Assess data quality issues"""
    print("\n" + "="*50)
    print("🔍 DATA QUALITY ASSESSMENT")
    print("="*50)
    
    # Missing values analysis
    print("Missing Values Analysis:")
    print("-" * 25)
    missing_stats = df.isnull().sum()
    missing_pct = (missing_stats / len(df)) * 100
    
    for col in df.columns:
        if missing_stats[col] > 0:
            print(f"{col:20} | Missing: {missing_stats[col]:4} ({missing_pct[col]:5.1f}%)")
    
    # Duplicate analysis
    print(f"\nDuplicate Records:")
    print("-" * 20)
    duplicates = df.duplicated().sum()
    print(f"Total duplicates: {duplicates}")
    
    if 'transcription' in df.columns:
        text_duplicates = df['transcription'].duplicated().sum()
        print(f"Duplicate transcriptions: {text_duplicates}")
    
    # Sample name consistency
    if 'sample_name' in df.columns:
        print(f"\nSample Name Consistency:")
        print("-" * 25)
        sample_names = df['sample_name'].dropna()
        print(f"Unique sample names: {sample_names.nunique()}")
        print(f"Average samples per class: {len(sample_names) / sample_names.nunique():.1f}")
        
        # Check for potential naming inconsistencies
        names_lower = sample_names.str.lower().value_counts()
        original_names = sample_names.value_counts()
        
        if len(names_lower) < len(original_names):
            print(f"⚠️  Potential case inconsistencies detected!")
            print(f"   Original unique names: {len(original_names)}")
            print(f"   Lowercase unique names: {len(names_lower)}")

def create_wordcloud_analysis(df):
    """Create word cloud analysis for text data"""
    print("\n" + "="*50)
    print("☁️  WORD CLOUD ANALYSIS")
    print("="*50)
    
    if 'transcription' not in df.columns:
        print("❌ No transcription data available for word cloud")
        return
    
    # Combine all transcriptions
    all_text = ' '.join(df['transcription'].dropna().astype(str))
    
    # Clean text for word cloud
    clean_text = re.sub(r'[^\w\s]', ' ', all_text.lower())
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         max_words=100,
                         collocations=False).generate(clean_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Medical Transcriptions', fontsize=16)
    plt.tight_layout()
    plt.savefig('eda_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Word cloud saved as 'eda_wordcloud.png'")

def main():
    """Main EDA pipeline"""
    print("🔬 MEDICAL TEXT CLASSIFICATION - EDA")
    print("="*50)
    
    # Load data
    df = load_data('../data/mtsamples.csv')
    
    # Run analysis pipeline
    basic_statistics(df)
    top_samples = analyze_target_variable(df)
    analyze_text_features(df)
    analyze_medical_specialties(df)
    data_quality_assessment(df)
    create_wordcloud_analysis(df)
    
    # Summary insights
    print("\n" + "="*50)
    print("📋 KEY INSIGHTS & RECOMMENDATIONS")
    print("="*50)
    
    print("✅ Dataset Quality:")
    print("   - Clean dataset with minimal missing values")
    print("   - Text data is rich and well-structured")
    
    print("\n✅ Classification Task:")
    print("   - Multi-class classification problem")
    print(f"   - {df['sample_name'].nunique()} unique sample types")
    print("   - Some class imbalance present")
    
    print("\n✅ Feature Engineering Opportunities:")
    print("   - Text length as feature")
    print("   - Medical specialty as categorical feature")
    print("   - TF-IDF on transcription text")
    print("   - Keyword extraction potential")
    
    print("\n✅ Model Strategy:")
    print("   - Start with Logistic Regression on TF-IDF features")
    print("   - Handle class imbalance with stratified sampling")
    print("   - Cross-validation with stratification")
    
    print("\n🎯 Next Steps:")
    print("   1. Implement text preprocessing pipeline")
    print("   2. Create TF-IDF features")
    print("   3. Train baseline classification model")
    print("   4. Evaluate model performance")
    
    print("\n✅ EDA Complete! Ready for preprocessing phase.")

if __name__ == "__main__":
    main() 