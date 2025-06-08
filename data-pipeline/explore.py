#!/usr/bin/env python3
"""
Medical Text Classification - Step-by-Step EDA
==============================================
Building EDA iteratively to understand the data properly.
Target: Predict medical_specialty from text
"""

import pandas as pd
import numpy as np
from collections import Counter

class MedicalTextEDA:
    """Configurable EDA class for medical text classification"""
    
    def __init__(self, data_path='data/mtsamples.csv', target_col='medical_specialty'):
        self.data_path = data_path
        self.target_col = target_col
        self.df = None
        self.analysis_results = {}
    
    def load_and_inspect_data(self):
        """Step 1: Load data and basic inspection"""
        print("🔍 STEP 1: Loading and Basic Inspection")
        print("=" * 40)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Target column: {self.target_col}")
        
        print("\nFirst few rows (basic info):")
        print(self.df.head(2))
        
        print("\nColumn data types:")
        print(self.df.dtypes)
        
        return self.df
    
    def examine_target_variable(self, top_n=10):
        """Step 2: Examine target variable"""
        print(f"\n🎯 STEP 2: Target Variable Analysis")
        print("=" * 40)
        
        print(f"Unique {self.target_col}: {self.df[self.target_col].nunique()}")
        print(f"Total records: {len(self.df)}")
        
        print(f"\nTop {top_n} most common {self.target_col}:")
        top_values = self.df[self.target_col].value_counts().head(top_n)
        for i, (name, count) in enumerate(top_values.items(), 1):
            pct = (count / len(self.df)) * 100
            print(f"{i:2}. {name[:35]:35} | {count:3} ({pct:4.1f}%)")
        
        # Store results
        self.analysis_results['target_distribution'] = top_values
        
        return top_values
    
    def check_data_quality(self):
        """Step 3: Data quality assessment"""
        print("\n🔍 STEP 3: Data Quality Check")
        print("=" * 40)
        
        # Missing values
        print("Missing values per column:")
        missing_analysis = {}
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            pct = (missing / len(self.df)) * 100
            missing_analysis[col] = {'count': missing, 'percentage': pct}
            status = f"{missing} ({pct:.1f}%)" if missing > 0 else "0 (0.0%)"
            print(f"  {col}: {status}")
        
        # Store results
        self.analysis_results['missing_data'] = missing_analysis
        
        print(f"\nDuplicate rows: {self.df.duplicated().sum()}")
        
        return missing_analysis
    
    def analyze_class_distribution(self, tier_thresholds=[100, 20, 5]):
        """Step 4: Analyze class distribution"""
        print("\n📊 STEP 4: Class Distribution Analysis")
        print("=" * 40)
        
        counts = self.df[self.target_col].value_counts()
        
        print(f"Total {self.target_col}: {len(counts)}")
        print(f"Total samples: {len(self.df)}")
        
        # Configurable tier analysis
        print(f"\nDistribution tiers:")
        tiers = {}
        
        # Create tiers based on thresholds
        for i, threshold in enumerate(tier_thresholds):
            if i == 0:
                tier_data = counts[counts >= threshold]
                tier_name = f"Tier {i+1} (≥{threshold} samples)"
            else:
                prev_threshold = tier_thresholds[i-1]
                tier_data = counts[(counts >= threshold) & (counts < prev_threshold)]
                tier_name = f"Tier {i+1} ({threshold}-{prev_threshold-1} samples)"
            
            tiers[f"tier_{i+1}"] = tier_data
            pct = (tier_data.sum() / len(self.df)) * 100
            print(f"  {tier_name}: {len(tier_data)} classes, {tier_data.sum()} samples ({pct:.1f}%)")
        
        # Last tier (below minimum threshold)
        min_threshold = min(tier_thresholds)
        tier_data = counts[counts < min_threshold]
        tiers['tier_rare'] = tier_data
        pct = (tier_data.sum() / len(self.df)) * 100
        print(f"  Tier Rare (<{min_threshold} samples): {len(tier_data)} classes, {tier_data.sum()} samples ({pct:.1f}%)")
        
        # Store results
        self.analysis_results['class_distribution'] = {
            'counts': counts,
            'tiers': tiers
        }
        
        return counts, tiers
    
    def analyze_text_characteristics(self, text_cols=['description', 'transcription']):
        """Step 5: Text characteristics analysis"""
        print("\n📝 STEP 5: Text Characteristics Analysis")
        print("=" * 40)
        
        text_stats = {}
        
        for col in text_cols:
            if col in self.df.columns:
                texts = self.df[col].dropna()
                char_lengths = texts.str.len()
                word_counts = texts.str.split().str.len()
                
                stats = {
                    'char_mean': char_lengths.mean(),
                    'char_median': char_lengths.median(),
                    'word_mean': word_counts.mean(),
                    'word_median': word_counts.median(),
                    'char_min': char_lengths.min(),
                    'char_max': char_lengths.max()
                }
                
                text_stats[col] = stats
                
                print(f"\n{col.capitalize()}:")
                print(f"  Avg: {stats['char_mean']:.0f} chars, {stats['word_mean']:.0f} words")
                print(f"  Range: {stats['char_min']}-{stats['char_max']} chars")
        
        # Store results
        self.analysis_results['text_characteristics'] = text_stats
        
        return text_stats
    
    def analyze_text_outliers(self, text_col='transcription'):
        """Step 6: Analyze text length outliers"""
        print("\n🔍 STEP 6: Text Length Outlier Analysis")
        print("=" * 40)
        
        if text_col not in self.df.columns:
            print(f"❌ Column '{text_col}' not found!")
            return
        
        texts = self.df[text_col].dropna()
        char_lengths = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        # Calculate outliers using IQR method
        def find_outliers(data, col_name):
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_low = data[data < lower_bound]
            outliers_high = data[data > upper_bound]
            
            print(f"\n{col_name} Outliers:")
            print(f"  IQR range: {q1:.0f} - {q3:.0f}")
            print(f"  Normal range: {lower_bound:.0f} - {upper_bound:.0f}")
            print(f"  Too short: {len(outliers_low)} samples (< {lower_bound:.0f})")
            print(f"  Too long: {len(outliers_high)} samples (> {upper_bound:.0f})")
            
            return outliers_low, outliers_high, lower_bound, upper_bound
        
        # Analyze character length outliers
        char_low, char_high, char_lower, char_upper = find_outliers(char_lengths, "Character Length")
        
        # Analyze word count outliers
        word_low, word_high, word_lower, word_upper = find_outliers(word_counts, "Word Count")
        
        # Show examples of extreme cases
        print(f"\n📝 Examples of Extreme Cases:")
        print("-" * 30)
        
        if len(char_high) > 0:
            longest_idx = char_lengths.idxmax()
            longest_text = texts.loc[longest_idx]
            specialty = self.df.loc[longest_idx, self.target_col]
            print(f"Longest text ({char_lengths.max()} chars, {specialty}):")
            print(f"  '{longest_text[:200]}...'")
        
        if len(char_low) > 0:
            shortest_idx = char_lengths.idxmin()
            shortest_text = texts.loc[shortest_idx]
            specialty = self.df.loc[shortest_idx, self.target_col]
            print(f"Shortest text ({char_lengths.min()} chars, {specialty}):")
            print(f"  '{shortest_text}'")
        
        # Data quality assessment
        print(f"\n⚠️  Data Quality Insights:")
        print("-" * 25)
        
        total_outliers = len(char_high) + len(char_low)
        outlier_pct = (total_outliers / len(texts)) * 100
        
        print(f"Total outliers: {total_outliers} ({outlier_pct:.1f}% of data)")
        
        if len(char_high) > len(texts) * 0.01:  # >1% very long
            print(f"⚠️  Many very long texts - check for data quality issues")
        
        if len(char_low) > len(texts) * 0.01:  # >1% very short  
            print(f"⚠️  Many very short texts - incomplete transcriptions?")
        
        if char_lengths.max() > char_lengths.median() * 10:  # 10x longer than median
            print(f"⚠️  Extreme length variation - consider preprocessing")
        
        # Store results
        outlier_analysis = {
            'char_outliers_high': len(char_high),
            'char_outliers_low': len(char_low), 
            'char_bounds': (char_lower, char_upper),
            'word_outliers_high': len(word_high),
            'word_outliers_low': len(word_low),
            'word_bounds': (word_lower, word_upper),
            'outlier_percentage': outlier_pct
        }
        
        self.analysis_results['outlier_analysis'] = outlier_analysis
        
        return outlier_analysis
    
    def analyze_text_quality(self, text_col='transcription'):
        """Step 7: Analyze text quality and potential issues"""
        print("\n🔍 STEP 7: Text Quality Analysis")
        print("=" * 40)
        
        if text_col not in self.df.columns:
            print(f"❌ Column '{text_col}' not found!")
            return
        
        texts = self.df[text_col].dropna()
        
        # Check for potential quality issues
        quality_issues = {}
        
        # 1. Non-ASCII characters
        non_ascii = texts.str.contains(r'[^\x00-\x7F]', regex=True).sum()
        quality_issues['non_ascii'] = non_ascii
        print(f"Texts with non-ASCII characters: {non_ascii} ({non_ascii/len(texts)*100:.1f}%)")
        
        # 2. HTML/XML tags
        html_tags = texts.str.contains(r'<[^>]+>', regex=True).sum()
        quality_issues['html_tags'] = html_tags
        print(f"Texts with HTML/XML tags: {html_tags} ({html_tags/len(texts)*100:.1f}%)")
        
        # 3. Multiple consecutive spaces/newlines
        excess_whitespace = texts.str.contains(r'\s{3,}', regex=True).sum()
        quality_issues['excess_whitespace'] = excess_whitespace
        print(f"Texts with excess whitespace: {excess_whitespace} ({excess_whitespace/len(texts)*100:.1f}%)")
        
        # 4. Very short texts (potential data quality issues)
        very_short = (texts.str.len() < 50).sum()
        quality_issues['very_short'] = very_short
        print(f"Very short texts (<50 chars): {very_short} ({very_short/len(texts)*100:.1f}%)")
        
        # 5. All caps texts (potential formatting issues)
        all_caps = texts.str.isupper().sum()
        quality_issues['all_caps'] = all_caps
        print(f"All caps texts: {all_caps} ({all_caps/len(texts)*100:.1f}%)")
        
        # 6. Medical report structure detection
        medical_sections = [
            ('SUBJECTIVE', texts.str.contains('SUBJECTIVE', case=False).sum()),
            ('OBJECTIVE', texts.str.contains('OBJECTIVE', case=False).sum()),
            ('ASSESSMENT', texts.str.contains('ASSESSMENT', case=False).sum()),
            ('PLAN', texts.str.contains('PLAN', case=False).sum()),
        ]
        
        print(f"\nMedical report structure patterns:")
        for section, count in medical_sections:
            pct = count / len(texts) * 100
            print(f"  {section}: {count} texts ({pct:.1f}%)")
        
        # Store results
        self.analysis_results['text_quality'] = quality_issues
        
        return quality_issues
    
    def analyze_vocabulary_richness(self, text_col='transcription', top_specialties=5):
        """Step 8: Analyze vocabulary richness and specialty distinctiveness"""
        print(f"\n📚 STEP 8: Vocabulary Richness Analysis")
        print("=" * 40)
        
        if text_col not in self.df.columns:
            print(f"❌ Column '{text_col}' not found!")
            return
        
        # Focus on top specialties for analysis
        top_specs = self.df[self.target_col].value_counts().head(top_specialties).index
        
        vocab_analysis = {}
        
        print(f"Analyzing vocabulary for top {top_specialties} specialties:")
        
        for specialty in top_specs:
            specialty_data = self.df[self.df[self.target_col] == specialty]
            all_text = ' '.join(specialty_data[text_col].dropna().astype(str))
            
            # Basic text processing
            words = all_text.lower().split()
            unique_words = set(words)
            
            # Calculate vocabulary richness
            total_words = len(words)
            unique_count = len(unique_words)
            richness_ratio = unique_count / total_words if total_words > 0 else 0
            
            vocab_analysis[specialty] = {
                'total_words': total_words,
                'unique_words': unique_count,
                'richness_ratio': richness_ratio
            }
            
            print(f"\n{specialty}:")
            print(f"  Total words: {total_words:,}")
            print(f"  Unique words: {unique_count:,}")
            print(f"  Vocabulary richness: {richness_ratio:.3f}")
        
        # Analyze vocabulary overlap between specialties
        if len(top_specs) >= 2:
            print(f"\nVocabulary overlap analysis:")
            
            spec1, spec2 = top_specs[0], top_specs[1]
            
            # Get vocabulary for each specialty
            text1 = ' '.join(self.df[self.df[self.target_col] == spec1][text_col].dropna().astype(str))
            text2 = ' '.join(self.df[self.df[self.target_col] == spec2][text_col].dropna().astype(str))
            
            vocab1 = set(text1.lower().split())
            vocab2 = set(text2.lower().split())
            
            overlap = vocab1.intersection(vocab2)
            overlap_pct = len(overlap) / len(vocab1.union(vocab2)) * 100
            
            print(f"  {spec1} ↔ {spec2}:")
            print(f"    Shared vocabulary: {len(overlap):,} words")
            print(f"    Overlap percentage: {overlap_pct:.1f}%")
            
            # Show some unique words for each specialty
            unique1 = vocab1 - vocab2
            unique2 = vocab2 - vocab1
            
            # Filter for meaningful words (length > 3, not common words)
            common_words = {'the', 'and', 'that', 'this', 'with', 'from', 'they', 'have', 'were', 'been', 'their'}
            unique1_filtered = [w for w in unique1 if len(w) > 3 and w not in common_words][:10]
            unique2_filtered = [w for w in unique2 if len(w) > 3 and w not in common_words][:10]
            
            print(f"    Unique to {spec1}: {', '.join(unique1_filtered[:5])}")
            print(f"    Unique to {spec2}: {', '.join(unique2_filtered[:5])}")
        
        # Store results
        self.analysis_results['vocabulary_richness'] = vocab_analysis
        
        return vocab_analysis
    
    def run_full_analysis(self, top_n=10, tier_thresholds=[100, 20, 5]):
        """Run complete EDA pipeline"""
        print("🔬 MEDICAL TEXT CLASSIFICATION - EDA")
        print("=" * 50)
        
        self.load_and_inspect_data()
        self.examine_target_variable(top_n=top_n)
        self.check_data_quality()
        self.analyze_class_distribution(tier_thresholds=tier_thresholds)
        self.analyze_text_characteristics()
        self.analyze_text_outliers()
        self.analyze_text_quality()           # New: Text quality analysis
        self.analyze_vocabulary_richness()    # New: Vocabulary analysis
        
        # Store the dataframe in results for access
        self.analysis_results['df'] = self.df
        
        return self.analysis_results

    def summarize_findings(self):
        print("📊 DATA SUMMARY:")
        print(f"  Classes: {self.df[self.target_col].nunique()}")
        print(f"  Text length: {self.analysis_results['text_characteristics']['transcription']['char_mean']:.0f} chars avg")
        print(f"  Class balance: {self.analysis_results['class_distribution']['tiers']['tier_1'].sum() / len(self.df) * 100:.1f}% in top tier")

# Backwards compatibility functions
def load_and_inspect_data():
    eda = MedicalTextEDA()
    return eda.load_and_inspect_data()

def analyze_specialty_language_patterns(df):
    """Step 9: Analyze language patterns by medical specialty"""
    print("\n🏥 STEP 9: Specialty-Specific Language Analysis")
    print("=" * 45)
    
    # Focus on top 5 specialties for detailed analysis
    top_5_specialties = df['medical_specialty'].value_counts().head(5).index
    
    print("Analyzing top 5 specialties for language patterns...")
    
    for specialty in top_5_specialties:
        print(f"\n{specialty}:")
        print("-" * len(specialty))
        
        specialty_data = df[df['medical_specialty'] == specialty]
        
        if 'transcription' in df.columns:
            # Combine all transcriptions for this specialty
            all_text = ' '.join(specialty_data['transcription'].dropna().astype(str))
            
            # Simple word frequency analysis
            words = all_text.lower().split()
            
            # Filter out common stop words and short words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'a', 'an'}
            filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
            
            # Count word frequency
            word_counts = Counter(filtered_words)
            
            # Show top 10 words for this specialty
            print(f"  Common terms:")
            for word, count in word_counts.most_common(10):
                print(f"    {word}: {count}")
            
            print(f"  Sample count: {len(specialty_data)}")
            print(f"  Avg text length: {specialty_data['transcription'].str.len().mean():.0f} chars")

def analyze_feature_engineering_opportunities(df):
    """Step 10: Available features for model training"""
    print("\n🔧 STEP 10: Available Features")
    print("=" * 30)
    
    print("Text features found in dataset:")
    
    # Text-based features
    if 'description' in df.columns:
        print("   ✅ Description text")
    if 'transcription' in df.columns:
        print("   ✅ Transcription text")
    
    # Categorical features
    if 'sample_name' in df.columns:
        print("   ✅ Sample name categories")
    
    # Keywords
    if 'keywords' in df.columns:
        missing_kw = df['keywords'].isnull().sum()
        pct_missing = (missing_kw / len(df)) * 100
        if pct_missing < 50:
            print("   ✅ Keywords (some missing values)")
        else:
            print("   ⚠️  Keywords (many missing values)")
    
    print(f"\nData ready for preprocessing and feature engineering.")

if __name__ == "__main__":
    # Run with configurable parameters
    eda = MedicalTextEDA(
        data_path='data/mtsamples.csv',
        target_col='medical_specialty'
    )
    
    results = eda.run_full_analysis(
        top_n=15,  # Show top 15 specialties
        tier_thresholds=[100, 20, 5]  # Configurable tier analysis
    )
    
    # Access results properly
    counts, tiers = results['class_distribution']['counts'], results['class_distribution']['tiers']
    text_stats = results['text_characteristics']
    df = results['df']  # Now properly stored
    
    # Run additional analysis with the dataframe
    analyze_specialty_language_patterns(df)
    analyze_feature_engineering_opportunities(df)

    # Summarize findings
    eda.summarize_findings() 