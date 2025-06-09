import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class MedicalTextFeatureEngine:
    """Configurable feature engineering for medical text classification"""

    def __init__(self, target_col="medical_specialty"):
        self.target_col = target_col
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.sample_name_encoder = None
        self.feature_names = []

    def analyze_text_artifacts(self, df, text_col="transcription", sample_size=100):
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
            "dates": r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b|\b\w+\s+\d{1,2},?\s+\d{4}\b",
            "times": r"\b\d{1,2}:\d{2}\b",
            "doctor_refs": r"\b(dr|doctor|physician|md|m\.d\.)\b",
            "patient_refs": r"\bpatient\b",
            "hospital_refs": r"\bhospital\b",
            "date_labels": r"\b(date|time)\s*[:\-]",
            "phone_numbers": r"\b\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4}\b",
            "medical_ids": r"\b[A-Z0-9]{3,}\b",  # Potential IDs
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
                "count": count,
                "examples": list(set(matches))[:5],  # Top 5 unique examples
            }

            if count > 0:
                print(f"  {pattern_name}: {count} matches")
                if matches:
                    examples = ", ".join(matches[:3])
                    print(f"    Examples: {examples}")
            else:
                print(f"  {pattern_name}: No matches found")

        return findings

    def preprocess_text(
        self, df, text_cols=["transcription", "description"], conservative_cleaning=True
    ):
        """Step 1: Data-driven text preprocessing"""
        print("\n📝 STEP 1: Text Preprocessing")
        print("=" * 35)

        # First analyze what artifacts exist
        if "transcription" in df.columns:
            artifacts = self.analyze_text_artifacts(df, "transcription")
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
                if artifacts.get("date_labels", {}).get("count", 0) > 0:
                    text = re.sub(r"\b(date|time)\s*[:\-]\s*[^\s,]+", "", text)

                # Remove phone numbers if found
                if artifacts.get("phone_numbers", {}).get("count", 0) > 0:
                    text = re.sub(r"\b\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4}\b", "", text)

                # Normalize common medical abbreviations (these are definitely in medical text)
                medical_abbrevs = {
                    r"\bp\.?o\.?\b": "oral",
                    r"\bi\.?v\.?\b": "intravenous",
                    r"\bq\.?i\.?d\.?\b": "daily",
                    r"\bb\.?i\.?d\.?\b": "twice daily",
                    r"\bt\.?i\.?d\.?\b": "three times daily",
                    r"\bprn\b": "as needed",
                    r"\bstat\b": "immediately",
                }

                for abbrev, full in medical_abbrevs.items():
                    text = re.sub(abbrev, full, text)

            else:
                # More aggressive cleaning (original approach)
                text = re.sub(
                    r"\b(date|time|patient|dr|md|hospital)\s*[:\-]?\s*\S+", "", text
                )

            # Remove punctuation but keep medical decimals (e.g., "2.5 mg")
            text = re.sub(r"[^\w\s\.]", " ", text)
            text = re.sub(
                r"\.(?!\d)", " ", text
            )  # Remove periods not followed by digits

            # Remove extra whitespace
            text = " ".join(text.split())

            return text

        # Process each text column
        for col in text_cols:
            if col in df_processed.columns:
                print(f"Processing {col}...")
                df_processed[f"{col}_cleaned"] = df_processed[col].apply(
                    clean_medical_text
                )

        # Create combined text AFTER processing all columns
        if (
            "transcription_cleaned" in df_processed.columns
            and "description_cleaned" in df_processed.columns
        ):
            print("Creating combined text...")
            df_processed["combined_text"] = (
                df_processed["description_cleaned"].fillna("")
                + " "
                + df_processed["transcription_cleaned"].fillna("")
            ).str.strip()

        print(f"✅ Text preprocessing complete")
        return df_processed

    def transform(self, texts):
        """Transform new text data using the fitted feature engine"""
        if not isinstance(texts, list):
            texts = [texts]

        # Apply the same preprocessing as during training
        processed_texts = []
        for text in texts:
            if not isinstance(text, str):
                text = ""

            # Convert to lowercase
            text = text.lower()

            # Apply same cleaning as during training
            # Remove punctuation but keep medical decimals
            text = re.sub(r"[^\w\s\.]", " ", text)
            text = re.sub(
                r"\.(?!\d)", " ", text
            )  # Remove periods not followed by digits

            # Remove extra whitespace
            text = " ".join(text.split())

            processed_texts.append(text)

        # Transform using the fitted TF-IDF vectorizer
        if self.tfidf_vectorizer is not None:
            tfidf_features = self.tfidf_vectorizer.transform(processed_texts)

            # Create additional features to match training pipeline
            additional_features = []

            for text in texts:
                # Length features (matching training pipeline)
                char_length = len(str(text)) if text else 0
                word_count = len(str(text).split()) if text else 0

                # Since we don't have separate transcription/description in inference,
                # we'll use the same text for both and set reasonable defaults
                transcription_char_length = char_length
                transcription_word_count = word_count
                description_char_length = (
                    0  # Default since we don't have separate description
                )
                description_word_count = (
                    0  # Default since we don't have separate description
                )
                text_length_ratio = transcription_char_length / (
                    description_char_length + 1
                )

                # Length category feature (based on training pipeline)
                # Bins: [0, 1608, 4011, 8000, inf] -> [short, medium, long, very_long]
                if transcription_char_length <= 1608:
                    text_length_category = 0  # short
                elif transcription_char_length <= 4011:
                    text_length_category = 1  # medium
                elif transcription_char_length <= 8000:
                    text_length_category = 2  # long
                else:
                    text_length_category = 3  # very_long

                # Medical structure features
                text_lower = str(text).lower() if text else ""
                has_subjective = 1 if "subjective" in text_lower else 0
                has_objective = 1 if "objective" in text_lower else 0
                has_assessment = 1 if "assessment" in text_lower else 0
                has_plan = 1 if "plan" in text_lower else 0

                # Sample name encoded - default to 0 (unknown) for inference
                sample_name_encoded = 0

                # Combine all additional features
                sample_features = [
                    transcription_char_length,
                    transcription_word_count,
                    description_char_length,
                    description_word_count,
                    text_length_ratio,
                    text_length_category,
                    sample_name_encoded,
                    has_subjective,
                    has_objective,
                    has_assessment,
                    has_plan,
                ]

                additional_features.append(sample_features)

            # Convert to numpy array
            additional_features = np.array(additional_features)

            # Combine TF-IDF with additional features
            tfidf_dense = tfidf_features.toarray()
            combined_features = np.hstack([tfidf_dense, additional_features])

            return combined_features
        else:
            raise ValueError("Feature engine not fitted. TF-IDF vectorizer is None.")

    def get_feature_names(self):
        """Get the feature names from the TF-IDF vectorizer"""
        if self.tfidf_vectorizer is not None:
            return self.tfidf_vectorizer.get_feature_names_out()
        else:
            return []
