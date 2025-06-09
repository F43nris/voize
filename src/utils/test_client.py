#!/usr/bin/env python3
"""
Simple test client for the Medical Document Classifier API
"""

import json
import sys
from typing import Any, Dict

import requests

API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_prediction(text: str):
    """Test prediction endpoint with given text"""
    print(f"\nTesting prediction with text: '{text[:50]}...'")
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json={"text": text})
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {data['prediction']}")
            print(f"Confidence: {data['confidence']:.4f}")
            print("Top predictions:")
            for i, pred in enumerate(data["top_predictions"][:3]):
                print(f"  {i+1}. {pred['class']}: {pred['confidence']:.4f}")
        else:
            print(f"Error: {response.text}")

        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Main function to run tests"""
    print("Medical Document Classifier API Test Client")
    print("=" * 50)

    # Test health first
    if not test_health():
        print("Health check failed. Is the API running?")
        sys.exit(1)

    # Sample medical texts for testing
    sample_texts = [
        "Patient presents with chest pain and shortness of breath. ECG shows normal sinus rhythm. Troponin levels elevated.",
        "Surgical procedure performed under general anesthesia. Incision made in the right lower quadrant. Appendix removed without complications.",
        "Patient has history of diabetes mellitus type 2. Blood glucose levels elevated at 250 mg/dL. Insulin therapy adjusted.",
        "Radiological findings show fracture of the left femur. No displacement noted. Patient stable for orthopedic consultation.",
        "Consultation requested for cardiology evaluation. Patient complains of palpitations and irregular heartbeat. Holter monitor recommended.",
        "Laboratory results show elevated white blood cell count. Patient presents with fever and malaise. Antibiotic therapy initiated.",
        "Endoscopy performed revealing gastric ulcer. H. pylori testing positive. Triple therapy prescribed for eradication.",
        "Patient underwent MRI of the brain. No acute findings. Headache workup negative for structural abnormalities.",
    ]

    # Test predictions
    print(f"\nTesting predictions with {len(sample_texts)} sample texts...")

    success_count = 0
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- Test {i}/{len(sample_texts)} ---")
        if test_prediction(text):
            success_count += 1

    print(f"\n" + "=" * 50)
    print(f"Results: {success_count}/{len(sample_texts)} tests passed")

    # Interactive mode
    print("\nEnter 'interactive' to test with your own text, or 'quit' to exit:")

    while True:
        user_input = input("\n> ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            break
        elif user_input.lower() == "interactive":
            print(
                "\nInteractive mode - enter medical text to classify (or 'back' to return):"
            )
            while True:
                text = input("Medical text: ").strip()
                if text.lower() == "back":
                    break
                elif text:
                    test_prediction(text)
                else:
                    print("Please enter some text.")
            print("Returning to main menu...")
        elif user_input:
            test_prediction(user_input)


if __name__ == "__main__":
    main()
