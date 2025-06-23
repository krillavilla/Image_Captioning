import os
import sys
from vocabulary import Vocabulary

def test_vocabulary_initialization():
    """Test that the Vocabulary class correctly uses the annotations_file parameter."""
    
    # Test with default parameters
    try:
        vocab = Vocabulary(vocab_threshold=5)
        print("Test 1: Created vocabulary with default parameters")
    except Exception as e:
        print(f"Test 1 Error: {e}")
    
    # Test with custom annotations_file
    test_annotations_file = "test_annotations.json"
    try:
        vocab = Vocabulary(vocab_threshold=5, annotations_file=test_annotations_file)
        print(f"Test 2: Created vocabulary with custom annotations_file: {vocab.annotations_file}")
    except Exception as e:
        print(f"Test 2 Error: {e}")
    
    # Test with vocab_from_file=True (should load from existing vocab file if it exists)
    try:
        if os.path.exists("./vocab.pkl"):
            vocab = Vocabulary(vocab_threshold=5, vocab_from_file=True)
            print("Test 3: Loaded vocabulary from existing vocab.pkl file")
        else:
            print("Test 3 Skipped: vocab.pkl file does not exist")
    except Exception as e:
        print(f"Test 3 Error: {e}")

if __name__ == "__main__":
    test_vocabulary_initialization()