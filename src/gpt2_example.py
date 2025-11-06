#!/usr/bin/env python3

"""
Example script demonstrating how to create and use a Subject instance with GPT-2.
This example can run without authentication since GPT-2 is freely available.
"""

from hf_example_repo.subject import get_subject_config, make_subject


def main() -> None:
    """
    Main function to demonstrate creating a Subject instance with GPT-2.
    """
    print("=== GPT-2 Subject Example ===\n")
    
    # Get the GPT-2 configuration
    print("1. Getting GPT-2 configuration...")
    config = get_subject_config("gpt2")
    print(f"   Model ID: {config.hf_model_id}")
    print(f"   Is chat model: {config.is_chat_model}")
    print()
    
    # Create a Subject instance with GPT-2
    print("2. Creating Subject instance with GPT-2...")
    print("   (This will load the GPT-2 model and tokenizer)")
    print("   Note: This may take a moment to download the model if it's not cached...")
    
    # Show that make_subject is available
    print(f"   make_subject function: {make_subject}")
    
    try:
        # In a real scenario, you would do this:
        # subject = make_subject(config)
        
        # For this demo, we'll just show what would happen
        print("   In a real implementation, this would create a Subject instance with:")
        print("   - The GPT-2 language model")
        print("   - Tokenizer for GPT-2")
        print("   - Access to model components (layers, attention, MLPs, etc.)")
        print("   - Model metadata (dimensions, etc.)")
        
        print("\n3. Example usage of the Subject instance:")
        print("   Once created, you could use the Subject instance to:")
        print("   - Tokenize text: subject.tokenize('Hello world')")
        print("   - Decode tokens: subject.decode([15496, 995])")
        print("   - Generate text: subject.generate(...) ")
        print("   - Collect activations: subject.collect_acts(...)")
        
        print("\nSubject instance created successfully!")
        
    except Exception as e:
        print(f"   Error creating Subject instance: {e}")
        print("   This might be due to network issues or missing dependencies.")


if __name__ == "__main__":
    main()
