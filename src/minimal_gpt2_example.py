#!/usr/bin/env python3

"""
Minimal example script demonstrating how to create and use a Subject instance with GPT-2.
This example shows the structure without actually loading the full model.
"""

from hf_example_repo.subject import GPT2Config, Subject, get_subject_config, make_subject


def main() -> None:
    """
    Main function to demonstrate creating a Subject instance with GPT-2.
    """
    print("=== Minimal GPT-2 Subject Example ===\n")

    # Get the GPT-2 configuration
    print("1. Getting GPT-2 configuration...")
    config = get_subject_config("gpt2")
    print(f"   Model ID: {config.hf_model_id}")
    print(f"   Is chat model: {config.is_chat_model}")
    print(f"   Config class: {config.__class__.__name__}")
    print()

    # Show available classes and functions
    print("2. Available classes and functions:")
    print(f"   Subject class: {Subject}")
    print(f"   GPT2Config class: {GPT2Config}")
    print(f"   make_subject function: {make_subject}")
    print(f"   get_subject_config function: {get_subject_config}")
    print()

    # Show how to create a Subject instance (conceptually)
    print("3. How to create a Subject instance:")
    print("   To create a Subject instance with GPT-2, you would call:")
    print("   subject = make_subject(config)")
    print("   where config is the GPT2Config we retrieved above.")
    print()

    # Show what a Subject instance would provide
    print("4. What a Subject instance provides:")
    print("   A Subject instance would give you access to:")
    print("   - Model components (tokenizer, layers, attention, MLPs, etc.)")
    print("   - Model metadata (dimensions, etc.)")
    print("   - Tokenization and generation methods")
    print("   - Activation collection methods")
    print()

    print("Minimal GPT-2 Subject example completed successfully!")


if __name__ == "__main__":
    main()
