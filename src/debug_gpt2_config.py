#!/usr/bin/env python3

"""
Debug script to check GPT-2 configuration attributes.
"""

from transformers import AutoConfig


def main() -> None:
    """
    Main function to debug GPT-2 configuration.
    """
    print("=== Debugging GPT-2 Configuration ===\n")

    # Load the GPT-2 configuration
    print("1. Loading GPT-2 configuration...")
    config = AutoConfig.from_pretrained("gpt2")
    print(f"   Config type: {type(config)}")
    print()

    # Check available attributes
    print("2. Checking available attributes...")
    attributes = ["n_inner", "n_embd", "vocab_size", "n_layer", "n_head"]
    for attr in attributes:
        value = config.__dict__.get(attr, "NOT FOUND")
        print(f"   {attr}: {value}")
    print()

    # Try to convert to int
    print("3. Trying to convert attributes to int...")
    for attr in attributes:
        value = config.__dict__.get(attr, "NOT FOUND")
        try:
            int_value = int(value) if value != "NOT FOUND" else "NOT FOUND"
            print(f"   {attr}: {value} -> {int_value}")
        except Exception as e:
            print(f"   {attr}: {value} -> ERROR: {e}")
    print()

    print("Debug completed!")


if __name__ == "__main__":
    main()
