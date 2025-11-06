#!/usr/bin/env python3

"""
Working example script demonstrating how to create and use a Subject instance with GPT-2.
This example can run without authentication since GPT-2 is freely available.
Note: This will actually load the GPT-2 model, which requires internet access for the first run.
"""

from hf_example_repo.subject import get_subject_config, make_subject


def main() -> None:
    """
    Main function to demonstrate creating a Subject instance with GPT-2.
    """
    print("=== Working GPT-2 Subject Example ===\n")
    
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
    
    try:
        # Create the Subject instance
        # We're using dispatch=False to avoid some potential issues with nnsight
        # We're also disabling FlashAttention2 to avoid dependency issues
        subject = make_subject(config, dispatch=False, disable_flash_attention=True)
        
        print("   Subject instance created successfully!")
        print(f"   Model name: {subject.model_name}")
        print(f"   Model dimensions - I: {subject.I}, D: {subject.D}, V: {subject.V}, L: {subject.L}")
        
        # Demonstrate some basic functionality
        print("\n3. Demonstrating basic functionality:")
        
        # Tokenize some text
        text = "Hello, world!"
        tokens = subject.tokenize(text)
        print(f"   Tokenizing '{text}': {tokens}")
        
        # Decode tokens back to text
        decoded = subject.decode(tokens)
        print(f"   Decoding tokens: '{decoded}'")
        
        # Get the pad token ID
        pad_token_id = subject.pad_token_id
        print(f"   Pad token ID: {pad_token_id}")
        
        print("\nWorking GPT-2 Subject example completed successfully!")
        
    except Exception as e:
        print(f"   Error creating Subject instance: {e}")
        print("   This might be due to network issues, missing dependencies, or model loading problems.")
        print("   For a demo without actually loading the model, try the hf_gpt2_example script.")


if __name__ == "__main__":
    main()
