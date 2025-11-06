#!/usr/bin/env python3

"""
Example script demonstrating how to use the ActivationGrabber with a Subject instance.
"""

from hf_example_repo.subject import get_subject_config, make_subject
from hf_example_repo.activiation_grabber import ActivationGrabber, ActivationData, ActivationConfig


def main() -> None:
    """
    Main function to demonstrate using ActivationGrabber with a Subject instance.
    """
    print("=== ActivationGrabber Example ===\n")

    # Get the GPT-2 configuration
    print("1. Getting GPT-2 configuration...")
    config = get_subject_config("gpt2")
    print(f"   Model ID: {config.hf_model_id}")
    print()

    # Create a Subject instance with GPT-2
    print("2. Creating Subject instance with GPT-2...")
    print("   (This will load the GPT-2 model and tokenizer)")
    subject = make_subject(config, dispatch=False, disable_flash_attention=True)
    print("   Subject instance created successfully!")
    print()

    # Create an ActivationGrabber instance
    print("3. Creating ActivationGrabber instance...")
    grabber = ActivationGrabber(subject)
    print(f"   ActivationGrabber: {grabber}")
    print(f"   Subject model name: {grabber.subject.model_name}")
    print()

    # Get activations for a sample input
    print("4. Getting activations for sample input...")
    input_text = "Hello, world!"
    print(f"   Input text: '{input_text}'")

    # Create activation configuration
    activation_config = ActivationConfig(
        layers=[0, subject.L - 1],  # First and last layer
        return_numpy=True,
    )
    print(f"   Layers: {activation_config.layers}")

    try:
        # Get activations
        activation_data: ActivationData = grabber.get_activations(input_sequence=input_text, config=activation_config)

        print("   Activations retrieved successfully!")
        print(f"   Activation data: {activation_data}")

        # Print details about the activation data
        print("\n5. Activation data details:")
        print(f"   Batch size: {len(activation_data.tokens)}")
        print(f"   Layers: {activation_data.layers}")
        print(f"   Tokens: {activation_data.tokens}")
        print(f"   Token IDs: {activation_data.token_ids}")
        print(f"   Activations shape: {activation_data.activations.shape}")
        print(f"   Attention mask shape: {activation_data.attention_mask.shape}")

        # Show some activation values
        print("\n6. Sample activation values:")
        batch_idx = 0
        layer_idx = 0
        token_idx = 0

        # Get a few neuron activations for the first token of the first layer
        if activation_data.activations.ndim >= 4:
            sample_activations = activation_data.activations[batch_idx, layer_idx, token_idx, :5]
            print(
                f"   First 5 activations for batch 0, layer {activation_data.layers[layer_idx]}, token '{activation_data.tokens[batch_idx][token_idx]}':"
            )
            print(f"   {sample_activations}")

        print("\nActivationGrabber example completed successfully!")

    except Exception as e:
        print(f"   Error getting activations: {e}")
        print("   This might be due to model-specific issues or implementation details.")


if __name__ == "__main__":
    main()
