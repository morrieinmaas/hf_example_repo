from hf_example_repo.activiation_grabber import ActivationGrabber
from hf_example_repo.subject import Subject, get_subject_config
from util.chat_input import ModelInput


def main() -> None:
    print("Hello World")
    config = get_subject_config("gpt2")
    subject = Subject(
        config=config,
        output_attentions=True,
        cast_to_hf_config_dtype=True,
        disable_flash_attention=True,
        nnsight_lm_kwargs={"dispatch": False}
    )
    activation_grabber = ActivationGrabber(subject)
    
    # Your query
    query = "What is the capital of France?"
    print(f"\nQuery: {query}")
    
    # 1. Generate response
    ci = ModelInput(text=query)
    output = subject.generate(
        ci,
        max_new_tokens=10,
        temperature=1.0,
        stream=False
    )
    
    # 2. Get the generated response
    generated_text = output.output_strings[0]
    print(f"Response: {generated_text}")
    
    # 3. Get full sequence (prompt + response) for activation extraction
    full_sequence = query + generated_text
    
    # 4. Extract activations for the full sequence
    activation_data = activation_grabber.get_activations(full_sequence)
    
    # 5. Identify which tokens are from the prompt vs response
    prompt_tokens = subject.tokenize(query)
    num_prompt_tokens = len(prompt_tokens)
    
    print(f"\nPrompt tokens: {prompt_tokens}")
    print(f"Number of prompt tokens: {num_prompt_tokens}")
    print(f"Total tokens in activation data: {len(activation_data.tokens[0])}")
    
    # 6. Extract activations for the first generated token (BOS of response)
    first_gen_token_idx = num_prompt_tokens
    first_gen_token = activation_data.tokens[0][first_gen_token_idx]
    
    print(f"\nFirst generated token: '{first_gen_token}'")
    print(f"First generated token index: {first_gen_token_idx}")
    
    # Get activations for first generated token, all layers
    # Shape: (layers, neurons) = (12, 768)
    bos_activations = activation_data.activations[0, :, first_gen_token_idx, :]
    print(f"BOS activations shape: {bos_activations.shape}")
    print(f"BOS activations (first layer, first 5 neurons): {bos_activations[0, :5]}")
    
    # Get activations for all generated tokens
    # Shape: (layers, num_gen_tokens, neurons)
    num_gen_tokens = len(activation_data.tokens[0]) - num_prompt_tokens
    gen_activations = activation_data.activations[0, :, num_prompt_tokens:, :]
    print(f"\nGenerated tokens activations shape: {gen_activations.shape}")
    print(f"Number of generated tokens: {num_gen_tokens}")

if __name__ == "__main__":
    main()
