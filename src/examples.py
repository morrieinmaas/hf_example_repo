#!/usr/bin/env python3

"""
Example script demonstrating how to instantiate and use Subject objects.
"""

# Import the necessary modules
try:
    from hf_example_repo.subject import Subject, LMConfig, llama3_8B_config, get_subject_config, make_subject, gpt2_config
    print("Successfully imported Subject and related classes")
    
    def show_config_info(config: LMConfig) -> None:
        """
        Display information about a configuration.
        """
        print(f"Model ID: {config.hf_model_id}")
        print(f"Is chat model: {config.is_chat_model}")
        print(f"Unembed module: {config.unembed_module_str}")
        print(f"Layer template: {config.layer_module_template}")
        print()
    
    def create_demo_subject() -> None:
        """
        Demonstrate how to create a Subject instance.
        Note: This is a demo that shows the structure without actually loading a model.
        """
        print("Creating a demo Subject instance...")
        
        # Show the config we would use
        config = gpt2_config
        print(f"Using config for model: {config.hf_model_id}")
        
        # Show that the Subject class is available
        print(f"Subject class: {Subject}")
        print(f"LMConfig class: {LMConfig}")
        
        # Show the GPT-2 configuration
        print("\nGPT-2 Configuration (free model, no authentication required):")
        show_config_info(gpt2_config)
        
        # In a real scenario, you would do:
        # subject = make_subject(config)
        # But we'll just show what the function does
        print("In a real scenario, you would call make_subject(config) to create a Subject instance")
        print("This would load the actual GPT-2 model and tokenizer (no authentication required)")
        
        # Show some properties that would be available
        print("\nSubject would have properties like:")
        print(" - model_name")
        print(" - tokenizer")
        print(" - I, D, V, L, Q, K (model dimensions)")
        print(" - Various model components (layers, attention, MLPs, etc.)")
    
    def main() -> None:
        """
        Main function to demonstrate Subject configurations and instantiation.
        """
        print("=== Subject Configuration Examples ===\n")
        
        # Show information about the Llama3 8B config
        print("1. Llama3 8B Configuration:")
        show_config_info(llama3_8B_config)
        
        # Demonstrate how to get a subject config by model ID
        print("2. Getting config by model ID:")
        config = get_subject_config("meta-llama/Meta-Llama-3-8B")
        show_config_info(config)
        
        # Show information about the GPT-2 config
        print("3. GPT-2 Configuration (Free model):")
        show_config_info(gpt2_config)
        
        # Show that we can create a Subject instance
        print("4. Creating a demo Subject instance:")
        create_demo_subject()
        
        # Show the available functions
        print("\n5. Available functions:")
        print(f"   make_subject function: {make_subject}")
        print(f"   get_subject_config function: {get_subject_config}")
        
        print("\nExamples created successfully!")
        print("Note: Actual model loading for Llama models requires valid HuggingFace tokens.")
        print("GPT-2 model can be loaded without authentication.")
        
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("This might be due to missing dependencies or version conflicts.")
    
    def main() -> None:
        print("Unable to import Subject due to dependency issues.")
        print("Please check your environment setup.")

if __name__ == "__main__":
    main()
