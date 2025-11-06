from hf_example_repo.activiation_grabber import ActivationGrabber
from hf_example_repo.subject import Subject, get_subject_config


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
    # Create activation grabber with the subject
    activation_grabber = ActivationGrabber(subject)
    
    # Use subject properties
    print(f"Model name: {subject.model_name}")
    print(f"Number of layers: {subject.L}")
    print(f"Hidden dimension: {subject.D}")
    
    # Get activations
    activation_data = activation_grabber.get_activations("Hello, world!")
    print(f"Activation data: {activation_data}")



if __name__ == "__main__":
    main()
