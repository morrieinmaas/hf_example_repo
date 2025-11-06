# HF Example Repo

This is a simple example project using Hugging Face transformers and nnsight to extract layers.

## Setup

This project uses `uv` for Python dependency management and virtual environments.

### Prerequisites

- uv (https://github.com/indygreg/uv)

### How to run

To install dependencies:

```bash
uv sync
```

To run the main project:

```bash
uv run hf_example_repo
```

To run the activation grabber example:

```bash
uv run hf_activation_grabber_example
```

Other available scripts:

- `uv run hf_examples`
- `uv run hf_gpt2_example`
- `uv run hf_working_gpt2_example`
- `uv run hf_minimal_gpt2_example`
- `uv run hf_debug_gpt2_config`
