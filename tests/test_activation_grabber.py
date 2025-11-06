#!/usr/bin/env python3

"""
Integration tests for the ActivationGrabber functionality.
"""

import numpy as np
import torch
from hf_example_repo.activiation_grabber import ActivationGrabber, ActivationConfig


def test_activation_grabber_initialization(gpt2_subject):
    """
    Test that ActivationGrabber can be initialized with a Subject.
    """
    grabber = ActivationGrabber(gpt2_subject)
    assert grabber.subject == gpt2_subject
    assert grabber.subject.model_name == "gpt2"


def test_get_activations_with_default_config(gpt2_subject):
    """
    Test getting activations with default configuration.
    """
    grabber = ActivationGrabber(gpt2_subject)

    # Test with a simple input
    input_text = "Hello, world!"
    activation_data = grabber.get_activations(input_text)

    # Verify the structure of the returned data
    assert activation_data.activations is not None
    assert activation_data.tokens is not None
    assert activation_data.token_ids is not None
    assert activation_data.attention_mask is not None
    assert activation_data.layers is not None

    # Verify shapes and types
    assert isinstance(activation_data.activations, np.ndarray)
    assert len(activation_data.tokens) == 1  # Single batch
    assert len(activation_data.token_ids) == 1  # Single batch

    # Verify tokens
    assert len(activation_data.tokens[0]) > 0
    assert "Hello" in activation_data.tokens[0][0]

    # Verify layers
    assert len(activation_data.layers) == gpt2_subject.L  # All layers

    # Verify activation shape: (batch, layers, tokens, neurons)
    expected_shape = (1, gpt2_subject.L, len(activation_data.tokens[0]), gpt2_subject.D)
    assert activation_data.activations.shape == expected_shape


def test_get_activations_with_custom_config(gpt2_subject):
    """
    Test getting activations with custom configuration.
    """
    grabber = ActivationGrabber(gpt2_subject)

    # Test with specific layers and numpy format
    config = ActivationConfig(
        layers=[0, gpt2_subject.L - 1],  # First and last layer
        return_numpy=True,
    )

    input_text = "Hello, world!"
    activation_data = grabber.get_activations(input_text, config=config)

    # Verify the structure of the returned data
    assert activation_data.activations is not None
    assert activation_data.tokens is not None
    assert activation_data.token_ids is not None
    assert activation_data.attention_mask is not None
    assert activation_data.layers is not None

    # Verify shapes and types
    assert isinstance(activation_data.activations, np.ndarray)
    assert len(activation_data.tokens) == 1  # Single batch
    assert len(activation_data.token_ids) == 1  # Single batch

    # Verify only specified layers are returned
    assert activation_data.layers == [0, gpt2_subject.L - 1]

    # Verify activation shape: (batch, layers, tokens, neurons)
    expected_shape = (1, 2, len(activation_data.tokens[0]), gpt2_subject.D)
    assert activation_data.activations.shape == expected_shape


def test_get_activations_with_torch_tensors(gpt2_subject):
    """
    Test getting activations with torch tensor format.
    """
    grabber = ActivationGrabber(gpt2_subject)

    # Test with torch tensor format
    config = ActivationConfig(
        layers=[0],  # First layer only
        return_numpy=False,
    )

    input_text = "Hello"
    activation_data = grabber.get_activations(input_text, config=config)

    # Verify the structure of the returned data
    assert activation_data.activations is not None
    assert activation_data.tokens is not None
    assert activation_data.token_ids is not None
    assert activation_data.attention_mask is not None
    assert activation_data.layers is not None

    # Verify shapes and types
    assert isinstance(activation_data.activations, torch.Tensor)
    assert len(activation_data.tokens) == 1  # Single batch
    assert len(activation_data.token_ids) == 1  # Single batch

    # Verify only specified layers are returned
    assert activation_data.layers == [0]

    # Verify activation shape: (batch, layers, tokens, neurons)
    expected_shape = (1, 1, len(activation_data.tokens[0]), gpt2_subject.D)
    assert activation_data.activations.shape == expected_shape


def test_get_activations_with_multiple_inputs(gpt2_subject):
    """
    Test getting activations with multiple input sequences.
    """
    grabber = ActivationGrabber(gpt2_subject)

    # Test with multiple inputs
    input_texts = ["Hello, world!", "How are you?"]
    activation_data = grabber.get_activations(input_texts)

    # Verify the structure of the returned data
    assert activation_data.activations is not None
    assert activation_data.tokens is not None
    assert activation_data.token_ids is not None
    assert activation_data.attention_mask is not None
    assert activation_data.layers is not None

    # Verify shapes and types
    assert isinstance(activation_data.activations, np.ndarray)
    assert len(activation_data.tokens) == 2  # Two batches
    assert len(activation_data.token_ids) == 2  # Two batches

    # Verify layers
    assert len(activation_data.layers) == gpt2_subject.L  # All layers

    # Verify activation shape: (batch, layers, tokens, neurons)
    # Note: tokens might be different lengths due to padding
    assert activation_data.activations.shape[0] == 2  # Two batches
    assert activation_data.activations.shape[1] == gpt2_subject.L  # All layers


def test_activation_data_string_representation(gpt2_subject):
    """
    Test the string representation of ActivationData.
    """
    grabber = ActivationGrabber(gpt2_subject)

    input_text = "Hello"
    activation_data = grabber.get_activations(input_text)

    # Verify string representation
    str_repr = str(activation_data)
    assert "ActivationData" in str_repr
    assert "batch_size=1" in str_repr
    assert f"layers={gpt2_subject.L}" in str_repr
    assert "tokens=" in str_repr
    assert "shape=" in str_repr
