#!/usr/bin/env python3

"""
pytest configuration file.
"""

import pytest


@pytest.fixture(scope="session")
def gpt2_subject():
    """
    Fixture to create a GPT-2 Subject instance for testing.
    """
    from hf_example_repo.subject import get_subject_config, make_subject
    
    config = get_subject_config("gpt2")
    subject = make_subject(config, dispatch=False, disable_flash_attention=True)
    return subject
