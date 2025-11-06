from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from .subject import Subject


@dataclass(frozen=True)
class ActivationConfig:
    """
    Configuration for activation grabbing.
    """

    layers: Optional[List[int]] = None
    return_numpy: bool = True


@dataclass
class ActivationData:
    """
    Data class to hold activation data and related information.
    """

    activations: np.ndarray | torch.Tensor  # Shape: (batch, layers, tokens, neurons)
    tokens: List[List[str]]
    token_ids: List[List[int]]
    attention_mask: np.ndarray | torch.Tensor
    layers: List[int]

    def __str__(self) -> str:
        """
        String representation of the activation data.
        """
        batch_size = len(self.tokens)
        num_layers = len(self.layers)
        num_tokens = len(self.tokens[0]) if self.tokens else 0
        return (
            f"ActivationData(batch_size={batch_size}, layers={num_layers}, "
            f"tokens={num_tokens}, shape={getattr(self.activations, 'shape', 'N/A')})"
        )


class ActivationGrabber:
    """
    Class to grab activations from a Subject model.
    """

    def __init__(self, subject: Subject):
        """
        Initialize the ActivationGrabber with a Subject.

        Args:
            subject: The Subject model to grab activations from.
        """
        self.subject = subject

    def get_activations(
        self, input_sequence: str | List[str], config: Optional[ActivationConfig] = None
    ) -> ActivationData:
        """
        Get activations for the given input sequence.

        Args:
            input_sequence: Input text or list of texts.
            config: Configuration for activation grabbing.

        Returns:
            ActivationData containing the activations and related information.
        """
        if config is None:
            config = ActivationConfig()

        if isinstance(input_sequence, str):
            input_sequence = [input_sequence]

        layers = config.layers
        if layers is None:
            layers = list(range(self.subject.L))

        tokenized = self.subject.tokenizer(input_sequence, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"]
        attn_mask = tokenized["attention_mask"]

        batch_size = input_ids.shape[0]

        acts_dict: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            with self.subject.model.trace({"input_ids": input_ids, "attention_mask": attn_mask}):
                for layer in layers:
                    # Get residual stream activations from layers
                    acts_dict[layer] = self.subject.layers[layer].input.detach().save()

        # Convert proxies to tensors and organize
        activations_BLTI = torch.stack(
            [acts_dict[layer].value.cpu().float() for layer in layers], dim=1
        )  # Shape: (batch, layers, tokens, neurons)

        if config.return_numpy:
            activations_BLTI = activations_BLTI.numpy()

        tokens_list = []
        token_ids_list = []
        for i in range(batch_size):
            token_ids = input_ids[i].tolist()
            tokens = [self.subject.decode(tid) for tid in token_ids]
            tokens_list.append(tokens)
            token_ids_list.append(token_ids)

        return ActivationData(
            activations=activations_BLTI,  # Shape: (batch, layers, tokens, neurons)
            tokens=tokens_list,
            token_ids=token_ids_list,
            attention_mask=attn_mask.numpy() if config.return_numpy else attn_mask,
            layers=layers,
        )
