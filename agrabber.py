from typing import Dict, List, Optional, Literal, Tuple
import torch
import numpy as np
from subject import Subject


class ActivationGrabber:
    
    def __init__(self, subject: Subject):
        self.subject = subject

    def get_tokens(self, input_sequence: str | List[str]):
        tokenized = self.subject.tokenizer(
            input_sequence, 
            return_tensors="pt",
            padding=True
        )
        return tokenized["input_ids"].tolist()
        
        
    def get_activations(
        self,
        input_sequence: str | List[str],
        layers: Optional[List[int]] = None,
        return_numpy: bool = True
    ) -> Dict[str, any]:
        
        if isinstance(input_sequence, str):
            input_sequence = [input_sequence]
        
        if layers is None:
            layers = list(range(self.subject.L))
        
        tokenized = self.subject.tokenizer(
            input_sequence, 
            return_tensors="pt",
            padding=True
        )
        input_ids = tokenized["input_ids"]
        attn_mask = tokenized["attention_mask"]
        
        batch_size = input_ids.shape[0]
        num_tokens = input_ids.shape[1]
        
        acts_dict: Dict[int, torch.Tensor] = {}
        
        with torch.no_grad():
            with self.subject.model.trace(
                {"input_ids": input_ids, "attention_mask": attn_mask}
            ):
                for layer in layers:
                    # original : using `w_outs` => post non-linearity
                    # acts_dict[layer] = self.subject.w_outs[layer].input.detach().save()
                    # new : we need `residual_stream` : w_outs => layers
                    acts_dict[layer] = self.subject.layers[layer].input.detach().save()
        
        # Convert proxies to tensors and organize
        activations_BLTI = torch.stack([
            acts_dict[layer].value.cpu().float() 
            for layer in layers
        ], dim=1)  # Shape: (batch, layers, tokens, neurons)
        
        if return_numpy:
            activations_BLTI = activations_BLTI.numpy()
        
        tokens_list = []
        token_ids_list = []
        for i in range(batch_size):
            token_ids = input_ids[i].tolist()
            tokens = [self.subject.decode(tid) for tid in token_ids]
            tokens_list.append(tokens)
            token_ids_list.append(token_ids)
        
        return {
            "activations": activations_BLTI,  # Shape: (batch, layers, tokens, neurons)
            "tokens": tokens_list,
            "token_ids": token_ids_list,
            "attention_mask": attn_mask.numpy() if return_numpy else attn_mask,
            "layers": layers,
        }
