from dataclasses import dataclass
import torch

@dataclass
class ActiviationGrabber:
    name: str = "ActiviationGrabber"

    def generate_layer(self, layer: torch.nn.Module | None) -> torch.nn.Module:
        if not layer:
            return self.name
        return torch.nn.Sequential(layer, self)
