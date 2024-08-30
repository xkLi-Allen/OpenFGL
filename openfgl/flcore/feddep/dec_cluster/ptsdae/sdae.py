from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from typing import Callable, Iterable, Optional, Tuple, List
import torch
import torch.nn as nn


def build_units(
    dimensions: Iterable[int], activation: Optional[torch.nn.Module]
) -> List[torch.nn.Module]:
    def single_unit(in_dimension: int, out_dimension: int) -> torch.nn.Module:
        unit = [("linear", nn.Linear(in_dimension, out_dimension))]
        if activation is not None:
            unit.append(("activation", activation))
        return nn.Sequential(OrderedDict(unit))

    return [
        single_unit(embedding_dimension, hidden_dimension)
        for embedding_dimension, hidden_dimension in sliding_window(2, dimensions)
    ]


def default_initialise_weight_bias_(
    weight: torch.Tensor, bias: torch.Tensor, gain: float
) -> None:
    nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0)


class StackedDenoisingAutoEncoder(nn.Module):
    def __init__(
        self,
        dimensions: List[int],
        activation: torch.nn.Module = nn.ReLU(),
        final_activation: Optional[torch.nn.Module] = nn.ReLU(),
        weight_init: Callable[
            [torch.Tensor, torch.Tensor, float], None
        ] = default_initialise_weight_bias_,
        gain: float = nn.init.calculate_gain("relu"),
    ):
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.dimensions = dimensions
        self.embedding_dimension = dimensions[0]
        self.hidden_dimension = dimensions[-1]
        # construct the encoder
        encoder_units = build_units(self.dimensions[:-1], activation)
        encoder_units.extend(
            build_units([self.dimensions[-2], self.dimensions[-1]], None)
        )
        self.encoder = nn.Sequential(*encoder_units)
        # construct the decoder
        decoder_units = build_units(reversed(self.dimensions[1:]), activation)
        decoder_units.extend(
            build_units([self.dimensions[1], self.dimensions[0]], final_activation)
        )
        self.decoder = nn.Sequential(*decoder_units)
        # initialise the weights and biases in the layers
        for layer in concat([self.encoder, self.decoder]):
            weight_init(layer[0].weight, layer[0].bias, gain)

    def get_stack(self, index: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        if (index > len(self.dimensions) - 2) or (index < 0):
            raise ValueError(
                "Requested subautoencoder cannot be constructed, index out of range."
            )
        return self.encoder[index].linear, self.decoder[-(index + 1)].linear

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)
