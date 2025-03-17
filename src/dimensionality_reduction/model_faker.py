import torch
from torch import nn, device
from copy import deepcopy

class ModelFaker():
    def __init__(
        self,
        model: nn.Module,
        map_location: device = 'cuda:3',
        copy: bool = True,
    ):
        super().__init__()
        self.map_location = map_location
        # to ensure no side effects use a copy of the model for modifications via hooks
        if copy:
            model_copy = deepcopy(model)
            model_copy.to(map_location)
            self.model = model_copy
            self.model.eval()
        else:
            self.model = model


    def get_modified_model_output(self, layer_name, inputs, modified_activations):
        layer = self.model.get_submodule(layer_name)

        def batch_swap_hook(module, input, output):
            return modified_activations.to(self.map_location)
        
        with torch.no_grad():
            handle = layer.register_forward_hook(batch_swap_hook)
            outputs = self.model(inputs.to(self.map_location))
            handle.remove()

        return outputs
    
