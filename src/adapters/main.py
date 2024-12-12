from torch import nn, Tensor

from adapters.layer_adapter import LayerAdapter
from adapters.model_adapter import ModelAdapter

def get_layers(model, limit_layers: int = 0):
    return [layer[0] for layer in model.named_modules() if 'conv' in layer[0]][-limit_layers:]

def capture_conv_layers(model, device: str, input_batch: Tensor, limit_layers: int = 0):
    """ wraps the model and captures all convolutional layer activations """
    conv_layer_names = get_layers(model, limit_layers)

    # create adapters for these layers and wrap them in a wrapper
    adapters = nn.ModuleList(
        [LayerAdapter(swivel) for swivel in conv_layer_names]
    )

    wrapper = ModelAdapter(
        model=model,
        adapters=adapters,
        map_location=device,
    )
    wrapper.hook_adapters()

    # do a forward pass and inspect results
    _ = wrapper.forward(input_batch.cuda())

    return wrapper, conv_layer_names
