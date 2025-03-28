from torch import nn, Tensor

from adapters.layer_adapter import LayerAdapter
from adapters.model_adapter import ModelAdapter

def get_convolution_layers(model, limit_layers: int = 0):
    return [layer[0] for layer in model.named_modules() if 'conv' in layer[0]][-limit_layers:]

def capture_convolution_layers(model, device: str, input_batch: Tensor,
                        selected_layer_names: [str] = [],
                        selected_layer_id: int = None, limit_layers: int = 0):
    """
    Wraps the model and captures all convolutional layer activations.
    If given `selected_layer_id`, then returns one layer in an array.
    """
    conv_layer_names = get_convolution_layers(model, limit_layers)

    if selected_layer_id:
        conv_layer_names = [conv_layer_names[selected_layer_id]]

    elif len(selected_layer_names) > 0:
        conv_layer_names = [l for l in conv_layer_names if l in selected_layer_names]

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
