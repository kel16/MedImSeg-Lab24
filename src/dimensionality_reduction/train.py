import torch
from torch import nn, device
from torch.nn.functional import softmax
from copy import deepcopy
# local imports
from adapters.main import capture_convolution_layers

class ModelFaker():
    def __init__(
        self,
        model: nn.Module,
        map_location: device = 'cuda:3',
        copy: bool = True,
    ):
        super().__init__()
        self.map_location = map_location
        # to ensure no side effects use a copy of the model for mdifications via hooks
        if copy:
            model_copy = deepcopy(model)
            model_copy.to(map_location)
            self.model = model_copy
            self.model.eval()
        else:
            self.model = model

    def _replace_layer_activations(self, fake_outputs):
        def swap_hook(module, input, output):
            # Replace output with a tensor of ones of the same shape
            modified_output = fake_outputs
            
            return modified_output.to(self.map_location)
        
        return swap_hook


    def get_modified_model_output(self, layer_name, inputs, modified_activations):
        # get the actual layer from the model
        layer = self.model.get_submodule(layer_name)
        
        adapter_handler = layer.register_forward_hook(self._replace_layer_activations(modified_activations))
        outputs = self.model.forward(inputs.to(self.map_location))
        adapter_handler.remove()

        return outputs

# ===========================================================================

DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_IMAGE = 1 # controls contribution of an image reconstruction loss
DEFAULT_WEIGHT_MODEL = 1 # constrols contribtuion of model swapping loss

def modified_train_dr(autoencoder,
                      datamodule,
                      model,
                      device,
                      selected_layer_names: list[str],
                      num_epochs: int,
                      learning_rate = DEFAULT_LR,
                      weight_i = DEFAULT_WEIGHT_IMAGE,
                      weight_m = DEFAULT_WEIGHT_MODEL,
                      get_weight_m = None,
                      logger = None):
    """
    Trains a DR Autoencoder on the whole dataset by modifying intermediate layer
    output and comparing loss.

    :get_weight_m: progressively trained model swap hyperparameter.
    weight_m will not be used.
    """
    model_fake = ModelFaker(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    batch_size = datamodule.batch_size
    layers_count = len(selected_layer_names)

    for epoch in range(num_epochs):
        epoch_loss = .0

        for _, data_batch in enumerate(iter(datamodule.val_dataloader())):
            # inputs = data_batch['data']
            inputs = data_batch['input']
            wrapper, _ = capture_convolution_layers(model, device, inputs, selected_layer_names=selected_layer_names)
            optimizer.zero_grad()
            # accumulates total loss (image+swap) across layers for this batch
            total_loss = .0
            # temporarily for logging purpose
            sum_image_loss = .0
            sum_model_loss = .0

            # go over each layer of selected resolutions
            for layer_name in selected_layer_names:
                layer_outputs = wrapper.layer_activations[layer_name]

                # go over each batch item and its hidden layer outputs
                for image, layer_output in zip(inputs, layer_outputs):
                    # forward pass
                    _, reconstructed = autoencoder(layer_output)

                    # reconstruction loss. Here it is of shape (C, H, W)
                    reconstructed_norm = softmax(reconstructed, dim=0)
                    layer_outputs_norm = softmax(layer_output, dim=0)
                    image_loss = criterion(reconstructed_norm, layer_outputs_norm)
                    
                    # swap the output of this image via hooks for a copied model and get an error
                    image = image.unsqueeze(0)
                    teacher_outputs = model.forward(image.to(device))
                    modified_outputs = model_fake.get_modified_model_output(layer_name, image, reconstructed.unsqueeze(0)).to(device)
                    # normalize with softmax
                    teacher_outputs = softmax(teacher_outputs.squeeze(0), dim=0)
                    modified_outputs = softmax(modified_outputs.squeeze(0), dim=0)

                    # compute loss
                    model_loss = criterion(modified_outputs, teacher_outputs)
                    
                    # in case of progressive training call the function
                    if callable(get_weight_m):
                        weight_m = get_weight_m(epoch)
                    
                    total_loss += weight_i * image_loss + weight_m * model_loss
                    
                    # for logging purpose
                    sum_image_loss += image_loss
                    sum_model_loss += model_loss
                
            # average out the sums
            samples_count = (batch_size * layers_count)
            total_loss /= samples_count
            sum_image_loss /= samples_count
            sum_model_loss /= samples_count

            if logger:
                logger({ "batch_image_loss": sum_image_loss.item() })
                logger({ "batch_model_loss": sum_model_loss.item() })
                logger({ "batch_train_loss": total_loss.item() })
            
            epoch_loss += total_loss.item()
            # backpropagation after all layers at the end of batch
            total_loss.backward()
            # update model parameters using the computed gradients
            optimizer.step()

        if logger:
            logger({ "epoch_loss": epoch_loss })
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
    return

# ===========================================================================

def train_dr(autoencoder, datamodule,
             model, device, logger,
             layer_names, num_epochs: int, learning_rate = DEFAULT_LR):
    """ Trains a DR autoencoder on the whole dataset for a given layer ID """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = .0
        for idx, data_batch in enumerate(iter(datamodule.val_dataloader())):
            optimizer.zero_grad()
            loss_sum = .0

            inputs = data_batch['input']
            wrapper, layer_names = capture_convolution_layers(model, device, inputs, selected_layer_names=layer_names)
            # Get activations of `layer_ID` for this `data_batch`
            layer_samples = wrapper.layer_activations[layer_names[0]]
            
            for image in layer_samples:
                # Forward pass
                _, reconstructed = autoencoder(image)
                loss = criterion(reconstructed, image)
                loss_sum += loss
            
            if logger:
                logger({ "train_batch_loss": loss_sum.item() })

            epoch_loss += loss_sum.item()
            # Backprop
            loss_sum.backward()
            optimizer.step()

        if logger:
            logger({ "epoch_loss": epoch_loss/len(layer_samples) })
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(layer_samples):.4f}")
    
    return autoencoder
