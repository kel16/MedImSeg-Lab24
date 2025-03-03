import time
import torch
from torch import nn, device
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

# ===========================================================================

DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_IMAGE = 1 # controls contribution of an image reconstruction loss
DEFAULT_WEIGHT_MODEL = 1 # constrols contribtuion of model swapping loss

def validate_dr(autoencoder, datamodule, model, device, selected_layer_names):
    autoencoder.eval() 
    val_loss = 0.0
    criterion = nn.MSELoss()
    # samples_count = len(datamodule.train_dataloader().dataset)

    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            inputs = batch['input'].to(device)
            
            wrapper, _ = capture_convolution_layers(model, device, inputs, 
                                                   selected_layer_names=selected_layer_names)
            
            batch_loss = 0.0
            for layer_name in selected_layer_names:
                layer_outputs = wrapper.layer_activations[layer_name]
                _, reconstructed = autoencoder(layer_outputs)
                batch_loss += criterion(reconstructed, layer_outputs)

            val_loss += batch_loss.item() / len(selected_layer_names)

    autoencoder.train()
    
    return { "val_loss": val_loss }


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
                      logger = None,
                      validate_every_n_epochs = None):
    """
    Trains a DR Autoencoder on the whole dataset by modifying intermediate layer
    output and comparing loss.

    :get_weight_m: progressively trained model swap hyperparameter.
    weight_m will not be used.
    """
    autoencoder.train()
    model.eval()

    model_fake = ModelFaker(model, map_location=device, copy=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    layers_count = len(selected_layer_names)

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = .0

        # in case of progressive training call the function
        if callable(get_weight_m):
            weight_m = get_weight_m(epoch)

        for _, data_batch in enumerate(iter(datamodule.train_dataloader())):
            inputs = data_batch['input'].to(device)
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

                # forward pass
                _, reconstructed = autoencoder(layer_outputs)

                # reconstruction loss
                image_loss = criterion(reconstructed, layer_outputs)
                
                # swap the output of this image via hooks for a copied model and get an error
                modified_outputs = model_fake.get_modified_model_output(layer_name, inputs, reconstructed)
                
                # compute loss
                teacher_outputs = model(inputs).to(device)
                model_loss = criterion(modified_outputs, teacher_outputs)
                
                total_loss += weight_i * image_loss + weight_m * model_loss
                
                # for logging purpose
                sum_image_loss += image_loss.detach().item()
                sum_model_loss += model_loss.detach().item()
                
            # average out the sums
            total_loss /= layers_count
            sum_image_loss /= layers_count
            sum_model_loss /= layers_count
            
            if logger:
                logger({
                    "batch_train_loss": total_loss.item(),
                    "batch_image_loss": image_loss.item(),
                    "batch_model_loss": model_loss.item(),
                })
            
            epoch_loss += total_loss.item()
            # backpropagation after all layers at the end of batch
            total_loss.backward()
            # update model parameters using the computed gradients
            optimizer.step()

        epoch_time = time.time() - start_time
        if logger:
            logger({
              "epoch_loss": epoch_loss,
              "epoch_time": epoch_time,
            })
        
        if validate_every_n_epochs:
            if epoch % validate_every_n_epochs == 0:
                log_info = validate_dr(autoencoder, datamodule, model, device, selected_layer_names)
                print(log_info)
                if logger:
                    logger(log_info)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        
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
        for idx, data_batch in enumerate(iter(datamodule.train_dataloader())):
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
