import torch
from torch import nn, device
from torch.nn.functional import softmax
from copy import deepcopy
# local imports
from adapters.main import capture_convolution_layers

DEFAULT_LR = 1e-5

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
        model_copy = deepcopy(model) if copy else model
        model_copy.to(map_location)
        self.model = model_copy
        self.model.eval()

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

DEFAULT_WEIGHT_IMAGE = 1 # controls contribution of an image reconstruction loss
DEFAULT_WEIGHT_MODEL = 1 # constrols contribtuion of model swapping loss

def modified_train_dr(autoencoder, datamodule,
             model, device,
             selected_layer_names: [str], num_epochs: int, learning_rate = DEFAULT_LR,
             weight_i = DEFAULT_WEIGHT_IMAGE,
             weight_m = DEFAULT_WEIGHT_MODEL,
             logger = None):
    """ Trains a DR Autoencoder on the whole dataset by modifying intermediate layer
    output and comparing loss """
    model_fake = ModelFaker(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = .0

        for idx, data_batch in enumerate(iter(datamodule.val_dataloader())):
            inputs = data_batch['input']
            wrapper, _ = capture_convolution_layers(model, device, inputs, selected_layer_names=selected_layer_names)
            optimizer.zero_grad()
            # accumulates total loss (image+swap) across layers for this batch
            total_loss = .0
            # temporarily for logging purpose
            sum_image_loss = .0
            sum_model_loss = .0

            # go over each layer of selected resolution
            for layer_name in selected_layer_names:
                layer_outputs = wrapper.layer_activations[layer_name]

                # go over each batch item and his hidden layer output
                for image, layer_output in zip(inputs, layer_outputs):
                    # forward pass
                    _, reconstructed = autoencoder(layer_output)

                    # reconstruction loss
                    reconstructed_norm = softmax(reconstructed, dim=1)
                    layer_outputs_norm = softmax(layer_output, dim=1)
                    image_loss = criterion(reconstructed_norm, layer_outputs_norm).cpu()
                    
                    # swap the output of this image via hooks for a copied model and get an error
                    image = image.unsqueeze(0)
                    teacher_outputs = model.forward(image.to(device))
                    modified_outputs = model_fake.get_modified_model_output(layer_name, image, reconstructed.unsqueeze(0)).to(device)
                    # normalize with softmax
                    teacher_outputs = softmax(teacher_outputs, dim=1)
                    modified_outputs = softmax(modified_outputs, dim=1)

                    # compute loss
                    model_loss = criterion(modified_outputs, teacher_outputs).cpu()
                    total_loss += weight_i * image_loss + weight_m * model_loss

                    # log
                    sum_image_loss += image_loss
                    sum_model_loss += model_loss
                
            if logger:
                logger({ "batch_image_loss": sum_image_loss.item() })
                logger({ "batch_model_loss": sum_model_loss.item() })
                logger({ "batch_train_loss": total_loss.item() })

            epoch_loss += total_loss.item()
            # Backpropagation after all layers at the end of batch
            total_loss.backward()
            # Update model parameters using the computed gradients
            optimizer.step()

        if logger:
            logger({ "epoch_loss": epoch_loss/len(layer_output) })
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(layer_output):.4f}")
            
    return
        
                

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
