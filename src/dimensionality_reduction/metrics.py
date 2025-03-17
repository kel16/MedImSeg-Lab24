import torch
from torch import nn, device
from torch.utils.data import DataLoader
# local imports
from adapters.main import capture_convolution_layers
from dimensionality_reduction.model_faker import ModelFaker

def get_avg_error(autoencoder, dataloader: DataLoader,
                  model, device: device,
                  selected_layer_names,
                  criterion = nn.MSELoss()):
    """
    gets avg batch loss on reconstruction
    criterion=MSELoss by default
    """
    is_train = autoencoder.training
    if is_train:
        autoencoder.eval()

    num_batches = 0
    loss = 0.0
    # samples_count = len(dataloader().dataset)

    with torch.no_grad():
        for batch in dataloader():
            inputs = batch['input'].to(device)
            
            wrapper, _ = capture_convolution_layers(model, device, inputs, 
                                                   selected_layer_names=selected_layer_names)
            
            batch_loss = 0.0
            for layer_name in selected_layer_names:
                layer_outputs = wrapper.layer_activations[layer_name]
                _, reconstructed = autoencoder(layer_outputs)
                batch_loss += criterion(reconstructed, layer_outputs)

            loss += batch_loss.item() / len(selected_layer_names)
            num_batches += 1

    if is_train:
        autoencoder.train()
    
    return loss / num_batches

def get_swapping_loss(autoencoder, dataloader: DataLoader,
                  model, device: device,
                  selected_layer_names,
                  criterion = nn.MSELoss()):
    """
    gets swapping loss measure
    """
    is_train = autoencoder.training
    if is_train:
        autoencoder.eval()

    layers_count = len(selected_layer_names)
    model_fake = ModelFaker(model, map_location=device, copy=True)

    with torch.no_grad():
        for data_batch in iter(dataloader()):
            inputs = data_batch['input'].to(device)
            wrapper, _ = capture_convolution_layers(model, device, inputs, selected_layer_names=selected_layer_names)
            avg_model_loss = .0

            # go over each layer of selected resolutions
            for layer_name in selected_layer_names:
                layer_outputs = wrapper.layer_activations[layer_name]

                # forward pass
                _, reconstructed = autoencoder(layer_outputs)
                
                # swap the output of this image via hooks for a copied model and get an error
                modified_outputs = model_fake.get_modified_model_output(layer_name, inputs, reconstructed)
                
                # compute the loss
                teacher_outputs = model(inputs).to(device)
                model_loss = criterion(modified_outputs, teacher_outputs)
                avg_model_loss += model_loss.detach().item()
                
            # average out the sum
            avg_model_loss /= layers_count
            
    if is_train:
        autoencoder.train()

    return avg_model_loss
