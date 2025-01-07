import torch
from torch import nn
# local imports
from adapters.main import capture_conv_layers

def train_dr(autoencoder, datamodule,
             model, device,
             layer_id: int, num_epochs: int):
    """ trains an autoencoder on the whole dataset for given a layer id """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        epoch_loss = .0
        for idx, data_batch in enumerate(iter(datamodule.val_dataloader())):
            inputs = data_batch['input']
            wrapper, layer_names = capture_conv_layers(model, device, inputs, selected_layer_id=layer_id)
            # get activations of `layer_ID` for this `data_batch`
            layer_samples = wrapper.layer_activations[layer_names[0]]
            
            for image in layer_samples:
                # Forward pass
                latent, reconstructed = autoencoder(image)
                loss = criterion(reconstructed, image)
                optimizer.zero_grad()

                # Compute Loss
                epoch_loss += loss.item()

                # Backprop
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(layer_samples):.4f}")
    
    return autoencoder
