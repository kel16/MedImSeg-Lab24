import torch
from torch import nn
# local imports
from adapters.main import capture_convolution_layers

DEFAULT_LR = 1e-3

def replace_layer_activations(device, fake_outputs):
    def swap_hook(module, input, output):
        # Replace output with a tensor of ones of the same shape
        modified_output = fake_outputs
        return modified_output.to(device)
    
    return swap_hook


def get_modified_model_output(model, device, layer_name, inputs, modified_activations):
    # get the actual layer from the model
    layer = model.get_submodule(layer_name)
    
    adapter_handler = layer.register_forward_hook(replace_layer_activations(device, modified_activations))
    outputs = model.forward(inputs.to(device))
    adapter_handler.remove()

    return outputs

DEFAULT_ALPHA = 1 # controls contribution of an image reconstruction loss
DEFAULT_BETA = 1 # constrols contribtuion of model swapping loss

def modified_train_dr(autoencoder, datamodule,
             model, device,
             selected_layer_names: [str], num_epochs: int, learning_rate = DEFAULT_LR,
             alpha = DEFAULT_ALPHA,
             beta = DEFAULT_BETA,
             logger = None):
    """ Trains a DR Autoencoder on the whole dataset by modifying intermediate layer
    output and comparing loss """
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
                    image_loss = criterion(reconstructed, layer_output)
                    
                    image = image.unsqueeze(0)
                    # swap the output of this image via hooks for a copy model and get an error
                    teacher_outputs = model.forward(image.to(device))
                    modified_output = get_modified_model_output(model, device, layer_name, image, reconstructed.unsqueeze(0))
                    model_loss = criterion(modified_output, teacher_outputs).cpu()
                    # balance the sum of losses
                    total_loss += alpha * image_loss + beta * model_loss

                    sum_image_loss =+ image_loss
                    sum_model_loss =+ model_loss
                
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
            
    return autoencoder
        
                

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
