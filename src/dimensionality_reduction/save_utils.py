import torch 

def get_autoencoder_file_path(layer_id: int):
    return f"autoencoder_layer_{layer_id}.pth"

def save_autoencoder_config(autoencoder, layer_id):
    # Define the file path to save the model
    autoencoder_save_path = get_autoencoder_file_path(layer_id)

    # Save the trained model's state dictionary
    torch.save(autoencoder.state_dict(), autoencoder_save_path)
    print(f"Autoencoder saved to {autoencoder_save_path}")

def load_autoencoder_config(autoencoder, layer_id):
    # Define the file path to save the model
    autoencoder_save_path = get_autoencoder_file_path(layer_id)

    # Load the state dictionary into the model
    autoencoder.load_state_dict(torch.load(autoencoder_save_path))

    # Set the model to evaluation mode
    autoencoder.eval()
    print("Autoencoder loaded successfully")
