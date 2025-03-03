import torch
from torch import nn

BASE_MODEL_SAVE_PATH = "."

class ModelSaver():
    def __init__(
        self,
        folder_name: str = BASE_MODEL_SAVE_PATH,
    ):
        super().__init__()
        self.folder_name = folder_name

    def _get_model_file_path(self, file_name: str):
        return f"{self.folder_name}/{file_name}.pth"

    def save_model_config(self, model: nn.Module, file_name: str):
        # Define the file path to save the model
        save_path = self._get_model_file_path(file_name)

        # Save the trained model's state dictionary
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_autoencoder_config(self, model: nn.Module, file_name: str):
        # Define the file path to save the model
        autoencoder_save_path = self._get_model_file_path(file_name)

        # Load the state dictionary into the model
        model.load_state_dict(torch.load(autoencoder_save_path))

        # Set the model to evaluation mode
        model.eval()
        print("Model loaded successfully")
