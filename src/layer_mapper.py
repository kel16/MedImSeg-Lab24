import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerMapper():
    """ wraps a model and performs dimensionality reduction and normalization
        (and for passed upscale_size also an upscaling) on selected layer_ids """
    def __init__(
        self,
        wrapped_model: nn.Module,
        layer_names: list[str],
        dimensionality_reducer,
        upscale_size: int,
    ):
        self.wrapped_model = wrapped_model
        self.dr = dimensionality_reducer
        self.upscale_size = upscale_size
        self.layer_names = layer_names

    def _reduce_layer(self, layer_name, batch_count):
        img_reduced = self.dr(self.wrapped_model.layer_activations[layer_name][batch_count].detach().numpy())
        
        return img_reduced

    def _upscale(self, feature_map):
        """ Nearest Neighbor Upsampling (each pixel is duplicated to fill the gaps) """
        feature_tensor = torch.tensor(feature_map).permute(2, 0, 1).unsqueeze(0)
        feature_upscaled = F.interpolate(feature_tensor, size=self.upscale_size, mode='nearest')
        # smoother version:
        # return F.interpolate(feature_map, size=(input_height, input_width), mode='bilinear', align_corners=False)
        
        return feature_upscaled.squeeze(0).permute(1, 2, 0).numpy()
    
    def get_dr(self):
        return self.dr
    
    def transform(self, layer_id: str, image_id: int):
        """
        returns a reduced feature

        :param layer_id: layer id
        :param image_id: id of an image in the batch
        """
        # step 1: perform dimensionality reduction
        feature_transformed = self._reduce_layer(self.layer_names[layer_id], image_id)
        # step 2: upscale an image (optional)
        if (self.upscale_size):
            feature_transformed = self._upscale(feature_transformed)

        return feature_transformed
    