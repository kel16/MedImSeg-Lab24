import torch
from monai.networks.nets import UNet
from omegaconf import OmegaConf

from data_utils import MNMv2DataModule
from unet import LightningSegmentationModel


def setup_device(device_location):
    """ sets up a device globally """
    device = torch.device(device_location if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and not device == 'cpu':
        cuda_device = int(device_location.split(":")[1])
        torch.cuda.set_device(cuda_device)

    return device


def setup_datamodule():
    """ loads and sets up an MNMV2 datamodule """
    mnmv2_config   = OmegaConf.load('../../configs/mnmv2.yaml')
    datamodule = MNMv2DataModule(
        data_dir=mnmv2_config.data_dir,
        vendor_assignment=mnmv2_config.vendor_assignment,
        batch_size=mnmv2_config.batch_size,
        binary_target=mnmv2_config.binary_target,
        non_empty_target=mnmv2_config.non_empty_target,
    )

    datamodule.setup()

    return datamodule

def load_model(checkpoint_path: str, device: str, load_as_lightning_module=True):
    """ loads from the given path as a lightning module or a pytorch """
    if load_as_lightning_module:
        unet_config    = OmegaConf.load('../../configs/monai_unet.yaml')
        unet = UNet(
            spatial_dims=unet_config.spatial_dims,
            in_channels=unet_config.in_channels,
            out_channels=unet_config.out_channels,
            channels=[unet_config.n_filters_init * 2 ** i for i in range(unet_config.depth)],
            strides=[2] * (unet_config.depth - 1),
            num_res_units=4
        )
        model = LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            model=unet,
            binary_target=True if unet_config.out_channels == 1 else False,
            lr=unet_config.lr,
            patience=unet_config.patience,
            # cfg=OmegaConf.to_container(unet_config)
        )

    else:
        # TODO: what model do we return?
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device)) #"cpu"
        model_state_dict = checkpoint['state_dict']
        model_state_dict = {k.replace('model.model.', 'model.'): v for k, v in model_state_dict.items() if k.startswith('model.')}
        model_config = checkpoint['hyper_parameters']['cfgs']

        print(model_config)

        unet = UNet(
            spatial_dims=model_config['unet']['spatial_dims'],
            in_channels=model_config['unet']['in_channels'],
            out_channels=model_config['unet']['out_channels'],
            channels=[model_config['unet']['n_filters_init'] * 2 ** i for i in range(model_config['unet']['depth'])],
            strides=[2] * (model_config['unet']['depth'] - 1),
            num_res_units=4
        )

        unet.load_state_dict(model_state_dict)

    print(f"Loaded as {'Lightning module' if load_as_lightning_module else 'PyTorch module'}")
    
    return model
