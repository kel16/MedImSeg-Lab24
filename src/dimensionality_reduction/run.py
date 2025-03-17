from argparse import ArgumentParser
from datetime import datetime
from torch import nn
import wandb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# local imports
from adapters.main import capture_convolution_layers
from dimensionality_reduction.feature_autoencoder import FeatureAutoencoder
from dimensionality_reduction.train import modified_train_dr
from setup import load_model, setup_datamodule, setup_device
from visualization_utils.layer_visualizer import LayerVisualizer
from model_saver import ModelSaver
from dimensionality_reduction.metrics import get_avg_error, get_swapping_loss

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--channels_num",
                        help="number of channels to reduce")
    parser.add_argument("-mp", "--model_path", default='../pre-trained/trained_UNets/mnmv2-00-02_22-11-2024-v1.ckpt',
                        help="path to your saved model")
    parser.add_argument("-an", "--autoencoder_name", default='encoder_00-06_05-03-2025',
                        help="path to your saved autoencoder model")
    parser.add_argument("-d", "--device", default="cuda:2",
                        choices=['cuda:0', 'cuda:1', 'cuda:2', 'cpu'],
                        help="device location: cpu or cuda")
    parser.add_argument("--should_log", default=False,
                        help="whether to log to wandb")
    parser.add_argument("-e", "--epochs", default=30,
                        help="epochs count")
    parser.add_argument("-lr", "--lr", default=1e-3,
                        help="learning rate")
    parser.add_argument("-mw", "--model_weight", default=0,
                        help="model downweight in [0..1]")
    parser.add_argument("-l", "--layers",
                        default='',
                        help="string with layer names separated by ','")
    parser.add_argument("-i", "--indices",
                        default='0',
                        help="string with indices separated by ','")
    args = parser.parse_args()

    device = setup_device(args.device)
    print('Working on a device:', device)

    print('Loading configs and setting up the datamodule...')
    datamodule = setup_datamodule()

    # ====================== PARAMETER INITIALIZATION
    AUTOENCODER_ID = 'autoencoder_' + datetime.now().strftime("%H-%M_%d-%m-%Y")
    
    BATCH_SIZE = datamodule.batch_size
    EPOCHS_COUNT = args.epochs
    LEARNING_RATE = args.lr
    WEIGHT_M = args.model_weight
    
    selected_layer_names = args.layers.split(',')
    selected_layer_names = list(map(str.strip, selected_layer_names))
    if args.layers == '':
        raise Exception("Must pass layer names")
    
    layers_count = len(selected_layer_names)
    print(f"{layers_count} layers to visualize: ", selected_layer_names)

    test_sample_ids = args.indices.split(',')
    test_sample_ids = list(map(int, test_sample_ids))
    print(f"Test sample IDs to visualize: ", test_sample_ids)

    # ====================== SEGMENTATION MODEL
    checkpoint_path = args.model_path
    if checkpoint_path:
        model = load_model(checkpoint_path, device)
    else:
        # TODO
        raise NotImplementedError
    
    # ====================== AUTOENCODER MODEL
    channels_num = args.channels_num
    if not channels_num:
        raise("Must pass the channel depth as argument via --channels_num=[VALUE]")
    
    autoencoder = FeatureAutoencoder(int(channels_num), 3)
    autoencoder_name = args.autoencoder_name
    autoencoder_saver = ModelSaver('./modified_autoencoders')
    # autoencoder_saver = ModelSaver('./simple_autoencoders')
    if autoencoder_name:
        print(f"Loading autoencoder named {autoencoder_name}")
        autoencoder_saver.load_autoencoder_config(autoencoder, autoencoder_name)
    else:
        logger = print
        if args.should_log:
            wandb.init(project="autoencoder-training", name=AUTOENCODER_ID,
            config={
              "learning_rate": LEARNING_RATE, "epochs": EPOCHS_COUNT,
              "batch_size": BATCH_SIZE,
            })
            wandb.watch(autoencoder, log="all", log_freq=10)
            logger = wandb.log

        modified_train_dr(autoencoder=autoencoder, datamodule=datamodule,
                                model=model, device=device,
                                logger=logger,
                                selected_layer_names=selected_layer_names,
                                num_epochs=EPOCHS_COUNT, learning_rate=LEARNING_RATE,
                                weight_m=WEIGHT_M,
                                validate_every_n_epochs=5)
        autoencoder_saver.save_model_config(autoencoder, file_name=AUTOENCODER_ID)

    # ====================== METRICS
    print(f"Average MSE: {get_avg_error(autoencoder=autoencoder, dataloader=datamodule.test_dataloader, model=model, device=device, selected_layer_names=selected_layer_names)}")
    print(f"Swapping Loss: {get_swapping_loss(autoencoder=autoencoder, dataloader=datamodule.test_dataloader, model=model, device=device, selected_layer_names=selected_layer_names)}")
    
    # ====================== VISUALIZE LAYERS
    layer_visualizer = LayerVisualizer()
    indices = test_sample_ids
    print(f"{len(indices)} samples are to visualize")

    for data_batch in datamodule.test_dataloader():
        if len(indices) == 0:
            break

        batch_indices = data_batch['index'].tolist()
        intersection = list(set(batch_indices) & set(indices))
        print(len(intersection), intersection)
        indices = list(set(indices) - set(intersection))
        print(f"Renewed indices ({len(indices)}):", indices)

        if len(intersection) > 0:
            batch_inputs = data_batch['input']
            batch_outputs = model(batch_inputs.to(device))
            wrapper, layer_names = capture_convolution_layers(model, device, batch_inputs)

            for sample_id in intersection:
                for layer_idx, layer_name in enumerate(selected_layer_names):
                    layer_samples = wrapper.layer_activations[layer_name]
                    selected_sample = layer_samples[sample_id]

                    original_np = selected_sample.detach().numpy()
                    latent, reconstructed = autoencoder(selected_sample)
                    latent_np = latent.detach().numpy()
                    reconstructed_np = reconstructed.detach().numpy()
                    
                    plot_title = f"ID={sample_id} for {layer_name} ({layer_idx + 1} / {layers_count})"
                    saved_plot_title = f"{datetime.now().strftime('%H-%M_%d-%m-%Y')}-{sample_id}-{layer_idx}"
                    text = f"MSE Error: {nn.MSELoss()(selected_sample, reconstructed):.5f}"
                    plot_title = f"{plot_title} ({text})"

                    layer_visualizer.plot(original_np, reconstructed_np, latent_np, plot_title, saved_plot_title)
                    layer_visualizer.map_components(latent_np, saved_plot_title)
