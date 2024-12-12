from argparse import ArgumentParser

# local imports
from dimensionality_reduction import pca_to_rgb
from adapters.main import capture_conv_layers
from setup import setup_device, setup_datamodule, load_model
from layer_mapper import LayerMapper
from feature_visualizer import FeatureVisualizer
from dataset_visualizer import DatasetVisualizer

def get_numbers_in_range(prompt: str, min_value: int, max_value: int) -> list[int]:
    while True:
        try:
            raw_input = input(prompt)
            numbers = [int(num.strip()) for num in raw_input.split(" ")]
            
            # Filter numbers in range and remove duplicates
            valid_numbers = sorted(set(num for num in numbers if min_value <= num <= max_value))
            
            return valid_numbers
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", default='./pre-trained/trained_UNets/mnmv2-00-02_22-11-2024-v1.ckpt',
                        help="path to your saved model")
    parser.add_argument("-d", "--device", default="cuda:2",
                        choices=['cuda:0', 'cuda:1', 'cuda:2', 'cpu'],
                        help="device location: cpu or cuda")
    args = parser.parse_args()

    device = setup_device(args.device)
    print('Working on device:', device)

    print('Loading configs and setting up the datamodule...')
    datamodule = setup_datamodule()

    # LOAD MODEL FROM A FILE
    checkpoint_path = args.file
    model = load_model(checkpoint_path, device)

    # GET LAYER ACTIVATIONS OF A BATCH
    first_batch = next(iter(datamodule.val_dataloader()))
    input_size = list(first_batch.size())[-2:]

    # HOOK THE MODEL TO CATCH CONV LAYERS
    wrapper, layer_names = capture_conv_layers(model, device, first_batch)

    # PRODUCE THE RESULTS
    layer_mapper = LayerMapper(wrapper, layer_names, pca_to_rgb, input_size)
    feature_visualizer = FeatureVisualizer()
    dataset_visualizer = DatasetVisualizer(device, model, layer_mapper, feature_visualizer)

    max_layer_id = len(layer_names)-1
    max_img_id = len(first_batch)-1
    layer_selection = get_numbers_in_range(f"Enter layer ids between 0 and {max_layer_id}, separated by spaces: ", max_value=max_layer_id)
    image_selection = get_numbers_in_range(f"Enter image ids between 0 and {max_img_id}, separated by spaces: ", max_value=max_img_id)

    print(layer_selection, image_selection)
    dataset_visualizer.set_data(first_batch)
    dataset_visualizer.visualize(image_selection, layer_selection)
