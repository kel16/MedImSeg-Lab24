import matplotlib.pyplot as plt

GRAYSCALE_COLORMAP = 'gray'

class DatasetVisualizer():
    def __init__(self, device, model, layer_mapper, feature_visualizer):
        self.device = device
        self.model = model
        self.layer_mapper = layer_mapper
        self.feature_visualizer = feature_visualizer
        self.images = []

    def set_data(self, data_batch):
        images = data_batch.to(self.device)

        self.outputs = self.model(images).cpu().detach().numpy()
        self.images = images.cpu()
    
    def visualize(self, image_idx, layer_idx):
        if len(self.images) == 0:
            raise Exception('Have to first feed with data_batch through set_data method')
        
        for image_id in image_idx:
            print()
            print('IMAGE ID', image_id)

            # display the input image
            plt.imshow(self.images[image_id].permute(1, 2, 0).numpy(), cmap=GRAYSCALE_COLORMAP)
            plt.title("Input image")
            plt.axis('off')
            plt.show()

            # visualize the selected hidden layers
            for layer_id in layer_idx:
                feature = self.layer_mapper.transform(layer_id=layer_id, image_id=image_id)
                self.feature_visualizer.plot_feature(feature)
            
            # final output visualization: channels separately and combined
            fig, axs = plt.subplots(1, 4, figsize=(12, 4))
            out = self.outputs[image_id]
            for idx, channel in enumerate(out):
                axs[idx].imshow(channel)
                axs[idx].set_title(f"Output. Channel {idx+1}")
                axs[idx].axis("off")
            plt.tight_layout()
            plt.show()
            plt.imshow(self.feature_visualizer.get_transform()(self.layer_mapper.get_dr()(out)), cmap=GRAYSCALE_COLORMAP)
            plt.title("Output image")
            plt.axis('off')
            plt.show()
