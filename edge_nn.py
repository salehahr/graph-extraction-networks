from model import VGG16
from tools import Config

base_path = "/graphics/scratch/schuelej/sar/tfgraph/"
name = "edge_nn"

if __name__ == "__main__":
    conf = Config("config.yaml")
    network = conf.network.edge_extraction

    n_filters = 4

    # build model
    edge_nn = VGG16(
        input_size=(*conf.img_dims, network.input_channels),
        n_filters=n_filters,
        pretrained_weights=None,
    )
    edge_nn.build()
