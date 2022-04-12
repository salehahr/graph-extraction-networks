import tensorflow as tf

from tools import AdjMatrPredictor, NetworkType, TestType, adj_matr, get_gedg, run

tf.config.run_functions_eagerly(False)

if __name__ == "__main__":
    # load data, model, predictor
    data_config, run_config = run.get_configs("config.yaml", "configs/edge_nn.yaml")
    graph_data = get_gedg(data_config, batch_size=1, shuffle=False)[TestType.TESTING]
    model = run.load_model(
        data_config, run_config, network=NetworkType.EDGE_NN, do_train=False
    )
    assert model.pretrained is True
    adj_matr_pred = AdjMatrPredictor(model, run_config.num_neighbours)

    # predict
    adj_matr.predict_loop(adj_matr_pred, graph_data)

    # preview
    # adj_matr.plot_in_loop(adj_matr_pred, graph_data)
