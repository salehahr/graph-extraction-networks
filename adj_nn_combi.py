import tensorflow as tf

from tools import AdjMatrPredictor, TestType, adj_matr, get_gedg, run

tf.config.run_functions_eagerly(False)

if __name__ == "__main__":
    # load data, model, predictor
    data_config, run_config = run.get_configs("config.yaml", "configs/edge_nn.yaml")
    graph_data = get_gedg(data_config, batch_size=1, shuffle=False)[TestType.VALIDATION]
    model = run.load_model(data_config, run_config, do_train=False)
    assert model.pretrained is True
    adj_matr_pred = AdjMatrPredictor(model, run_config.num_neighbours)

    # predict
    print("\nIMAGE 1------------")
    edge_nn_input, adj_matr_true, filepath = graph_data.get_single_data_point(0)
    print("First run with plot")
    adj_matr_pred.predict(edge_nn_input, do_preview=True)
    print("Second run with plot")
    adj_matr_pred.predict(edge_nn_input, do_preview=True)
    print("Third run, NO plot")
    adj_matr_pred.predict(edge_nn_input)

    print("\nIMAGE 2------------")
    edge_nn_input2, _, _ = graph_data.get_single_data_point(2)
    print("First run with plot")
    adj_matr_pred.predict(edge_nn_input2, do_preview=True)
    print("Second run with plot")
    adj_matr_pred.predict(edge_nn_input2, do_preview=True)
    print("Third run, NO plot")
    adj_matr_pred.predict(edge_nn_input2)

    print("\nLOOP------------")
    # only save the matrices/skel_imgs, plot later
    plot_array = []
    for i in range(0, 6):
        edge_nn_inputs, _, _ = graph_data.get_single_data_point(i)
        plot_data = adj_matr_pred.predict(edge_nn_inputs)
        plot_array.append(plot_data)

    for pd in plot_array:
        adj_matr.preview(*pd)
