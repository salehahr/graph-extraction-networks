from tools import EdgeDGSingle, TestType, get_gedg, run, timer


@timer
def iterate(data_generator, model_):
    iteration = 0
    while not data_generator.checked_all_nodes and data_generator.num_combos > 0:
        # todo: calculate batch size

        # loop over all nearest neighbour combinations
        for i in range(len(data_generator)):
            data_generator.update_adjacencies(i, model_)
            # noinspection PyUnreachableCode
            if __debug__:
                edge_dg.preview(title=f"$A_{iteration}$")

        num_neighbours = data_generator.num_neighbours * 2
        data_generator.update_neighbours(num_neighbours)

        iteration += 1


if __name__ == "__main__":
    # configs
    data_config, run_config = run.get_configs("config.yaml", "configs/edge_nn.yaml")

    # load EdgeNN model
    model = run.load_model(data_config, run_config, do_train=False)
    assert model.pretrained is True

    # generate data
    graph_data = get_gedg(data_config, batch_size=1)[TestType.VALIDATION]
    (skel_img, node_pos, degrees), adj_matr_true = graph_data.get_single_data_point(0)
    edge_dg = EdgeDGSingle(run_config.num_neighbours, skel_img, node_pos, degrees)

    # calculate A
    iterate(edge_dg, model)
    edge_dg.preview(title="$A_\mathrm{\mathsf{end}}$")
