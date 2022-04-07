from tools import NetworkType, get_eedg, run


def train():
    # init new run
    run.start(run_config, is_sweep=True)

    # init model/reload model on resumed run
    edge_nn = run.load_model(
        data_config, run_config, network=NetworkType.EDGE_NN, do_sweep=True
    )

    # train and save for next run
    run.train(edge_nn, edge_data, max_num_images=run_config.parameters.train_imgs)

    # terminate run
    run.end()


if __name__ == "__main__":
    # configs
    data_config, run_config = run.get_configs(
        "config_sweep.yaml", "configs/edge_nn_sweep.yaml"
    )

    # generate data
    edge_data = get_eedg(data_config, run_config)

    # start sweep
    sweep_id = "6k2b57dp"
    run.sweep(run_config, train, sweep_id=sweep_id)
