from tools import get_eedg, run

if __name__ == "__main__":
    # configs
    data_config, run_config = run.get_configs(
        "config.yaml", "configs/vgg_mod_large.yaml"
    )

    # generate data
    edge_data = get_eedg(data_config, run_config)

    # init new run
    run.start(run_config)

    # init model/reload model on resumed run
    edge_nn = run.load_model(data_config, run_config)

    # train and save for next run
    run.train(edge_nn, edge_data)

    # terminate run
    run.end()
