from tools import NetworkType, get_eedg, run

if __name__ == "__main__":
    # configs
    data_config, run_config = run.get_configs("config.yaml", "configs/86ae75jb.yaml")

    # generate data
    edge_data = get_eedg(data_config, run_config)

    # init new run
    run.start(run_config)

    # init model/reload model on resumed run
    edge_nn = run.load_model(data_config, run_config, network=NetworkType.EDGE_NN)

    # train and save for next run
    run.train(edge_nn, edge_data, max_num_images=run_config.parameters.train_imgs)

    # terminate run
    run.end()
