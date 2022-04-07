from tools import NetworkType, get_nedg, run

if __name__ == "__main__":
    # configs, data generation
    data_config, run_config = run.get_configs(
        "config.yaml",
        "configs/unet_test.yaml",
    )
    data = get_nedg(data_config)

    # training
    run.start(run_config)
    model = run.load_model(
        data_config, run_config, network=NetworkType.NODES_NN, eager=True
    )
    run.train(model, data, max_num_images=run_config.parameters.train_imgs)
    run.end()
