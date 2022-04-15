from tools import NetworkType, RunConfig, get_eedg, run

models = ["1m3yxeop", "rqvcpm69", "skaoai90", "5qooe1a2"]
metric_headers = ["tp", "tn", "fp", "fn", "precision", "recall", "f1"]
network = NetworkType.EDGE_NN

if __name__ == "__main__":
    # generate data
    data_config, eval_config = run.get_configs(
        "config.yaml", "configs/eval_edge_nn.yaml"
    )
    edge_data = get_eedg(data_config, eval_config, test=True)

    for model in models:
        run_config = RunConfig(f"configs/{model}.yaml", data_config=data_config)
        run_config.set_evaluate(eval_config)
        run_config.set_pretrained_weights(f"wandb/{model}.h5")

        # load model
        edge_nn = run.load_model(
            data_config,
            run_config,
            network=network,
            do_train=False,
            eager=True,
        )
        assert edge_nn.pretrained

        run.evaluate(
            edge_nn,
            edge_data,
            name=model,
            metric_headers=metric_headers,
            network=network,
        )

        del edge_nn
