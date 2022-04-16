from tools import NetworkType, RunConfig, get_nedg, run

models = ["5xo1nzt1", "3ee26r9o", "91pmt9xd"]
metric_headers = [
    "node_pos_loss",
    "degrees_loss",
    "node_types_loss",
    "node_pos_accuracy",
    "degrees_accuracy",
    "node_types_accuracy",
]
network = NetworkType.NODES_NN

if __name__ == "__main__":
    # generate data
    data_config, eval_config = run.get_configs(
        "config.yaml", "configs/eval_nodes_nn.yaml"
    )
    test_data = get_nedg(data_config, test=True)

    for model in models:
        run_config = RunConfig(f"configs/{model}.yaml", data_config=data_config)
        run_config.set_evaluate(eval_config)
        run_config.set_pretrained_weights(f"wandb/{model}.h5")

        # load model
        nodes_nn = run.load_model(
            data_config,
            run_config,
            network=network,
            do_train=False,
            eager=True,
        )
        assert nodes_nn.pretrained

        run.evaluate(
            nodes_nn,
            test_data,
            name=model,
            metric_headers=metric_headers,
            network=network,
        )

        del nodes_nn
