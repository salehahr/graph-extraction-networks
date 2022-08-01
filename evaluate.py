from tools import NetworkType, RunConfig, get_eedg, get_nedg, run
from tools.plots import save_prediction_images

network = NetworkType.NODES_NN
model_ids, metric_headers, test_data = [], [], []

if __name__ == "__main__":
    # generate data
    data_config, eval_config = run.get_configs(
        "config.yaml", "configs/eval_nodes_nn.yaml"
    )

    # define models to evaluate and headers
    if network == NetworkType.EDGE_NN:
        test_data = get_eedg(data_config, eval_config, test=True)
        model_ids = ["1m3yxeop", "rqvcpm69", "skaoai90", "5qooe1a2"]
        metric_headers = ["tp", "tn", "fp", "fn", "precision", "recall", "f1"]
    elif network == NetworkType.NODES_NN:
        test_data = get_nedg(data_config, test=True)
        model_ids = ["2uircygo", "pqphq89g"]
        metric_headers = [
            "loss",
            "L_pos",
            "L_degs",
            "L_types",
            "acc_pos",
            "acc_degs",
            "acc_types",
        ]

    # iterate over models
    for id_ in model_ids:
        run_config = RunConfig(f"configs/{id_}.yaml", data_config=data_config)
        run_config.set_evaluate(eval_config)
        run_config.set_pretrained_weights(f"wandb/{id_}.h5")

        # load model
        model = run.load_model(
            data_config,
            run_config,
            network=network,
            do_train=False,
            eager=True,
        )
        assert model.pretrained

        # visualise the predictions
        for batch_num in [0, 2, 5, 7]:
            save_prediction_images(model, test_data, batch=batch_num, prefix=id_)

        # metric evaluation per test image
        run.evaluate_single(
            model,
            test_data,
            name=id_,
            metric_headers=metric_headers,
            network=network,
        )

        # # metric evaluations over batches
        # # deprecated, get_nedg with test=True option automatically sets batch size
        # # to one.
        # run.evaluate_batch(
        #     model,
        #     test_data,
        #     name=id_,
        #     metric_headers=metric_headers,
        #     network=network,
        # )

        del model
