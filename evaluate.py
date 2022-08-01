from tools import NetworkType, RunConfig, run
from tools.evaluate import get_test_data
from tools.plots import save_prediction_images

network = NetworkType.NODES_NN
model_ids = ["smnvgh0v"]

if __name__ == "__main__":
    # generate data
    data_config, eval_config = run.get_configs(
        "config.yaml", "configs/eval_nodes_nn.yaml"
    )

    # define models to evaluate and headers
    test_data, metric_headers = get_test_data(network, data_config, eval_config)

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

        del model
