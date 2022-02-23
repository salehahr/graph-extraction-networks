from tools import TestType, get_eedg, get_gedg, run

# user input
run_name = "edge_nn_16f"
start_image_id = 1

if __name__ == "__main__":
    # configs
    data_config, run_config = run.get_configs("config.yaml", run_name)
    # run.check_weights(run_config)

    # generate data
    graph_data = get_gedg(data_config, batch_size=1)
    num_images = len(graph_data[run_config.run_type])

    # loop over all images
    old_run_id = run_config.run_id
    edge_nn = None

    for i in range(start_image_id, num_images):
        # init new run
        do_resume = False if i == 0 else "must"
        run.start(run_config, resume=do_resume, reinit=True, _id=old_run_id)

        # generate node combinations
        edge_data = get_eedg(
            data_config, run_config.node_pairs_in_batch, graph_data, step_num=i
        )

        # init model/reload model on resumed run
        edge_nn = run.load_model(data_config, run_config, model_=edge_nn)

        # train and save for next run
        run.train(edge_nn, edge_data)
        old_run_id = run.save(edge_nn, run_config.model_filename)

        # predict for 5 images
        if i < 5:
            run.predict(
                edge_nn,
                edge_data[TestType.VALIDATION],
                max_pred=7,
                only_adj_nodes=True,
            )

        # terminate run
        run.end()
