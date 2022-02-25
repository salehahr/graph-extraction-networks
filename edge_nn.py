import math

from tools import TestType, get_eedg, get_gedg, run

# user input
run_name = "edge_nn"
max_num_images = 50

if __name__ == "__main__":
    # configs
    data_config, run_config = run.get_configs("config.yaml", run_name)

    # generate data
    graph_data = get_gedg(data_config)
    edge_data = get_eedg(data_config, run_config, graph_data)

    # init new run
    run.start(run_config)

    # init model/reload model on resumed run
    edge_nn = run.load_model(data_config, run_config)

    # train and save for next run
    # one step contains n = run_config.batch_size images
    steps = math.ceil(max_num_images / run_config.batch_size)
    run.train(edge_nn, edge_data, steps_in_epoch=steps)
    run.save(edge_nn, run_config.model_filename)

    # predict for 5 combinations
    run.predict(edge_nn, edge_data[TestType.VALIDATION], max_pred=5)

    # terminate run
    run.end()
