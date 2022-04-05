import wandb

from tools.postprocessing import smooth

api = wandb.Api()


def update_config(runs):
    for r in runs:
        run = api.run(f"salehah/edge_extraction/{r}")
        run.config["adjacency_frac"] = 0.2
        run.config["train_imgs"] = 1000
        # run.config["batch_norm"] = True
        run.update()


def update_metric(runs):
    metric = "val_precision"
    sum_metric = f"best_{metric}"

    for run in runs:
        history = [h[metric] for h in run.history(keys=[metric], pandas=False)]

        if not history:
            print(f"Did not update metric for run {run.path[-1]}")
            continue

        best_val = max(smooth(history))
        run.summary[sum_metric] = best_val
        run.summary.update()


def update_sweep_metric():
    sweep = api.sweep("salehah/edge_extraction/6k2b57dp")
    update_metric(sweep.runs)


if __name__ == "__main__":
    # runs = ["2pik0ra1", "wn6u03dw", "34lnvf06", "1wa53k5i", "2jaghqie", "2ni5aly4", "wcxdtu93"]
    # runs = ["msju3tfw"]
    # runs = ["9x1zx0g6", "6yzdfvh7", "87t6xd8t"]
    # runs = ["gxmn7ntn", "skaoai90"]

    runs = api.runs("salehah/edge_extraction")
    update_metric(runs)
