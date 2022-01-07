import wandb

api = wandb.Api()

run = api.run("salehah/node_extraction/19wesr02")
run.config["n_filters"] = 64
run.config["depth"] = 5
run.config["extended"] = False
run.update()
