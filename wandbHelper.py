import wandb


def wandb_init(project, entity, learning_rate, epochs, batch_size):
    wandb.init(project=project, entity=entity)
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }


def wandb_log(log_dict):
    wandb.log(log_dict)
