import wandb


def init_wandb(project: str, entity: str = None):
    # wandb.init(project=project, entity="graham")
    wandb.init(project=project)
    return wandb
