import wandb

class Logger:

    def __init__(self,config = {}) -> None:
        self.config = config

    def make_run(self,**kwargs):
        return wandb.init(
            project="NDE_as_Mental_Models",
            config=self.config,
            **kwargs
        )


