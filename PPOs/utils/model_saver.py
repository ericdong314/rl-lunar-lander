from time import strptime
import torch
from datetime import datetime
from os.path import dirname, join


class ModelSaver():

    def __init__(self, folder="saved_models"):
        parent = dirname(dirname(__file__))
        self.path_prefix = join(parent, folder)

    def save(self, actor_model, critic_model, epoch):
        time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        torch.save(actor_model.state_dict(),
                   join(self.path_prefix, f"{(time)}-{epoch}-ac"))
        torch.save(critic_model.state_dict(),
                   join(self.path_prefix, f"{(time)}-{epoch}-cr"))

    def load(self, file):
        print(f"load from {join(self.path_prefix, file)}")
        return torch.load(join(self.path_prefix, file))
