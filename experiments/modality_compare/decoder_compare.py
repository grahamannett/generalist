from generalist.train import train
import hydra
from hydra import compose, initialize
from generalist.utils.utils import get_hostname


# @hydra.main(config_path="./", config_name="experiment", version_base=None)
# def exp_one(cfg):
def exp_one():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="iapetus")
        breakpoint()
    cfg = hydra.compose(config_name="experiment")
    # breakpoint()
    train(cfg)


if __name__ == "__main__":
    exp_one()
