from engine import Runner
from argparse import ArgumentParser
from config.builder import build as cfg_build

def parse_cfg():
    parser = ArgumentParser("calm")
    parser.add_argument("--dataset", 
                        type=str, 
                        choices=["handwritten"],
                        help="config file",
                        default="handwritten")

    config = parser.parse_args()
    cfg = cfg_build(config.dataset)
    return cfg, config



def train(cfg, config=None):
    runner = Runner(arch_cfg=cfg['Arch'],
                    data_cfg=cfg['Data'],
                    global_cfg=cfg['Global'],
                    loss_cfg=cfg['Loss'],
                    metric_cfg=cfg['Metric'])
    
    runner.train()



if __name__ == "__main__":
    cfg, config = parse_cfg()
    train(cfg, config)
