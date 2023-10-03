import yaml

config_file = {"handwritten": "./config/handwritten.yaml",
               "scene15": "./config/scene15.yaml",
               "landUse21": "./config/landUse21.yaml"}

def build(dataset):
    file = config_file[dataset]
    with open(file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg