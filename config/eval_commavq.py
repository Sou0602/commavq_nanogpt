import torch


class ModelConfig:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compile = True
        self.seed_offset = 0
        self.SEED = 1337


class PathsConfig:
    def __init__(self):
        self.out_dir = "out-commavq"


class DatasetConstants:
    def __init__(self):
        ## constants
        self.BOS_TOKEN = 1024
        self.TOKENS_PER_FRAME = 129
        self.BS = 10
        self.CONTEXT_SIZE_FRAMES = 20
        self.N_FRAMES = 1200
        self.N = self.N_FRAMES - 20
