from typing import Dict, List
import numpy as np
import pickle
import os

class Logger:
    """
    A simple logger for logging train/val losses and dynamically reading them from disk
    """
    def __init__(self, exp_name: str, config: Dict, n_vars: int) -> None:
        data_dir = os.path.join("./logs", exp_name)
        os.makedirs(data_dir, exist_ok=True)
        self.mmap = np.memmap(os.path.join(data_dir, "vals.npy"), np.float32, mode="w+", shape=(config["steps"], n_vars))
        self.idx = 1
        config["n_vars"] = n_vars
        with open(os.path.join(data_dir, "config.pkl"), 'wb') as f:
            pickle.dump(config, f)
        
    def __call__(self, log_var: List[float]) -> None:
        self.mmap[self.idx] =  np.array(log_var)
        self.mmap[0][0] = self.idx + 1
        self.mmap.flush()
        self.idx += 1
    
    @staticmethod
    def read(exp_name: str):
        fpath = os.path.join("./logs", exp_name)
        with open(os.path.join(fpath, "config.pkl"), 'rb') as f:
            config = pickle.load(f)
        
        while True:
            data = np.memmap(os.path.join(fpath, "vals.npy"), np.float32, mode="r", shape=(config["steps"], config["n_vars"]))
            yield data