import random
import time

import numpy as np

from scripts import txt2img

if __name__ == "__main__":
    opt = txt2img.parse_args()
    opt.ckpt = 'E:/Workspace/Data/sd-ckpt/v2-1_512-ema-pruned.ckpt'
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    opt.seed = random.randint(min_seed_value, max_seed_value)
    opt.n_iter = 1
    opt.n_samples = 1
    opt.n_rows = 1
    opt.steps = 50
    opt.scale = 7
    opt.precision = 'full'
    opt.H = 512
    opt.W = 512
    opt.prompt = 'A patriotic cat'
    print('Option:', opt)
    start_time = time.time()
    txt2img.main(opt)
    print("Execution time: %s seconds" % (time.time() - start_time))
