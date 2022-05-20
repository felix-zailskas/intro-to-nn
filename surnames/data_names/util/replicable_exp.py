##From - https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras/52897216#52897216

## INCLUDE THIS AT THE BEGINNING OF YOUR PYTHON PROGRAM TO MAKE YOUR EXPERIMENTS REPLICABLE

def set_seed(seed):
    import os
    os.environ['PYTHONHASHSEED']=str(seed)
    # 2. Set python built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed)
    # 3. Set numpy pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed)
    # 4. Set tensorflow pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed)
