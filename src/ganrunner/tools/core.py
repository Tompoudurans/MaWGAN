import os
#import tensorflow as tf


def set_core(number_of_cores):
    """
    Set the number of cores.
    """
    os.environ["MKL_NUM_THREADS"] = str(number_of_cores)
    os.environ["GOTO_NUM_THREADS"] = str(number_of_cores)
    os.environ["OMP_NUM_THREADS"] = str(number_of_cores)
    #tf.config.threading.set_inter_op_parallelism_threads(number_of_cores)
    #tf.config.threading.set_intra_op_parallelism_threads(number_of_cores)
