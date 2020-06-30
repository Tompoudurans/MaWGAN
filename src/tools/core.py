import os
import tensorflow as tf

def set_core(number_of_core):
    """
    deduce the maximum number of cores that code uses
    """
    os.environ['MKL_NUM_THREADS'] = str(number_of_core)
    os.environ['GOTO_NUM_THREADS'] = str(number_of_core)
    os.environ['OMP_NUM_THREADS'] = str(number_of_core)
    #os.environ['openmp'] = 'False'
    #Set number of threads used for parallelism between independent operations.
    tf.config.threading.set_inter_op_parallelism_threads(number_of_core)
    #Set number of threads used within an individual op for parallelism.
    tf.config.threading.set_intra_op_parallelism_threads(number_of_core)
