import os
import tensorflow as tf

def set_core():
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['GOTO_NUM_THREADS'] = '2'
    os.environ['OMP_NUM_THREADS'] = '2'
    #os.environ['openmp'] = 'False'
    tf.config.threading.set_inter_op_parallelism_threads(2) #Set number of threads used for parallelism between independent operations.
    tf.config.threading.set_intra_op_parallelism_threads(2) #Set number of threads used within an individual op for parallelism.
