from ganrunner.tools.fid import calculate_fid
from ganrunner.tools.prepocessing import *
from ganrunner.tools.sqlman import *
from ganrunner.tools.compute_LS import *
import logging

def setup_log(filepath):
    """
    creates a log file
    """
    logging.basicConfig(
        filename=filepath,
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: \n%(message)s",
    )
