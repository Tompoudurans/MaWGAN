from Mawgan.tools.fid import calculate_fid
from Mawgan.tools.prepocessing import *
from Mawgan.tools.sqlman import *
from Mawgan.tools.compute_LS import *
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
