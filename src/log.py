import logging
from logging.handlers import RotatingFileHandler
import os

log_file_name = "workdir/log_replaceAnything.log"
os.makedirs(os.path.dirname(log_file_name), exist_ok=True)

format = '[%(levelname)s] %(asctime)s "%(filename)s", line %(lineno)d, %(message)s'
logging.basicConfig(
    format=format,
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(name="WordArt_Studio")

fh = RotatingFileHandler(log_file_name, maxBytes=20000000, backupCount=3)
formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
logger.addHandler(fh)