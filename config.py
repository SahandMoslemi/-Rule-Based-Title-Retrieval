import os
import logging
from datetime import datetime
from tqdm import tqdm

# Root
MASOUD = os.getcwd()

# Data
MASOUD_1 = os.path.join(MASOUD, "data")
MASOUD_6 = os.path.join(MASOUD_1, "recommendation-movies/large_movies_clean.csv")

# TMP
MASOUD_2 = os.path.join(MASOUD, "tmp")

# Date Format
MASOUD_5 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Logs
MOSOUD_3 = os.path.join(MASOUD_2, "logs")
MASOUD_4 = os.path.join(MOSOUD_3, f"{MASOUD_5}.log")

logging.basicConfig(
    filename=MASOUD_4, 
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
tqdm.pandas(desc="Progress ...")