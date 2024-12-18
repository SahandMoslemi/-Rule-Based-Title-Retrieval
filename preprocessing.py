import kagglehub
import os
import logging
import shutil
from config import MASOUD_1


class Masoud:
    def __init__(self, masoud_2, masoud_1):
        # Dataset Name
        self.masoud_2 = masoud_2

        # Data Path
        self.masoud_1 = masoud_1

        self._masoud()

    def _masoud(self):

        # Download if not exists
        if not os.path.exists(self.masoud_1):
            masoud_3 = kagglehub.dataset_download(self.masoud_2)

            logging.info(f"Downloading {self.masoud_2} to {masoud_3}...")
            logging.info(f"Copying {self.masoud_2} to {self.masoud_1}...")

            shutil.copytree(masoud_3, self.masoud_1)

        else:
            logging.info(f"Data folder {self.masoud_1} for {self.masoud_2} already exists ...")

        return self.masoud_1

if __name__ == "__main__":
    Masoud("bharatkumar0925/tmdb-movies-clean-dataset", MASOUD_1)