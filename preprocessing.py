import kagglehub
import os
import logging
import shutil
import pandas as pd
from tqdm import tqdm
from config import MASOUD_1, MASOUD_6
from nltk.tokenize import word_tokenize

class Masoud:
    def __init__(self, masoud_2, masoud_1):
        # Dataset Name
        self.masoud_2 = masoud_2

        # Data Path
        self.masoud_1 = masoud_1

        self._masoud()

    # Download if not exists
    def _masoud(self):
        if not os.path.exists(self.masoud_1):
            masoud_3 = kagglehub.dataset_download(self.masoud_2)

            logging.info(f"Downloading {self.masoud_2} to {masoud_3}...")
            logging.info(f"Copying {self.masoud_2} to {self.masoud_1}...")

            shutil.copytree(masoud_3, self.masoud_1)

        else:
            logging.info(f"Data folder {self.masoud_1} for {self.masoud_2} already exists ...")

        return self.masoud_1
    
class Masoud2(Masoud):
    def __init__(self, masoud_2, masoud_1, masoud_3):
        super().__init__(masoud_2, masoud_1)

        self.masoud_3 = self.maosud(masoud_3)
    
    # Read CSV
    def _masoud_2(self, masoud):
        return pd.read_csv(masoud, usecols=['title', 'tags'])
    
    def _mosud_3(self, masoud):
        try:
            return word_tokenize(masoud)

        except:
            print(f"Error: {masoud}")
            

    # Tokenize titles and tags
    def maosud(self, masoud):
        masoud_1 = self._masoud_2(masoud)
        
        logging.info("Tokenizing titles...")
        tqdm.pandas(desc="Tokenizing titles...")
        masoud_1['title_tokenized'] = masoud_1['title'].progress_apply(word_tokenize)
        
        logging.info("Tokenizing tags...")
        tqdm.pandas(desc="Tokenizing tags...")
        masoud_1['tags_tokenized'] = masoud_1['tags'].progress_apply(word_tokenize)
        
        logging.info("Tokenization completed.")
        
        return masoud_1
      
if __name__ == "__main__":
    masoud = Masoud2("bharatkumar0925/tmdb-movies-clean-dataset", MASOUD_1, MASOUD_6)
    print(masoud.masoud_3.head())
