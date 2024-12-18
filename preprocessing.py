import kagglehub
import os
import logging
import shutil
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
from config import MASOUD_1, MASOUD_6, MASOUD_7, MASOUD_8
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords

class Masoud:
    def __init__(self, masoud_2, masoud_1):
        # Dataset Name
        self.masoud_2 = masoud_2

        # Data Path
        self.masoud_1 = masoud_1

        # Download
        self._masoud()

    # Download if not exists
    def _masoud(self):
        if not os.path.exists(self.masoud_1):
            masoud_3 = kagglehub.dataset_download(self.masoud_2)

            logging.info(f"Downloading {self.masoud_2} to {masoud_3}...")
            logging.info(f"Copying {masoud_3} to {self.masoud_1}...")

            shutil.copytree(masoud_3, self.masoud_1)

        else:
            logging.info(f"Data folder {self.masoud_1} for {self.masoud_2} already exists ...")

        return self.masoud_1
    
class Masoud2(Masoud):
    def __init__(self, masoud_2, masoud_1, masoud_3, masoud_5, masoud_4, masoud_6):
        super().__init__(masoud_2, masoud_1)

        # Models Directory
        self.masoud_4 = masoud_4

        # Model Name
        self.masoud_5 = masoud_5

        # Download
        self._masoud_1()
        self.masoud_6 = masoud_6

        # Embed
        self.masoud_3 = self.maosud(masoud_3)

    # Download if not exists
    def _masoud_1(self):
        if not os.path.exists(self.masoud_4):
            masoud_3 = kagglehub.dataset_download(self.masoud_5)

            logging.info(f"Downloading {self.masoud_5} to {masoud_3}...")
            logging.info(f"Copying {masoud_3} to {self.masoud_4}...")

            shutil.copytree(masoud_3, self.masoud_4)

        else:
            logging.info(f"Data folder {self.masoud_4} for {self.masoud_5} already exists ...")

        return self.masoud_4
    
    # Read CSV
    def _masoud_2(self, masoud):
        return pd.read_csv(masoud, usecols=['title', 'tags'])
    
    # Embed
    def _masoud_3(self, m, masoudd):
        mas = []
        for word in m:
            if word in masoudd:
                mas.append(masoudd[word])

            else:
                mas.append(np.zeros(masoudd.vector_size))

        return mas
    
    # Get Embeddings
    def maosud(self, masoud):
        masoud_1 = self._masoud_2(masoud)

        # Dropping NaN 
        logging.info("Dropping NaN values...")
        masoud_1.dropna(inplace=True)
        logging.info(f"Shape after dropping NaN values: {masoud_1.shape}")

        masoud_1['title_tokenized'] = masoud_1['title']
        masoud_1['tags_tokenized'] = masoud_1['tags']

        # Lowercase
        masoud_1['title_tokenized'] = masoud_1['title_tokenized'].str.lower()
        masoud_1['tags_tokenized'] = masoud_1['tags_tokenized'].str.lower()

        # Special Characters
        masoud_2 = re.compile(r'[^a-zA-Z0-9]')
        masoud_1['title_tokenized'] = masoud_1['title_tokenized'].apply(lambda x: masoud_2.sub(' ', x)) #
        masoud_1['tags_tokenized'] = masoud_1['tags_tokenized'].apply(lambda x: masoud_2.sub(' ', x)) #

        # Stopwords
        masoud_1['title_tokenized'] = masoud_1['title_tokenized'].apply(remove_stopwords)
        masoud_1['tags_tokenized'] = masoud_1['tags_tokenized'].apply(remove_stopwords)

        # Tokenize
        logging.info("Tokenizing cleaned titles...")
        tqdm.pandas(desc="Tokenizing titles...")
        masoud_1['title_tokenized'] = masoud_1['title_tokenized'].progress_apply(word_tokenize)

        logging.info("Tokenizing cleaned tags...")
        tqdm.pandas(desc="Tokenizing tags...")
        masoud_1['tags_tokenized'] = masoud_1['tags_tokenized'].progress_apply(word_tokenize)

        logging.info("Tokenization completed.")

        # Embed
        logging.info("Embedding titles...")
        masoud_2 = KeyedVectors.load_word2vec_format(self.masoud_6, binary=True)

        masoud_1["title_word2vec"] = masoud_1["title_tokenized"].apply(lambda m: self._masoud_3(m, masoud_2))
        masoud_1["tags_word2vec"] = masoud_1["tags_tokenized"].apply(lambda m: self._masoud_3(m, masoud_2))

        return masoud_1
      
if __name__ == "__main__":
    masoud = Masoud2("bharatkumar0925/tmdb-movies-clean-dataset", MASOUD_1, MASOUD_6, "leadbest/googlenewsvectorsnegative300", MASOUD_7, MASOUD_8)
    print(masoud.masoud_3.head())
