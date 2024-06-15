import json
import os
import random
from dataclasses import dataclass

import pandas as pd
from pymilvus import MilvusClient

from preprocessing import preprocess


class Recommender:

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.anime_df = self.load_dataset()
        self.vector_db = self.load_vector_db()

    def load_dataset(self) -> pd.DataFrame:
        # Load dataset
        return pd.read_csv(os.path.join(self.config['DATA_DIR'], 'raw', 'anime_data.csv'))
    
    def load_vector_db(self) -> MilvusClient:
        # Load vector database
        return MilvusClient(self.config['vector_dv'])

    def get_user_anime_list(self, user_name):
        # Get users anime list through MAL api
        user_anime_list = []
        return user_anime_list
    
    def get_random_recommendations(self, limit):
        # Get random recommendations
        return self.anime_df.sample(n=limit)
    
    def get_anime_info(self, anime_ids):
        # Get anime info from MAL api
        # TODO use data_loader.py to get anime info
        anime_info = []
        # TODO add new anime info to dataset
        # TODO prepeocess new anime info (this also adds their embeddings to the vector database)

    def get_combined_recommendations(self, user_anime_embeddings: dict, limit: int):
        # TODO implement bulk vector search
        pass


    def recommend(self, user_name: str, limit: int):
        # Get users anime list
        user_anime_list = self.get_user_anime_list(user_name)
        
        # If user has no anime in their list, return random recommendations
        if len(user_anime_list) == 0:
            return self.get_random_recommendations(limit)
        
        # If user has anime in their list, that are not in the dataset, get info on those anime from MAL api
        anime_ids = [anime['id'] for anime in user_anime_list]
        anime_not_in_dataset = anime_ids[~anime_ids.isin(self.anime_df['id'])]
        if len(anime_not_in_dataset) > 0:
            # Get info on anime from MAL api
            self.get_anime_info(anime_not_in_dataset)
            pass
        
        # Get embeddings for users anime list
        user_anime_embeddings = self.vector_db.get(
            collection_name=self.config['collection_name'],
            ids=anime_ids
        )

        #recommendations = self.get_combined_recommendations(user_anime_embeddings, limit)

        """ # Get topk combined recommendations
        recommendations_combined = self.vector_db.search(
            collection_name=self.config['collection_name'],
            query_records=user_anime_embeddings,
            top_k=limit
        ) """


@dataclass
class Recommendation:
    anime_id: int
    title: str
    similarity: float