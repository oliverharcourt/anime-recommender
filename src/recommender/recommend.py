import dataclasses
import os

import pandas as pd
import requests
from pymilvus import (AnnSearchRequest, Collection, WeightedRanker,
                      connections)

from data_loading import data_loader
from preprocessing import preprocess

@dataclasses.dataclass
class Recommendation:
    """
    A simple wrapper class for a recommendation.

    Attributes:
        anime_id (int): The ID of the anime.
        title (str): The title of the anime.
        similarity (float): The similarity score of the recommendation.
    """
    anime_id: int
    title: str
    similarity: float

class Recommender:

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.anime_df = self._load_dataset()
        self.anime_collection = self._load_collection()
        self.BASE_URL = config['BASE_URL']

    def _load_dataset(self) -> pd.DataFrame:
        # Load dataset
        return pd.read_csv(os.path.join(self.config['DATA_DIR'], 'raw', 'anime_data.csv'))

    def _load_collection(self) -> Collection:
        # Load vector database
        connections.connect("default", host=self.config['VECTOR_DB']['HOST'],
                                         port=self.config['VECTOR_DB']['PORT'])
        collection = Collection(name=self.config['VECTOR_DB']['COLLECTION_NAME'])
        collection.load()
        return collection

    def _get_user_anime_list(self, user_name: str) -> list[dict]:
        # TODO: implement this
        # Get users anime list through MAL api
        user_anime_list = []
        try:
            # request user anime list from MAL api
            user_anime_list = requests.get(f'{self.BASE_URL}/user/{user_name}/animelist', timeout=10).json()
        except Exception as e:
            print(e)
        return user_anime_list

    def _get_random_recommendations(self, limit: int) -> list[Recommendation]:
        # TODO: check and test this method
        # Get random recommendations
        random_sample = self.anime_df.sample(n=limit)
        recommendations = []
        for _, anime in random_sample.iterrows():
            recommendations.append(Recommendation(anime_id=anime['id'],
                                                  title=anime['title'],
                                                  similarity=-1.0))
        return recommendations


    def _get_new_anime_info(self, user_anime_list: pd.DataFrame) -> pd.DataFrame | None:
        # TODO: implement this
        # 1. Get anime info from MAL api --> use data_loader.py to get anime info
        # 3. add new anime info to dataset
        # 4. prepeocess new anime info (this also adds their embeddings to the vector database)

        # If user has anime in their list, that are not in the dataset, get info on those anime from MAL api
        anime_ids = [anime['id'] for anime in user_anime_list]
        anime_not_in_dataset = anime_ids[~anime_ids.isin(self.anime_df['id'])]
        if len(anime_not_in_dataset) <= 0:
            return None
        # Get info on anime from MAL api
        headers = {}
        collector = data_loader.DataCollector(headers=headers,
                                              base_url=self.config['BASE_URL'],
                                              data_dir=self.config['DATA_DIR'],
                                              request_delay=1.0
                                              )
        new_info, new_anime_ids = collector.collect(anime_not_in_dataset, return_new_ids=True)
        with open(os.path.join(self.config['DATA_DIR'], 'raw', 'new_anime_ids.txt'), 'a', encoding='utf-8') as f:
            for anime_id in new_anime_ids:
                f.write(f'{anime_id}\n')
        return new_info

    def _get_topk_recommendations(self, data: pd.DataFrame, limit: int) -> list[Recommendation]:
        # TODO: check and test this method

        # Get search vectors
        hybrid_query_synopsis = data['synopsis_embedding'].tolist()
        hybrid_query_related = data['related_embedding'].tolist()
        hybrid_query_genres = data['genres'].tolist()
        hybrid_query_studios = data['studios'].tolist()

        # Create search params
        search_params_synopsis = {
            "data": hybrid_query_synopsis,
            "anns_field": "synopsis_embedding",
            "param": {
                "metric_type": "COSINE",
                "params": {"nprobe": self.config['nprobe']['text']}
            },
            "limit": limit
        }

        search_params_related = {
            "data": hybrid_query_related,
            "anns_field": "related_embedding",
            "param": {
                "metric_type": "COSINE",
                "params": {"nprobe": self.config['nprobe']['text']}
            },
            "limit": limit
        }

        search_params_genres = {
            "data": hybrid_query_genres,
            "anns_field": "genres",
            "param": {
                "metric_type": "L2",
                "params": {"nprobe": self.config['nprobe']['other']}
            },
            "limit": limit
        }

        search_params_studios = {
            "data": hybrid_query_studios,
            "anns_field": "studios",
            "param": {
                "metric_type": "L2",
                "params": {"nprobe": self.config['nprobe']['other']}
            },
            "limit": limit
        }

        # Create request objects
        req_synopsis = AnnSearchRequest(**search_params_synopsis)
        req_related = AnnSearchRequest(**search_params_related)
        req_genres = AnnSearchRequest(**search_params_genres)
        req_studios = AnnSearchRequest(**search_params_studios)

        reqs = [req_synopsis, req_related, req_genres, req_studios]

        # Define reranking strategy
        # (0.2, 0.1, 0.6, 0.1) seemed to work well in one case
        rerank = WeightedRanker(0.2, 0.1, 0.6, 0.1)

        # Get topk combined recommendations
        res = self.anime_collection.hybrid_search(
            reqs=reqs,
            rerank=rerank,
            limit=limit
        )

        # Transform SearchResult to list of Recommendations
        recommendations = []
        for rec in res[0]:
            anime = self.anime_df[self.anime_df['id'] == rec.pk]
            recommendations.append(Recommendation(anime_id=rec.pk,
                                                  title=anime['title'].values[0],
                                                  similarity=rec.score))
        return recommendations


    def recommend(self, user_name: str, limit: int) -> list[Recommendation]:
        # TODO: implement this

        # 1. Get users anime list
        # 2. If user has no anime in their list, return random recommendations
        # 3. If user has anime in their list, that are not in the dataset, get info on those anime from MAL api
        # 4. preprocess new anime info
        # 5. perform hybrid vector search with users anime embeddings
        # 6. add new anime info to the vector database and dataset (after generating recommendations)
        # 7. return recommendations
        # 8. profit

        # Get users anime list
        user_anime_list = self._get_user_anime_list(user_name)

        # If user has no anime in their list, return random recommendations
        if len(user_anime_list['data']) == 0:
            return self._get_random_recommendations(limit)

        # If user has anime in their list, that are not in the dataset, get info on those anime from MAL api
        new_anime_info = self._get_new_anime_info(user_anime_list)

        # Preprocess new anime info
        new_anime_info_preprocessed = preprocess.process(new_anime_info, config=self.config['preprocessing'])

        # Generate recommendations
        recommendations = self._get_topk_recommendations(new_anime_info_preprocessed, limit)

        # Add new anime info to the vector database
        self.anime_collection.insert(new_anime_info_preprocessed)

        # Add new anime info to the dataset
        self.anime_df.append(new_anime_info, ignore_index=True)
        self.anime_df.to_csv(os.path.join(self.config['DATA_DIR'], 'raw', 'anime_data.csv'), index=False)

        return recommendations


t = """
'id,' # Anime ID (integer)\
🔵🛑'title,' # Anime title (string)\
🔵✅'synopsis,' # Anime synopsis (string or null)\
🔵✅'mean,' # Mean score (float or null)\
🔵✅'popularity,' # Popularity rank (integer or null)\
🔵🛑'num_list_users,' # Number of users who have the anime in their list (integer)\
🔵✅'num_scoring_users,' # Number of users who have scored the anime (integer)\
🔵✅'nsfw,' # NSFW classification (white=sfw, gray=partially, black=nsfw) (string or null)\
🔵✅'genres,' # Genres (array of objects)\
🔵✅'studios,' # Studios (array of objects)\
🔵✅'num_episodes,' # Number of episodes (integer)\
🔵✅'average_episode_duration,' # Average duration of an episode (integer or null)\
🔵✅'status,' # Airing status (string)\
🔵✅'rating,' # Age rating (string or null) (g, pg, pg_13, r, r+, rx)\
🔵✅'source,' # Source (string or null)\
🔵✅'media_type,' # Media type (string)\
🔵🛑'created_at,' # Date of creation (string <date-time>)\
🔵🛑'updated_at,' # Date of last update (string <date-time>)\
🔵✅'start_season,' # Start season (object or null)\
🔵✅'start_date,' # Start date (string or null)\
🔵✅'end_date,' # End date (string or null)\
✅'related_anime,' # Related anime (array of objects)\
🛑'related_manga,' # Related manga (array of objects)\
🛑'recommendations,' # Recommendations (array of objects)\
✅'statistics' # Statistics (object or null)
"""