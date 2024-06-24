import dataclasses
import itertools

import pandas as pd
import requests
from pymilvus import AnnSearchRequest, SearchResult, WeightedRanker


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
    link: str


class Recommender:

    def __init__(self, config, model, collection, dataset):
        self.config = config
        self.model = model
        self.anime_collection = collection
        self.anime_df = dataset
        self.BASE_URL = config['BASE_URL']
        self.request_headers = {
            'Authorization': f"Bearer {config['MAL_ACCESS_TOKEN']}"}

    def _get_user_anime_list(self, user_name: str,
                             limit: int = 1000) -> pd.DataFrame:
        # Get users anime list through MAL api
        user_anime_df = pd.DataFrame({'id': [], 'title': [], 'score': [], 'status': [
        ], 'episodes_watched': [], 'rewatching': []}).astype(
            {'id': 'int32', 'title': 'string', 'score': 'int32', 'status': 'string', 'episodes_watched': 'int32', 'rewatching': 'bool'})
        print(f"Created empty user anime list dataframe: {user_anime_df}")

        #  Set initial offset
        offset = 0

        # While there are more animes to fetch
        while True:
            try:
                # request user anime list from MAL api
                print(
                    f"Requesting user anime list from MAL api for user: {user_name}")

                params = {
                    'fields': 'list_status',
                    'limit': limit,
                    'offset': offset,
                    # 'status': 'completed',
                }
                user_anime_list = requests.get(
                    url=f"{self.BASE_URL}/users/{user_name}/animelist",
                    headers=self.request_headers,
                    params=params,
                    timeout=10
                ).json()

                print(
                    f"Received user anime list from MAL api: {user_anime_list}")

            except Exception as e:
                print(e)

            if not user_anime_list['paging']:
                break

            offset += limit

        print("Adding user anime list to dataframe...")
        for entry in user_anime_list['data']:
            merged = entry['node'] | entry['list_status']
            new_row = {'id': int(merged['id']), 'title': merged['title'],
                       'score': int(merged['score']), 'status': merged['status'],
                       'episodes_watched': int(merged['num_episodes_watched']),
                       'rewatching': merged['is_rewatching']}
            user_anime_df = pd.concat(
                [user_anime_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        return user_anime_df

    def _get_random_recommendations(self, limit: int) -> list[Recommendation]:
        # TODO: check and test this method
        # Get random recommendations
        random_sample = self.anime_df.sample(n=limit)
        recommendations = []
        for _, anime in random_sample.iterrows():
            recommendations.append(Recommendation(anime_id=anime['id'],
                                                  title=anime['title'],
                                                  similarity=-1.0,
                                                  link=f"https://myanimelist.net/anime/{anime['id']}"))
        return recommendations

    def _do_hybrid_search(self, hybrid_query_vec: dict,
                          limit: int) -> SearchResult:

        # Create search params
        search_params_synopsis = {
            "data": hybrid_query_vec['synopsis_embedding'],
            "anns_field": "synopsis_embedding",
            "param": {
                "metric_type": "COSINE",
                "params": {"nprobe": self.config['nprobe']['text']}
            },
            # Exclude the query anime from the search results
            "expr": f"id not in {hybrid_query_vec['id']}",
            "limit": limit
        }

        search_params_related = {
            "data": hybrid_query_vec['related_embedding'],
            "anns_field": "related_embedding",
            "param": {
                "metric_type": "COSINE",
                "params": {"nprobe": self.config['nprobe']['text']}
            },
            "limit": limit
        }

        search_params_genres = {
            "data": hybrid_query_vec['genres'],
            "anns_field": "genres",
            "param": {
                "metric_type": "L2",
                "params": {"nprobe": self.config['nprobe']['other']}
            },
            "limit": limit
        }

        search_params_studios = {
            "data": hybrid_query_vec['studios'],
            "anns_field": "studios",
            "param": {
                "metric_type": "L2",
                "params": {"nprobe": self.config['nprobe']['other']}
            },
            "limit": limit
        }

        # Create request objects
        reqs = [
            AnnSearchRequest(**search_params_synopsis),
            AnnSearchRequest(**search_params_related),
            AnnSearchRequest(**search_params_genres),
            AnnSearchRequest(**search_params_studios)
        ]

        # Define reranking strategy
        #  order: synopsis, related, genres, studios
        # (0.2, 0.1, 0.6, 0.1) seemed to work well in one case
        rerank = WeightedRanker(0.2, 0.1, 0.6, 0.1)

        # Get topk combined recommendations
        res = self.anime_collection.hybrid_search(
            reqs=reqs,
            rerank=rerank,
            limit=limit
        )

        return res

    def _get_db_entries_by_id(self, ids: list) -> SearchResult:
        # Get anime embeddings from vector database
        res = self.anime_collection.query(
            expr=f"id in {ids}",
            output_fields=["*"]
        )

        return res

    def _get_topk_recommendations(
            self, anime_ids: list, limit: int) -> list[Recommendation]:
        # TODO: check and test this method

        data = self.anime_df[self.anime_df['id'].isin(anime_ids)]

        present_animes = self._get_db_entries_by_id(
            data['id'].drop_duplicates().tolist())

        res = []

        # Perform hybrid search
        for anime in present_animes:
            res.append(self._do_hybrid_search(anime, limit))

        return res

    def _scale_recommendations(
            self, recommendations: list[list[dict]]) -> list[list[dict]]:
        # TODO: implement this method
        # This method will scale the similarity scores of the recommendations
        # by the users score for the anime that produced those recommendations
        # For animes where there is no user score available, default to some value
        # This method must be called before _format_recommendations, since _format_recommendations
        #  flattens the list of recommendations
        pass

    def _format_recommendations(
            self, recommendations: list[SearchResult]) -> list[Recommendation]:
        # TODO: implement this method
        # Transform SearchResult to list of Recommendations
        recommendations = list(itertools.chain(recommendations))
        res = []
        for rec in recommendations:
            anime = self.anime_df[self.anime_df['id'] == rec.pk]
            res.append(Recommendation(anime_id=rec.pk,
                                      title=anime['title'].values[0],
                                      similarity=rec.score,
                                      link=f"https://myanimelist.net/anime/{rec.pk}"))
        return res

    def recommend(self, user_name: str, limit: int) -> list[Recommendation]:
        # TODO: implement this

        # New process (after updating data_loader.py with full anime download functionality):
        # 1. Get users anime list
        # 2. If user has no anime in their list, return random recommendations
        # 3. Get anime ids from users list (they should all be in the dataset)
        # 4. Get anime embeddings from vector database
        # 5. Perform hybrid vector search

        user_anime_list = self._get_user_anime_list(user_name)

        if len(user_anime_list) == 0:
            return self._get_random_recommendations(limit)

        anime_ids = user_anime_list['id'].tolist()

        recommendations = self._get_topk_recommendations(
            anime_ids, limit)

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
