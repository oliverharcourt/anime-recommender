import dataclasses
import itertools

import pandas as pd
import requests
from pymilvus import AnnSearchRequest, SearchResult, WeightedRanker, Collection


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

    def __init__(self, config, collection, dataset):
        self.config: dict = config
        self.anime_collection: Collection = collection
        self.anime_df: pd.DataFrame = dataset
        self.BASE_URL: str = config['BASE_URL']
        self.request_headers = {
            'Authorization': f"Bearer {config['MAL_ACCESS_TOKEN']}"}

    def _get_user_anime_list(self, user_name: str,
                             limit: int = 1000) -> pd.DataFrame:
        # Get users anime list through MAL api
        user_anime_df = pd.DataFrame({'id': [], 'title': [], 'score': [], 'status': [
        ], 'episodes_watched': [], 'rewatching': []}).astype(
            {'id': 'int32', 'title': 'string', 'score': 'int32', 'status': 'string', 'episodes_watched': 'int32', 'rewatching': 'bool'})
        print(f"Created empty user anime list dataframe: {user_anime_df}")

        # Set initial offset
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

                # print(
                # f"Received user anime list from MAL api: {user_anime_list}")

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
                          limit: int, exclude_ids: list) -> SearchResult:

        # Create search params
        search_params_synopsis = {
            "data": [hybrid_query_vec['synopsis_embedding']],
            "anns_field": "synopsis_embedding",
            "param": {
                "metric_type": "COSINE",
                "params": {"nprobe": self.config['nprobe']['text']}
            },
            # Exclude the query anime from the search results
            "expr": f"id not in {exclude_ids}",
            "limit": limit
        }

        search_params_related = {
            "data": [hybrid_query_vec['related_embedding']],
            "anns_field": "related_embedding",
            "param": {
                "metric_type": "COSINE",
                "params": {"nprobe": self.config['nprobe']['text']}
            },
            # Exclude the query anime from the search results
            "expr": f"id not in {exclude_ids}",
            "limit": limit
        }

        search_params_genres = {
            "data": [hybrid_query_vec['genres']],
            "anns_field": "genres",
            "param": {
                "metric_type": "L2",
                "params": {"nprobe": self.config['nprobe']['other']}
            },
            # Exclude the query anime from the search results
            "expr": f"id not in {exclude_ids}",
            "limit": limit
        }

        search_params_studios = {
            "data": [hybrid_query_vec['studios']],
            "anns_field": "studios",
            "param": {
                "metric_type": "L2",
                "params": {"nprobe": self.config['nprobe']['other']}
            },
            # Exclude the query anime from the search results
            "expr": f"id not in {exclude_ids}",
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
        # order: synopsis, related, genres, studios
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
            self, user_anime_list: pd.DataFrame, limit: int) -> dict[int, tuple[int, list]]:
        print(f"Columns of user anime list: {user_anime_list.columns}")

        # Filter out animes that are not in the dataset and drop duplicate
        # animes
        relevant_animes = user_anime_list.loc[user_anime_list['id'].isin(
            self.anime_df['id']), ['id', 'score', 'status']].drop_duplicates(subset='id')

        # Only consider animes that the user has completed
        relevant_animes = relevant_animes[relevant_animes['status']
                                          == 'completed']  # .head(3)

        print(f"Relevant animes: {relevant_animes}")

        # ids of animes to fetch from the database to use as the query vectors
        # of the hybrid search
        ids = relevant_animes['id'].tolist()

        exclude_ids_from_status = {
            'dropped', 'on_hold', 'watching', 'completed'
        }
        exclude_ids = user_anime_list.loc[user_anime_list['status'].isin(
            exclude_ids_from_status)].drop_duplicates(subset='id')['id'].tolist()

        print(f"Number of ids to exclude: {len(exclude_ids)}")

        print(f"exclude_ids: {sorted(exclude_ids)}")

        # Get anime embeddings from vector database
        present_animes = self._get_db_entries_by_id(ids)

        res = {}

        # Perform hybrid search
        for anime in present_animes:
            anime_id = anime['id']
            anime_score = relevant_animes[relevant_animes['id']
                                          == anime_id]['score'].values[0]

            # wrap query vectors in outer list to match the format of the
            # hybrid search method
            anime['synopsis_embedding'] = anime['synopsis_embedding']
            anime['related_embedding'] = anime['related_embedding']
            anime['genres'] = anime['genres']
            anime['studios'] = anime['studios']

            query_res = self._do_hybrid_search(
                anime, limit=limit, exclude_ids=exclude_ids)
            # We need a map from query anime id to recommendations generated
            # from that anime to scale the similarity scores later
            res[anime_id] = (anime_score, query_res)

        return res

    def _scale_recommendations(
            self, recommendations: dict[int, tuple[int, list]], default_score: int = 7) -> dict[int, list]:
        """Scales recommendations by multiplying similarity scores with user scores.

        Args:
            recommendations: A dictionary mapping anime IDs to tuples of
                (user_score, list_of_recommendations).

        Returns:
            A dictionary mapping anime IDs to scaled recommendation lists.
        """
        scaled_recommendations = {}  # Initialize the result dictionary

        for content_id, (user_score,
                         recommendation_list) in recommendations.items():
            # Iterate over each content ID and its associated tuple

            scaled_list = []
            # Iterate over each recommendation
            for rec in recommendation_list[0]:
                # Initialize the scaled recommendation as a dictionary, since
                # rec is a pymilvus.Hit object
                # print(f"rec: {rec}")
                scaled_rec = {}
                scaled_rec['id'] = rec.pk  # Copy the anime ID
                scaled_rec['distance'] = rec.score  # Copy the similarity score
                factor = user_score if user_score != 0 else default_score
                scaled_rec['user_score'] = user_score
                scaled_rec['factor'] = factor
                scaled_rec["scaled_distance"] = scaled_rec['distance'] * factor
                scaled_list.append(scaled_rec)  # Add the scaled recommendation

            scaled_recommendations[content_id] = scaled_list

        return scaled_recommendations

    def _flatsort_recommendations(
            self, recommendations: dict[int, list]) -> list[dict]:
        flattend_recommendations = []
        for recommendation_list in recommendations.values():
            flattend_recommendations.extend(recommendation_list)
        return sorted(flattend_recommendations,
                      key=lambda x: x['scaled_distance'], reverse=True)

    def recommend(self, user_name: str, limit: int) -> pd.DataFrame:
        """Recommends anime to a user based on their anime list.

        Args:
            user_name (str): The name of the user to recommend anime to.
            limit (int): The number of anime to recommend.

        Returns:
            pd.DataFrame: A dataframe containing the recommended anime.
        """
        user_anime_list = self._get_user_anime_list(user_name)

        if len(user_anime_list) == 0:
            return self._get_random_recommendations(limit)

        recommendations = self._get_topk_recommendations(
            user_anime_list, limit)

        recommendations = self._scale_recommendations(recommendations)

        recommendations = pd.DataFrame(
            self._flatsort_recommendations(recommendations))

        recommendations.drop_duplicates(subset='id', inplace=True)

        recommendations['link'] = recommendations.apply(
            lambda x: f"https://myanimelist.net/anime/{int(x['id'])}", axis=1)

        # recommendations = self._format_recommendations(recommendations)

        return recommendations.head(limit)

    def recommend_by_id(self, anime_id: int, limit: int) -> pd.DataFrame:
        """Recommends anime similar to the given anime.

        Args:
            anime_id (int): The ID of the anime to recommend similar anime to.
            limit (int): The number of anime to recommend.

        Returns:
            pd.DataFrame: A dataframe containing the recommended anime.
        """

        print(f"Recommendations for anime with id: {anime_id}")

        # Get the animes embedding
        vectors = self._get_db_entries_by_id([anime_id])
        #  There should only be one anime with the given id
        vector = vectors[0]

        # Perform hybrid search
        query_res = self._do_hybrid_search(
            vector, limit=limit, exclude_ids=[anime_id])

        res = []

        for rec in query_res[0]:
            res.append(
                {
                    'id': rec.pk,
                    'distance': rec.score,
                    'link': f"https://myanimelist.net/anime/{rec.pk}"
                }
            )

        return pd.DataFrame(res)
