import sys

import pandas as pd
import requests
from pymilvus import AnnSearchRequest, Collection, SearchResult, WeightedRanker


class Recommender:
    """Recommends anime to users based on their anime list or to an anime based on its ID."""

    def __init__(self, config, collection, dataset):
        self.config: dict = config
        self.anime_collection: Collection = collection
        self.anime_df: pd.DataFrame = dataset
        self.base_url: str = config['BASE_URL']
        self.request_headers = {
            'Authorization': f"Bearer {config['MAL_ACCESS_TOKEN']}"
        }

    def _get_user_anime_list(self, user_name: str,
                             limit: int = 1000) -> pd.DataFrame:
        # Get users anime list through MAL api
        user_anime_df = pd.DataFrame({
            'id': [],
            'title': [],
            'score': [],
            'status': [],
            'episodes_watched': [],
            'rewatching': []
        }).astype({
            'id': 'int32',
            'title': 'string',
            'score': 'int32',
            'status': 'string',
            'episodes_watched': 'int32',
            'rewatching': 'bool'
        })

        # request user anime list from MAL api
        # print(f"Fetching anime list for user: {user_name}")
        offset = 0

        # While there are more animes to fetch
        while True:
            try:
                params = {
                    'fields': 'list_status',
                    'limit': limit,
                    'offset': offset,
                    # 'status': 'completed',
                }
                user_anime_list = requests.get(
                    url=f"{self.base_url}/users/{user_name}/animelist",
                    headers=self.request_headers,
                    params=params,
                    timeout=10
                )

            except Exception as e:
                print(e)

            if user_anime_list.status_code != 200:
                print(
                    f"Failed to fetch anime list for user: {user_name}, status code: {user_anime_list.status_code}")
                sys.exit(1)

            user_anime_list = user_anime_list.json()

            if not user_anime_list['paging']:
                break

            offset += limit

        for entry in user_anime_list['data']:
            merged = entry['node'] | entry['list_status']
            new_row = {
                'id': int(merged['id']),
                'title': merged['title'],
                'score': int(merged['score']),
                'status': merged['status'],
                'episodes_watched': int(merged['num_episodes_watched']),
                'rewatching': merged['is_rewatching']
            }
            user_anime_df = pd.concat(
                [user_anime_df, pd.DataFrame(new_row, index=[0])],
                ignore_index=True
            )

        return user_anime_df

    def _get_random_recommendations(self, limit: int) -> pd.DataFrame:
        # Get random recommendations
        random_sample = self.anime_df.sample(n=limit)[['id', 'title']]

        random_sample['link'] = random_sample.apply(
            lambda x: f"https://myanimelist.net/anime/{int(x['id'])}",
            axis=1
        )
        return random_sample

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
        res = self.anime_collection.query(
            expr=f"id in {ids}",
            output_fields=["*"]
        )
        return res

    def _get_topk_recommendations(
            self, user_anime_list: pd.DataFrame, limit: int) -> dict[int, tuple[int, list]]:

        # Filter out animes that are not in the dataset and drop duplicate
        # animes
        relevant_animes = user_anime_list.loc[user_anime_list['id'].isin(
            self.anime_df['id']), ['id', 'score', 'status']]

        # print(f"Num anime in user list: {len(user_anime_list)}, Num relevant animes: {len(relevant_animes)}")
        # print(f"Animes not in dataset: {user_anime_list[~user_anime_list['id'].isin(self.anime_df['id'])]}")
        # TODO: Add never seen animes to the local dataset

        relevant_animes.drop_duplicates(subset='id', inplace=True)

        # print(f"Num relevant animes after dropping duplicates: {len(relevant_animes)}")

        # Recommendations only based on animes the user has completed
        relevant_animes = relevant_animes[relevant_animes['status']
                                          == 'completed']

        ids = relevant_animes['id'].tolist()

        excluded_status = {
            'dropped', 'on_hold', 'watching', 'completed', 'plan_to_watch'
        }
        excluded_ids = user_anime_list.loc[user_anime_list['status'].isin(
            excluded_status)].drop_duplicates(subset='id')['id'].tolist()

        present_animes = self._get_db_entries_by_id(ids)

        res = {}

        # Perform hybrid search
        for anime in present_animes:
            anime_id = anime['id']
            anime_score = relevant_animes[
                relevant_animes['id'] == anime_id
            ]['score'].values[0]

            query_res = self._do_hybrid_search(
                anime,
                limit=limit,
                exclude_ids=excluded_ids
            )
            # Map from query anime id to recommendations generated
            # from that anime for later weighting
            res[anime_id] = (anime_score, query_res)
        return res

    def _scale_recommendations(
            self, recommendations: dict[int, tuple[int, list]], default_scaling_factor: float = 7.0) -> dict[int, list]:
        """Scales recommendations.

        Args:
            recommendations: A dictionary mapping anime IDs to tuples of
                (user_score, list_of_recommendations).
                default_scaling_factor: Factor to scale the similarity scores with, defaults to 7.0.

        Returns:
            A dictionary mapping anime IDs to scaled recommendation lists.
        """

        scaled_recommendations = {}

        for content_id, (user_score,
                         recommendation_list) in recommendations.items():

            scaled_list = []
            # Iterate over each recommendation
            for rec in recommendation_list[0]:
                # Initialize the scaled recommendation as a dictionary, since
                # rec is a pymilvus.Hit object
                scaled_rec = {}
                scaled_rec['id'] = rec.pk  # Copy the anime ID
                scaled_rec['distance'] = rec.score  # Copy the similarity score
                factor = user_score if user_score != 0 else default_scaling_factor
                scaled_rec['user_score'] = user_score
                scaled_rec['factor'] = factor
                scaled_rec["scaled_distance"] = scaled_rec['distance'] * factor
                scaled_list.append(scaled_rec)

            scaled_recommendations[content_id] = scaled_list

        return scaled_recommendations

    def _flatsort_recommendations(
            self, recommendations: dict[int, list]) -> list[dict]:
        """Flattens and sorts recommendations.

        Args:
            recommendations: A dictionary mapping anime IDs to lists of recommendations.

        Returns:
            A flattened, sorted list of recommendations.
        """
        flattend_recommendations = []
        for recommendation_list in recommendations.values():
            flattend_recommendations.extend(recommendation_list)
        return sorted(flattend_recommendations,
                      key=lambda x: x['scaled_distance'], reverse=True)

    def recommend_by_username(self, user_name: str, limit: int) -> pd.DataFrame:
        """Recommends anime to a user based on their anime list.

        Args:
            user_name (str): The name of the user to recommend anime to.
            limit (int): The number of anime to recommend.

        Returns:
            pd.DataFrame: A dataframe containing the recommended anime.
        """
        user_anime_list = self._get_user_anime_list(user_name)

        # Median score for later weighting
        median_score = user_anime_list[user_anime_list['score'] != 0]['score'].median(
        )

        if len(user_anime_list) == 0:
            return self._get_random_recommendations(limit)

        recommendations = self._get_topk_recommendations(
            user_anime_list, limit)

        recommendations = self._scale_recommendations(
            recommendations, default_scaling_factor=median_score)

        recommendations = pd.DataFrame(
            self._flatsort_recommendations(recommendations))

        recommendations.drop_duplicates(subset='id', inplace=True)

        recommendations['link'] = recommendations.apply(
            lambda x: f"https://myanimelist.net/anime/{int(x['id'])}", axis=1)

        recommendations.reset_index(inplace=True, drop=True)

        recommendations.drop(
            columns=["distance", "user_score", "factor"], inplace=True)

        recommendations.rename(
            columns={"scaled_distance": "distance"}, inplace=True)

        return recommendations.head(limit)

    def recommend_by_id(self, anime_id: int, limit: int) -> pd.DataFrame:
        """Recommends anime similar to the given anime.

        Args:
            anime_id (int): The ID of the anime to recommend similar anime to.
            limit (int): The number of anime to recommend.

        Returns:
            pd.DataFrame: A dataframe containing the recommended anime.
        """

        # Get the animes embedding
        anime_id = int(anime_id)
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
