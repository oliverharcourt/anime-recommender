import time

import pandas as pd
import requests


class DataLoader:
    def __init__(self, headers: dict, base_url: str, request_delay: float = 4.15):
        r"""Initializes the data loader.

        :param headers: Headers with Authorization
        :param base_url: Base URL for the MAL API
        :param request_delay: (optional) Delay between API requests, defaults to 4.15
        """

        self.headers = headers
        self.base_url = base_url
        self.request_delay = request_delay

    def collect_anime_data(self, limit: int) -> pd.DataFrame:
        # NOTE This method cannot retrieve the related_anime, related_manga,
        # recommendations, and statistics fields

        expected_columns = [
            "id",
            "title",
            "synopsis",
            "mean",
            "popularity",
            "num_list_users",
            "num_scoring_users",
            "nsfw",
            "genres",
            "studios",
            "num_episodes",
            "average_episode_duration",
            "status",
            "rating",
            "source",
            "media_type",
            "created_at",
            "updated_at",
            "start_date",
            "end_date",
            "main_picture.medium",
            "main_picture.large",
            "start_season.year",
            "start_season.season",
        ]

        anime_details_df = pd.DataFrame(columns=expected_columns)

        field_params = {
            "fields": (
                "id,"
                # Anime ID (integer)
                "title,"
                # Anime title (string)
                "synopsis,"
                # Anime synopsis (string or null)
                "mean,"
                # Mean score (float or null)
                "popularity,"
                # Popularity rank (integer or null)
                "num_list_users,"
                # Number of users who have the anime in their list
                # (integer)
                "num_scoring_users,"
                # Number of users who have scored the anime (integer)
                "nsfw,"
                # NSFW classification (white=sfw, gray=partially,
                # black=nsfw) (string or null)
                "genres,"
                # Genres (array of objects)
                "studios,"
                # Studios (array of objects)
                "num_episodes,"
                # Number of episodes (integer)
                "average_episode_duration,"
                # Average duration of an episode (integer or null)
                "status,"
                # Airing status (string)
                "rating,"
                # Age rating (string or null) (g, pg, pg_13, r, r+, rx)
                "source,"
                # Source (string or null)
                "media_type,"
                # Media type (string)
                "created_at,"
                # Date of creation (string <date-time>)
                "updated_at,"
                # Date of last update (string <date-time>)
                "start_season,"
                # Start season (object or null)
                "start_date,"
                # Start date (string or null)
                "end_date"
                # End date (string or null)
            )
        }

        offset = 0

        print("Starting data collection...")
        while True:
            other_params = {"limit": 500, "ranking_type": "all", "offset": offset}

            combined_params = field_params | other_params

            try:
                response = requests.get(
                    f"{self.base_url}/anime/ranking",
                    headers=self.headers,
                    params=combined_params,
                    timeout=10.0,
                )
            except requests.Timeout:
                print(f"Request timed out at offset: {offset}")
            except requests.RequestException as e:
                print(f"Request failed: {str(e)}")
                break

            if response.status_code != 200:
                print(f"Failed to fetch top anime: {response.status_code}")
                break

            response = response.json()
            if not response["data"]:
                break

            for anime in response["data"]:
                new_row = pd.json_normalize(anime["node"])
                anime_details_df = pd.concat(
                    [anime_details_df, new_row], ignore_index=True
                )

            if not response["paging"]:
                break

            offset += 500
            print(f"Offset: {offset}")
            if offset >= limit * 500:
                print("Limit reached.")
                break
            time.sleep(self.request_delay)

        anime_details_df["synopsis"] = anime_details_df["synopsis"].apply(
            lambda x: x.replace("\n", " ") if pd.notnull(x) else x
        )

        return anime_details_df

    def collect_manga_data(self, offset: int = 0) -> pd.DataFrame:
        manga_details_df = pd.DataFrame(columns=["id", "title", "synopsis"])

        field_params = {
            "fields": (
                "id,"
                # Manga ID (integer)
                "title,"
                # Manga title (string)
                "synopsis"
            )
            # Manga synopsis (string or null)
        }

        print("Starting data collection...")
        while True:
            other_params = {"limit": 500, "ranking_type": "all", "offset": offset}

            combined_params = field_params | other_params

            try:
                response = requests.get(
                    f"{self.base_url}/manga/ranking",
                    headers=self.headers,
                    params=combined_params,
                    timeout=10.0,
                )
            except requests.Timeout:
                print(f"Request timed out at offset: {offset}")
            except requests.RequestException as e:
                print(f"Request failed: {str(e)}")
                break

            if response.status_code != 200:
                print(f"Failed to fetch top manga: {response.status_code}")
                break

            response = response.json()
            if not response["data"]:
                break

            for manga in response["data"]:
                new_row = pd.json_normalize(manga["node"])
                manga_details_df = pd.concat(
                    [manga_details_df, new_row], ignore_index=True
                )

            if not response["paging"]:
                break

            offset += 500
            if offset % 1000 == 0:
                print(f"Offset: {offset}")
            time.sleep(self.request_delay)

        manga_details_df["synopsis"] = manga_details_df["synopsis"].apply(
            lambda x: x.replace("\n", " ") if pd.notnull(x) else x
        )

        return manga_details_df

    def run_collection(
        self, media_type: str, output_path: str, limit: int = 100_000
    ) -> None:
        r"""Runs the data collection process.

        Args:
            media_type (str): Type of media to collect data for (anime or manga)
            output_path (str): Path to save the collected data
            limit (int): Number of records to collect in batches of 500 (for anime mode only), defaults to 100_000
        """

        data: pd.DataFrame = (
            self.collect_anime_data(limit)
            if media_type == "anime"
            else self.collect_manga_data()
        )

        data.to_json(output_path, orient="records")
