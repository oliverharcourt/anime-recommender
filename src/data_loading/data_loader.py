import json
import logging
import os
import pickle
import signal
import sys
import time
from typing import Any, Dict

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm


class DataCollector:

    def __init__(self, headers: dict, base_url: str, data_dir: str,
                 request_delay: float = 4.15, max_iterations: int = 100, debug: bool = False):
        r"""Initializes the data collector.

        :param headers: Headers with Authorization
        :param base_url: Base URL for the MAL API
        :param data_dir: Path to the data directory
        :param request_delay: (optional) Delay between API requests, defaults to 4.15
        :param max_iterations: (optional) Maximum number of iterations, defaults to 100
        :param debug: (optional) Debug mode, defaults to False
        """

        self.headers = headers
        self.base_url = base_url
        self.data_dir = data_dir
        self.request_delay = request_delay
        self.debug = debug
        self.max_iterations = max_iterations
        self.seen = set()
        self.download_queue = []
        self.anime_details = []
        self.session = requests.Session()
        signal.signal(signal.SIGINT, self._signal_handler)


    def _signal_handler(self, sig, frame):
        r"""Handles the SIGINT signal by saving the state and exiting.

        :param sig: Signal number
        :param frame: Current stack frame
        """

        print("Caught SIGINT, saving state and exiting.")
        self._save_state()
        sys.exit(0)


    def _save_state(self):
        r"""Saves the state of the data collection process.
            Saves the seen anime ids and the download queue to pickle files.
            Saves the anime details to a CSV file.

        :param seen: Set of seen anime ids
        :param download_queue: List of anime ids to download
        """

        seen_path = os.path.join(self.data_dir, 'interim', 'seen.pkl')
        queue_path = os.path.join(self.data_dir, 'interim', 'queue.pkl')

        with open(seen_path, 'wb') as f:
            pickle.dump(self.seen, f)
        with open(queue_path, 'wb') as f:
            pickle.dump(self.download_queue, f)
        
        anime_df = self._create_dataframe()

        output_path = os.path.join(self.data_dir, 'raw', 'anime_data.csv')
        anime_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)


    def _load_state(self):
        r"""Loads the state of the data collection process.
        """

        seen_path = os.path.join(self.data_dir, 'interim', 'seen.pkl')
        queue_path = os.path.join(self.data_dir, 'interim', 'queue.pkl')

        if os.path.exists(seen_path):
            with open(seen_path, 'rb') as f:
                self.seen = pickle.load(f)

        if os.path.exists(queue_path):
            with open(queue_path, 'rb') as f:
                self.download_queue = pickle.load(f)


    def _fetch_top_anime(self, limit: int = 500):
        r"""Fetches the top anime from MyAnimeList and adds them to the download queue.
            This function is only used to start the data collection process.

        :param limit: (optional) Number of anime to fetch, defaults to 500
        :raises Exception: Failed to fetch top anime
        """
        
        anime_list_url = f"{self.base_url}/anime/ranking"
        response = self.session.get(anime_list_url, headers=self.headers, params={'limit': limit, 'ranking_type': 'all'})
        if response.status_code == 200:
            for anime in response.json()['data']:
                if anime['node']['id'] not in self.seen:
                    self.download_queue.append(anime['node']['id'])
                    self.seen.add(anime['node']['id'])
        else:
            logging.error(f"Failed to fetch top anime: {response.status_code if response else None}")
            raise Exception("Failed to fetch top anime")


    def _fetch_anime_details(self, anime_id: int, max_retries: int = 5, base_delay: float = 1.0, timeout: float = 10.0) -> dict:
        r"""Fetches the details of an anime from MyAnimeList.
        
        :param anime_id: Anime ID
        :param max_retries: (optional) Maximum number of retries, defaults to 5
        :param base_delay: (optional) Base delay in seconds for exponential backoff, defaults to 1.0
        :param timeout: (optional) Maximum time in seconds to wait for a response, defaults to 10.0
        :return: JSON object with fetched anime details or an empty dictionary if the request failed
        :rtype: dict
        """

        # Include fields you are interested in, refer to API for more options
        params = {
            'fields': ('id,' # Anime ID (integer)
                    'title,' # Anime title (string)
                    'synopsis,' # Anime synopsis (string or null)
                    'mean,' # Mean score (float or null)
                    'popularity,' # Popularity rank (integer or null)
                    'num_list_users,' # Number of users who have the anime in their list (integer)
                    'num_scoring_users,' # Number of users who have scored the anime (integer)
                    'nsfw,' # NSFW classification (white=sfw, gray=partially, black=nsfw) (string or null)
                    'genres,' # Genres (array of objects)
                    'studios,' # Studios (array of objects)
                    'num_episodes,' # Number of episodes (integer)
                    'average_episode_duration,' # Average duration of an episode (integer or null)
                    'status,' # Airing status (string)
                    'rating,' # Age rating (string or null) (g, pg, pg_13, r, r+, rx)
                    'source,' # Source (string or null)
                    'media_type,' # Media type (string)
                    'created_at,' # Date of creation (string <date-time>)
                    'updated_at,' # Date of last update (string <date-time>)
                    'start_season,' # Start season (object or null)
                    'start_date,' # Start date (string or null)
                    'end_date,' # End date (string or null)
                    'related_anime,' # Related anime (array of objects)
                    'related_manga,' # Related manga (array of objects)
                    'recommendations,' # Recommendations (array of objects)
                    'statistics') # Statistics (object or null)
        }

        url = f"{self.base_url}/anime/{anime_id}"
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, headers=self.headers, params=params, timeout=timeout)
                
                if response.status_code == 200:
                    # Successful request
                    #logging.info(f"Request successful for anime id: {anime_id}\nHeaders: {response.headers}")
                    return response.json()
                else:
                    # Received a response other than 200 OK, handle: log, wait, and possibly retry
                    print(f"Request failed with status code: {response.status_code}")
                    logging.warning(f"Request failed with status code: {response.status_code}\nHeaders: {response.headers}")
                    backoff = base_delay * (2 ** attempt)
                    print(f"Retrying in {backoff} seconds.")
                    time.sleep(backoff)
            except requests.Timeout:
                # The request timed out, log and potentially retry
                print(f"Request timed out after {timeout} seconds.")
                logging.warning(f"Request timed out after {timeout} seconds.")
                backoff = base_delay * (2 ** attempt)
                print(f"Retrying in {backoff} seconds.")
                time.sleep(backoff)
            except requests.RequestException as e:
                # General requests exception (includes ConnectionError, HTTPError, etc.)
                print(f"Request failed: {str(e)}")
                logging.error(f"Request failed: {str(e)}")
                backoff = base_delay * (2 ** attempt)
                print(f"Retrying in {backoff} seconds.")
                time.sleep(backoff)

        # All retries failed
        print("Max retries exceeded.")
        return {}


    def _get_related_anime_ids(self, anime: Dict[str, Any], include_recommended: bool = True) -> list:
        r"""Extracts the related anime ids from an anime json object.

        :param anime: Anime json object
        :param include_recommended: (optional) Whether to include recommended anime ids, defaults to True
        :return: List of related anime ids
        :rtype: list
        :raises KeyError: Structure of the anime json object is not as expected
        """

        try:
            related_anime = []
            for related in anime['related_anime']:
                related_anime.append(related['node']['id'])
            if include_recommended:
                for recommendation in anime['recommendations']:
                    related_anime.append(recommendation['node']['id'])
            return related_anime
        except KeyError as e:
            print(f"KeyError: {str(e)}")
            raise e


    def _create_dataframe(self) -> pd.DataFrame:
        r"""Converts a list of anime details to a cleaned DataFrame.

        :return: Cleaned anime dataframe
        :rtype: pd.DataFrame
        :raises Exception: Failed to clean synopsis
        """

        anime_df = pd.json_normalize(self.anime_details)

        expected_columns = ['id', 'title', 'synopsis', 'mean', 'popularity', 'num_list_users',
                            'num_scoring_users', 'nsfw', 'genres', 'studios', 'num_episodes',
                            'average_episode_duration', 'status', 'rating', 'source', 'media_type',
                            'created_at', 'updated_at', 'start_date', 'end_date', 'related_anime',
                            'related_manga', 'recommendations', 'main_picture.medium',
                            'main_picture.large', 'start_season.year', 'start_season.season',
                            'statistics.status.watching', 'statistics.status.completed',
                            'statistics.status.on_hold', 'statistics.status.dropped',
                            'statistics.status.plan_to_watch', 'statistics.num_list_users']

        anime_df = anime_df.reindex(columns=expected_columns)

        try:
            anime_df['synopsis'] = anime_df['synopsis'].apply(lambda x: x.replace("\n", " ") if pd.notnull(x) else x)
            anime_df['synopsis'] = anime_df['synopsis'].apply(lambda x: x.strip() if pd.notnull(x) else x)
        except Exception as e:
            with open('data.json', 'w', encoding='utf-8') as f:
                json.dump(self.anime_details, f, ensure_ascii=False, indent=4)
            raise e

        return anime_df


    def _add_ids_to_queue(self, anime_ids: list):
        r"""Adds not seen anime ids to the download queue.

        :param anime_ids: List of anime ids
        """

        for id in anime_ids:
            if id not in self.seen:
                self.download_queue.append(id)
                self.seen.add(id)


    def _collect_data(self):
        r"""Main data collection loop.
        """

        pbar = tqdm(total=self.max_iterations, desc="Fetching anime details")

        iterations = 0

        # While there are anime to fetch and the iteration limit has not been reached
        while self.download_queue and iterations < self.max_iterations:
            id = self.download_queue.pop(0)

            # Refresh the session every 100 iterations
            if iterations % 100 == 0 and iterations > 0:
                self.session = requests.Session()
                if self.debug:
                    print("Refreshing session.")

            fetched_details = self._fetch_anime_details(anime_id=id)

            # Request failed, recover the id and exit
            if not fetched_details:
                self.download_queue.append(id)
                print("Request failed, exiting...")
                break

            self.anime_details.append(fetched_details)

            iterations += 1
            pbar.update(1)

            # Sleep to respect the rate limit (keep at >= 4.5 seconds to avoid rate limit errors)
            time.sleep(self.request_delay)

            try:
                related_anime_ids = self._get_related_anime_ids(fetched_details)
            except KeyError as e:
                self.download_queue.append(id)
                print(f"KeyError: {str(e)}")
                break

            self._add_ids_to_queue(related_anime_ids)

        pbar.close()

        print('Length of seen:', len(self.seen), 'Length of download queue:', len(self.download_queue))
        if self.debug:
            print('Next anime to fetch:', self.download_queue[0] if self.download_queue else None)


    def run(self):
        r"""Runs the full data collection process.
        """

        self._load_state()

        if not self.download_queue:
            self._fetch_top_anime()

        self._collect_data()

        self.session.close()

        self._save_state()

    def collect(self, anime_ids: list):
        r"""Collects the details of a list of anime ids.

        :param anime_ids: List of anime ids
        :return: List of anime details
        :rtype: list
        """

        anime_details = []

        for anime_id in anime_ids:
            fetched_details = self._fetch_anime_details(anime_id=anime_id)
            if fetched_details:
                anime_details.append(fetched_details)
            time.sleep(self.request_delay)

        return anime_details


if __name__ == "__main__":
    load_dotenv()

    # Define base URL for the MAL API
    BASE_URL = "https://api.myanimelist.net/v2"

    data_dir = os.getenv('DATA_DIR')

    access_token = os.getenv('MAL_ACCESS_TOKEN')

    if not access_token:
        raise ValueError("Missing MAL_ACCESS_TOKEN. Check .env file.")

    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    logging.basicConfig(filename=os.path.join(os.getenv('LOGS_DIR'), 'data_collector.log'),
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    collector = DataCollector(headers=headers, base_url=BASE_URL,
                              data_dir=data_dir, request_delay=4.15,
                              max_iterations=500, debug=False)

    collector.run()
