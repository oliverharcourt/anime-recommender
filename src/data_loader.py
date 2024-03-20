from typing import Any, Dict
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import time
import json
import re
from tqdm import tqdm
import pickle

# Define base URL for the MAL API
base_url = "https://api.myanimelist.net/v2"

def fetch_top_anime(headers: dict, limit: int = 500) -> dict:
    r"""Fetches the top anime from MyAnimeList.

    :param headers: Headers with Authorization
    :param limit: (optional) Number of anime to fetch, defaults to 500
    :return: List of anime
    :rtype: list
    :raises Exception: Failed to fetch top anime
    """
    
    anime_list_url = f"{base_url}/anime/ranking"
    response = requests.get(anime_list_url, headers=headers, params={'limit': limit, 'ranking_type': 'all'})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch top anime")


def fetch_anime_details(anime_id: int, headers: dict, session: requests.Session, max_retries: int = 5, base_delay: float = 1.0, timeout: float = 10.0) -> dict:
    r"""Fetches the details of an anime from MyAnimeList.
    
    :param anime_id: Anime ID
    :param headers: Headers with Authorization
    :param max_retries: (optional) Maximum number of retries, defaults to 5
    :param base_delay: (optional) Base delay in seconds for exponential backoff, defaults to 0.5
    :param timeout: (optional) Maximum time in seconds to wait for a response, defaults to 10.0
    :return: The fetched anime details or an empty dictionary if the request failed
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

    url = f"{base_url}/anime/{anime_id}"
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, params=params, timeout=timeout)
            
            if response.status_code == 200:
                # Successful request
                return response.json()
            else:
                # Received a response other than 200 OK, handle: log, wait, and possibly retry
                print(f"Request failed with status code: {response.status_code}")
                backoff = base_delay * (2 ** attempt)
                print(f"Retrying in {backoff} seconds.")
                time.sleep(backoff)
        except requests.Timeout:
            # The request timed out, log and potentially retry
            print(f"Request timed out after {timeout} seconds.")
            backoff = base_delay * (2 ** attempt)
            print(f"Retrying in {backoff} seconds.")
            time.sleep(backoff)
        except requests.RequestException as e:
            # General requests exception (includes ConnectionError, HTTPError, etc.)
            print(f"Request failed: {str(e)}")
            backoff = base_delay * (2 ** attempt)
            print(f"Retrying in {backoff} seconds.")
            time.sleep(backoff)
    
    # All retries failed
    print("Max retries exceeded.")
    return {}
    

def get_related_anime_ids(anime: Dict[str, Any], include_recommended: bool = True) -> list:
    r"""Extracts the related anime ids from an anime json object.

    :param anime: Anime json object
    :return: List of related anime ids
    :rtype: list
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


def create_dataframe(anime_details: list) -> pd.DataFrame:
    r"""Converts a list of anime details to a cleaned DataFrame.

    :param anime_details: List of anime details
    :return: Cleaned anime dataframe
    :rtype: pd.DataFrame
    """

    # Convert json to dataframe
    anime_df = pd.json_normalize(anime_details)

    try:
        # Clean up the dataframe
        # Remove mal rewrite from synopsis
        anime_df['synopsis'] = anime_df['synopsis'].apply(lambda x: x.replace("[Written by MAL Rewrite]", "") if pd.notnull(x) else x)
        # Remove newlines
        anime_df['synopsis'] = anime_df['synopsis'].apply(lambda x: x.replace("\n", " ") if pd.notnull(x) else x)
        # Remove leading and trailing double quotes
        anime_df['synopsis'] = anime_df['synopsis'].apply(lambda x: x.strip("\"") if pd.notnull(x) else x)
        anime_df['synopsis'] = anime_df['synopsis'].apply(lambda x: x.strip() if pd.notnull(x) else x)
        # Escape double quotes and wrap in double quotes
        anime_df['synopsis'] = anime_df['synopsis'].apply(lambda x: x.replace('"', '""') if pd.notnull(x) else x)
        anime_df['synopsis'] = anime_df['synopsis'].apply(lambda x: re.sub(r'"{2,}', '"', x) if pd.notnull(x) else x)
    except Exception as e:
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(anime_details, f, ensure_ascii=False, indent=4)
        raise e

    return anime_df


def save_state(seen: set, download_queue: list, seen_path: str = None, queue_path: str = None):
    r"""Saves the state of the data collection process.

    :param seen: Set of seen anime ids
    :param download_queue: List of anime ids to download
    :param seen_path: (optional) Path to save the seen set, defaults to 'data/seen.pkl'
    :param queue_path: (optional) Path to save the download queue, defaults to 'data/queue.pkl'
    """

    if seen_path is None:
        seen_path = os.path.join(os.getenv('DATA_DIR'), 'interim', 'seen.pkl')
    if queue_path is None:
        queue_path = os.path.join(os.getenv('DATA_DIR'), 'interim', 'queue.pkl')

    with open(seen_path, 'wb') as f:
        pickle.dump(seen, f)
    with open(queue_path, 'wb') as f:
        pickle.dump(download_queue, f)


def load_state(seen_path: str = None, queue_path: str = None) -> tuple:
    r"""Loads the state of the data collection process.

    :param seen_path: (optional) Path to load the seen set, defaults to 'data/seen.pkl'
    :param queue_path: (optional) Path to load the download queue, defaults to 'data/queue.pkl'
    :return: Tuple of seen set and download queue
    :rtype: tuple
    """

    if seen_path is None:
        seen_path = os.path.join(os.getenv('DATA_DIR'), 'interim', 'seen.pkl')
    if queue_path is None:
        queue_path = os.path.join(os.getenv('DATA_DIR'), 'interim', 'queue.pkl')

    if os.path.exists(seen_path) and os.path.exists(queue_path):
        with open(seen_path, 'rb') as f:
            seen = pickle.load(f)
        with open(queue_path, 'rb') as f:
            download_queue = pickle.load(f)
    else:
        seen = set()
        download_queue = []
    return seen, download_queue


# Main function to collect data
def collect_data(headers: dict, debug: bool = False, max_iterations: int = 100) -> pd.DataFrame:
    r"""Fetches the top and related anime from MyAnimeList and downloads relevant info.

    :param headers: Headers with Authorization
    :param debug: (optional) Debug mode, defaults to False
    :param max_iterations: (optional) Maximum number of iterations, defaults to 100
    :return: DataFrame with anime details
    :rtype: pd.DataFrame
    """

    # Load ot initialize seen and download_queue
    seen, download_queue = load_state()

    # Fetch top anime only if needed
    if not download_queue:
        top_anime = fetch_top_anime(headers=headers)
        for anime in top_anime['data']:
            if anime['node']['id'] not in seen:
                download_queue.append(anime['node']['id'])
                seen.add(anime['node']['id'])

    # Colleted anime details
    anime_details = []
    
    # Debugging: Limit the number of iterations
    iteration_limit = 1 if debug else max_iterations
    if debug:
        print("Debug mode enabled: Limiting the number of iterations to", iteration_limit)
    
    # Prepare the progress bar
    pbar = tqdm(total=iteration_limit, desc="Fetching anime details")

    iterations = 0
    session = requests.Session()

    # While there are anime to fetch and the iteration limit has not been reached
    while download_queue and iterations < iteration_limit:
        id = download_queue.pop(0)

        # Refresh the session every 100 iterations
        if iterations % 100 == 0 and iterations > 0:
            session = requests.Session()

        fetched_details = fetch_anime_details(anime_id=id, headers=headers, session=session)

        # Request failed, recover the id and exit
        if not fetched_details:
            download_queue.append(id)
            session.close()
            print("Request failed, exiting")
            break

        anime_details.append(fetched_details)

        iterations += 1
        pbar.update(1)

        # Sleep to respect the rate limit (keep at >= 4.5 seconds to avoid rate limit errors)
        time.sleep(4.15)
        
        try:
            related_anime_ids = get_related_anime_ids(fetched_details)
        except KeyError as e:
            download_queue.append(id)
            break

        # Add not seen related anime to the download queue
        for related_id in related_anime_ids:
            if related_id not in seen:
                download_queue.append(related_id)
                seen.add(related_id)

    print('Length of seen:', len(seen), 'Length of download queue:', len(download_queue))
    if debug:
        print('Next anime to fetch:', download_queue[0] if download_queue else None)

    pbar.close()

    # Convert the list of anime details to a DataFrame
    try:
        anime_df = create_dataframe(anime_details)
    except Exception as e:
        print(f'Error creating dataframe: {str(e)}')
    
    # Save the state
    save_state(seen, download_queue)

    return anime_df


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve your access token from the environment variable
    access_token = os.getenv('MAL_ACCESS_TOKEN')

    # Check if the access token is loaded correctly
    if not access_token:
        raise ValueError("Missing MAL_ACCESS_TOKEN. Check .env file.")

    # Set up headers with Authorization
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    anime_df = collect_data(headers=headers, debug=False, max_iterations=300)

    # Add data to the CSV file '../data/raw/anime_data.csv'
    output_path = os.path.join(os.getenv('DATA_DIR'), 'raw', 'anime_data.csv')
    anime_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
