
import ast
import os
import pickle
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import textacy.preprocessing as tprep
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler,
                                   MultiLabelBinarizer, PowerTransformer,
                                   StandardScaler)

#####################################
#  Anime features and their format  #
#####################################
# 'id,' # Anime ID (integer)\
# 🛑'title,' # Anime title (string)\
# ✅'synopsis,' # Anime synopsis (string or null)\
# ✅'mean,' # Mean score (float or null)\
# ✅'popularity,' # Popularity rank (integer or null)\
# 🛑'num_list_users,' # Number of users who have the anime in their list (integer)\
# ✅'num_scoring_users,' # Number of users who have scored the anime (integer)\
# ✅'nsfw,' # NSFW classification (white=sfw, gray=partially, black=nsfw) (string or null)\
# ✅'genres,' # Genres (array of objects)\
# ✅'studios,' # Studios (array of objects)\
# ✅'num_episodes,' # Number of episodes (integer)\
# ✅'average_episode_duration,' # Average duration of an episode (integer or null)\
# ✅'status,' # Airing status (string)\
# ✅'rating,' # Age rating (string or null) (g, pg, pg_13, r, r+, rx)\
# ✅'source,' # Source (string or null)\
# ✅'media_type,' # Media type (string)\
# 🛑'created_at,' # Date of creation (string <date-time>)\
# 🛑'updated_at,' # Date of last update (string <date-time>)\
# ✅'start_season,' # Start season (object or null)\
# ✅'start_date,' # Start date (string or null)\
# ✅'end_date,' # End date (string or null)\
# ✅'related_anime,' # Related anime (array of objects)\
# 🛑'related_manga,' # Related manga (array of objects)\
# 🛑'recommendations,' # Recommendations (array of objects)\
# ✅'statistics' # Statistics (object or null)

def load_model(path: str) -> object:
    """
    Loads a scaler object from the given path.

    Args:
        path (str): The path to the scaler object.

    Returns:
        object: The loaded scaler object.
    """
    assert os.path.exists(path), f"Path {path} does not exist."
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def process_dates(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the dates in the given anime data batch.
    Preprocesses: start_date, end_date, start_season.year

    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The preprocessed data. Time difference between
                            start_date and end_date calculated and added as
                            feature, start_season.year scaled. start_date and
                            end_date columns dropped.
    """
    # 'start_date' (string or null)
    # 'end_date' (string or null)
    # 'start_season' (object or null)
    # 'start_season.year' (object or null)
    def safe_date_convert(date) -> datetime.date:
        if pd.isna(date):
            return None
        if type(date) is float:
            return datetime.strptime(str(int(date)), '%Y').date()
        if type(date) is str:
            if re.compile("\d{4}-\d{2}-\d{2}").match(date):
                return datetime.strptime(date, '%Y-%m-%d').date()
            elif re.compile("\d{4}-\d{2}").search(date):
                return datetime.strptime(date, '%Y-%m').date()
            else:
                return datetime.strptime(date, '%Y').date()
        raise ValueError(f"Invalid date format: {date}, {type(date)}")

    def time_diff(start_date, end_date):
        if pd.isna(start_date) or pd.isna(end_date):
            return None
        if start_date <= end_date:
            return (end_date - start_date).days
        else:
            return (start_date - end_date).days

    # Convert dates to datetime objects
    data['start_date'] = data['start_date'].apply(safe_date_convert)
    data['end_date'] = data['end_date'].apply(safe_date_convert)
    # Calculate time difference
    data['time_diff'] = data.apply(lambda x: time_diff(x['start_date'], x['end_date']), axis=1)
    data = data.drop(columns=['start_date', 'end_date'])
    # Scale time_diff
    td_scaler = load_model(config['time_diff'])
    data['time_diff'] = td_scaler.transform(data['time_diff'].values.reshape(-1, 1))
    # Scale start_season.year
    year_scaler = load_model(config['start_season_year'])
    data['start_season.year'] = year_scaler.transform(data['start_season.year'].values.reshape(-1, 1))
    return data


def preprocess_season(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'start_season' column in the given data batch by performing cyclical encoding.
    Preprocesses: start_season.season
    
    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The preprocessed data with cyclical encoding applied to the 'start_season.season' column.
    """
    def cyclical_encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return data

    season_encoder = load_model(config['start_season_season'])
    data['start_season.season'] = season_encoder.transform(data['start_season.season'])

    # Apply the cyclical_encode function to create sine and cosine features
    data = cyclical_encode(data, 'start_season.season', max_val=len(season_encoder.classes_))
    data = data.drop(columns=['start_season.season'])
    return data


def preprocess_text(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses synopsis and title data anime data batch.
    Preprocesses: synopsis, title, related_anime

    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The DataFrame with the preprocessed text data.
    """
    # 'synopsis' (string or null)
    # 'related_anime' (array of objects)
    def clean_text(text):
        text = unicodedata.normalize('NFKC', text)  # Unicode normalization
        text = text.replace('\u2013', '\u002d')  # Replace en dash with hyphen
        text = text.replace('\u00d7', '\u0078')  # Replace multiplication sign with x
        text = tprep.normalize.hyphenated_words(text)  # Normalize hyphenated words
        text = tprep.normalize.quotation_marks(text)  # Normalize quotation marks
        text = tprep.normalize.bullet_points(text)  # Normalize bullet points
        text = tprep.normalize.whitespace(text)  # Normalize whitespace
        text = tprep.remove.accents(text)  # Remove accents
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags if any
        text = re.sub(r"\([\s+]?source.*?\)+", "", text, flags=re.IGNORECASE)  # Remove source citations
        text = re.sub(r"\[Writ.*?by.*?\]", "", text)  # Remove MAL citations
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = text.strip()  # Strip whitespace from the beginning and the end
        return text
    
    def preprocess_related(data):
        # Join the title of the anime with the related anime titles
        f = lambda x: ' '.join([entry['node']['title'] for entry in ast.literal_eval(x)])
        data['related'] = data['title'] + ' ' + data['related_anime'].apply(f)
        data = data.drop(columns=['title', 'related_anime'])
        return data

    data['synopsis'] = data['synopsis'].apply(clean_text)
    data = preprocess_related(data)
    data['related'] = data['related'].apply(clean_text)
    return data


def preprocess_genres(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'genres' column in the given data batch.
    Preprocesses: genres

    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The data with the 'genres' column preprocessed by MultiLabelBinarizer.

    """
    # 'genres' (array of objects)

    def process(entry):
        genres_set = set(genre['name'] for genre in entry)
        return genres_set

    data['genres'] = data['genres'].apply(ast.literal_eval)
    data['genres'] = data['genres'].apply(process)
    
    genre_mlb = load_model(config['genres'])
    data['genres'] = data['genres'].apply(lambda x: np.squeeze(genre_mlb.transform([x])))
    return data


def preprocess_studios(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'studios' column in the given data using MultiLabelBinarizer.
    Preprocesses: studios

    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The preprocessed data with the 'studios' column transformed.

    """
    # 'studios' (array of objects, may be empty)

    def process(entry):
        studios_set = set(studio['name'] for studio in entry)
        return studios_set

    data['studios'] = data['studios'].apply(ast.literal_eval)
    data['studios'] = data['studios'].apply(process)

    studio_mlb = load_model(config['studios'])
    data['studios'] = data['studios'].apply(lambda x: np.squeeze(studio_mlb.transform([x])))
    return data


def preprocess_nsfw(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'nsfw' column in the given data batch using LabelEncoder.
    Preprocesses: nsfw

    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The preprocessed data with the 'nsfw' column encoded.

    """
    # 'nsfw' (white=sfw, gray=partially, black=nsfw) (string or null)
    nsfw_encoder = load_model(config['nsfw'])
    data['nsfw'] = nsfw_encoder.transform(data['nsfw'])
    return data


def preprocess_source(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'source' column in the given data using LabelEncoder.
    Preprocesses: source

    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The data with the 'source' column encoded.

    """
    # 'source' (string or null)
    source_encoder = load_model(config['source'])
    data['source'] = source_encoder.transform(data['source'])
    return data


def preprocess_status(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'status' column of the given data using LabelEncoder.
    Preprocesses: status

    Parameters:
    data (pandas.DataFrame): The input data batch.

    Returns:
    pandas.DataFrame: The data with the 'status' column encoded.

    """
    # 'status' (string)
    status_encoder = load_model(config['status'])
    data['status'] = status_encoder.transform(data['status'])
    return data


def preprocess_media_type(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'media_type' column of the given data using LabelEncoder.
    Preprocesses: media_type

    Parameters:
    data (pandas.DataFrame): The input data batch.

    Returns:
    pandas.DataFrame: The data with the 'media_type' column encoded.

    """
    # 'media_type' (string)
    # We might want to change media_type to better reflect the data
    # We might use the following rules:
    # movie = 'avg_ep_dur'>1800
    # tv = ('avg_ep_dur'<=1800 & 'num_episodes'>=6)
    # special = ('avg_ep_dur'<=1800 & 'num_episodes'<6) | ('avg_ep_dur' < 240)
    # This covers all cases, but the duration and num_ep thresholds seem suboptimal after some testing
    # thus we skip this for now
    def filter_media_type(anime):
        d = anime['average_episode_duration']
        n = anime['num_episodes'] 
        if d > 1800:
            anime['media_type'] = 'movie'
        elif d <= 1800 and n >= 6:
            anime['media_type'] = 'tv'
        elif (d <= 1800 and n < 6) or d < 240:
            anime['media_type'] = 'special'
        return anime

    data['media_type'] = data['media_type'].apply(lambda x: 'special' if x in {'ona', 'ova', 'tv_special'} else x)
    media_type_encoder = load_model(config['media_type'])
    data['media_type'] = media_type_encoder.transform(data['media_type'])
    return data


def preprocess_rating(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'rating' column in the given data DataFrame.$
    Preprocesses: rating

    Args:
        data (DataFrame): The input data batch.

    Returns:
        DataFrame: The modified DataFrame with the 'rating' column mapped to numerical values.
    """
    # 'rating' (string or null) (g, pg, pg_13, r, r+, rx)
    rating_map = {
        "g": 0,
        "pg": 1,
        "pg_13": 2,
        "r": 3,
        "r+": 4,
        "rx": 5
    }
    data['rating'] = data['rating'].map(rating_map)
    return data


def preprocess_mean(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'mean' feature of the given data using a standard scaler.
    Preprocesses: mean

    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The preprocessed data with the 'mean' feature transformed using a standard scaler.
    """
    # 'mean' (float or null)
    # This feature looks similar to a normal distribution, so we try a standard scaler
    mean_scaler = load_model(config['mean'])
    data['mean'] = mean_scaler.transform(data['mean'].values.reshape(-1, 1))
    return data


def preprocess_popularity(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'popularity' feature in the given data using standard scaling.
    Preprocesses: popularity

    Parameters:
    data (pandas.DataFrame): The input data batch.

    Returns:
    pandas.DataFrame: The preprocessed data with the 'popularity' feature scaled using standard scaling.
    """
    # 'popularity' (integer or null)
    # The distribution of this feature seems to get messed up for anything other than standard scaler
    popularity_scaler = load_model(config['popularity'])
    data['popularity'] = popularity_scaler.transform(data['popularity'].values.reshape(-1, 1))
    return data


def preprocess_num_scoring_users(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'num_scoring_users' feature in the given data using PowerTransformer (yeo-johnson).
    Preprocesses: num_scoring_users

    Parameters:
    data (pandas.DataFrame): The input data batch.

    Returns:
    pandas.DataFrame: The preprocessed data with the 'num_scoring_users' feature scaled.
    """
    # 'num_scoring_users' (integer)
    # This feature exhibits a long tail distribution, we try power transformer (yeo-johnson)
    # This might not the best way to handle this feature
    # Perhaps try https://arxiv.org/abs/2111.05956#:~:text=The%20visual%20world%20naturally%20exhibits,models%20based%20on%20deep%20learning.

    popularity_scaler = load_model(config['num_scoring_users'])
    data['num_scoring_users'] = popularity_scaler.transform(data['num_scoring_users'].values.reshape(-1, 1))
    return data


def preprocess_num_episodes(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'num_episodes' feature in the given data using a power transformer (yeo-johnson).
    Preprocesses: num_episodes

    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The preprocessed data with the 'num_episodes' feature transformed.

    """
    # 'num_episodes' (integer)
    # This feature exhibits a long tail distribution, we again try power transformer (yeo-johnson)
    num_episodes_scaler = load_model(config['num_episodes'])
    data['num_episodes'] = num_episodes_scaler.transform(data['num_episodes'].values.reshape(-1, 1))
    return data


def preprocess_average_episode_duration(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'average_episode_duration' feature in the given data using PowerTransformer (yeo-johnson).
    Preprocesses: average_episode_duration

    Parameters:
    data (pandas.DataFrame): The input data batch.

    Returns:
    pandas.DataFrame: The data with the 'average_episode_duration' feature preprocessed.
    """
    # 'average_episode_duration' (integer or null)
    # This feature might also benefit from power transformer (yeo-johnson)
    avg_ep_scaler = load_model(config['average_episode_duration'])
    data['average_episode_duration'] = avg_ep_scaler.transform(data['average_episode_duration'].values.reshape(-1, 1))
    return data


def preprocess_stats(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the statistics features of the given data using PowerTransformer (yeo-johnson).
    Preprocesses: statistics.status.watching, statistics.status.completed, statistics.status.on_hold,

    Args:
        data (pandas.DataFrame): The input data batch.

    Returns:
        pandas.DataFrame: The preprocessed data with transformed statistics features.
    """
    # 'statistics' (object or null)
    # The feature 'num_list_users' contains inconsistent data
    # We will drop this feature, and instead use 'statistics.num_list_users'
    data = data.drop(columns=['num_list_users'])
    watching_scaler = load_model(config['statistics_status_watching'])
    data['statistics.status.watching'] = watching_scaler.transform(data['statistics.status.watching'].values.reshape(-1, 1))
    completed_scaler = load_model(config['statistics_status_completed'])
    data['statistics.status.completed'] = completed_scaler.transform(data['statistics.status.completed'].values.reshape(-1, 1))
    on_hold_scaler = load_model(config['statistics_status_on_hold'])
    data['statistics.status.on_hold'] = on_hold_scaler.transform(data['statistics.status.on_hold'].values.reshape(-1, 1))
    dropped_scaler = load_model(config['statistics_status_dropped'])
    data['statistics.status.dropped'] = dropped_scaler.transform(data['statistics.status.dropped'].values.reshape(-1, 1))
    plan_to_watch_scaler = load_model(config['statistics_status_plan_to_watch'])
    data['statistics.status.plan_to_watch'] = plan_to_watch_scaler.transform(data['statistics.status.plan_to_watch'].values.reshape(-1, 1))
    num_list_users_scaler = load_model(config['statistics_num_list_users'])
    data['statistics.num_list_users'] = num_list_users_scaler.transform(data['statistics.num_list_users'].values.reshape(-1, 1))
    return data


def process(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the given data based on the provided configuration.

    Args:
        data (pd.DataFrame): The input data batch to be preprocessed.
        config (dict): The configuration containing paths to sklearn scalers for each preprocessing step.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    data = process_dates(data, config['dates'])
    data = preprocess_season(data, config['season'])
    data = preprocess_text(data, config['text'])
    data = preprocess_genres(data, config['genres'])
    data = preprocess_studios(data, config['studios'])
    data = preprocess_nsfw(data, config['nsfw'])
    data = preprocess_source(data, config['source'])
    data = preprocess_status(data, config['status'])
    data = preprocess_media_type(data, config['media_type'])
    data = preprocess_rating(data, config['rating'])
    data = preprocess_mean(data, config['mean'])
    data = preprocess_popularity(data, config['popularity'])
    data = preprocess_num_scoring_users(data, config['num_scoring_users'])
    data = preprocess_num_episodes(data, config['num_episodes'])
    data = preprocess_average_episode_duration(data, config['average_episode_duration'])
    data = preprocess_stats(data, config['stats'])
    return data
