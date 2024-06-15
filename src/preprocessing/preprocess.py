
import ast
import os
import pickle
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import textacy.preprocessing as tprep
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

#####################################
#  Anime features and their format  #
#####################################
# 'id,' # Anime ID (integer)
# ✅'title,' # Anime title (string)
# ✅'synopsis,' # Anime synopsis (string or null)
# ✅'mean,' # Mean score (float or null)
# ✅'popularity,' # Popularity rank (integer or null)
# 🛑'num_list_users,' # Number of users who have the anime in their list (integer)
# ✅'num_scoring_users,' # Number of users who have scored the anime (integer)
# ✅'nsfw,' # NSFW classification (white=sfw, gray=partially, black=nsfw) (string or null)
# ✅'genres,' # Genres (array of objects)
# ✅'studios,' # Studios (array of objects)
# ✅'num_episodes,' # Number of episodes (integer)
# ✅'average_episode_duration,' # Average duration of an episode (integer or null)
# ✅'status,' # Airing status (string)
# ✅'rating,' # Age rating (string or null) (g, pg, pg_13, r, r+, rx)
# ✅'source,' # Source (string or null)
# ✅'media_type,' # Media type (string)
# 🛑'created_at,' # Date of creation (string <date-time>)
# 🛑'updated_at,' # Date of last update (string <date-time>)
# ✅'start_season,' # Start season (object or null)
# ✅'start_date,' # Start date (string or null)
# ✅'end_date,' # End date (string or null)
# ✅'related_anime,' # Related anime (array of objects)
# 🛑'related_manga,' # Related manga (array of objects)
# 🛑'recommendations,' # Recommendations (array of objects)
# ✅'statistics' # Statistics (object or null)

def _load_model(path: str) -> object:
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


def _preprocess_dates(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the dates in the given DataFrame according to the provided configuration.

    Args:
    data (pd.DataFrame): The DataFrame to preprocess.
    config (dict): The configuration dict containing paths to the time difference and
    start season year scalers.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """

    # 'start_date' (string or null)
    # 'end_date' (string or null)
    # 'start_season' (object or null)
    # 'start_season.year' (object or null)

    td_scaler_path = config['time_diff']
    year_scaler_path = config['start_season_year']

    def safe_date_convert(date) -> datetime.date:
        if pd.isna(date):
            return None
        if isinstance(date, float):
            return datetime.strptime(str(int(date)), '%Y').date()
        if isinstance(date, str):
            if re.compile(r"\d{4}-\d{2}-\d{2}").match(date):
                return datetime.strptime(date, '%Y-%m-%d').date()
            if re.compile(r"\d{4}-\d{2}").search(date):
                return datetime.strptime(date, '%Y-%m').date()
            return datetime.strptime(date, '%Y').date()
        raise ValueError(f"Invalid date format: {date}, {type(date)}")

    def time_diff(start_date, end_date):
        if pd.isna(start_date) or pd.isna(end_date):
            return None
        if start_date <= end_date:
            return (end_date - start_date).days
        return (start_date - end_date).days

    # Convert dates to datetime objects
    data['start_date'] = data['start_date'].apply(safe_date_convert)
    data['end_date'] = data['end_date'].apply(safe_date_convert)
    # Calculate time difference
    data['time_diff'] = data.apply(lambda x: time_diff(x['start_date'], x['end_date']), axis=1)
    data = data.drop(columns=['start_date', 'end_date'])
    # Scale time_diff
    td_scaler = _load_model(td_scaler_path)
    data['time_diff'] = td_scaler.transform(data['time_diff'].values.reshape(-1, 1))
    # Scale start_season.year
    year_scaler = _load_model(year_scaler_path)
    data['start_season.year'] = year_scaler.transform(data['start_season.year']
                                                      .values.reshape(-1, 1))
    return data


def _preprocess_season(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'start_season.season' column in the given DataFrame according to the provided configuration.
    The preprocessing includes applying a cyclical encoding to create sine and cosine features.

    Args:
    data (pd.DataFrame): The DataFrame to preprocess.
    config (dict): The configuration dict containing the path to the season encoder model.

    Returns:
    pd.DataFrame: The preprocessed DataFrame with the original 'start_season.season' column replaced
    by its sine and cosine features.
    """

    def cyclical_encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return data

    season_encoder = _load_model(config['start_season_season'])
    data['start_season.season'] = season_encoder.transform(data['start_season.season'])

    # Apply the cyclical_encode function to create sine and cosine features
    data = cyclical_encode(data, 'start_season.season', max_val=len(season_encoder.classes_))
    data = data.drop(columns=['start_season.season'])
    return data


def _preprocess_text(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'synopsis' 'title', and 'related_anime' columns in the given DataFrame.
    The preprocessing includes cleaning the text, normalizing Unicode characters,
    removing HTML tags, source citations, and MAL citations, and normalizing whitespace.
    It also preprocesses related anime by sorting them and joining them into a single string.

    Args:
    data (pd.DataFrame): The DataFrame to preprocess.
    config (dict): Not used in this method but included for consistency with
    other preprocessing methods.

    Returns:
    pd.DataFrame: The preprocessed DataFrame with the original 'synopsis' 'title',
    'related_anime' columns replaced by their cleaned and preprocessed versions.
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
        f = lambda x: [entry['node']['title'] for entry in ast.literal_eval(x)]
        cr = lambda x: [clean_text(i) for i in f(x)]
        g = lambda x: [clean_text(x)]
        data['related'] = data['title'].apply(g) + data['related_anime'].apply(cr)
        data['related'] = data['related'].apply(sorted)
        data = data.drop(columns=['title', 'related_anime'])
        return data

    data['synopsis'] = data['synopsis'].apply(clean_text)
    data = preprocess_related(data)
    data['related'] = data['related'].apply(' '.join)
    return data


def _preprocess_genres(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'genres' column in the given DataFrame.
    The preprocessing includes transforming the list of genres using a pre-trained MultiLabelBinarizer.

    Args:
    data (pd.DataFrame): The DataFrame to preprocess.
    config (dict): The configuration dict containing the path to the genre MultiLabelBinarizer model.

    Returns:
    pd.DataFrame: The preprocessed DataFrame with the original 'genres' column replaced
    by a binary array representing the presence of each genre.
    """

    # 'genres' (array of objects)
    def extract_genres(entry):
        genres_set = set(genre['name'] for genre in entry)
        return genres_set

    data['genres'] = data['genres'].apply(ast.literal_eval)
    data['genres'] = data['genres'].apply(extract_genres)

    genre_mlb = _load_model(config['genres'])
    data['genres'] = genre_mlb.transform(data['genres']).tolist()
    return data


def _preprocess_studios(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'studios' column in the given DataFrame.
    The preprocessing includes transforming the list of studios using a pre-trained FeatureHasher.

    Args:
    data (pd.DataFrame): The DataFrame to preprocess.
    config (dict): The configuration dict containing the path to the studio FeatureHasher model.

    Returns:
    pd.DataFrame: The preprocessed DataFrame with the original 'studios' column replaced
    by a binary arrays representing the presence of each studio.
    """

    # 'studios' (array of objects, may be empty)
    def extract_studio_names(entry):
        studios_set = [studio['name'] for studio in entry]
        return studios_set

    data['studios'] = data['studios'].apply(ast.literal_eval)
    data['studios'] = data['studios'].apply(extract_studio_names)

    studio_enc = _load_model(config['studios'])
    data['studios'] = studio_enc.transform(data['studios']).toarray().tolist()
    return data


def _preprocess_media_type(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'media_type' column in the given DataFrame.
    The preprocessing includes recategorizing ona, ova and tv_special to special,
    and then applying a pre-trained LabelEncoder.

    Args:
    data (pd.DataFrame): The DataFrame to preprocess.
    config (dict): The configuration dict containing the path to the media_type encoder model.

    Returns:
    pd.DataFrame: The preprocessed DataFrame with the original 'media_type' column
    replaced by the encoded media_type.
    """

    # 'media_type' (string)
    # We might want to change media_type to better reflect the data
    # We might use the following rules:
    # movie = 'avg_ep_dur'>1800
    # tv = ('avg_ep_dur'<=1800 & 'num_episodes'>=6)
    # special = ('avg_ep_dur'<=1800 & 'num_episodes'<6) | ('avg_ep_dur' < 240)
    # This covers all cases, but the duration and num_ep thresholds seem suboptimal
    # after some testing, thus we skip this for now

    f = lambda x: 'special' if x in {'ona', 'ova', 'tv_special'} else x
    data['media_type'] = data['media_type'].apply(f)
    media_type_encoder = _load_model(config['media_type'])
    data['media_type'] = media_type_encoder.transform(data['media_type'])
    return data


def _preprocess_rating(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Preprocesses the 'rating' column in the given DataFrame.
    The preprocessing includes mapping the 'rating' column from a string representation
    of a rating to an integer representation.

    Args:
    data (pd.DataFrame): The DataFrame to preprocess.
    config (dict): Not used in this method but included for consistency
    with other preprocessing methods.

    Returns:
    pd.DataFrame: The preprocessed DataFrame with the original 'rating' column replaced
    by the integer representation of the rating.
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


class TextDataset(Dataset):
    """
    A custom Dataset class for handling text data.

    This class is designed to work with a DataFrame that contains text data,
    and a tokenizer for processing the text.
    It also supports setting a maximum length for the processed text.

    Attributes:
    data (pd.DataFrame): The DataFrame containing the text data.
    tokenizer: The tokenizer used to process the text.
    max_length (int): The maximum length for the processed text.

    Methods:
    __len__(): Returns the number of items in the dataset.
    __getitem__(idx): Returns the processed text at the given index.
    """

    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        col = 'synopsis' if 'synopsis' in self.data.columns else 'related'
        text = self.data.iloc[idx][col]  # Adjust column name as necessary
        inputs = self.tokenizer(text, padding='max_length', truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        return idx, inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)


def _generate_embeddings(data: pd.DataFrame, model_path: str, device: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates embeddings for the synopsis and related text in the given DataFrame.

    This method uses a pretrained DistilBert model to generate embeddings for the synopsis and
    related text in the given DataFrame.
    The embeddings are then stored in new DataFrames, which are returned.

    Args:
    data (pd.DataFrame): The DataFrame containing the text data.
    model_path (str): The path to the pretrained DistilBert model.
    device (str): The device to use for the DistilBert model.
    config (dict): A configuration dictionary containing the embedding dimension.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames.
    The first DataFrame contains the id and synopsis embeddings. The second DataFrame contains
    the id and related embeddings.
    """

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Extract relevant columns
    synopsis_df = data[['id', 'synopsis']].copy()
    related_df = data[['id', 'related']].copy()

    # Create dataset
    synopsis_dataset = TextDataset(synopsis_df, tokenizer)
    related_dataset = TextDataset(related_df, tokenizer)

    # Create DataLoaders
    loader_synopsis = DataLoader(synopsis_dataset, batch_size=32, shuffle=False, num_workers=2)
    loader_related = DataLoader(related_dataset, batch_size=32, shuffle=False, num_workers=2)
    # make this a dict for access to specific loaders from calling code
    loaders = [loader_synopsis, loader_related]

    # Pre-allocate embedding tensors
    embedding_sizes = config['embedding_dim']
    embedding_tensors = {
        'synopsis_emb': torch.zeros(len(synopsis_dataset), embedding_sizes),
        'related_emb': torch.zeros(len(related_dataset), embedding_sizes),
    }

    # Helper function to get the right embedding tensor key
    def get_tensor_key(loader):
        if loader == loader_synopsis:
            return 'synopsis_emb'
        elif loader == loader_related:
            return 'related_emb'

    model = DistilBertForMaskedLM.from_pretrained(model_path, device_map=device).base_model

    # Generate embeddings
    for loader in loaders:
        tensor_key = get_tensor_key(loader)
        print(f'Processing {tensor_key}')
        for indices, input_ids, attention_masks in loader:
            if indices[0] % 1024 == 0:
                print(indices[0])
            input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)
            with torch.no_grad():
                # Extract the CLS token embeddings from the last hidden state
                embeddings = model(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state[:, 0, :].detach().cpu()
                embedding_tensors[tensor_key][indices.cpu()] = embeddings

    # Dataframes of embeddings
    synopsis_tensor = embedding_tensors['synopsis_emb'].clone().detach().numpy()
    related_tensor = embedding_tensors['related_emb'].clone().detach().numpy()

    synopsis_vectors = [synopsis_tensor[i].tolist() for i in range(synopsis_tensor.shape[0])]
    related_vectors = [related_tensor[i].tolist() for i in range(related_tensor.shape[0])]

    # We only need the id column from the original dataframes
    synopsis_df.drop(columns=['synopsis'], inplace=True)
    related_df.drop(columns=['related'], inplace=True)

    # Merge the embeddings with the id column
    synopsis_df_merge = pd.DataFrame({'id': synopsis_df['id'], 'synopsis_embedding': synopsis_vectors})
    related_df_merge = pd.DataFrame({'id': related_df['id'], 'related_embedding': related_vectors})

    return synopsis_df_merge, related_df_merge


def _create_embeddings(data: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates embeddings for the synopsis and related text in the given DataFrame.

    This method is a wrapper for the _generate_embeddings method.

    Args:
    data (pd.DataFrame): The DataFrame containing the text data.
    config (dict): A configuration dictionary containing the model path.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames. The first DataFrame contains the synopsis embeddings. The second DataFrame contains the related embeddings.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = config['model_path']
    synopsis_df, related_df = _generate_embeddings(data, model_path, device, config)

    data.drop(columns=['synopsis', 'related'], inplace=True)
    return synopsis_df, related_df


def process(data: pd.DataFrame, config: dict) -> None:
    """
    This is the main method for preprocessing the given DataFrame.

    This method performs a series of preprocessing steps on the given DataFrame,
    including dropping unnecessary columns, handling missing values, and applying various
    preprocessing functions to specific columns. It also creates embeddings for the synopsis and
    related anime using a pretrained model.

    Args:
    data (pd.DataFrame): The DataFrame to be processed.
    config (dict): A configuration dictionary specifying the preprocessing model paths and
    embedding dimension.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """

    data = data.copy()

    columns = ['created_at', 'updated_at', 'related_manga',
               'recommendations', 'main_picture.medium', 'main_picture.large']
    data.drop(columns=columns, inplace=True)
    data.dropna(inplace=True)

    # Handle features with special preprocessing methods
    func_map = {
        'start_date end_date start_season.year': _preprocess_dates,
        'start_season.season': _preprocess_season,
        'synopsis title related_anime': _preprocess_text,
        'genres': _preprocess_genres,
        'studios': _preprocess_studios,
        'media_type': _preprocess_media_type,
        'rating': _preprocess_rating
    }

    for col, func in func_map.items():
        cols = set(col.split())
        if not cols.issubset(data.columns):
            print(f"Columns {col} not found in the dataset, skipping...")
            continue
        data = func(data, config)

    # Handle features with simple preprocessing methods
    for col, path in config['simple'].items():
        if col not in data.columns:
            print(f"Column {col} not found in the dataset, skipping...")
            continue
        model = _load_model(path)
        data[col] = model.transform(data[col].values.reshape(-1, 1))

    synopsis_df, related_df = _create_embeddings(data, config)

    # Join all data
    data = data.merge(synopsis_df, on='id', how='left')
    data = data.merge(related_df, on='id', how='left')

    print(f'Final dim: {data.shape}')

    print(f"Data shape at process method end: {data.shape}")

    return data
