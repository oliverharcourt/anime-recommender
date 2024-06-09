
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
from pymilvus import MilvusClient
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


def preprocess_dates(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    # 'start_date' (string or null)
    # 'end_date' (string or null)
    # 'start_season' (object or null)
    # 'start_season.year' (object or null)

    td_scaler_path = config['time_diff']
    year_scaler_path = config['start_season_year']

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
    td_scaler = load_model(td_scaler_path)
    data['time_diff'] = td_scaler.transform(data['time_diff'].values.reshape(-1, 1))
    # Scale start_season.year
    year_scaler = load_model(year_scaler_path)
    data['start_season.year'] = year_scaler.transform(data['start_season.year'].values.reshape(-1, 1))
    return data


def preprocess_season(data: pd.DataFrame, config: dict) -> pd.DataFrame:
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


def preprocess_genres(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    # 'genres' (array of objects)
    def process(entry):
        genres_set = set(genre['name'] for genre in entry)
        return genres_set

    data['genres'] = data['genres'].apply(ast.literal_eval)
    data['genres'] = data['genres'].apply(process)
    
    genre_mlb = load_model(config['genres'])
    data['genres'] = data['genres'].apply(lambda x: genre_mlb.transform([x]).reshape(1, -1))
    return data


def preprocess_studios(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    # 'studios' (array of objects, may be empty)
    def process(entry):
        studios_set = [studio['name'] for studio in entry]
        return studios_set

    data['studios'] = data['studios'].apply(ast.literal_eval)
    data['studios'] = data['studios'].apply(process)

    studio_enc = load_model(config['studios'])
    studios_hashed = studio_enc.transform(data['studios']).toarray()
    data['studios'] = [hash for hash in studios_hashed]
    data['studios'] = data['studios'].apply(lambda x: x.reshape(1, -1))
    return data


def preprocess_media_type(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    # 'media_type' (string)
    # We might want to change media_type to better reflect the data
    # We might use the following rules:
    # movie = 'avg_ep_dur'>1800
    # tv = ('avg_ep_dur'<=1800 & 'num_episodes'>=6)
    # special = ('avg_ep_dur'<=1800 & 'num_episodes'<6) | ('avg_ep_dur' < 240)
    # This covers all cases, but the duration and num_ep thresholds seem suboptimal after some testing
    # thus we skip this for now

    data['media_type'] = data['media_type'].apply(lambda x: 'special' if x in {'ona', 'ova', 'tv_special'} else x)
    media_type_encoder = load_model(config['media_type'])
    data['media_type'] = media_type_encoder.transform(data['media_type'])
    return data


def preprocess_rating(data: pd.DataFrame, config: dict) -> pd.DataFrame:
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
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.copy()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        col = 'synopsis' if 'synopsis' in self.data.columns else 'related'
        text = self.data.iloc[idx][col]  # Adjust column name as necessary
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return idx, inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)


def generate_embeddings(data: pd.DataFrame, model_path: str, device: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:

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

    synopsis_df2 = pd.DataFrame({'synopsis_embedding': synopsis_vectors})
    related_df2 = pd.DataFrame({'related_embedding': related_vectors})

    # We only need the id column from the original dataframes
    synopsis_df.drop(columns=['synopsis'], inplace=True)
    related_df.drop(columns=['related'], inplace=True)

    # Concatenate the embeddings with the id column
    synopsis_df_concat = pd.concat([synopsis_df, synopsis_df2], axis=1, ignore_index=True)
    synopsis_df_concat.rename(columns={0: 'id', 1: 'synopsis_embedding'}, inplace=True)

    related_df_concat = pd.concat([related_df, related_df2], axis=1, ignore_index=True)
    related_df_concat.rename(columns={0: 'id', 1: 'related_embedding'}, inplace=True)

    return synopsis_df_concat, related_df_concat


def handle_studio_genres(data: pd.DataFrame) -> pd.DataFrame:
    genres_studios_df = data[['id', 'genres', 'studios']].copy()

    f = lambda x: np.array(x).reshape(-1,)
    genres_studios_df['genres'] = genres_studios_df['genres'].apply(f)
    genres_studios_df['studios'] = genres_studios_df['studios'].apply(f)

    g = lambda x: np.concatenate((np.array([x['genres']]), np.array([x['studios']])), axis=1).reshape(-1,)
    genres_studios_df['genres_studios_flattened'] = genres_studios_df.apply(g, axis=1)
    genres_studios_df.drop(columns=['genres', 'studios'], inplace=True)

    data.drop(columns=['genres', 'studios'], inplace=True)

    ret = pd.DataFrame(genres_studios_df)
    #print(f'genres_studios_df: {ret}')

    return ret


def create_embeddings(data: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = config['model_path']
    synopsis_df, related_df = generate_embeddings(data, model_path, device, config)
    
    data.drop(columns=['synopsis', 'related'], inplace=True)
    return synopsis_df, related_df


def get_existing_ids(data: pd.DataFrame, vector_db: MilvusClient, collection_name: str) -> list:

    ids = data['id'].tolist()
    
    vectors = vector_db.get(
        collection_name=collection_name,
        ids=ids
    )
    print(f"Found {len(ids)} embeddings in the database.")
    
    retrieved_ids = {vector.id for vector in vectors if vector is not None}

    return retrieved_ids

def sep_and_concat_cols(data: pd.DataFrame, data_np: np.ndarray, cols_to_seperate: list) -> np.ndarray:
    
    col_idxs = {}

    seperate_columns = {}

    for col in cols_to_seperate:
        col_idx = data.columns.get_loc(col)
        col_idxs[col] = col_idx

    for col, idx in col_idxs.items():
        if isinstance(data_np[0, idx], np.ndarray) or isinstance(data_np[0, idx], list):
            col_data = data_np[:, idx]
            print(f"col_idx: {idx} col_name: {col} col_data: {col_data}")
            print(f'data_np shape: {data_np.shape} col_data shape: {col_data.shape}')
            col_data = np.vstack(col_data)
            data_np = np.concatenate([data_np, col_data], axis=1)

            # seperate text embedding columns for seperate distance calculations
            if col != 'genres_studios_flattened':
                seperate_columns[col] = col_data
            print(f'new data_np shape: {data_np.shape}')
    
    print(f'final data_np shape: {data_np.shape}')
    
    data_np = np.delete(data_np, list(col_idxs.values()), axis=1)
    
    return data_np, seperate_columns


def process(data: pd.DataFrame, config: dict, vector_db: MilvusClient) -> None:

    data = data.copy()

    columns = ['created_at', 'updated_at', 'related_manga',
               'recommendations', 'main_picture.medium', 'main_picture.large']
    data.drop(columns=columns, inplace=True)
    data.dropna(inplace=True)
    
    #seen_ids = get_existing_ids(data, vector_db, config['collection_name'])

    #data = data[~data['id'].isin(seen_ids)]
    
    # Handle features with special preprocessing methods
    func_map = {
        'start_date end_date start_season.year': preprocess_dates,
        'start_season.season': preprocess_season,
        'synopsis title related_anime': preprocess_text,
        'genres': preprocess_genres,
        'studios': preprocess_studios,
        'media_type': preprocess_media_type,
        'rating': preprocess_rating
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
        model = load_model(path)
        data[col] = model.transform(data[col].values.reshape(-1, 1))

    synopsis_df, related_df = create_embeddings(data, config)
    genres_studios_df = handle_studio_genres(data)
    
    # join all data, transform to np array and insert into vector db

    print(f'synsopsis_df columns: {synopsis_df.columns}')
    print(f'related_df columns: {related_df.columns}')
    print(f'genres_studios_df columns: {genres_studios_df.columns}')
    print(f'data columns: {data.columns}')

    data = data.merge(synopsis_df, on='id', how='left')
    data = data.merge(related_df, on='id', how='left')
    data = data.merge(genres_studios_df, on='id', how='left')

    print(f'Final dim: {data.shape}')

    #return data

    data_np = data.to_numpy()

    columns_to_seperate = ['genres_studios_flattened', 'synopsis_embedding', 'related_embedding']

    data_np, seperated_columns = sep_and_concat_cols(data, data_np, columns_to_seperate)

    synopsis_embeddings = seperated_columns['synopsis_embedding']
    related_embeddings = seperated_columns['related_embedding']

    print(f'data_np shape: {data_np.shape}')

    #return data, data_np, synopsis_embeddings, related_embeddings

    res_data = vector_db.insert(
        collection_name=config['collection_name']['data'],
        records=data_np
    )
    print(f"res_data: {res_data}")

    res_synopsis = vector_db.insert(
        collection_name=config['collection_name']['synopsis'],
        records=synopsis_embeddings
    )
    print(f"res_synopsis: {res_synopsis}")

    res_related = vector_db.insert(
        collection_name=config['collection_name']['related'],
        records=related_embeddings
    )
    print(f"res_related: {res_related}")