import re
import unicodedata

import cleantext
import pandas as pd
import torch
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer
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
# ✅'num_list_users,' # Number of users who have the anime in their list (integer)
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
# 🚫(no longer supported)'related_anime,' # Related anime (array of objects)
# 🚫(no longer supported)'related_manga,' # Related manga (array of objects)
# 🚫(no longer supported)'recommendations,' # Recommendations (array of objects)
# 🚫(no longer supported)'statistics' # Statistics (object or null)


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
        return idx, inputs['input_ids'].squeeze(
            0), inputs['attention_mask'].squeeze(0)


def _generate_embeddings(data: pd.DataFrame, model_path: str,
                         device: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    loader_synopsis = DataLoader(
        synopsis_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2)
    loader_related = DataLoader(
        related_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2)
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

    model = DistilBertForMaskedLM.from_pretrained(
        model_path, device_map=device).base_model

    # Generate embeddings
    for loader in loaders:
        tensor_key = get_tensor_key(loader)
        print(f'Processing {tensor_key}')
        for indices, input_ids, attention_masks in loader:
            if indices[0] % 1024 == 0:
                print(indices[0])
            input_ids, attention_masks = input_ids.to(
                device), attention_masks.to(device)
            with torch.no_grad():
                # Extract the CLS token embeddings from the last hidden state
                embeddings = model(input_ids=input_ids,
                                   attention_mask=attention_masks).last_hidden_state[:,
                                                                                     0,
                                                                                     :].detach().cpu()
                embedding_tensors[tensor_key][indices.cpu()] = embeddings

    # Dataframes of embeddings
    synopsis_tensor = embedding_tensors['synopsis_emb'].clone(
    ).detach().numpy()
    related_tensor = embedding_tensors['related_emb'].clone().detach().numpy()

    synopsis_vectors = [synopsis_tensor[i].tolist()
                        for i in range(synopsis_tensor.shape[0])]
    related_vectors = [related_tensor[i].tolist()
                       for i in range(related_tensor.shape[0])]

    # We only need the id column from the original dataframes
    synopsis_df.drop(columns=['synopsis'], inplace=True)
    related_df.drop(columns=['related'], inplace=True)

    # Merge the embeddings with the id column
    synopsis_df_merge = pd.DataFrame(
        {'id': synopsis_df['id'], 'synopsis_embedding': synopsis_vectors})
    related_df_merge = pd.DataFrame(
        {'id': related_df['id'], 'related_embedding': related_vectors})

    return synopsis_df_merge, related_df_merge


def _create_embeddings(
        data: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    synopsis_df, related_df = _generate_embeddings(
        data, model_path, device, config)

    data.drop(columns=['synopsis', 'related'], inplace=True)
    return synopsis_df, related_df


def make_embeddings(data: pd.DataFrame, studios_n_feat: int, config: dict) -> pd.DataFrame:
    """
    This method only processes the title, synopsis, genres and studios columns for embedding generation.

    Args:
        data (pd.DataFrame): The DataFrame to be processed.
        studios_n_feat (int): The number of features to use for the studio FeatureHasher.
        config (dict): Config containing the model path.

    Returns:
        pd.DataFrame: The processed DataFrame. This DataFrame only contains id, synopsis, genres, studios and related columns.
    """

    data = data[['id', 'synopsis', 'title',
                 'genres', 'studios']].copy()
    data.dropna(inplace=True)

    print("Cleaning text data...")

    def clean_text(text):
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        text = cleantext.clean(
            text,
            fix_unicode=True,
            to_ascii=True,
            lower=False,
            normalize_whitespace=True,
            no_line_breaks=True,
            strip_lines=True,
            keep_two_line_breaks=False,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=False,
            no_emoji=True,
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            replace_with_punct="",
            lang="en"
        )
        """
        # Replace en dash with hyphen
        text = text.replace('\u2013', '\u002d')
        # Replace multiplication sign with x
        text = text.replace('\u00d7', '\u0078')
        # Normalize hyphenated words
        text = tprep.normalize.hyphenated_words(
            text)
        # Normalize quotation marks
        text = tprep.normalize.quotation_marks(
            text)
        # Normalize bullet points
        text = tprep.normalize.bullet_points(text)
        # Normalize whitespace
        text = tprep.normalize.whitespace(text)
        # Remove accents
        text = tprep.remove.accents(text)
        """
        # Remove HTML tags if any
        text = re.sub(r'<.*?>', '', text)
        # Remove source citations
        text = re.sub(r"\([\s+]?source.*?\)+", "", text,
                      flags=re.IGNORECASE)
        # Remove MAL citations
        text = re.sub(r"\[Writ.*?by.*?\]", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Strip leading and trailing whitespace
        text = text.strip()
        return text

    data['synopsis'] = data['synopsis'].apply(clean_text)
    data['related'] = data['title'].apply(clean_text)
    data.drop(columns=['title'], inplace=True)

    print("Preprocessing genres...")

    genres = {
        'Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love', 'Comedy',
        'Drama', 'Fantasy', 'Girls Love', 'Gourmet', 'Horror', 'Mystery', 'Romance',
        'Sci-Fi', 'Slice of Life', 'Sports', 'Supernatural', 'Suspense', 'Ecchi',
        'Erotica', 'Hentai', 'Adult Cast', 'Anthropomorphic', 'CGDCT', 'Childcare',
        'Combat Sports', 'Crossdressing', 'Delinquents', 'Detective', 'Educational',
        'Gag Humor', 'Gore', 'Harem', 'High Stakes Game', 'Historical', 'Idols (Female)',
        'Idols (Male)', 'Isekai', 'Iyashikei', 'Love Polygon', 'Magical Sex Shift',
        'Mahou Shoujo', 'Martial Arts', 'Mecha', 'Medical', 'Military', 'Music',
        'Mythology', 'Organized Crime', 'Otaku Culture', 'Parody', 'Performing Arts',
        'Pets', 'Psychological', 'Racing', 'Reincarnation', 'Reverse Harem',
        'Romantic Subtext', 'Samurai', 'School', 'Showbiz', 'Space',
        'Strategy Game', 'Super Power', 'Survival', 'Team Sports', 'Time Travel',
        'Vampire', 'Video Game', 'Visual Arts', 'Workplace', 'Josei',
        'Kids', 'Seinen', 'Shoujo', 'Shounen',
    }

    assert len(
        genres) == 76, "Incorrect number of genres, check list and compare to MAL"

    def f(entry):
        genres_set = set(genre['name'] for genre in entry)
        return genres_set

    data['genres'] = data['genres'].apply(f)

    genre_mlb = MultiLabelBinarizer()
    genre_mlb.fit([genres])
    data['genres'] = data['genres'].apply(
        lambda x: genre_mlb.transform([x]).reshape(-1,))

    print("Preprocessing studios...")

    def g(entry):
        studios_set = [studio['name'] for studio in entry]
        return studios_set

    data['studios'] = data['studios'].apply(g)

    # Use FeatureHasher to encode studios
    studio_hasher = FeatureHasher(
        n_features=studios_n_feat, input_type='string')
    studios_hashed = studio_hasher.transform(data['studios']).toarray()
    data['studios'] = [hash for hash in studios_hashed]
    data['studios'] = data['studios'].apply(lambda x: x.reshape(-1,))

    # Generate embeddings
    print("Generating embeddings...")
    synopsis_df, related_df = _create_embeddings(data, config)

    # Join all data
    print("Merging data...")
    data = data.merge(synopsis_df, on='id', how='left')
    data = data.merge(related_df, on='id', how='left')

    print(f"Final data shape: {data.shape}")
    return data
