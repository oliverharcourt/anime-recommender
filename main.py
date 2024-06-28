import argparse
import json
import os

import pandas as pd
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)
from thefuzz import fuzz, process

from src.recommendation.recommend import Recommender


def _replace_placeholders(data) -> dict:
    """Recursively replaces environment variables in the data.

    Args:
        data (dict|list|str): The data to replace the placeholders in.

    Returns:
        dict|list|str: The data with the placeholders replaced.
    """
    if isinstance(data, dict):
        return {key: _replace_placeholders(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_replace_placeholders(item) for item in data]
    if isinstance(data, str):
        return os.path.expandvars(data)
    return data


def _ammend_paths(config: dict, project_root: str) -> dict:
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = _ammend_paths(value, project_root)
        elif isinstance(value, str) and not value.startswith('/'):
            if os.path.exists(os.path.join(project_root, value)):
                config[key] = os.path.join(project_root, value)
    return config


def _load_config(config_path: str) -> dict:
    """Loads the config file from the path.

    Args:
        config_path (str): The path to the config file.

    Returns:
        dict: The config file as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """

    # project_root = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config directory {config_path} does not exist.")
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)

    config = _replace_placeholders(config)

    # this might not be necessary
    # config = _ammend_paths(config, project_root)

    return config


def _load_collection(config: dict) -> Collection:
    """Loads the vector database collection with anime embeddings.

    Args:
        collection_name (str): The name of the collection.

    Returns:
        Collection: The collection object.
    """
    connections.connect("default", host=config['host'], port=config['port'])

    collection_name = config['collection_name']

    if collection_name in utility.list_collections():
        print("Collection found, loading...")
        ret_col = Collection(collection_name)
        ret_col.load()
        return ret_col

    print("Collection not found, creating...")

    collection_fields = [
        FieldSchema(
            name='id',
            dtype=DataType.INT64,
            is_primary=True),
        FieldSchema(
            name='synopsis_embedding',
            dtype=DataType.FLOAT_VECTOR,
            dim=config['text_embedding_dim']),
        FieldSchema(
            name='related_embedding',
            dtype=DataType.FLOAT_VECTOR,
            dim=config['text_embedding_dim']),
        FieldSchema(
            name='genres',
            dtype=DataType.FLOAT_VECTOR,
            dim=config['genres_dim']),
        FieldSchema(
            name='studios',
            dtype=DataType.FLOAT_VECTOR,
            dim=config['studios_dim'])
    ]

    combined_schema = CollectionSchema(
        fields=collection_fields,
        description="Main anime collection.")
    collection = Collection(
        name=collection_name, schema=combined_schema)

    index_params_embedd = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 512}
    }
    index_params_gen_stud = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 64}
    }
    collection.create_index(
        field_name="synopsis_embedding", index_params=index_params_embedd)
    collection.create_index(
        field_name="related_embedding", index_params=index_params_embedd)
    collection.create_index(
        field_name="genres", index_params=index_params_gen_stud)
    collection.create_index(
        field_name="studios", index_params=index_params_gen_stud)

    collection.load()
    vector_dataset = _load_dataset(config["dataset_path"])

    chunk_size = 1000
    for i in range(0, len(vector_dataset), chunk_size):
        print(
            f"Inserting chunk {i//chunk_size + 1} of {len(vector_dataset)//chunk_size + 1}")
        collection.insert(
            vector_dataset.iloc[i:i+chunk_size].to_dict(orient='records'))

    return collection


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    """Loads the raw anime dataset from the specified path.

    Args:
        dataset_path (str): The path to the raw anime dataset.

    Returns:
        pd.DataFrame: The raw anime dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset file {dataset_path} does not exist.")
    return pd.read_json(dataset_path, orient='records')


def _dispatch_user_recommendation(user_name: str, limit: int, collection: Collection, dataset: pd.DataFrame, config: dict):
    """Dispatches the recommendation task for a user to the recommender.

    Args:
        user_name (str): The name of the user to recommend anime to.
        limit (int): The number of anime to recommend.
        collection (Collection): The vector database collection.
        dataset (pd.DataFrame): The raw anime dataset.
        config (dict): The recommender configuration.
    """

    recommender = Recommender(
        config=config,
        collection=collection,
        dataset=dataset)
    # print is just for testing
    print(recommender.recommend_by_username(user_name=user_name, limit=limit))


def _dispatch_anime_recommendations(anime_id: int, limit: int, collection: Collection, dataset: pd.DataFrame, config: dict):
    """Dispatches the recommendation task for an anime to the recommender.

    Args:
        anime_id (int): The ID of the anime to recommend similar anime for.
        limit (int): The number of anime to recommend.
        collection (Collection): The vector database collection.
        dataset (pd.DataFrame): The raw anime dataset.
        config (dict): The recommender configuration.
    """

    recommender = Recommender(
        config=config,
        collection=collection,
        dataset=dataset)
    # print is just for testing
    print(recommender.recommend_by_id(anime_id=anime_id, limit=limit))


def _find_anime(title_query: str, dataset: pd.DataFrame) -> int | None:
    """Looks for animes matching the query string.

    Args:
        title_query (str): The query string to search for.
        dataset (pd.DataFrame): The raw anime dataset.

    Returns:
        int | None: The ID of the anime found, or None if the search was aborted.
    """
    res = process.extract(
        title_query, dataset['title'], scorer=fuzz.partial_ratio)
    print("The following animes were found:")
    for i, (title, _, _) in enumerate(res):
        print(f"{i+1}. {title}")

    while True:
        choice = input(
            "Select the anime you are looking for, or n to abort search (1/2/.../n): ")

        if choice.lower() == 'n':
            print("Search aborted.")
            return None
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(res):
            print("Invalid choice.")
            continue
        else:
            break

    return dataset.iloc[res[int(choice)-1][2]]['id']


def main(args):

    config = _load_config(config_path="config.json")

    # Load the raw dataset
    dataset = _load_dataset(config['dataset'])
    # Load the vector database collection
    collection = _load_collection(config=config['vector_database'])
    # print(f"args: {args}")
    # Dispatch the recommendation task
    if args.username:
        _dispatch_user_recommendation(user_name=args.username, limit=args.limit,
                                      collection=collection, dataset=dataset, config=config['recommender'])
    elif args.anime:
        anime_id = _find_anime(title_query=args.anime, dataset=dataset)
        _dispatch_anime_recommendations(anime_id=anime_id, limit=args.limit,
                                        collection=collection, dataset=dataset, config=config['recommender'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anime Recommendation System')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--username", help="Username for recommendations")
    group.add_argument(
        "-a", "--anime", help="Anime title for similar recommendations")

    parser.add_argument("-l", "--limit", type=int, default=10,
                        help="Number of recommendations to return")

    args = parser.parse_args()
    main(args)
