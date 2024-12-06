import argparse
import json
import os

import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from thefuzz import fuzz, process

if os.getenv("ENV") == "dev":
    from anime_recommender import data_loader, preprocess
else:
    data_loader = None
    preprocess = None
from anime_recommender.recommend import Recommender


class AnimeRecommender:
    """Main class for the anime recommendation system."""

    def __init__(
        self,
        config_path: str,
    ):
        self.config = self._load_config(config_path=config_path)
        self.dataset = self._load_dataset(self.config["dataset"])
        self.collection = self._load_collection(config=self.config["vector_database"])

    def _replace_placeholders(self, data) -> dict:
        """Recursively replaces environment variables in the data.

        Args:
            data (dict|list|str): The data to replace the placeholders in.

        Returns:
            dict|list|str: The data with the placeholders replaced.
        """
        if isinstance(data, dict):
            return {
                key: self._replace_placeholders(value) for key, value in data.items()
            }
        if isinstance(data, list):
            return [self._replace_placeholders(item) for item in data]
        if isinstance(data, str):
            return os.path.expandvars(data)
        return data

    def _load_config(self, config_path: str) -> dict:
        """Loads the config file from the path.

        Args:
            config_path (str): The path to the config file.

        Returns:
            dict: The config file as a dictionary.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config directory {config_path} does not exist.")
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        config = self._replace_placeholders(config)

        return config

    def _load_collection(self, config: dict, chunk_size: int = 1000) -> Collection:
        """Loads the vector database collection with anime embeddings.

        Args:
            config (dict): The configuration for the vector database.

        Returns:
            Collection: The collection object.
        """
        connections.connect(
            "default", host=os.getenv("HOST"), port=os.getenv("HOST_PORT")
        )

        collection_name = config["collection_name"]

        if collection_name in utility.list_collections():
            ret_col = Collection(collection_name)
            ret_col.load()
            return ret_col

        print("Collection not found, creating...")

        collection_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(
                name="synopsis_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=config["text_embedding_dim"],
            ),
            FieldSchema(
                name="related_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=config["text_embedding_dim"],
            ),
            FieldSchema(
                name="genres", dtype=DataType.FLOAT_VECTOR, dim=config["genres_dim"]
            ),
            FieldSchema(
                name="studios", dtype=DataType.FLOAT_VECTOR, dim=config["studios_dim"]
            ),
        ]

        combined_schema = CollectionSchema(
            fields=collection_fields, description="Main anime collection."
        )
        collection = Collection(name=collection_name, schema=combined_schema)

        index_params_embedd = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 512},
        }
        index_params_gen_stud = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 64},
        }
        collection.create_index(
            field_name="synopsis_embedding", index_params=index_params_embedd
        )
        collection.create_index(
            field_name="related_embedding", index_params=index_params_embedd
        )
        collection.create_index(field_name="genres", index_params=index_params_gen_stud)
        collection.create_index(
            field_name="studios", index_params=index_params_gen_stud
        )

        collection.load()

        # Could add download and embedding generation here

        vector_dataset = self._load_dataset(config["embedded_dataset_path"])

        for i in range(0, len(vector_dataset), chunk_size):
            print(
                f"Inserting chunk {i//chunk_size + 1} of {len(vector_dataset)//chunk_size + 1}"
            )
            collection.insert(
                vector_dataset.iloc[i : i + chunk_size].to_dict(orient="records")
            )

        return collection

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Loads the raw anime dataset from the specified path.

        Args:
            dataset_path (str): The path to the raw anime dataset.

        Returns:
            pd.DataFrame: The raw anime dataset.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")
        return pd.read_json(dataset_path, orient="records")

    def _find_anime(
        self, title_query: str, dataset: pd.DataFrame, autoselect: bool = False
    ) -> int | None:
        """Looks for animes matching the query string.

        Args:
            title_query (str): The query string to search for.
            dataset (pd.DataFrame): The raw anime dataset.

        Returns:
            int | None: The ID of the anime found, or None if the search was aborted.
        """
        res = process.extract(title_query, dataset["title"], scorer=fuzz.partial_ratio)

        if autoselect:
            return dataset.iloc[res[0][2]]["id"]

        print("The following animes were found:")
        for i, (title, _, _) in enumerate(res):
            print(f"{i}. {title}")

        while True:
            choice = input(
                "Select the anime you are looking for, or n to abort search (0/1/.../n): "
            )

            if choice.lower() == "n":
                print("Search aborted.")
                return None
            if not choice.isdigit() or int(choice) < 0 or int(choice) >= len(res):
                print("Invalid choice.")
                continue
            break

        return dataset.iloc[res[int(choice)][2]]["id"]

    def rebuild(self) -> None:
        access_token = self.config["recommender"]["MAL_ACCESS_TOKEN"]
        headers = {"Authorization": f"Bearer {access_token}"}
        collector = data_loader.DataCollector(
            headers=headers,
            base_url=self.config["recommender"]["BASE_URL"],
            request_delay=4.15,
        )
        collector.run_collection(
            media_type="anime", output_path=self.config["dataset"], limit=self.rebuild
        )

        # Preprocess the dataset
        dataset_raw = pd.read_json(self.config["dataset"], orient="records")
        dataset_preprocessed = preprocess.make_embeddings(
            data=dataset_raw,
            studios_n_feat=self.config["vector_database"]["studios_dim"],
            config=self.config["preprocessing"],
        )
        dataset_preprocessed.to_json(
            self.config["vector_database"]["embedded_dataset_path"], orient="records"
        )

    def run(
        self,
        search_str: str,
        anime_mode: bool,
        limit: int = 10,
        autoselect: bool = False,
    ) -> pd.DataFrame:
        """Main function for the anime recommendation system.

        Returns:
            pd.DataFrame: The recommendations as a DataFrame.
        """

        recommender = Recommender(
            config=self.config["recommender"],
            collection=self.collection,
            dataset=self.dataset,
        )

        recommendations = None

        # Dispatch the recommendation task
        if not anime_mode:
            # print(f"Recommendations for user {self.search_str}:")
            recommendations = recommender.recommend_by_username(
                user_name=search_str, limit=limit
            )
        else:
            anime_id = self._find_anime(
                title_query=search_str, dataset=self.dataset, autoselect=autoselect
            )
            if anime_id is None:
                return
            # print(f"Recommendations for \'{dataset[dataset['id'] == anime_id]['title']}\':")
            recommendations = recommender.recommend_by_id(
                anime_id=anime_id, limit=limit
            )

        return recommendations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anime Recommendation System")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--username", help="Username for recommendations")
    group.add_argument("-a", "--anime", help="Anime title for similar recommendations")
    group.add_argument(
        "-r",
        "--rebuild",
        # action="store_true",
        default=0,
        type=int,
        help="Rebuild the dataset from scratch",
    )

    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=10,
        help="Number of recommendations to return",
    )

    args = parser.parse_args()

    if args.rebuild:
        rec = AnimeRecommender(config_path="config.json")
        rec.rebuild()

    if args.username:
        rec = AnimeRecommender(config_path="config.json")
        print(rec.run(search_str=args.username, anime_mode=False, limit=args.limit))
    elif args.anime:
        rec = AnimeRecommender(config_path="config.json")
        print(rec.run(search_str=args.anime, anime_mode=True, limit=args.limit))
