import os
from models.autoencoder import Autoencoder
from recommender.recommender import Recommender
from config.config_manager import ConfigManager

def main():

    configs_dir = os.path.join(os.getenv('ROOT_DIR'), 'configs')
    
    configs = ConfigManager().load_configs()

    model = Autoencoder(configs['model'])
    recommender = Recommender(configs['recommender'], model)
    recommender.recommend('Alice', 5)