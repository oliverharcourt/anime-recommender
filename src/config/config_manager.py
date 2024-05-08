import os
import json

class ConfigManager:

    def replace_placeholders(self, config: dict) -> dict:        
        # Replace placeholders with environment variable values
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value.strip('${}')
                env_var = env_var.rstrip('}')
                assert len(env_var) > 0, f'Invalid placeholder: {value}'
                assert env_var in os.environ, f'Missing environment variable: {env_var}'
                config[key] = os.getenv(env_var)
        return config

    def load_configs(self, config_dir: str) -> dict:
        configs = {}
        for file_name in os.listdir(config_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(config_dir, file_name)
                with open(file_path, 'r') as file:
                    config = json.load(file)
                name = file_name.rstrip('.json').split('_')[-1]
                configs[name] = self.replace_placeholders(config)
        return configs

if __name__ == '__main__':
    config_manager = ConfigManager()
    root_dir = os.getenv('ROOT_DIR')
    configs_dir = os.path.join(root_dir, 'configs')
    configs = config_manager.load_configs(configs_dir)
    print(json.dumps(configs, indent=4))