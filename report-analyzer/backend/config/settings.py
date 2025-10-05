from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional, Literal
import yaml
import os
from pathlib import Path

def load_yaml_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    def replace_env_vars(config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                replace_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config_dict[key] = os.environ.get(env_var)
    
    replace_env_vars(config)
    return config

class Settings(BaseSettings):
    config: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = load_yaml_config()
    
    @property
    def cloud_provider(self) -> str:
        return self.config["cloud"]["provider"]
    
    @property
    def cloud_config(self) -> Dict[str, Any]:
        return self.config["cloud"][self.cloud_provider]
    
    @property
    def vector_store_config(self) -> Dict[str, Any]:
        return self.config["vector_store"]
    
    @property
    def tools_config(self) -> Dict[str, Any]:
        return self.config["tools"]
    
    class Config:
        env_file = ".env"