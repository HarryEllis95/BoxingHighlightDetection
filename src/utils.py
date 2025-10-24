import tomllib

def get_max_upload_size(config_path=".streamlit/config.toml") -> int:
    """Read maxUploadSize from Streamlit config"""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("server", {}).get("maxUploadSize", 200)  # default fallback