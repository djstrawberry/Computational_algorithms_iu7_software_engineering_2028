import config as config
from .file_reader import load_tables
from .model import get_model_params

def main():

    params = config.build_params()
    tables = load_tables("data")

if __name__ == "__main__":
    main()