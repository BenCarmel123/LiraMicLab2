from dotenv import load_dotenv
import os

load_dotenv()

kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")

os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

from kaggle.api.kaggle_api_extended import KaggleApi

try:
    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated successfully.")
except Exception as e:
    print(f"Error authenticating with Kaggle API: {e}")

    