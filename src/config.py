from dotenv import load_dotenv

load_dotenv()

PROJECT_ID_ENV_NAME: str = "GOOGLE_CLOUD_PROJECT"
PROJECT_ID_ENV_PREFIX: str = "GOOGLE_CLOUD_PROJECT_"
LOCATION: str = "global"
MODEL: str = "google/gemma-4-26b-a4b-it-maas"
ENABLE_THINKING: bool = False
