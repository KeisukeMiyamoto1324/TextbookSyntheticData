from dotenv import load_dotenv

load_dotenv()

PROJECT_ID_ENV_NAME: str = "GOOGLE_CLOUD_PROJECT"
LOCATION: str = "global"
MODEL: str = "google/gemini-3-flash-preview"
ENABLE_THINKING: bool = False
