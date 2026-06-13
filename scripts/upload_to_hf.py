from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


REPO_ID = "MK0727/SyntheticTextbook-jp"
LOCAL_FILE = Path("results/textbook.parquet")
REMOTE_FILE = "textbook.parquet"


def main() -> None:
    # ---------------------------------------------------------
    # Load HF_TOKEN from .env and upload the parquet file.
    # ---------------------------------------------------------
    load_dotenv()

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    commit = api.upload_file(
        path_or_fileobj=LOCAL_FILE,
        path_in_repo=REMOTE_FILE,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Upload cleaned textbook parquet",
    )

    print(commit)


if __name__ == "__main__":
    main()
