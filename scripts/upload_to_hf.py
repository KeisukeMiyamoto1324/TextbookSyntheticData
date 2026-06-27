from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi


REPO_ID = "MK0727/SyntheticTextbook-jp"
PARQUET_FILES = sorted(Path("results").glob("textbook-[0-9][0-9][0-9][0-9][0-9].parquet"))
README_FILE = Path("README.md")


def main() -> None:
    # ---------------------------------------------------------
    # Load HF_TOKEN from .env and upload dataset files.
    # ---------------------------------------------------------
    load_dotenv()

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=[
            *[
                CommitOperationAdd(
                    path_in_repo=parquet_file.name,
                    path_or_fileobj=parquet_file,
                )
                for parquet_file in PARQUET_FILES
            ],
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=README_FILE),
        ],
        commit_message="Upload textbook dataset shards",
    )

    print(commit.commit_url)


if __name__ == "__main__":
    main()
