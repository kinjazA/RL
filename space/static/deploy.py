import os
from huggingface_hub import HfApi

api = HfApi()
repo = "Shawnno/Interview_Assistant"
folder = os.path.dirname(os.path.abspath(__file__))

api.upload_file(
    path_or_fileobj=folder + r"\index.html",
    path_in_repo="index.html",
    repo_id=repo, repo_type="space",
)
api.upload_file(
    path_or_fileobj=folder + r"\README.md",
    path_in_repo="README.md",
    repo_id=repo, repo_type="space",
)
for name in ["app.py", "requirements.txt"]:
    try:
        api.delete_file(path_in_repo=name, repo_id=repo, repo_type="space")
        print("deleted", name)
    except Exception as e:
        print("skip delete", name, type(e).__name__)
print("deployed static showcase")
