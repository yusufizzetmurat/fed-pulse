import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download

p = snapshot_download(
    repo_id="yusufizzetmurat/fed-pulse-training-package",
    repo_type="dataset",
    revision="b0320d8731753bc5bf4a05b9dcd57898c3f84fbd",
    local_dir="/root/fed-pulse/data/processed/canonical",
    local_dir_use_symlinks=False,
)
print("DOWNLOAD_COMPLETE", p)
