from huggingface_hub import snapshot_download
import os

local_dir = "./models/bge-m3"
os.makedirs(local_dir, exist_ok=True)

print("Downloading BAAI/bge-m3 model from HuggingFace Hub...")
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 실제 파일 복사
)
print(f"Model saved to {os.path.abspath(local_dir)}")
