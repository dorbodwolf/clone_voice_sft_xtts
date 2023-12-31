from huggingface_hub import snapshot_download
from huggingface_hub import login
login()

model_id="openai/whisper-large-v2"
model_id="guillaumekln/faster-whisper-large-v2"
snapshot_download(repo_id=model_id,
                  local_dir_use_symlinks=False, revision="main")
