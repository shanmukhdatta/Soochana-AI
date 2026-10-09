"""Optional one-time downloads FROM repositories containing the user's own trained weights.

Set EVENT_MODEL_REPO / SENTIMENT_MODEL_REPO / IMPACT_MODEL_REPO first.
This script never creates or fabricates weights. Private repos need HF_TOKEN.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from app.config import Settings


def main():
    s = Settings.from_env()
    token = os.environ.get('HF_TOKEN') or None
    repo_specs = [
        ('EVENT_MODEL_REPO', s.event_model_dir),
        ('SENTIMENT_MODEL_REPO', s.sentiment_model_dir),
        ('IMPACT_MODEL_REPO', s.impact_model_dir),
    ]
    for var, local_dir in repo_specs:
        repo = os.environ.get(var)
        if not repo:
            print(var, 'not set; skipping')
            continue
        local_dir.mkdir(parents=True,exist_ok=True)
        print('Downloading owned model repository', repo, 'to', local_dir)
        snapshot_download(repo_id=repo, repo_type='model', local_dir=str(local_dir), token=token)
    print('Download process complete; run scripts.check_models next.')

if __name__ == '__main__':
    main()
