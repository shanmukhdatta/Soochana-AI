"""Local HF inference. Separate tokenizers are required: model vocabularies differ."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from app.engine.sentiment_model import FinBertSentimentRegressor


class ModelCompatibilityError(RuntimeError):
    pass


def normalize_label(s: str) -> str:
    import re
    return re.sub(r'[^a-z]+', '', str(s).lower())


class NLPModels:
    def __init__(self, event_dir: Path, sentiment_dir: Path,
                 labels: dict[int, str], device: str='cpu', max_length: int=128):
        if device not in {'cpu', 'cuda', 'auto'}:
            raise ValueError('MODEL_DEVICE must be cpu, cuda or auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('MODEL_DEVICE=cuda but CUDA is unavailable')
        self.device = torch.device(device)
        self.max_length = max_length
        self.labels = labels

        event_dir = Path(event_dir)
        sentiment_dir = Path(sentiment_dir)
        for name, p in [('event model', event_dir), ('sentiment model', sentiment_dir)]:
            if not (p/'config.json').exists():
                raise FileNotFoundError(
                    f'Missing {name} config.json at {p}; place ORIGINAL model weights+tokenizer, not reference configs'
                )

        self.event_tok = AutoTokenizer.from_pretrained(event_dir, local_files_only=True)
        self.event_model, load_info = AutoModelForSequenceClassification.from_pretrained(
            event_dir, local_files_only=True, output_loading_info=True
        )
        if int(self.event_model.config.num_labels) != 5:
            raise ModelCompatibilityError('Event model must have exactly five labels')
        got = {int(i): normalize_label(v) for i, v in self.event_model.config.id2label.items()}
        wanted = {int(i): normalize_label(v) for i, v in labels.items()}
        if got != wanted:
            raise ModelCompatibilityError(f'Event id2label mismatch: {got} != {wanted}')
        missing_event_head = [k for k in load_info.get('missing_keys', []) if 'classifier' in k]
        if missing_event_head:
            raise ModelCompatibilityError(f'Event classifier head not loaded: {missing_event_head}')
        if int(self.event_model.config.vocab_size) != int(self.event_tok.vocab_size):
            raise ModelCompatibilityError('Event tokenizer vocab_size mismatch')

        self.sent_tok = AutoTokenizer.from_pretrained(sentiment_dir, local_files_only=True)
        config = AutoConfig.from_pretrained(sentiment_dir, local_files_only=True)
        if getattr(config, 'problem_type', None) != 'regression' or int(config.num_labels) != 1:
            raise ModelCompatibilityError('Sentiment model is not one-output regression')
        if getattr(config, 'sentiment_head', None) != 'Dropout -> Linear(hidden_size,1) -> tanh':
            raise ModelCompatibilityError('Sentiment model head metadata differs from training')
        self.sent_model, load_info = FinBertSentimentRegressor.from_pretrained(
            sentiment_dir, config=config, local_files_only=True, output_loading_info=True
        )
        missing_sent_head = [k for k in load_info.get('missing_keys', []) if k.startswith('regressor.')]
        if missing_sent_head:
            raise ModelCompatibilityError(f'Sentiment regression head not loaded: {missing_sent_head}')
        if int(config.vocab_size) != int(self.sent_tok.vocab_size):
            raise ModelCompatibilityError('Sentiment tokenizer vocab_size mismatch')

        self.event_model.to(self.device).eval()
        self.sent_model.to(self.device).eval()

    def analyze(self, text: str) -> tuple[dict[str, object], float]:
        event_inputs = self.event_tok(
            text, return_tensors='pt', truncation=True,
            max_length=self.max_length,
        )
        sentiment_inputs = self.sent_tok(
            text, return_tensors='pt', truncation=True,
            max_length=self.max_length,
        )
        event_inputs = {k: v.to(self.device) for k, v in event_inputs.items()}
        sentiment_inputs = {k: v.to(self.device) for k, v in sentiment_inputs.items()}
        with torch.inference_mode():
            logits = self.event_model(**event_inputs).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
            sentiment = float(self.sent_model(**sentiment_inputs).logits.view(-1)[0].cpu())
        if probabilities.shape != (5,) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError('Non-finite or wrong-length event probabilities')
        if not np.isfinite(sentiment):
            raise RuntimeError('Non-finite sentiment score')
        selected = int(np.argmax(probabilities))
        return (
            {
                'class_name': self.labels[selected],
                'confidence': float(probabilities[selected]),
                'probabilities': {
                    self.labels[i]: float(probabilities[i]) for i in range(5)
                },
                'probability_vector': [float(p) for p in probabilities],
                'class_id': selected,
            },
            float(np.clip(sentiment, -1, 1)),
        )
