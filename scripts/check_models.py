"""Run before starting the server. Verifies actual weight and tokenizer artifacts."""
import json
from pathlib import Path
from app.config import Settings


def check():
    settings = Settings.from_env()
    dirs = {
        'event model': settings.event_model_dir,
        'sentiment model': settings.sentiment_model_dir,
    }
    errors = []
    for name, d in dirs.items():
        if not (d/'config.json').exists():
            errors.append(f'{name}: missing config.json in {d}')
        if not (d/'model.safetensors').exists() and not (d/'pytorch_model.bin').exists() and not list(d.glob('model-*.safetensors')):
            errors.append(f'{name}: model weights missing in {d}')
        if not any((d/n).exists() for n in ['tokenizer.json','vocab.txt']):
            errors.append(f'{name}: tokenizer.json/vocab.txt missing in {d}')
        if not (d/'tokenizer_config.json').exists():
            errors.append(f'{name}: tokenizer_config.json missing in {d}')
        print(f'{name}: {d}')
    impact = settings.impact_model_dir
    for file in ['impact_model_final.joblib','impact_preprocessor.joblib']:
        if not (impact/file).is_file():
            errors.append(f'impact model: missing {impact/file}')
    print('impact model:', impact)
    for file in [
        'event_labels.json',
        'sector_mapping.json',
        'supported_tickers.json',
        'impact_model_feature_manifest.json',
        'impact_decile_thresholds.json',
        'impact_step2_market_regime_thresholds.json',
    ]:
        if not (settings.config_dir/file).is_file():
            errors.append(f'config: missing {settings.config_dir/file}')
    if errors:
        print('\nNOT READY:')
        for err in errors:
            print(' -', err)
        raise SystemExit(1)
    print('\nModel file presence checks: PASSED (not a complete loading test)')

if __name__ == '__main__':
    check()
