# Soochana AI — Model Training Source Code

This folder contains the original training notebooks and scripts for all three AI models. Adaptations make dataset input/output paths configurable, avoid hard dependency on Kaggle Secrets, and preserve the original model-training logic. Scripts do not modify the production `app/` or `models/` directories.

## Training Modules

* [**Event Classifier Training**](event_classifier/README.md): 5-class FinBERT event classifier (`yiyanghkust/finbert-tone`) trained on 16,576 cleaned financial records.
* [**Sentiment Regressor Training**](sentiment/README.md): Continuous FinBERT regression model (`ProsusAI/finbert`) predicting sentiment in `[-1.0, +1.0]` on the FiQA corpus.
* [**Impact Severity Engine Training**](impact_engine/README.md): XGBoost / LightGBM abnormal reaction severity decile model trained on 27,460 pre-publication financial features.

**Python 3.11 recommended; GPU recommended for BERT fine-tuning.** From repository root:

```bash
python -m pip install -r training/event_classifier/requirements.txt
python -m pip install -r training/sentiment/requirements.txt
python training/run.py check
python training/run.py event-gather
python training/run.py event-preprocess
python training/run.py event-train
python training/run.py sentiment-prepare
python training/run.py sentiment-train
```

Run stages sequentially. They download public datasets/model foundations, and require internet connectivity. Intermediate files are written to `training_runs/event_classifier/` or `training_runs/sentiment/`. To use previously prepared split files, pass `--input-dir PATH` to a training stage. To choose an output directory, pass `--workdir PATH`.

The original event experiment trains on `event_train` (real-data split); **synthetic generation is optional and separate**, requiring your own Groq key and an explicit call to `python training/run.py event-synthetic-optional` (may incur API charges). Synthetic output is not automatically used for the documented event training stage.

The model checkpoints are not bundled, and underlying dataset revisions are not pinned. Re-execution depends on external dataset availability, package versions, compute and pretrained weights. These are source-level reproduction instructions, not a guarantee of identical weights to the existing deployed models. Event fine-tuning in the original source initializes `nickmuchi/finbert-tone-finetuned-finance-topic-classification`; the published deployed classifier config references `yiyanghkust/finbert-tone`, so exact checkpoint identity cannot be assumed.

Never commit API tokens, private datasets or trained weights without checking provider licenses and file size limits.
