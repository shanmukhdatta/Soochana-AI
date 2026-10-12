# Reference Model Configurations (`configs/reference/`)

This directory houses frozen architectural blueprints used exclusively by automated configuration tests to prevent drift between training experiments and production runtimes.

---

## 1. File Inventory

* [`event_model_config_CHECK_ONLY.json`](event_model_config_CHECK_ONLY.json): Exact vocabulary size ($30,873$), number of labels ($5$), hidden dimension ($768$), and id-to-label ordering required for the FinBERT Event Classifier.
* [`sentiment_model_config_CHECK_ONLY.json`](sentiment_model_config_CHECK_ONLY.json): Expected vocabulary size ($30,522$), problem type (`regression`), and regression head metadata for the Continuous Sentiment Regressor.

---

## 2. Testing Role

These files are evaluated during test suite execution by [`backend/tests/test_configuration.py`](../../backend/tests/test_configuration.py) to guarantee that any new model export strictly satisfies the runtime interface.
