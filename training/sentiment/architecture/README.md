# FinBERT Sentiment Regressor Architecture (`training/sentiment/architecture/`)

This directory provides the PyTorch neural network module defining the custom continuous regression head on top of the FinBERT backbone.

---

## 1. Module Definition

* [`modeling_finbert_sentiment_regressor.py`](modeling_finbert_sentiment_regressor.py): Defines `FinBertSentimentRegressor`, extracting the $[CLS]$ pooled hidden state $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^{768}$, applying dropout ($p=0.10$), projecting via a linear layer $W \in \mathbb{R}^{768 \times 1} + b$, and applying a hyperbolic tangent activation:
  $$\hat{s} = \tanh\left(\mathbf{w}^T \mathbf{h}_{\text{CLS}} + b\right) \in [-1.0, +1.0]$$
