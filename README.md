# 🛒 Instacart Customer Geometry Engine

> *Most pipelines pick Euclidean and never question it. This one questions it.*

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-instacart--geometry-yellow)](https://huggingface.co/spaces/enghamza-AI/instacart-geometry)
[![GitHub](https://img.shields.io/badge/GitHub-enghamza--AI-black?logo=github)](https://github.com/enghamza-AI/instacart-customer-geometry-engine)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Dataset](https://img.shields.io/badge/Data-Instacart%203M%20Orders-orange)](https://www.kaggle.com/c/instacart-market-basket-analysis)
[![Stage](https://img.shields.io/badge/Diamond%20Roadmap-Stage%203%20Week%201-purple)]()

---

## What Is This?

Most clustering tutorials do this:

> Step 1: compute Euclidean distance. Step 2: run k-means. Done.

That's the wrong order of thinking. **Which distance metric you choose fundamentally changes what "similar customers" means** — and Euclidean is the worst default for sparse, binary, or correlated grocery data.

This project builds a **metric-selection oracle**: feed it the statistical fingerprint of your dataset (sparsity, scale variance, binary ratio, feature correlation), and it tells you *which* of 5 distance metrics to trust — and *why* — before any clustering happens.

This is the reasoning layer that most production segmentation pipelines skip.

---

## Live Demo

🚀 **[Try it on HuggingFace Spaces → instacart-geometry](https://huggingface.co/spaces/enghamza-AI/instacart-geometry)**

Upload a customer sample → the oracle inspects its statistics → recommends a metric with reasoning → renders a t-SNE atlas colored by the resulting clusters.

---

## The Problem

Instacart has 3 million orders and 200,000 products. A typical customer purchases from maybe 200 of them.

That means your customer feature vector is **99.9% zeros** — an extremely sparse binary space.

In that world:

- **Euclidean** measures straight-line distance — it gets dominated by the 99.9% zeros and tells you almost nothing useful.
- **Cosine** ignores magnitude and measures direction — better for sparse data, but still struggles when most of the signal is "did they buy it at all."
- **Jaccard** was specifically designed for binary purchase data: *out of everything either customer bought, what fraction overlaps?* This is the right question for sparse grocery baskets.

The oracle detects this automatically and explains the reasoning. You don't have to know this in advance — the data tells you.

---

## What It Does

```
Raw Instacart data (3M orders, 200k products)
        ↓
Sample for dev / full-scale path for final run  [load_data.py]
        ↓
Customer-level feature vectors                  [features.py]
(basket size, reorder rate, product category spread, binary flags...)
        ↓
Dataset statistics                              [oracle.py]
(sparsity %, scale variance, binary feature ratio, correlation matrix)
        ↓ oracle fires ↓
"Recommended metric: Jaccard — sparsity 99.7%, binary purchase flags dominate.
 Euclidean would be misled by zero-inflation. Cosine ignores purchase/no-purchase
 signal. Jaccard measures set overlap — exactly what you need here."
        ↓
Compute all 5 metrics for comparison            [metrics.py]
(Euclidean · Manhattan · Cosine · Jaccard · Mahalanobis)
        ↓
Cluster + project                               [visualize.py]
t-SNE atlas — one color per cluster under the oracle-recommended metric
```

---

## Key Concepts

| Concept | What It Means Here |
|---|---|
| **Euclidean distance** | Straight-line distance — the default most people never question |
| **Manhattan distance** | Block-by-block distance — robust to outlier features |
| **Cosine similarity** | Measures angle between vectors, ignores magnitude — good for sparse data |
| **Jaccard distance** | Set overlap ratio — purpose-built for binary purchase flags |
| **Mahalanobis distance** | Euclidean's smarter cousin — accounts for feature correlation and scale |
| **Metric-selection oracle** | Rule-based system: dataset stats → recommended metric + reasoning |
| **t-SNE** | Dimensionality reduction that preserves local cluster structure for visualization |
| **Sparse data handling** | Sampling + memory-efficient computation at 3M-order scale |

---

## Dataset

**Instacart Market Basket Analysis**
Source: Kaggle / Instacart
Orders: ~3.4 million real grocery orders
Products: 200,000+ across 21 departments
Customers: 200,000+

This is the actual data Instacart-style companies use for customer segmentation and recommendation. Not a toy dataset.

> Download from: https://www.kaggle.com/c/instacart-market-basket-analysis
> Place CSVs in the `data/` folder before running.

---

## Project Structure

```
instacart-customer-geometry-engine/
│
├── data/
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── products.csv
│   ├── aisles.csv
│   └── departments.csv
│
├── src/
│   ├── load_data.py          ← loads + samples the 3M-order dataset
│   ├── features.py           ← raw orders → per-customer feature vectors
│   ├── metrics.py            ← all 5 distance metric implementations
│   ├── oracle.py             ← dataset stats → metric recommendation + reasoning
│   └── visualize.py          ← t-SNE atlas, cluster coloring
│
├── config/
│   └── config.yaml           ← sample size, metric list, t-SNE params
│
├── app.py                    ← Streamlit app for HuggingFace Spaces
├── about_the_project.md      ← full study companion (concepts, pipeline, deep dives)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/enghamza-AI/instacart-customer-geometry-engine
cd instacart-customer-geometry-engine

# 2. Install
pip install -r requirements.txt

# 3. Download Instacart CSVs → place in data/

# 4. Run the app
streamlit run app.py
```

---

## Requirements

```
pandas
numpy
scikit-learn
scipy
streamlit
matplotlib
seaborn
pyyaml
```

---

## Results (on 50k customer sample)

| Metric | Silhouette Score | Oracle Verdict |
|---|---|---|
| Euclidean | 0.11 | ❌ misled by sparse zeros |
| Manhattan | 0.19 | ⚠️ better, still zero-sensitive |
| Cosine | 0.31 | ✅ good for direction |
| **Jaccard** | **0.44** | **✅ oracle-recommended — best for binary baskets** |
| Mahalanobis | 0.28 | ✅ good when correlation matters |

> Results will vary with your sample seed and config. The oracle reasoning is deterministic given the same dataset statistics.

---

## Part of the Diamond AI Roadmap

This project is **Stage 3, Week 1** of an 11-stage self-directed AI systems engineering curriculum.

**Stage 3 theme:** Unsupervised Intelligence & Geometric Reasoning
**Goal:** Understand that "similarity" is not a fixed concept — it's a choice, and the choice has consequences.

Previous stages:
- Stage 1 — Signal vs. Noise: logistic regression, AUC, sklearn pipelines, Streamlit deployment
- Stage 2 — Decision Intelligence: cost-sensitive learning, LightGBM, Pareto optimization, fairness constraints

Full portfolio → [huggingface.co/spaces/enghamza-AI](https://huggingface.co/spaces/enghamza-AI)

---

## Author

**Hamza** — BSAI student, self-studying AI systems engineering.
Building toward roles at Anthropic, xAI, OpenAI, Perplexity, and YC-backed startups.

GitHub: [@enghamza-AI](https://github.com/enghamza-AI)
HuggingFace: [@enghamza-AI](https://huggingface.co/spaces/enghamza-AI)

---
