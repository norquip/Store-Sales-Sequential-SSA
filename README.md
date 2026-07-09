

# 📈 Store Sales Analysis with Sequential Singular Spectrum Analysis (SSA)

A Streamlit application demonstrating Sequential Singular Spectrum Analysis (SSA) for trend extraction, oscillatory mode identification, and signal reconstruction in retail sales time series.
---

## 📌 Overview

This project presents a complete Sequential SSA workflow applied to a daily retail sales dataset.

The analysis is performed in two stages:

1. **Stage 1**
   - Extract the long-term trend using a small embedding dimension.

2. **Stage 2**
   - Apply SSA to the detrended residual using a larger embedding dimension to identify oscillatory modes.

The application provides interactive visualizations that explain each step of the decomposition process.

---

## 🚀 Features

- Sequential SSA decomposition
- Trend extraction
- Scree plot visualization
- Leading eigenvector analysis
- Eigenvector pair plots
- W-correlation matrix
- Oscillatory mode grouping
- Relative and cumulative variance contributions
- Reconstruction of oscillatory modes
- Final signal reconstruction
- Interactive Streamlit interface

---

## 🔬 Methodology

The workflow consists of:

Original Time Series

↓

SSA (Small Window Length)

↓

Trend + Residual

↓

SSA (Large Window Length)

↓

Oscillatory Groups + Residual Components

↓

Final Reconstruction

---

## 📊 Main Results

The analysis identified:

- Weekly oscillation (≈7 days)
- Two independent oscillatory modes with period ≈3.5 days
- Two independent oscillatory modes with period ≈2.3 days
- Longer oscillations around 20, 15.3 and 11.7 days
- Accurate reconstruction of the original series (numerical error ≈10⁻¹²)

---

## 🖥 Application

The Streamlit application includes:

- Data overview
- Sequential SSA decomposition
- Scree plots
- Eigenvectors
- Pair plots
- W-correlation analysis
- Oscillatory mode reconstruction
- Final reconstruction

---

## 🛠 Technologies

- Python
- NumPy
- Pandas
- Plotly
- Streamlit
- PyTorch

---

## 📂 Project Structure

```text
Store-Sales-Sequential-SSA
│
├── app/
├── data/
├── figures/
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

---

## ⚙ Installation

```bash
git clone https://github.com/norquip/Store-Sales-Sequential-SSA.git

cd Store-Sales-Sequential-SSA

pip install -r requirements.txt

streamlit run app/app.py
```

---

## 📚 References

- Golyandina, N., Korobeynikov, A., & Zhigljavsky, A.
*Singular Spectrum Analysis with R.*

-  Jordan D'Arcy, [Kaggle](https://www.kaggle.com/code/jdarcy/introducing-ssa-for-time-series-decomposition),  Introducing SSA for Time Series Decomposition

- Golyandina, N., Zhigljavsky,[ResearchGate](https://www.researchgate.net/publication/260124592_Singular_Spectrum_Analysis_for_Time_Series) A.*Singular Spectrum Analysis for Time Series*



## 👤 Author

**Norma Quiroz**

PhD in Theoretical Physics

University of Guadalajara
