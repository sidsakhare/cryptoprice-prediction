# 📈 Crypto Volatility Prediction Using Machine Learning

A machine learning project that predicts the **7-day rolling volatility of cryptocurrencies** using historical daily market data, statistical features, and technical indicators.

The project covers the complete machine learning workflow, including exploratory data analysis, feature engineering, model development, evaluation, and model serialization.

## 🎯 Project Objective

Cryptocurrency markets are highly volatile. Understanding and estimating price volatility can help analyze market behavior and identify periods of increased price fluctuations.

The objective of this project is to develop a regression model that predicts cryptocurrency volatility using historical market information.

**Problem Type:** Regression

**Target Variable:** `volatility`

**Target Definition:** 7-day rolling standard deviation of daily log returns.

## 📊 Dataset Overview

- **Total Observations:** 58,679
- **Cryptocurrencies:** 36
- **Data Frequency:** Daily
- **Input Features:** 57
- **Train-Test Split:** Chronological 80:20

The dataset contains historical cryptocurrency market information, including:

- Open, High, Low, Close (OHLC)
- Trading Volume
- Market Capitalization
- Historical timestamps
- Cryptocurrency names

Only cryptocurrencies with more than 750 historical observations were retained.

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)

Notebook: `EDA.ipynb`

Performed exploratory analysis to understand cryptocurrency price movements, distributions, relationships, and historical trends.

**Key Activities:**

- Removed unnecessary index and duplicate timestamp columns.
- Parsed timestamps into datetime format.
- Filtered cryptocurrencies with sufficient historical observations.
- Removed zero-value records.
- Excluded selected stablecoin and wrapped-asset duplicates.
- Performed univariate and bivariate analysis.
- Conducted time-series analysis for major cryptocurrencies.
- Created correlation heatmaps.
- Investigated outliers and feature relationships.
- Generated automated profiling reports using `ydata-profiling`.

### 2. Feature Engineering

Notebook: `Feature_engeenering.ipynb`

Created additional features to capture market trends, momentum, historical price movements, and volatility patterns.

**Engineered Features**

| Feature Group | Features |
|---|---|
| Calendar | year, month, day, day_of_week |
| Returns | log returns |
| Price & Liquidity | price_range, liquidity_ratio |
| Trend | MA_7, MA_30, EMA_12, EMA_26, MACD |
| Momentum | RSI (14) |
| Bollinger Bands | Upper, Lower, Width |
| Lag Features | close_lag1, volatility_lag1, volatility_lag2 |
| Categorical | One-hot encoded crypto_name |

**Additional Processing:**

- Removed highly correlated OHLC columns to reduce redundancy.
- Retained engineered price and return information.
- Conducted Variance Inflation Factor (VIF) analysis.
- Applied one-hot encoding to cryptocurrency names.

**Final Feature Matrix:** 57 features.

### 3. Model Development & Selection

Notebook: `Model_selection.ipynb`

Developed and compared three tree-based regression algorithms.

**Algorithms Used:**

1. CatBoost Regressor
2. XGBoost Regressor
3. Random Forest Regressor

**Evaluation Metrics:**

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### 📊 Model Performance

The models were evaluated using a chronological 80:20 train-test split.

| Model | MAE | RMSE | Train R² | Test R² |
|---|---:|---:|---:|---:|
| **CatBoost** | **0.2799** | **0.3989** | 0.9494 | **0.9330** |
| XGBoost | 0.2841 | 0.4007 | 0.9566 | 0.9324 |
| Random Forest | 0.3099 | 0.4303 | 0.9905 | 0.9221 |

**Final Model: CatBoost Regressor**

CatBoost achieved the lowest MAE and RMSE and the highest test R² among the evaluated models.

XGBoost demonstrated comparable performance, while Random Forest showed a larger train-test performance gap, indicating greater overfitting.

### ⚙️ Final CatBoost Configuration

```python
CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    early_stopping_rounds=50,
    random_seed=42
)
```

The selected model was serialized and saved as `model.pkl`.

## 🗂️ Project Structure

```text
cryptoprice-prediction/
│
├── notebook/
│   ├── EDA.ipynb
│   ├── Feature_engeenering.ipynb
│   ├── Model_selection.ipynb
│   │
│   └── data/
│       ├── raw/
│       │   └── Crypto_dataset.csv
│       │
│       ├── eda/
│       │   └── crypto.csv
│       │
│       ├── train/
│       │   ├── X_train.csv
│       │   └── y_train.csv
│       │
│       ├── test/
│       │   ├── X_test.csv
│       │   └── y_test.csv
│       │
│       └── results/
│           └── results.csv
│
├── model.pkl
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/cryptoprice-prediction.git

cd cryptoprice-prediction
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv

.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks

Execute the notebooks in the following order:

1. `EDA.ipynb`
2. `Feature_engeenering.ipynb`
3. `Model_selection.ipynb`

**Note:** The notebooks currently contain absolute local file paths. Update the paths according to your local directory structure before execution.

## 💻 Using the Trained Model

The trained CatBoost model is available as `model.pkl`.

You can load the serialized model and generate predictions using the following code:

```python
import pickle
import pandas as pd

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load test data
X_new = pd.read_csv("notebook/data/test/X_test.csv")

# Generate predictions
predictions = model.predict(X_new)

print(predictions)
```

**Important:** The input dataset must contain the same 57 features, with matching column names and order, used during model training.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Statistical Analysis | Statsmodels |
| Machine Learning | Scikit-learn |
| Gradient Boosting | CatBoost, XGBoost |
| Data Profiling | ydata-profiling |
| Model Serialization | Pickle |
| Development | Jupyter Notebook, VS Code |


## 🔮 Future Improvements

- Perform all rolling and lag calculations independently for each cryptocurrency.
- Introduce walk-forward cross-validation.
- Optimize hyperparameters using Optuna.
- Explore additional technical indicators.
- Integrate cryptocurrency sentiment and news-based features.
- Deploy the trained model using Flask, FastAPI, or Streamlit.
- Develop an interactive cryptocurrency volatility prediction dashboard.

## 👨‍💻 Author

**Siddheshwar Sakhare**

Aspiring Healthcare Data Scientist | Data Analytics | Machine Learning

GitHub: https://github.com/sidsakhare

LinkedIn: https://www.linkedin.com/in/siddheshwar-sakhare-1a0499201/

---

⭐ If you find this project interesting, consider giving the repository a star!
