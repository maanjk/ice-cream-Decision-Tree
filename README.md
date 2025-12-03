# Ice Cream Sales – Decision Tree Regression (Kaggle + Streamlit)

This project demonstrates an end‑to‑end **regression** pipeline:

1. Train and evaluate a **Decision Tree Regressor** on an Ice Cream Sales dataset using **Kaggle**.
2. Deploy the trained model as an interactive **Streamlit** web app hosted from **GitHub**.

---

## Links

- **Kaggle training notebook**  
  https://www.kaggle.com/code/musmanaslamawan/ice-cream-decisiontree  

- **Live Streamlit app**  
  https://ice-cream-decision-tree-cwp38sfppwwoagr4ge9tqs.streamlit.app/

---

## Dataset

- **Name:** Ice Cream Sales Dataset  
- **Source (Kaggle):** https://www.kaggle.com/datasets/sakshisatre/ice-cream-sales-dataset  
- **File used:** `Ice Cream.csv`  
- **Problem type:** Regression  

**Features**

- `Temperature` – outside temperature (°C)

**Target**

- `Revenue` – ice‑cream sales revenue

The goal is to predict **Revenue** from **Temperature**.

---

## Model

- **Algorithm:** `DecisionTreeRegressor` (from `sklearn.tree`)
- **Example hyperparameters:**  
  - `max_depth` (e.g., 3)  
  - `random_state=42`

In the Kaggle notebook:

- The dataset is loaded and explored.
- Data is split into **train** and **test** sets (e.g., 80/20).
- A Decision Tree Regressor is trained on the training set.
- Performance is evaluated using:
  - RMSE (Root Mean Squared Error)
  - R² (coefficient of determination)
- The final trained model is exported to a **`.pkl`** file for deployment.

---

## Streamlit Web App

The Streamlit app provides a simple UI to use the trained Decision Tree model:

**Features**

- A slider to choose **Temperature (°C)**.
- On clicking **“Predict”**, the app uses the Decision Tree model to predict:
  - **Expected revenue** for that temperature.
- Optionally shows a plot of:
  - Actual data points (Temperature vs Revenue)
  - The Decision Tree prediction curve

**Live demo:**  
https://ice-cream-decision-tree-cwp38sfppwwoagr4ge9tqs.streamlit.app/

---

## Running the App Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ice-cream-decision-tree.git
cd ice-cream-decision-tree
