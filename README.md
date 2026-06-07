# IPL Win Probability Predictor

A professional machine learning portfolio project that predicts the win probability of the chasing team in an IPL match at any delivery during the second-innings run chase. 

The application is built entirely in Python, utilizing an XGBoost Classifier for prediction, SHAP for feature explainability, and Streamlit for a premium, interactive web interface.

---

## 🚀 Key Features

*   **Match State Predictor**: Select batting team, bowling team, venue, target score, current score, overs completed, and wickets lost. The application instantly predicts and visualizes the win probability via a dynamic, color-coded gauge (Green $> 60\%$, Amber $40\%-60\%$, Red $< 40\%$).
*   **Model Comparison Dashboard**: Compare the performance of the baseline Logistic Regression model and the final XGBoost Classifier. View accuracy, precision, recall, F1-score, and AUC-ROC side-by-side, along with interactive ROC curves.
*   **SHAP Explainability Page**: Real-time explainability using SHAP values. A custom, interactive horizontal bar chart visualizes which features (e.g., runs left, wickets in hand, pressure index) had the greatest impact on driving the current prediction up or down.
*   **Interactive EDA & Ball-by-Ball Match Tracker**: 
    *   **Venue Win Rates**: View historical chasing win rates across different stadiums.
    *   **Head-to-Head Stats**: View a historical head-to-head pie chart between any two teams.
    *   **Match Tracker**: Select any historical match from the database and track how the chasing team's win probability fluctuated ball-by-ball. Falling wickets are marked dynamically on the line graph.

---

## 🛠️ Technology Stack

*   **Core Language**: Python 3.13.2
*   **Data Tools**: Pandas, Numpy, Built-in JSON
*   **Machine Learning**: Scikit-Learn (Logistic Regression), XGBoost (XGBClassifier), SHAP (TreeExplainer), Joblib
*   **Visualizations**: Plotly (Interactive charts/gauges), Matplotlib, Seaborn
*   **App Framework**: Streamlit (Premium dark-themed dashboard)
*   **Environment**: Jupyter Notebook (Interactive report)

---

## 📂 Project Structure

```text
ipl-win-predictor/
├── data/
│   ├── raw/                  # Extracted Cricsheet JSON files (gitignored)
│   ├── plots/                # Pre-computed EDA and SHAP PNG plots
│   └── ipl_data.csv          # Cleaned, parsed ball-by-ball dataset (committed)
├── models/
│   ├── ipl_model.pkl         # Serialized XGBoost model
│   ├── lr_model.pkl          # Serialized Logistic Regression model
│   ├── model_cols.pkl        # List of one-hot encoded feature column names
│   └── metrics.json          # Pre-computed test performance metrics
├── notebooks/
│   └── analysis.ipynb        # Pre-compiled EDA and Model Training Notebook
├── app.py                    # Multi-page Streamlit web application
├── parse_cricsheet.py        # Automated download & data parsing script
├── train.py                  # Feature engineering & ML training script
├── create_notebook.py        # Helper to generate Jupyter Notebook programmatically
├── requirements.txt          # Python package dependency list
└── README.md                 # Project documentation
```

---

## 💻 Installation & Setup

Follow these steps to run the project locally on your machine:

### 1. Clone the repository and navigate to the project directory
```bash
git clone <your-repo-url>
cd ipl-win-predictor
```

### 2. Create and activate a virtual environment
**On Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Fetch and clean the Cricsheet data
Run the parsing script. This script will automatically download the Cricsheet IPL matches dataset zip file (~15MB), extract the match JSONs, parse them into a ball-by-ball tabular format, and save the result as `data/ipl_data.csv`:
```bash
python parse_cricsheet.py
```

### 5. Train the models
Train the machine learning classifiers and save the metrics and serialize the pipeline:
```bash
python train.py
```

### 6. Run the Streamlit web application
Launch the Streamlit server locally:
```bash
streamlit run app.py
```
Open the local URL (standardly `http://localhost:8501`) in your browser to interact with the dashboard!

---

## 🔬 Exploratory Data Analysis & Notebook

The file `notebooks/analysis.ipynb` contains a complete report of the dataset's features, visual explorations, model training logs, classification reports, and SHAP explanations.

You can run and explore the notebook interactively by launching Jupyter:
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 📊 Evaluation Summary

The models are evaluated on a temporal test split (chases from years **2023-2024**), trained on chases from years **2008-2022**:

| Metric | Logistic Regression (Baseline) | XGBoost Classifier (Final) |
| :--- | :---: | :---: |
| **Accuracy** | **75.04%** | **74.82%** |
| **Precision** | **83.97%** | **83.13%** |
| **Recall** | **62.66%** | **63.07%** |
| **F1-Score** | **71.76%** | **71.72%** |
| **AUC-ROC** | **0.8519** | **0.8390** |

XGBoost and Logistic Regression perform extremely well on this task, achieving $\sim 75\%$ accuracy on the unseen, unpredictable chases of the 2023 and 2024 IPL seasons.
