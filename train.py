import os
import json
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
# pyrefly: ignore [missing-import]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

DATA_CSV = "data/ipl_data.csv"
MODELS_DIR = "models"

def load_and_engineer_features():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Parsed data not found at {DATA_CSV}. Run parse_cricsheet.py first.")
        
    df = pd.read_csv(DATA_CSV)
    
    completed_overs = df['over'].astype(int)
    balls_in_over = ((df['over'] - completed_overs) * 10).round().astype(int)
    balls_bowled = completed_overs * 6 + balls_in_over
    
    df['balls_left'] = (120 - balls_bowled).clip(lower=0)
    df['runs_left'] = (df['target'] - df['runs_scored_so_far']).clip(lower=0)
    df['wickets_in_hand'] = 10 - df['wickets_fallen']
    
    df['current_run_rate'] = np.where(
        balls_bowled > 0,
        (df['runs_scored_so_far'] * 6) / balls_bowled,
        0.0
    )
    
    df['required_run_rate'] = np.where(
        df['balls_left'] > 0,
        (df['runs_left'] * 6) / df['balls_left'],
        0.0
    )
    df['required_run_rate'] = np.where(df['runs_left'] <= 0, 0.0, df['required_run_rate'])
    df['crr_vs_rrr'] = df['current_run_rate'] - df['required_run_rate']
    
    return df

def train_and_evaluate():
    df = load_and_engineer_features()
    
    cat_cols = ['batting_team', 'bowling_team', 'city']
    num_cols = ['balls_left', 'runs_left', 'wickets_in_hand', 'current_run_rate', 'required_run_rate', 'crr_vs_rrr']
    
    df_features = df[cat_cols + num_cols]
    df_encoded = pd.get_dummies(df_features, columns=cat_cols, dtype=int)
    
    feature_cols = list(df_encoded.columns)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "model_cols.pkl"))
    
    # Split temporally (train on 2008-2022, test on 2023-2024)
    train_mask = df['year'] <= 2022
    test_mask = df['year'] >= 2023
    
    X_train = df_encoded[train_mask]
    y_train = df.loc[train_mask, 'result']
    X_test = df_encoded[test_mask]
    y_test = df.loc[test_mask, 'result']
    
    print(f"Train set size: {len(X_train)} deliveries")
    print(f"Test set size: {len(X_test)} deliveries")
    
    lr_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
    lr_model.fit(X_train, y_train)
    
    xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    
    lr_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_lr)),
        'precision': float(precision_score(y_test, y_pred_lr)),
        'recall': float(recall_score(y_test, y_pred_lr)),
        'f1': float(f1_score(y_test, y_pred_lr)),
        'auc_roc': float(roc_auc_score(y_test, y_prob_lr))
    }
    
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_xgb)),
        'precision': float(precision_score(y_test, y_pred_xgb)),
        'recall': float(recall_score(y_test, y_pred_xgb)),
        'f1': float(f1_score(y_test, y_pred_xgb)),
        'auc_roc': float(roc_auc_score(y_test, y_prob_xgb))
    }
    
    print("\nLogistic Regression:")
    for k, v in lr_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\nXGBoost:")
    for k, v in xgb_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    
    # Downsample coordinates slightly for smaller JSON size
    step_lr = max(1, len(fpr_lr) // 200)
    step_xgb = max(1, len(fpr_xgb) // 200)
    
    metrics_json = {
        'lr': lr_metrics,
        'xgb': xgb_metrics,
        'roc_lr': {
            'fpr': fpr_lr[::step_lr].tolist(),
            'tpr': tpr_lr[::step_lr].tolist()
        },
        'roc_xgb': {
            'fpr': fpr_xgb[::step_xgb].tolist(),
            'tpr': tpr_xgb[::step_xgb].tolist()
        }
    }
    
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=4)
        
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "lr_model.pkl"))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "ipl_model.pkl"))
    print("\nModels and metrics successfully serialized to models/ directory.")

if __name__ == "__main__":
    train_and_evaluate()
