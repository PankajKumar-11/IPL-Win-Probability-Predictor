import os
import json
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import shap

st.set_page_config(
    page_title="IPL Win Probability Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
    <style>
        .main, .stApp { background-color: #0e1117; color: #ffffff; }
        .css-1d391kg { background-color: #1a1c23; }
        .metric-card {
            background-color: #1f2937;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #374151;
            text-align: center;
        }
        .metric-val { font-size: 36px; font-weight: bold; color: #10b981; }
        .metric-label { font-size: 14px; color: #9ca3af; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

MODELS_DIR = "models"
DATA_CSV = "data/ipl_data.csv"
MODEL_PATH = os.path.join(MODELS_DIR, "ipl_model.pkl")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "lr_model.pkl")
COLS_PATH = os.path.join(MODELS_DIR, "model_cols.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

@st.cache_resource
def load_models_and_metadata():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH)):
        return None, None, None
    return joblib.load(MODEL_PATH), joblib.load(LR_MODEL_PATH), joblib.load(COLS_PATH)

@st.cache_data
def load_historical_data():
    return pd.read_csv(DATA_CSV) if os.path.exists(DATA_CSV) else None

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

def generate_input_df(batting_team, bowling_team, city, target, current_score, over_float, wickets_lost, model_cols):
    completed_overs = int(over_float)
    balls_in_over = int(round((over_float - completed_overs) * 10))
    balls_bowled = completed_overs * 6 + balls_in_over
    
    balls_left = max(0, 120 - balls_bowled)
    runs_left = max(0, target - current_score)
    wickets_in_hand = 10 - wickets_lost
    
    current_run_rate = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0.0
    required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0.0
    if runs_left <= 0:
        required_run_rate = 0.0
        
    crr_vs_rrr = current_run_rate - required_run_rate
    
    input_dict = {
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'city': city,
        'balls_left': balls_left,
        'runs_left': runs_left,
        'wickets_in_hand': wickets_in_hand,
        'current_run_rate': current_run_rate,
        'required_run_rate': required_run_rate,
        'crr_vs_rrr': crr_vs_rrr
    }
    
    input_df = pd.DataFrame([input_dict])
    input_df_encoded = pd.get_dummies(input_df, columns=['batting_team', 'bowling_team', 'city'], dtype=int)
    input_df_encoded = input_df_encoded.reindex(columns=model_cols, fill_value=0)
    
    return input_df_encoded, input_dict

def render_predict_page(teams_list, cities_list, xgb_model, lr_model, model_cols):
    st.title("🏏 Live Win Probability Predictor")
    st.markdown("Specify the current match state of the run chase to calculate win probabilities.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Match Settings")
        bat_team_idx = teams_list.index(st.session_state['batting_team']) if st.session_state['batting_team'] in teams_list else 0
        batting_team = st.selectbox("Batting Team (Chasing)", teams_list, index=bat_team_idx)
        
        bowling_choices = [t for t in teams_list if t != batting_team]
        bowl_team_idx = bowling_choices.index(st.session_state['bowling_team']) if st.session_state['bowling_team'] in bowling_choices else 0
        bowling_team = st.selectbox("Bowling Team (Defending)", bowling_choices, index=bowl_team_idx)
        
        city_idx = cities_list.index(st.session_state['city']) if st.session_state['city'] in cities_list else 0
        city = st.selectbox("City Venue", cities_list, index=city_idx)
        target = st.number_input("Target Score", min_value=1, max_value=350, value=st.session_state['target'], step=1)
        
    with col2:
        st.subheader("Chase Situation")
        current_score = st.number_input(
            "Current Score", 
            min_value=0, 
            max_value=int(target - 1) if target > 1 else 350, 
            value=min(int(st.session_state['current_score']), int(target - 1)) if target > 1 else 0,
            step=1
        )
        completed_overs = st.slider("Completed Overs", 0, 19, int(st.session_state['overs']))
        balls_in_over = st.slider("Balls Bowled in Over", 0, 5, int(round((st.session_state['overs'] - int(st.session_state['overs'])) * 10)) % 6)
        over_float = completed_overs + balls_in_over / 10
        wickets_lost = st.slider("Wickets Lost", 0, 9, int(st.session_state['wickets_lost']))

    st.session_state['batting_team'] = batting_team
    st.session_state['bowling_team'] = bowling_team
    st.session_state['city'] = city
    st.session_state['target'] = target
    st.session_state['current_score'] = current_score
    st.session_state['overs'] = over_float
    st.session_state['wickets_lost'] = wickets_lost

    st.markdown("---")
    
    input_df_encoded, input_dict = generate_input_df(
        batting_team, bowling_team, city, target, current_score, over_float, wickets_lost, model_cols
    )
    
    prob_xgb = xgb_model.predict_proba(input_df_encoded)[0][1]
    prob_lr = lr_model.predict_proba(input_df_encoded)[0][1]
    
    st.subheader("Win Probability Results")
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        gauge_color = "#2ecc71"
        if prob_xgb < 0.40:
            gauge_color = "#ff4d4d"
        elif prob_xgb <= 0.60:
            gauge_color = "#ffb366"
            
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = round(prob_xgb * 100, 1),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{batting_team} Win Probability", 'font': {'size': 20, 'color': "white"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': gauge_color},
                'bgcolor': "#1f2937",
                'borderwidth': 2,
                'bordercolor': "#374151",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 77, 77, 0.15)'},
                    {'range': [40, 60], 'color': 'rgba(255, 179, 102, 0.15)'},
                    {'range': [60, 100], 'color': 'rgba(46, 204, 113, 0.15)'}
                ],
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Arial"}, height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with res_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">XGBoost (Final Model)</div>
            <div class="metric-val" style="color: {gauge_color};">{prob_xgb * 100:.1f}%</div>
            <div class="metric-label" style="margin-top: 15px;">Logistic Regression (Baseline)</div>
            <div class="metric-val" style="font-size: 24px; color: #9ca3af;">{prob_lr * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Match Equation")
    eq_col1, eq_col2, eq_col3, eq_col4 = st.columns(4)
    with eq_col1:
        st.metric("Runs Needed", f"{input_dict['runs_left']} runs")
    with eq_col2:
        st.metric("Balls Remaining", f"{input_dict['balls_left']} balls")
    with eq_col3:
        st.metric("Wickets in Hand", f"{input_dict['wickets_in_hand']} wickets")
    with eq_col4:
        st.metric("Required Run Rate", f"{input_dict['required_run_rate']:.2f}")

def render_model_comparison(metrics_data):
    st.title("📊 Model Performance Metrics")
    st.markdown("Performance comparison between the baseline Logistic Regression and the final XGBoost Classifier.")

    if metrics_data is None:
        st.warning("Metrics file `models/metrics.json` not found. Run train.py first.")
        return

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        "Logistic Regression (Baseline)": [
            f"{metrics_data['lr']['accuracy']*100:.2f}%",
            f"{metrics_data['lr']['precision']*100:.2f}%",
            f"{metrics_data['lr']['recall']*100:.2f}%",
            f"{metrics_data['lr']['f1']*100:.2f}%",
            f"{metrics_data['lr']['auc_roc']:.4f}"
        ],
        "XGBoost Classifier (Final)": [
            f"{metrics_data['xgb']['accuracy']*100:.2f}%",
            f"{metrics_data['xgb']['precision']*100:.2f}%",
            f"{metrics_data['xgb']['recall']*100:.2f}%",
            f"{metrics_data['xgb']['f1']*100:.2f}%",
            f"{metrics_data['xgb']['auc_roc']:.4f}"
        ]
    })
    st.table(metrics_df.set_index("Metric"))
    
    st.markdown("""
    > **Note**: XGBoost is superior at capturing non-linear interactions. In cricket, the value of wickets-in-hand is highly dependent on balls remaining, which a linear model struggles to model directly.
    """)

    st.subheader("ROC Curves Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics_data['roc_lr']['fpr'], y=metrics_data['roc_lr']['tpr'],
        mode='lines', name=f"Logistic Regression (AUC = {metrics_data['lr']['auc_roc']:.3f})",
        line=dict(color='#ff7f0e', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=metrics_data['roc_xgb']['fpr'], y=metrics_data['roc_xgb']['tpr'],
        mode='lines', name=f"XGBoost Classifier (AUC = {metrics_data['xgb']['auc_roc']:.3f})",
        line=dict(color='#10b981', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name="Random Baseline", line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a1c23', font={'color': "white"},
        legend=dict(x=0.5, y=0.15, bgcolor='rgba(26, 28, 35, 0.8)'), height=500,
        xaxis=dict(gridcolor='#374151', range=[0, 1]), yaxis=dict(gridcolor='#374151', range=[0, 1])
    )
    st.plotly_chart(fig, use_container_width=True)

def render_shap_explanation(xgb_model, model_cols):
    st.title("🔍 Predictor Explainer (SHAP)")
    st.markdown("Visualizing the impact of features on the live win probability calculation.")

    input_df_encoded, input_dict = generate_input_df(
        st.session_state['batting_team'], st.session_state['bowling_team'],
        st.session_state['city'], st.session_state['target'],
        st.session_state['current_score'], st.session_state['overs'],
        st.session_state['wickets_lost'], model_cols
    )
    
    with st.spinner("Calculating SHAP values..."):
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(input_df_encoded)
        
    shap_vals = shap_values.values[0]
    
    contrib_df = pd.DataFrame({
        'Feature_OHE': input_df_encoded.columns,
        'Value': input_df_encoded.iloc[0].values,
        'SHAP_Value': shap_vals
    })
    
    def clean_feature_name(col):
        if col.startswith('batting_team_'):
            return f"Batting Team: {col.replace('batting_team_', '')}"
        elif col.startswith('bowling_team_'):
            return f"Bowling Team: {col.replace('bowling_team_', '')}"
        elif col.startswith('city_'):
            return f"City: {col.replace('city_', '')}"
        return col.replace('_', ' ').title()
            
    contrib_df['Feature'] = contrib_df['Feature_OHE'].apply(clean_feature_name)
    active_mask = (contrib_df['Value'] != 0) | (contrib_df['Feature_OHE'].isin(['balls_left', 'runs_left', 'wickets_in_hand', 'current_run_rate', 'required_run_rate', 'crr_vs_rrr']))
    filtered_contribs = contrib_df[active_mask].copy()
    
    filtered_contribs['Abs_SHAP'] = filtered_contribs['SHAP_Value'].abs()
    filtered_contribs = filtered_contribs.sort_values(by='Abs_SHAP', ascending=True)
    
    colors = ['#2ecc71' if x >= 0 else '#ff4d4d' for x in filtered_contribs['SHAP_Value']]
    hover_texts = []
    for _, row in filtered_contribs.iterrows():
        val = row['Value']
        val_str = f"{val:.1f}" if isinstance(val, (float, np.floating)) else str(val)
        hover_texts.append(f"<b>{row['Feature']}</b><br>Value: {val_str}<br>SHAP: {row['SHAP_Value']:.4f}")
        
    fig = go.Figure(go.Bar(
        x=filtered_contribs['SHAP_Value'], y=filtered_contribs['Feature'],
        orientation='h', marker_color=colors, hovertext=hover_texts, hoverinfo='text'
    ))
    fig.update_layout(
        title={'text': "Feature Contribution to Log-Odds of Winning", 'font': {'size': 18, 'color': 'white'}},
        xaxis_title="SHAP Value (Log-Odds Impact)", yaxis_title="Feature",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a1c23', font={'color': "white"},
        height=450, xaxis=dict(gridcolor='#374151')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Drivers")
    top_pos = contrib_df.sort_values(by='SHAP_Value', ascending=False).iloc[0]
    top_neg = contrib_df.sort_values(by='SHAP_Value', ascending=True).iloc[0]
    
    col_p, col_n = st.columns(2)
    with col_p:
        st.markdown("#### 👍 Driving win probability up")
        p_name = clean_feature_name(top_pos['Feature_OHE'])
        if top_pos['SHAP_Value'] > 0:
            st.markdown(f"**{p_name}** has the strongest positive impact on the chase.")
        else:
            st.markdown("No significant positive drivers.")
            
    with col_n:
        st.markdown("#### 👎 Dragging win probability down")
        n_name = clean_feature_name(top_neg['Feature_OHE'])
        if top_neg['SHAP_Value'] < 0:
            st.markdown(f"**{n_name}** has the strongest negative impact.")
        else:
            st.markdown("No significant negative drivers.")

def render_eda_insights(df_data, model_cols, xgb_model):
    st.title("📈 Match Analysis & Tracker")
    tab1, tab2 = st.tabs(["Ball-by-Ball Match Tracker", "Historical Metrics"])
    
    with tab1:
        st.subheader("Trace a Historical Run Chase")
        
        match_summary = df_data.groupby('match_id').agg({
            'year': 'first', 'batting_team': 'first', 'bowling_team': 'first'
        }).reset_index()
        match_summary['display_name'] = match_summary.apply(
            lambda r: f"{r['year']} | {r['batting_team']} vs {r['bowling_team']} (ID: {r['match_id']})", axis=1
        )
        
        selected_display = st.selectbox("Select Match", match_summary['display_name'].tolist())
        selected_match_id = match_summary.loc[match_summary['display_name'] == selected_display, 'match_id'].values[0]
        
        match_balls = df_data[df_data['match_id'] == selected_match_id].sort_values(by='over').copy()
        
        completed_overs = match_balls['over'].astype(int)
        balls_in_over = ((match_balls['over'] - completed_overs) * 10).round().astype(int)
        balls_bowled = completed_overs * 6 + balls_in_over
        
        match_balls['balls_left'] = (120 - balls_bowled).clip(lower=0)
        match_balls['runs_left'] = (match_balls['target'] - match_balls['runs_scored_so_far']).clip(lower=0)
        match_balls['wickets_in_hand'] = 10 - match_balls['wickets_fallen']
        match_balls['current_run_rate'] = np.where(balls_bowled > 0, (match_balls['runs_scored_so_far'] * 6) / balls_bowled, 0.0)
        match_balls['required_run_rate'] = np.where(match_balls['balls_left'] > 0, (match_balls['runs_left'] * 6) / match_balls['balls_left'], 0.0)
        match_balls['required_run_rate'] = np.where(match_balls['runs_left'] <= 0, 0.0, match_balls['required_run_rate'])
        match_balls['crr_vs_rrr'] = match_balls['current_run_rate'] - match_balls['required_run_rate']
        
        feat_df = match_balls[['batting_team', 'bowling_team', 'city', 'balls_left', 'runs_left', 'wickets_in_hand', 'current_run_rate', 'required_run_rate', 'crr_vs_rrr']]
        feat_df_encoded = pd.get_dummies(feat_df, columns=['batting_team', 'bowling_team', 'city'], dtype=int)
        feat_df_encoded = feat_df_encoded.reindex(columns=model_cols, fill_value=0)
        
        probs = xgb_model.predict_proba(feat_df_encoded)[:, 1]
        match_balls['win_prob'] = probs * 100
        match_balls['wicket_fell'] = match_balls['wickets_fallen'].diff().fillna(0).astype(int)
        wickets_df = match_balls[match_balls['wicket_fell'] > 0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=match_balls['over'], y=match_balls['win_prob'], mode='lines', name='Win Probability (%)',
            line=dict(color='#10b981', width=3),
            hoverinfo='text',
            hovertext=[
                f"Over: {over:.1f}<br>Score: {score}<br>Wickets: {w}<br>Prob: {p:.1f}%"
                for over, score, w, p in zip(match_balls['over'], match_balls['runs_scored_so_far'], match_balls['wickets_fallen'], match_balls['win_prob'])
            ]
        ))
        
        if len(wickets_df) > 0:
            fig.add_trace(go.Scatter(
                x=wickets_df['over'], y=wickets_df['win_prob'], mode='markers', name='Wicket Fallen',
                marker=dict(color='#ff4d4d', size=10, symbol='x'),
                hoverinfo='text', hovertext=[f"Wicket Fallen!<br>Over: {over:.1f}<br>Wickets lost: {w}" for over, w in zip(wickets_df['over'], wickets_df['wickets_fallen'])]
            ))
            
        fig.update_layout(
            title={'text': f"Chase Win Probability Trend ({match_balls.iloc[0]['batting_team']} vs {match_balls.iloc[0]['bowling_team']})", 'font': {'color': 'white'}},
            xaxis_title="Overs", yaxis_title="Win Probability (%)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a1c23', font={'color': "white"},
            height=400, xaxis=dict(gridcolor='#374151', tickmode='linear', dtick=2), yaxis=dict(gridcolor='#374151', range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        bat_team = match_balls.iloc[0]['batting_team']
        st.markdown(f"""
        **Outcome**: **{"Won" if match_balls.iloc[-1]['result'] == 1 else "Lost"}** by {bat_team} (Final Score: {match_balls.iloc[-1]['runs_scored_so_far']}/{match_balls.iloc[-1]['wickets_fallen']} chasing {match_balls.iloc[0]['target']}).
        """)

    with tab2:
        st.subheader("Historical Analytics")
        
        match_cities = df_data.drop_duplicates(subset=['match_id']).copy()
        match_city_stats = match_cities.groupby('city').agg(
            total_matches=('result', 'count'), chase_wins=('result', 'sum')
        ).reset_index()
        match_city_stats['chase_win_rate'] = (match_city_stats['chase_wins'] / match_city_stats['total_matches']) * 100
        top_cities = match_city_stats.sort_values(by='total_matches', ascending=False).head(15).copy()
        
        fig_city = px.bar(
            top_cities, x='chase_win_rate', y='city', orientation='h',
            title='Chasing Win Rate by City (Top 15)',
            labels={'chase_win_rate': 'Win Rate (%)', 'city': 'City'},
            color='chase_win_rate', color_continuous_scale=px.colors.sequential.Viridis
        )
        fig_city.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1a1c23', font={'color': "white"},
            height=400, xaxis=dict(range=[0, 100], gridcolor='#374151'), yaxis=dict(categoryorder='total ascending')
        )
        st.plotly_chart(fig_city, use_container_width=True)
        
        st.markdown("#### Head-to-Head Statistics")
        teams_list = sorted(list(df_data['batting_team'].unique()))
        h2h_bat = st.selectbox("Batting Team", teams_list, key="h2h_bat")
        h2h_bowl = st.selectbox("Bowling Team", [t for t in teams_list if t != h2h_bat], key="h2h_bowl")
        
        h2h_matches = match_cities[(match_cities['batting_team'] == h2h_bat) & (match_cities['bowling_team'] == h2h_bowl)]
        total_plays = len(h2h_matches)
        
        if total_plays == 0:
            st.info(f"No match history for {h2h_bat} chasing against {h2h_bowl}.")
        else:
            bat_wins = h2h_matches['result'].sum()
            bowl_wins = total_plays - bat_wins
            
            st.markdown(f"**Total matches**: {total_plays} with {h2h_bat} chasing.")
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f"{h2h_bat} Wins", f"{h2h_bowl} Wins"],
                values=[bat_wins, bowl_wins], hole=.3, marker_colors=['#10b981', '#ef4444']
            )])
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

def main():
    load_css()
    
    xgb_model, lr_model, model_cols = load_models_and_metadata()
    df_data = load_historical_data()
    metrics_data = load_metrics()
    
    if xgb_model is None or df_data is None:
        st.title("🏏 IPL Win Probability Predictor")
        st.warning("⚠️ **Model files or data not found!** Run parse_cricsheet.py and train.py first.")
        st.stop()
        
    st.sidebar.title("🏏 IPL Win Predictor")
    st.sidebar.markdown("Predict live match probabilities using Machine Learning.")
    page = st.sidebar.radio("Navigate", ["Predict Probability", "Model Comparison", "Why This Prediction (SHAP)", "EDA Insights"])
    
    teams_list = sorted(list(set(df_data['batting_team'].unique()) | set(df_data['bowling_team'].unique())))
    cities_list = sorted(list(df_data['city'].dropna().unique()))
    
    if 'batting_team' not in st.session_state:
        st.session_state['batting_team'] = 'Mumbai Indians'
    if 'bowling_team' not in st.session_state:
        st.session_state['bowling_team'] = 'Chennai Super Kings'
    if 'city' not in st.session_state:
        st.session_state['city'] = 'Mumbai'
    if 'target' not in st.session_state:
        st.session_state['target'] = 180
    if 'current_score' not in st.session_state:
        st.session_state['current_score'] = 120
    if 'overs' not in st.session_state:
        st.session_state['overs'] = 14.0
    if 'wickets_lost' not in st.session_state:
        st.session_state['wickets_lost'] = 3
        
    if page == "Predict Probability":
        render_predict_page(teams_list, cities_list, xgb_model, lr_model, model_cols)
    elif page == "Model Comparison":
        render_model_comparison(metrics_data)
    elif page == "Why This Prediction (SHAP)":
        render_shap_explanation(xgb_model, model_cols)
    elif page == "EDA Insights":
        render_eda_insights(df_data, model_cols, xgb_model)

if __name__ == "__main__":
    main()
