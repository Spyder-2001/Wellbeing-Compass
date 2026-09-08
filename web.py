import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time

st.set_page_config(
    page_title="Wellbeing Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_artifacts():
    base = Path("./deployment_artifacts")
    model_binary = joblib.load(base / "model_binary.pkl")
    model_severity = joblib.load(base / "model_severity.pkl")
    features = joblib.load(base / "features.pkl")
    severity_map = joblib.load(base / "severity_map.pkl")
    scaler_mean = joblib.load(base / "scaler_mean.pkl")
    scaler_scale = joblib.load(base / "scaler_scale.pkl")
    encoder_categories = joblib.load(base / "encoder_categories.pkl")
    feature_cols = joblib.load(base / "feature_cols.pkl")
    categorical_feature_cols = joblib.load(base / "categorical_feature_cols.pkl")
    severity_reverse = {v: k for k, v in severity_map.items()}
    return (model_binary, model_severity, features, severity_reverse,
            scaler_mean, scaler_scale, encoder_categories, feature_cols, categorical_feature_cols)

(model_binary, model_severity, features, severity_reverse,
 scaler_mean, scaler_scale, encoder_categories, feature_cols, categorical_feature_cols) = load_artifacts()

def preprocess_input(input_df):
    num_cols = feature_cols
    cat_cols = categorical_feature_cols
    num_data = input_df[num_cols].values
    cat_data = input_df[cat_cols].values
    num_scaled = (num_data - scaler_mean) / scaler_scale
    cat_encoded = np.hstack([
        np.array([[1 if val == cat else 0 for cat in encoder_categories[i]] for val in cat_data[:, i]])
        for i in range(len(cat_cols))
    ])
    return np.hstack([num_scaled, cat_encoded])

def generate_recommendations(user_data, binary_pred, severity_label):
    recs = []
    if binary_pred == 1:
        recs.append("🟡 Mental health risk identified. Please consult a licensed professional counselor or psychiatrist immediately.")
    else:
        recs.append("🟢 No immediate mental health risk detected based on current data. Continue maintaining a healthy routine.")
    if severity_label in ['Medium', 'High']:
        recs.append(f"⚠️ Severity level is *{severity_label}*. This indicates a notable condition requiring professional evaluation.")
    if user_data.get('Sleep_Hours', 10) < 6:
        recs.append("😴 Sleep duration is less than 6 hours. Aim for 7‑8 hours of quality sleep daily.")
    elif user_data.get('Sleep_Hours', 0) > 9:
        recs.append("😴 Sleep duration exceeds 9 hours. Oversleeping can be a sign of depression; consider tracking your mood.")
    if user_data.get('Work_Hours', 40) > 50:
        recs.append("💼 Working more than 50 hours per week. Ensure you take regular breaks and maintain work‑life balance.")
    if user_data.get('Physical_Activity_Hours', 0) < 3:
        recs.append("🚶 Physical activity is low. Incorporate at least 30 minutes of walking or exercise into your daily routine.")
    if user_data.get('Stress_Level', 'Low') in ['High', 'Medium']:
        recs.append("🧘 Stress level is high/medium. Practice mindfulness, meditation, or engage in hobbies to manage stress.")
    if user_data.get('Consultation_History', 'No') == 'Yes':
        recs.append("📋 You have a history of consultation. Regular follow‑ups are highly recommended.")
    if not recs:
        recs.append("✅ All your indicators look good. Keep up the healthy habits!")
    return recs

all_countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas",
    "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize",
    "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil",
    "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba",
    "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
    "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada",
    "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras",
    "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan",
    "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia",
    "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro",
    "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal",
    "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine",
    "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland",
    "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain",
    "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo",
    "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan",
    "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City",
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe", "Other"
]

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main-header {
        background: linear-gradient(-45deg, #2AA198, #1a7a6e, #0e4d44, #2AA198);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 2.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(42, 161, 152, 0.3);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.05) 0%, transparent 60%);
        pointer-events: none;
    }

    .main-header .icon {
        font-size: 4.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 10px 20px rgba(0,0,0,0.1));
    }

    .main-header h1 {
        color: white;
        font-size: 3.8rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 0.2rem 0 0.2rem 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    .main-header .tagline {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        font-weight: 300;
        letter-spacing: 2px;
        margin: 0;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin: 1.5rem 0 2.5rem 0;
        animation: fadeInUp 0.8s ease;
    }

    .feature-card {
        background: white;
        padding: 1.2rem 1rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
        cursor: default;
    }

    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 35px rgba(42, 161, 152, 0.12);
        border-color: #2AA19840;
    }

    .feature-card .feat-icon {
        font-size: 2.5rem;
        margin-bottom: 0.3rem;
    }

    .feature-card .feat-title {
        font-weight: 700;
        color: #1a3b3a;
        font-size: 1rem;
        margin: 0.2rem 0;
    }

    .feature-card .feat-desc {
        font-size: 0.85rem;
        color: #777;
        margin: 0;
        line-height: 1.4;
    }

    .form-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2rem 2rem 1.8rem 2rem;
        border-radius: 24px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.06);
        border: 1px solid rgba(255,255,255,0.8);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }

    .form-card:hover {
        box-shadow: 0 12px 50px rgba(0,0,0,0.08);
    }

    .form-card h3 {
        color: #1a3b3a;
        font-weight: 700;
        font-size: 1.4rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.8rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2AA198 0%, #1a7a6e 100%);
        color: white;
        font-weight: 700;
        padding: 0.8rem 2.5rem;
        border-radius: 50px;
        border: none;
        font-size: 1.15rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(42, 161, 152, 0.3);
        width: 100%;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 8px 35px rgba(42, 161, 152, 0.45);
        background: linear-gradient(135deg, #36b3a8 0%, #208b7d 100%);
    }

    .stButton > button:active {
        transform: scale(0.97);
    }

    .cta-placeholder {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, #f8faf9 0%, #eef5f3 100%);
        border-radius: 24px;
        border: 2px dashed #c0d6d2;
        margin: 1rem 0;
    }

    .cta-placeholder h3 {
        color: #2c5e56;
        font-weight: 600;
    }

    .cta-placeholder p {
        color: #777;
        font-size: 1.1rem;
    }

    .result-card {
        padding: 1.8rem 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        transition: all 0.3s ease;
        border-left: 6px solid #2AA198;
        background: linear-gradient(135deg, #f8fffe 0%, #f0f9f7 100%);
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    }

    .result-card.risk {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #fff5f5 0%, #fde8e8 100%);
    }

    .result-card.safe {
        border-left-color: #2AA198;
        background: linear-gradient(135deg, #f0faf8 0%, #e0f5f0 100%);
    }

    .metric-box {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 2px 15px rgba(0,0,0,0.04);
        border: 1px solid #f0f0f0;
    }

    .metric-box .label {
        font-size: 0.8rem;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-box .value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a3b3a;
        margin-top: 0.2rem;
    }

    .metric-box .value.risk {
        color: #e74c3c;
    }

    .metric-box .value.safe {
        color: #2AA198;
    }

    .recommendation-item {
        background: white;
        padding: 0.9rem 1.2rem;
        border-radius: 12px;
        margin: 0.6rem 0;
        border-left: 4px solid #7559D9;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }

    .recommendation-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(117, 89, 217, 0.08);
    }

    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 12px !important;
        border: 1px solid #e8e8e8 !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.2s ease;
        background: rgba(255,255,255,0.😎 !important;
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #2AA198 !important;
        box-shadow: 0 0 0 4px rgba(42, 161, 152, 0.12) !important;
    }

    .stSelectbox label, .stNumberInput label {
        font-weight: 600 !important;
        color: #2c3e50 !important;
        font-size: 0.9rem !important;
    }

    .css-1d391kg {
        background: linear-gradient(180deg, #f8faf9 0%, #eef5f3 100%);
    }

    .sidebar-brand {
        text-align: center;
        padding: 1rem 0;
    }

    .sidebar-brand h2 {
        color: #1a3b3a;
        font-weight: 800;
    }

    .sidebar-brand .tagline {
        color: #666;
        font-size: 0.9rem;
    }

    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.85rem;
        padding: 2rem 0 0.5rem 0;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }

    .footer span {
        color: #2AA198;
        font-weight: 600;
    }

    @media (max-width: 768px) {
        .main-header h1 { font-size: 2.2rem; }
        .main-header .tagline { font-size: 1rem; }
        .feature-grid { grid-template-columns: repeat(2, 1fr); gap: 0.8rem; }
        .form-card { padding: 1.2rem; }
        .main-header { padding: 1.5rem 1rem; }
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size: 3.5rem; line-height: 1;">🧭</div>
        <h2 style="margin: 0.2rem 0 0 0;">Wellbeing Compass</h2>
        <p class="tagline">AI-Powered Mental Wellbeing</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    ### 📊 How It Works
    1. Enter your details
    2. Our AI analyzes risk factors
    3. Get personalized recommendations
    """)
    st.markdown("---")
    st.markdown("""
    ### 🔒 Privacy First
    All data is processed locally and *never stored*.
    """)
    st.markdown("---")
    st.caption("Version 2.0 | Group 7")

st.markdown("""
<div class="main-header">
    <div class="icon">🧭</div>
    <h1>Wellbeing Compass</h1>
    <p class="tagline">Listen to Your Mind, Care for Your Wellbeing</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feat-icon">🧠</div>
        <div class="feat-title">AI-Powered Assessment</div>
        <div class="feat-desc">Advanced ML models analyze your lifestyle factors</div>
    </div>
    <div class="feature-card">
        <div class="feat-icon">📊</div>
        <div class="feat-title">Multi-Factor Analysis</div>
        <div class="feat-desc">Sleep, work, activity, stress & demographics</div>
    </div>
    <div class="feature-card">
        <div class="feat-icon">💡</div>
        <div class="feat-title">Personalized Insights</div>
        <div class="feat-desc">Tailored recommendations for your wellbeing</div>
    </div>
    <div class="feature-card">
        <div class="feat-icon">🔒</div>
        <div class="feat-title">Privacy First</div>
        <div class="feat-desc">Zero data stored. 100% confidential</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div class="form-card">
        <h3>📝 Enter Your Details</h3>
    </div>
    """, unsafe_allow_html=True)

    with st.form("user_input_form", clear_on_submit=False):
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            age = st.number_input("🎂 Age", min_value=10, max_value=100, value=30, step=1)
            gender = st.selectbox("👤 Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
            occupation = st.selectbox("💼 Occupation", ["IT", "Sales", "Education", "Healthcare", "Finance", "Engineering", "Other"])
            country = st.selectbox("🌍 Country", all_countries)

        with col2:
            sleep_hours = st.number_input("😴 Sleep Hours (per day)", min_value=0.0, max_value=15.0, value=7.0, step=0.5)
            work_hours = st.number_input("💻 Work Hours (per week)", min_value=0, max_value=100, value=40, step=1)
            physical_activity = st.number_input("🏃 Physical Activity (hrs/week)", min_value=0, max_value=20, value=3, step=1)
            stress_level = st.selectbox("🧘 Stress Level", ["Low", "Medium", "High"])
            consultation_history = st.selectbox("📋 Previous Consultation History", ["No", "Yes"])

        submitted = st.form_submit_button("🧠 Assess My Wellbeing", use_container_width=True)

if submitted:
    user_dict = {
        'Age': age, 'Sleep_Hours': sleep_hours, 'Work_Hours': work_hours,
        'Physical_Activity_Hours': physical_activity, 'Gender': gender,
        'Occupation': occupation, 'Country': country,
        'Consultation_History': consultation_history, 'Stress_Level': stress_level
    }

    sleep_deficit = max(0, 7 - sleep_hours)
    overtime_hours = max(0, work_hours - 40)
    work_sleep_ratio = work_hours / (sleep_hours + 0.01)
    low_activity = 1 if physical_activity < 3 else 0
    sleep_score = 0 if sleep_hours < 6 else (1 if sleep_hours <= 8 else 0.5)
    activity_score = 1 if physical_activity >= 5 else (0.5 if physical_activity >= 3 else 0)
    stress_score = {'Low': 0, 'Medium': 1, 'High': 2}[stress_level]
    lifestyle_risk_score = (sleep_deficit/7)*0.3 + (overtime_hours/40)*0.3 + (1 - activity_score)*0.2 + (stress_score/2)*0.2

    all_feature_dict = {
        'Age': age, 'Sleep_Hours': sleep_hours, 'Work_Hours': work_hours,
        'Physical_Activity_Hours': physical_activity, 'Sleep_Deficit': sleep_deficit,
        'Overtime_Hours': overtime_hours, 'Work_Sleep_Ratio': work_sleep_ratio,
        'Low_Activity': low_activity, 'Lifestyle_Risk_Score': lifestyle_risk_score,
        'Gender': gender, 'Occupation': occupation, 'Country': country,
        'Consultation_History': consultation_history, 'Stress_Level': stress_level
    }

    input_df = pd.DataFrame([all_feature_dict])[features]
    processed = preprocess_input(input_df)

    binary_pred = model_binary.predict(processed)[0]
    severity_pred = model_severity.predict(processed)[0]
    severity_label = severity_reverse[severity_pred]

    risk_score = 0
    if sleep_hours < 4:
        risk_score += 2
    if work_hours > 70:
        risk_score += 2
    if physical_activity < 2:
        risk_score += 1
    if stress_level == 'High':
        risk_score += 2

    if risk_score >= 4:
        binary_pred = 1
        if severity_label in ['None', 'Low']:
            severity_label = 'Medium'

    recs = generate_recommendations(user_dict, binary_pred, severity_label)

    st.markdown("---")
    st.markdown("## 📊 Assessment Results")

    col_res1, col_res2 = st.columns([1, 1.5], gap="medium")

    with col_res1:
        if binary_pred == 0:
            st.markdown("""
            <div class="result-card safe">
                <h2 style="margin:0; color:#2AA198;">✅ No Risk Detected</h2>
                <p style="margin:0.3rem 0 0 0; color:#555;">Your responses indicate a healthy status.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card risk">
                <h2 style="margin:0; color:#e74c3c;">⚠️ Potential Risk</h2>
                <p style="margin:0.3rem 0 0 0; color:#555;">Consult a professional for proper guidance.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-box">
            <div class="label">Severity Level</div>
            <div class="value">{}</div>
        </div>
        """.format(severity_label), unsafe_allow_html=True)

    with col_res2:
        st.markdown("#### 💡 Personalized Recommendations")
        for rec in recs:
            st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📈 Risk Factor Profile")

    risk_factors = {
        'Sleep Deficit': sleep_deficit,
        'Overtime': overtime_hours,
        'Low Activity': low_activity,
        'Stress Level': stress_score
    }

    fig = go.Figure(data=[
        go.Bar(
            x=list(risk_factors.keys()),
            y=list(risk_factors.values()),
            marker_color=['#2AA198', '#7559D9', '#F26B6B', '#E9B949'],
            text=[round(v, 2) for v in risk_factors.values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>'
        )
    ])

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=40),
        yaxis_title="Risk Score",
        xaxis_title="Factor",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif"),
        yaxis=dict(gridcolor='#e0e0e0', range=[0, max(list(risk_factors.values())) + 1]),
        xaxis=dict(tickfont=dict(size=13))
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View your input data"):
        st.dataframe(input_df.style.highlight_max(axis=0, color='#e0f5f0'), use_container_width=True)

else:
    st.markdown("""
    <div class="cta-placeholder">
        <div style="font-size: 4rem; margin-bottom: 0.5rem;">🧭</div>
        <h3>Ready to check your wellbeing?</h3>
        <p>Fill in the details above and click <strong>"Assess My Wellbeing"</strong> to get your personalized report.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Wellbeing Compass © 2026 | <span>Group 7</span> | Powered by AI 🧠
</div>
""", unsafe_allow_html=True)