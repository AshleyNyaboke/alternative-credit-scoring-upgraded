import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Unbanked Kenya Credit Scorer",
    page_icon="🤝",
    layout="wide"
)

# Custom CSS for warm card based layout
st.markdown("""
<style>
    .main { background-color: #FFF8F0; }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #E07B2A;
        margin-bottom: 16px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.07);
    }
    .green-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2E7D32;
        margin-bottom: 16px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.07);
    }
    .metric-box {
        background-color: #FFF3E0;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 12px;
    }
    .big-number {
        font-size: 48px;
        font-weight: bold;
        color: #E07B2A;
    }
    .section-title {
        color: #E07B2A;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    div[data-testid="stButton"] button {
        background-color: #E07B2A;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-size: 16px;
        padding: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='background: linear-gradient(135deg, #E07B2A, #2E7D32);
     padding: 30px; border-radius: 16px; margin-bottom: 24px;'>
    <h1 style='color: white; margin:0;'>🤝 Credit For All Kenyans</h1>
    <p style='color: #FFE0B2; font-size: 16px; margin-top: 8px;'>
    Predicting credit risk for boda boda riders, mama mbogas, hawkers 
    and jua kali artisans — using mobile money behaviour, not payslips.
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 2000
    occupations = [
        "Boda Boda Rider", "Mama Mboga", "Hawker",
        "Jua Kali Artisan", "Content Creator",
        "Casual Labourer", "Fish Monger",
        "Salon/Barber", "Matatu Operator", "Farmer"
    ]
    counties = [
        "Nairobi", "Mombasa", "Kisumu", "Nakuru",
        "Eldoret", "Kiambu", "Machakos", "Meru",
        "Nyeri", "Kakamega", "Kisii", "Kilifi",
        "Garissa", "Kitui", "Bungoma", "Uasin Gishu",
        "Muranga", "Embu", "Kericho", "Migori"
    ]
    county_weights = [
        0.20,0.08,0.07,0.07,0.06,0.06,0.05,
        0.04,0.04,0.04,0.04,0.03,0.03,0.03,
        0.03,0.03,0.03,0.02,0.02,0.02
    ]
    county_weights = [w/sum(county_weights) for w in county_weights]

    df = pd.DataFrame({
        "county": np.random.choice(counties, n, p=county_weights),
        "occupation": np.random.choice(occupations, n),
        "age": np.random.randint(18, 55, n),
        "monthly_income_ksh": np.random.choice(
            [5000, 10000, 15000, 25000, 40000, 60000],
            n, p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05]
        ),
        "mobile_money_freq_per_month": np.random.randint(0, 60, n),
        "mobile_money_savings_ksh": np.random.choice(
            [0, 500, 1000, 3000, 5000, 10000],
            n, p=[0.25, 0.20, 0.20, 0.15, 0.12, 0.08]
        ),
        "owns_smartphone": np.random.choice([0, 1], n, p=[0.40, 0.60]),
        "years_in_business": np.random.randint(0, 15, n),
        "has_prior_loan": np.random.choice([0, 1], n, p=[0.65, 0.35]),
        "repaid_prior_loan": np.random.choice([0, 1], n, p=[0.30, 0.70]),
        "urban": np.random.choice([0, 1], n, p=[0.38, 0.62]),
    })

    credit_score = (
        0.30 * (df["mobile_money_freq_per_month"] / 60) +
        0.25 * (df["monthly_income_ksh"] / 60000) +
        0.20 * df["repaid_prior_loan"] * df["has_prior_loan"] +
        0.15 * (df["mobile_money_savings_ksh"] / 10000) +
        0.10 * df["owns_smartphone"]
    )
    noise = np.random.normal(0, 0.08, n)
    df["low_risk"] = ((credit_score + noise).clip(0, 1) > 0.45).astype(int)

    le_county = LabelEncoder()
    le_occ = LabelEncoder()
    df["county_enc"] = le_county.fit_transform(df["county"])
    df["occupation_enc"] = le_occ.fit_transform(df["occupation"])

    features = ["county_enc", "occupation_enc", "age",
                "monthly_income_ksh", "mobile_money_freq_per_month",
                "mobile_money_savings_ksh", "owns_smartphone",
                "years_in_business", "has_prior_loan",
                "repaid_prior_loan", "urban"]

    X = df[features]
    y = df["low_risk"]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf, le_county, le_occ, df, features

rf, le_county, le_occ, df, features = train_model()

# Portfolio stats row
st.markdown("<p class='section-title'>📊 Portfolio Overview</p>",
            unsafe_allow_html=True)

total = len(df)
low = df["low_risk"].sum()
high = total - low
low_pct = round((low/total)*100)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-box'>
        <div style='color:#777;font-size:13px'>Total Applicants</div>
        <div class='big-number'>{total:,}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-box'>
        <div style='color:#777;font-size:13px'>Low Risk</div>
        <div class='big-number' style='color:#2E7D32'>{low:,}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-box'>
        <div style='color:#777;font-size:13px'>High Risk</div>
        <div class='big-number' style='color:#C62828'>{high:,}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-box'>
        <div style='color:#777;font-size:13px'>Approval Rate</div>
        <div class='big-number'>{low_pct}%</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Two column layout
left, right = st.columns([1, 1])

with left:
    st.markdown("""<div class='card'>
        <p class='section-title'>📋 Applicant Profile</p>
    </div>""", unsafe_allow_html=True)

    county = st.selectbox("County", sorted([
        "Nairobi", "Mombasa", "Kisumu", "Nakuru",
        "Eldoret", "Kiambu", "Machakos", "Meru",
        "Nyeri", "Kakamega", "Kisii", "Kilifi",
        "Garissa", "Kitui", "Bungoma", "Uasin Gishu",
        "Muranga", "Embu", "Kericho", "Migori"
    ]))
    occupation = st.selectbox("Occupation", [
        "Boda Boda Rider", "Mama Mboga", "Hawker",
        "Jua Kali Artisan", "Content Creator",
        "Casual Labourer", "Fish Monger",
        "Salon/Barber", "Matatu Operator", "Farmer"
    ])
    age = st.slider("Age", 18, 55, 28)
    income = st.selectbox("Monthly Income (KSH)",
        [5000, 10000, 15000, 25000, 40000, 60000], index=2)
    mm_freq = st.slider("Mobile Money Transactions per Month", 0, 60, 20)
    mm_savings = st.selectbox("Mobile Money Savings (KSH)",
        [0, 500, 1000, 3000, 5000, 10000], index=2)
    smartphone = st.radio("Owns Smartphone?", ["Yes", "No"])
    years_biz = st.slider("Years in Business", 0, 15, 3)
    prior_loan = st.radio("Has Prior Loan?", ["Yes", "No"])
    repaid = st.radio("Repaid Prior Loan?", ["Yes", "No"])
    location = st.radio("Location", ["Urban", "Rural"])
    predict_btn = st.button("🔍 Assess Credit Risk")

with right:
    st.markdown("""<div class='green-card'>
        <p class='section-title' style='color:#2E7D32'>
        🎯 Credit Risk Result</p>
    </div>""", unsafe_allow_html=True)

    if predict_btn:
        county_enc = le_county.transform([county])[0]
        occ_enc = le_occ.transform([occupation])[0]

        input_data = pd.DataFrame([{
            "county_enc": county_enc,
            "occupation_enc": occ_enc,
            "age": age,
            "monthly_income_ksh": income,
            "mobile_money_freq_per_month": mm_freq,
            "mobile_money_savings_ksh": mm_savings,
            "owns_smartphone": 1 if smartphone == "Yes" else 0,
            "years_in_business": years_biz,
            "has_prior_loan": 1 if prior_loan == "Yes" else 0,
            "repaid_prior_loan": 1 if repaid == "Yes" else 0,
            "urban": 1 if location == "Urban" else 0
        }])

        prob = rf.predict_proba(input_data)[0][1]

        st.markdown(f"""<div class='metric-box' style='margin-top:20px'>
            <div style='color:#777;font-size:14px'>
            Low Risk Probability</div>
            <div class='big-number'>{prob:.0%}</div>
        </div>""", unsafe_allow_html=True)

        if prob >= 0.65:
            st.success("✅ LOW RISK — Fast Track for loan approval!")
            st.markdown("""<div class='green-card'>
                <b>Recommendation:</b> This applicant shows strong 
                mobile money behaviour and repayment history. 
                Eligible for automated mobile disbursement.
            </div>""", unsafe_allow_html=True)
        elif prob >= 0.40:
            st.warning("⚠️ MEDIUM RISK — Manual review recommended")
            st.markdown("""<div class='card'>
                <b>Recommendation:</b> This applicant shows moderate 
                creditworthiness. A loan officer should review further 
                before approval.
            </div>""", unsafe_allow_html=True)
        else:
            st.error("❌ HIGH RISK — Loan not recommended at this time")
            st.markdown("""<div style='background:#FFF3F3;padding:16px;
                border-radius:10px;border-left:5px solid #C62828'>
                <b>Recommendation:</b> Insufficient mobile money 
                activity or prior default history. Consider financial 
                literacy support first.
            </div>""", unsafe_allow_html=True)

        st.progress(float(prob))
        st.divider()

        st.markdown("<p class='section-title'>📊 Key Risk Drivers</p>",
                    unsafe_allow_html=True)
        feat_imp = pd.Series(
            rf.feature_importances_,
            index=["County","Occupation","Age","Income",
                   "MM Frequency","MM Savings","Smartphone",
                   "Years in Biz","Prior Loan",
                   "Repaid Loan","Urban"]
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#2E7D32" if v > feat_imp.median()
                  else "#E07B2A" for v in feat_imp.values]
        feat_imp.plot(kind="barh", ax=ax, color=colors)
        ax.set_title("What Drives This Credit Decision?")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.markdown("""<div style='text-align:center;
            padding:60px 20px;color:#999;'>
            <h3>👈 Fill in the applicant profile</h3>
            <p>and click Assess Credit Risk to see the result</p>
        </div>""", unsafe_allow_html=True)

st.divider()

# County insights
st.markdown("<p class='section-title'>📍 Low Risk Rate by County</p>",
            unsafe_allow_html=True)
st.markdown("""<div class='card'>
Which counties have the highest concentration of creditworthy 
unbanked applicants?
</div>""", unsafe_allow_html=True)

county_risk = df.groupby("county")["low_risk"].mean().sort_values(
    ascending=False)
fig2, ax2 = plt.subplots(figsize=(12, 4))
county_risk.plot(kind="bar", ax=ax2, color="#E07B2A")
ax2.set_title("Low Risk Rate by County")
ax2.set_ylabel("Low Risk Rate")
ax2.set_xlabel("County")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig2)

st.divider()

# Occupation breakdown
st.markdown("<p class='section-title'>💼 Risk by Occupation</p>",
            unsafe_allow_html=True)
occ_risk = df.groupby("occupation")["low_risk"].mean().sort_values(
    ascending=False)
fig3, ax3 = plt.subplots(figsize=(10, 4))
occ_risk.plot(kind="bar", ax=ax3, color="#2E7D32")
ax3.set_title("Low Risk Rate by Occupation")
ax3.set_ylabel("Low Risk Rate")
ax3.set_xlabel("Occupation")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig3)

st.caption("""Built by Ashley Nyaboke Kibwogo | 
Alternative Credit Scoring for Unbanked Kenyans | 
🇰🇪 Financial Inclusion Project""")
