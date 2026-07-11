import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Insurance AI",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e293b,#334155);
color:white;
}

h1,h2,h3{
text-align:center;
color:white;
}

.big-font{
font-size:18px;
font-weight:bold;
}

.card{
background:#1e293b;
padding:25px;
border-radius:20px;
box-shadow:0px 5px 15px rgba(0,0,0,.35);
margin-bottom:20px;
}

.metric{
background:#334155;
padding:20px;
border-radius:15px;
text-align:center;
}

footer{
visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance_data.csv")

df = load_data()

# -----------------------------
# Train Model
# -----------------------------
X = df[["age"]]
y = df["bought_insurance"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙ Prediction Settings")

age = st.sidebar.slider(
    "Select Age",
    min_value=18,
    max_value=80,
    value=30
)

predict = st.sidebar.button("Predict")

# -----------------------------
# Header
# -----------------------------
st.title("🛡 Smart Insurance AI Dashboard")
st.markdown(
"<center>Machine Learning Powered Insurance Purchase Prediction</center>",
unsafe_allow_html=True
)

st.write("")

# -----------------------------
# Metrics
# -----------------------------
col1,col2,col3=st.columns(3)

with col1:
    st.metric("Dataset Size",len(df))

with col2:
    st.metric("Training Accuracy",f"{train_acc*100:.2f}%")

with col3:
    st.metric("Testing Accuracy",f"{test_acc*100:.2f}%")

st.write("---")

# -----------------------------
# Prediction
# -----------------------------
left,right=st.columns([1,1])

with left:

    st.markdown("## 👤 Customer Details")

    st.info(f"Selected Age : **{age} years**")

    if predict:

        sample=np.array([[age]])

        probability=model.predict_proba(sample)[0][1]
        prediction=model.predict(sample)[0]

        st.subheader("Prediction Probability")

        st.progress(float(probability))

        st.metric(
            "Chance of Buying Insurance",
            f"{probability*100:.2f}%"
        )

        if prediction==1:

            st.success("✅ Customer is likely to purchase insurance.")

        else:

            st.error("❌ Customer is unlikely to purchase insurance.")

with right:

    st.markdown("## 📈 Dataset Overview")

    st.dataframe(
        df,
        use_container_width=True,
        height=300
    )

# -----------------------------
# Statistics
# -----------------------------
st.write("---")

st.subheader("📊 Dataset Statistics")

st.dataframe(df.describe(),use_container_width=True)

# -----------------------------
# Chart
# -----------------------------
st.write("---")

st.subheader("📉 Age Distribution")

chart=df["age"].value_counts().sort_index()

st.bar_chart(chart)

# -----------------------------
# Footer
# -----------------------------
st.write("---")

st.markdown(
"""
<center>
Built with ❤️ using Streamlit & Scikit-Learn
</center>
""",
unsafe_allow_html=True
)

