
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px

st.set_page_config(page_title="Luxury House Price AI", page_icon="🏡", layout="wide")

st.markdown("""
<style>
.main{background:linear-gradient(135deg,#0f172a,#1e293b);}
h1{color:white}
.block-container{padding-top:1rem;}
.metric{border-radius:12px}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load():
    for f in ["houseprice.csv","houseprice (2).csv"]:
        try:
            return pd.read_csv(f)
        except:
            pass
    st.error("Dataset not found.")
    st.stop()

@st.cache_resource
def model(df):
    X=df[["area"]]
    y=df["price"]
    m=LinearRegression().fit(X,y)
    return m

df=load()
m=model(df)
preds=m.predict(df[["area"]])
score=r2_score(df["price"],preds)

st.title("🏡 Luxury House Price AI Dashboard")
st.caption("Modern premium house value prediction app")

c1,c2,c3=st.columns(3)
area=c1.slider("Area (sq.ft)",int(df.area.min()),int(df.area.max()),int(df.area.median()),50)
price=float(m.predict([[area]])[0])
c1.metric("Area",f"{area} sq.ft")
c2.metric("Predicted Price",f"₹ {price:,.0f}")
c3.metric("R² Score",f"{score:.3f}")

fig = px.scatter(
    df,
    x="area",
    y="price",
    title="Area vs Price"
)
st.plotly_chart(fig,use_container_width=True)

st.dataframe(df,use_container_width=True)

out=pd.DataFrame({"Area":[area],"Predicted Price":[price]})
st.download_button("📥 Download Prediction",out.to_csv(index=False),"prediction.csv","text/csv")
