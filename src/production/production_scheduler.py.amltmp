import streamlit as st
import pandas as pd
import plotly.express as px
from pyprojroot import here
import os

# Set the path and read the data
path_data = here("./data")
os.chdir(path_data)
data = pd.read_csv("data_shap_hospital.csv")[["Hospital ID", "Driver", "Impact"]]

# Streamlit title
st.title("Hospital Drivers Analysis")

st.markdown("""
This is a demo with synthetic data based on a model predicting an ER visit within the last 30 days. 
The results below show the average impact of the driver on the outcome in percentage.
""")

# Sidebar for hospital selection
hospital_id = st.sidebar.selectbox("Select Hospital ID", options=data["Hospital ID"].unique())

# Filter data based on selected hospital
df = data[data["Hospital ID"] == hospital_id]
df["AbsImpact"] = df["Impact"].abs()
df = df.sort_values(by="AbsImpact", ascending=False)
df = df.iloc[::-1]
df = df.drop("AbsImpact", axis=1)
df["ImpactText"] = df["Impact"].apply(lambda x: f"{x:.0%}")
df["Color"] = df["Impact"].apply(lambda x: "blue" if x > 0 else "red")

# Create horizontal bar chart using Plotly
fig = px.bar(
    df,
    x="Impact",
    y="Driver",
    orientation="h",
    text="ImpactText",
    color="Color",
    labels={"Impact": "Impact Value", "Driver": "Driver Factor"},
)

# Customizing the layout
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showticklabels=False, title=None),
    showlegend=False,
)
fig.update_traces(marker_coloraxis=None)
fig.update_traces(texttemplate="%{text}", textposition="inside")

# Display the plot in Streamlit
st.plotly_chart(fig)
