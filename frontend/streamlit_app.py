import yaml
import requests
import streamlit as st

from collect_news import collect_news
from utils import exec_preprocessing

with open("config.yaml", "r") as fh:
    config = yaml.safe_load(fh)
    
# API end points
api_url = config["api"]["url"]
#print(api_url)


@st.cache_data
def get_data():
    df_ini = collect_news() # collect news from all the sources
    df = exec_preprocessing(df_ini, api_url)
    return df

# UI
st.title("📰 News Aggregator and Fake News Classifier")

# Extract preprocessed data
df = get_data()

# Filter by source
sources = df["Source"].unique().tolist()
selected_sources = st.multiselect("Select news source(s)", sources, default=sources)

src_filt_df = df[df["Source"].isin(selected_sources)]

categories = df["Category"].unique().tolist()
selected_cats = st.multiselect("Select Categories", categories, default=categories)
cat_filt_df = src_filt_df[src_filt_df["Category"].isin(selected_cats)]


# Show news and predictions
for idx, row in cat_filt_df.iterrows():
    if row['Label'] == "Genuine":
        symb = "✅"
    else:
        symb = "⚠️"

    st.markdown(f"### {row['Headline']}")
    st.markdown(f"**Source:** {row['Source']}  |  **Published:** {row['Date']}   | {symb}  | Score:{int(row['Confidence']*100)}")
    st.markdown(f"[Read more]({row['Link']})")
    st.write(row["Description"])
