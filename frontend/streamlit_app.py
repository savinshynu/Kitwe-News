import requests
import streamlit as st
#from config import BACKEND_URL

"""
Plan: What should the app do?
1. This app collects news from RSS feeds at regular intervals. Maybe 15-30 minutes intervals from all the sources. Do I need 
   need to keep the collected information in the memory or stored somewhere?
2. The collected data will be preprocessed and convert to dataframes. 
3. This data will be fed to the API end points to get the confidence scores and the class.
4. Different news links can be displayed along with their predictions and scores.
5. Once you click on the links, it should go to the actual news link.
6. You should have the ability to group news based on category.
7. Feedback loop where user can mark if right or wrong.
8. You can actually manually select the news channels.
9. Save logs and predictions so you can retrain the data later.
10. Ability to select languages and ability to translate them.
"""

st.title("📰 News Aggregator with Fake News Detection")

# Step 1: Select Source
sources = ["BBC", "Reuters", "CNN"]
source = st.selectbox("Select news source", sources)

# Step 2: Fetch news articles
#news_items = get_news_items(source)
selected_news = st.selectbox("Choose a headline", [item['title'] for item in news_items])

# Step 3: Show full article
selected_item = next(item for item in news_items if item['title'] == selected_news)
st.markdown(f"**Source:** {selected_item['source']}")
st.markdown(f"**Published:** {selected_item['published']}")
st.write(selected_item['summary'])

# Step 4: Send to backend for prediction
if st.button("Check if this news is fake"):
    response = requests.post(
        f"{BACKEND_URL}/predict",
        json={"text": selected_item['summary']}
    )
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        if prediction == "fake":
            st.error("⚠️ This news might be FAKE.")
        else:
            st.success("✅ This news appears to be REAL.")
    else:
        st.error("Failed to get prediction from backend.")
