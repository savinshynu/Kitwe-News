"""
Collect RSS feeds from different news sources and parses the contents into a pandas dataframe
Used for frontend display and model inference.
"""
import csv
import pandas as pd
import feedparser
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
from urllib.parse import urlparse


def get_base_url(full_url):
    """
    Convert a full URL into a base URL
    """
    parsed = urlparse(full_url)
    return f"{parsed.scheme}://{parsed.netloc}"

def collect_news():

    # List of news feeds to subscribe
    # Adding kitwe in the search to make sure that the data collected is related to Kitwe city in Zambia
    
    feed_urls = {
        'Copperbelt Energy': 'https://cecinvestor.com/search/kitwe/feed/rss2/',
        #'ZNBC' : 'https://znbc.co.zm/news/search/kitwe/feed/rss2/',
        'News Invasion 24': 'https://newsinvasion24.com/search/kitwe/feed/rss2/',
        'Mwebantu': 'https://www.mwebantu.com/search/kitwe/feed/rss2/',
        'Lusaka Times': 'https://www.lusakatimes.com/search/kitwe/feed/rss2/',
        'Kitwe Online': 'https://kitweonline.com/search/kitwe/feed/rss2/',
        'Daily Revelation Zambia': 'https://dailyrevelationzambia.com/search/kitwe/feed/rss2/',
        'Zambia Monitor': 'https://www.zambiamonitor.com/search/kitwe/feed/rss2/',
        'Tech Africa News': 'https://www.techafricanews.com/search/kitwe/feed/rss2/',
        'Zambian Eye': 'https://zambianeye.com/search/kitwe/feed/rss2/',
        'DailyMail': 'https://www.daily-mail.co.zm/search/kitwe/feed/rss2/'
    }


    # Define your date window
    cutoff_date = datetime.now() - timedelta(days=14)  # last 2 days
    #cutooff_date = cutoff_date.replace(tzinfo=timezone.utc)
    #print(cutoff_date)

    # The format to convert datetime string from the published dates of news articles.
    format_string = "%a, %Y-%m-%d %H:%M:%S"
   
    # Create a dictionary for each entry and append it to a list
    # this can be later converted to a pandas dataframe later
    news_list = []
    for source_name, url in feed_urls.items():
        print(f"Fetching data from {source_name}")

        # Each url might contain mulitiple pages of feeds. Lets' collect feeds from the first 10 pages
        npages = 2 # Changes this to increase or decrease the number of feeds
        for page in range(1, npages+1):
            print(f"Collecting page {page} from {source_name}")
            feed = feedparser.parse(f"{url}?paged={page}")
            entries = feed.entries
            if not entries:
                print(f"No entries at page: {page}")
                break
            # Access feed entries
            for entry in entries:
                #print(entry)
                if 'published' in entry:
                    date_string = entry.published
                    print(date_string)
                    try:
                        # Try to convert the date string to the format we want
                        #date = datetime.strptime(date_string, format_string)
                        # Using dateutils for flexibility, also introducing the timezone awarness to avoid comparison problems
                        date = dateparser.parse(date_string)
                        date = date.replace(tzinfo=None) # converting to UTC timezone.
                        print(date)
                    except ValueError:
                        print("Invalid date format")
                        continue

                    if date >= cutoff_date:
                        headline = entry.title if 'title' in entry else 'N/A'
                        desc = entry.description if 'description' in entry else 'N/A'
                        author = entry.author if 'author' in entry else 'N/A'
                        link = entry.link if 'link' in entry else get_base_url(url) # if no news link, replace with the homepage of news channels
                        cat_list = [tag.term for tag in entry.tags] if 'tags' in entry else 'N/A' 
                        category = ','.join(cat_list)

                        #print(entry.title, "\n", entry.link, "\n", entry.description, "\n", entry.published, "\n", entry.tags, "\n", category)

                        news_list.append({'Source': source_name, 'Category': category, 'Headline': headline, 'Link':link, 'Description': desc, 'Date':date, 'Author': author})
    
    return pd.DataFrame(news_list)

if __name__ == "__main__":
    df = collect_news()
    print(df.shape)
    print(df.head())