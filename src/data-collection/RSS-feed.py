"""
This script will be used to collect data from various sources using RSS feed and 
store them into a CSV file further preprocessing and modeling.

RSS (Really Simple Syndication) feeds are used to access web feeds. This will the help user/application to 
know about updates/changes to the websites without actually visiting their websites.
RSS feeds are in the XML format which is generally used for storing and transporting data. This format is simple plain text
and independent of the software and hardware making it a good choice. XML is extensible(add/remove data) and does not have predefine
tags like html. Feedparser is used to parse the xml documents.
"""
import csv
import feedparser

# List of news feeds to subscribe
# Adding kitwe in the search to make sure that the data collected is related to Kitwe city in Zambia
feed_urls = {
     'Copperbelt Energy': 'https://cecinvestor.com/search/kitwe/feed/rss2/',
     'ZNBC' : 'https://znbc.co.zm/news/search/kitwe/feed/rss2/',
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


# Open a  csv file to store the information
with open("/Users/savin/Omdena-Projects/Kitwe-News/data/raw-new.csv", "w", encoding='UTF8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ['Source', 'Category', 'Headline', 'Link', 'Description', 'Date', 'Author']
    writer.writerow(header)

    for source_name, url in feed_urls.items():
        print(f"Fetching data from {source_name}")

        # Each url might contain mulitiple pages of feeds. Lets' collect feeds from the first 10 pages
        npages = 30 # Changes this to increase or decrease the number of feeds
        for page in range(1, npages+1):
            print(f"Collecting page {page} from {source_name}")
            feed = feedparser.parse(f"{url}?paged={page}")
            entries = feed.entries
            if not entries:
                print(f"No entries after page: {page}")
                break
            # Access feed entries
            for entry in entries:
                headline = entry.title if 'title' in entry else 'N/A'
                desc = entry.description if 'description' in entry else 'N/A'
                author = entry.author if 'author' in entry else 'N/A'
                date = entry.published if 'published' in entry else 'N/A'
                cat_list = [tag.term for tag in entry.tags] if 'tags' in entry else 'N/A' 
                category = ','.join(cat_list)

                #print(entry.title, "\n", entry.link, "\n", entry.description, "\n", entry.published, "\n", entry.tags, "\n", category)

                # writing the collected info as a row to the csv file
                writer.writerow([source_name, category, headline, url, desc, date, author])