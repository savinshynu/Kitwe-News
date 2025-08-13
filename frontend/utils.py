# Contains classes for preprocessing the data and also categorization
import spacy
import re
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bf
from textblob import TextBlob
from urllib.parse import urlparse
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier


class Preprocesser:
    """
    A classs to do preprocessing of the data like
    dropping duplicates, dropping rows with description containing Nan values,
    stripping html tags, etc
    """
    def __init__(self, df):
        self.df = df.copy() # data frame
    
    def clean_data(self):
        # Let's drop the duplicates
        self.df.drop_duplicates(inplace=True)

        # let's drop the rows which has nan values for the description
        self.df.dropna(subset=["Description"], inplace=True)

        # Fill out NaN values
        self.df.fillna(value = "", inplace=True)

    @staticmethod
    def strip_html_tags(text):
        """
        Strip html tags in a text
        """
        soup = bf(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text
    
    def preprocess(self):
        """
        return the cleaned data frame
        """
        self.clean_data()
        
        # Now let's apply this function on the headline and the description column
        self.df['Headline'] = self.df['Headline'].astype('str').apply(self.strip_html_tags)
        self.df['Description'] = self.df['Description'].astype('str').apply(self.strip_html_tags)

        return self.df
    
    
class TextCategorizer:
    """
    A class to assign a set of pre-selected categories and 
    assign KNN based classifier to assign the nearest category using
    tf-idf vectorization
    """
    def __init__(self, data, n_neighbors=5, max_features=5000):
        self.data = data.copy() # pandas data frame
        self.n_neighbors = n_neighbors # no of nearest neighbours to consider
        self.max_features = max_features # maximum number of features to consider
        self.vectorizer = TfidfVectorizer(max_features=max_features) #tf-dif vectorizer
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        
        # Define category keywords directly in the class
        self.categories_keywords = {
            'sports': ['football', 'soccer', 'basketball', 'tennis', 'cricket', 'olympics', 'athlete', 'sports'],
            'politics': ['government', 'election', 'politician', 'policy', 'parliament', 'minister', 'president', 'vote'],
            'education': ['school', 'university', 'education', 'college', 'students', 'learning', 'teacher', 'scholarship'],
            'health and wellness': ['health', 'hospital', 'doctor', 'wellness', 'mental health', 'fitness', 'medicine', 'disease'],
            'development': ['development', 'infrastructure', 'construction', 'road', 'bridge', 'building', 'urbanization'],
            'narcotics': ['narcotics', 'drug', 'cocaine', 'heroin', 'meth', 'drug trafficking', 'illegal drugs'],
            'fashion': ['fashion', 'clothing', 'designer', 'runway', 'model', 'style', 'apparel', 'trends'],
            'career': ['job', 'career', 'employment', 'opportunity', 'work', 'recruitment', 'hiring', 'position'],
            'local news': ['local', 'community', 'city', 'town', 'village', 'municipality', 'neighborhood', 'region'],
            'economy news': ['economy', 'economic', 'finance', 'market', 'stocks', 'currency', 'inflation', 'gdp'],
            'business news': ['business', 'company', 'corporation', 'entrepreneur', 'startup', 'industry', 'investment', 'profit']
        }
        
    def prioritize_category(self, description):
        """Assign a single category based on highest keyword count."""

        keyword_count = {}
        for category, keywords in self.categories_keywords.items():
            count = sum(description.lower().count(keyword) for keyword in keywords)
            if count > 0:
                keyword_count[category] = count
        return max(keyword_count, key=keyword_count.get) if keyword_count else 'uncategorized'
    
    def assign_single_categories(self):
        """Apply single category based on keyword prioritization."""
        self.data['Single_Category'] = self.data['Description'].apply(self.prioritize_category)

    def train_knn_classifier(self):
        """Train the KNN model to predict categories for uncategorized entries."""
        desc = self.data['Description']
        
        cat = self.data['Single_Category'] != 'uncategorized'
        uncat = self.data['Single_Category'] == 'uncategorized'
        
        if len(cat) > self.n_neighbors:
         
            # Train data
            X_train = desc[cat]
            y_train = self.data['Single_Category'][cat]

            # test data
            X_test = desc[uncat]
            
            # Convert text to TF-IDF vectors
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            
            # Train KNN classifier
            self.knn.fit(X_train_tfidf, y_train)
            
            # Predict uncategorized entries
            if any(uncat):
                X_test_tfidf = self.vectorizer.transform(X_test) # vectorize the text to TF-IDF
                y_pred = self.knn.predict(X_test_tfidf)
                self.data.loc[uncat, 'Single_Category'] = y_pred
    
    def categorize(self):
        """Run all categorization steps in sequence, and replace 'Category' with 'Single_Category'."""
        self.assign_single_categories()
        self.train_knn_classifier()

        # Drop the initial category column
        self.data.drop(columns=["Category"], inplace=True)

        # Renaming the column name and changing the order
        self.data.rename(columns={"Single_Category":"Category"}, inplace=True)

        # Reordering the column values
        self.data = self.data[["Source", "Category", "Headline", "Description", "Link", "Date", "Author"]]

        return self.data

class PredictClass:
    """
    Predict the class and confidence scores of the news data
    using the trained Bert/DistillBert LLM model. Add it to the pandas dataframe as well
    """
    def __init__(self, df, API):
        self.df = df.copy()
        self.api_point = API
    
    def get_prediction(self):
        df_inp = (self.df['Headline'] + " " + self.df['Description']).astype('str')
        #print(df_inp)
        
        response = requests.post(self.api_point, json={'texts':df_inp.tolist()})
        #response = requests.get(self.api_point)
        print(response.status_code)
        if response.status_code == 200:
            print("Request Successfull")
            response = response.json() # converting the response into json
            self.df['Label'] = response['label']
            self.df['Label'] = self.df['Label'].replace({0:"Genuine", 1:"Fake"}) #apply(lambda x: "Genuine" if 0 else "Fake"
            self.df['Confidence'] = response['confidence']
            
        else:
            print("Request Unsuccessfull")
            self.df['Label'] = np.nan
            self.df['Confidence'] = np.nan
        
        return self.df


def exec_preprocessing(df, api):
    """
    Just apply cleaning and categorization to the data
    """ 
    clean_ob = Preprocesser(df)
    df_clean = clean_ob.preprocess()
    print(df_clean.shape)

    cat_ob = TextCategorizer(df_clean)
    df_cat = cat_ob.categorize()

    pred_ob = PredictClass(df_cat, api)
    df_fin = pred_ob.get_prediction()

    return df_fin 


if __name__ == "__main__":
    df = pd.read_csv('/home/savin/Omdena/Kitwe-News/data/raw-new.csv')
    api = "http://localhost:8000/predict"
    #api = "http://localhost:8000"
    df_test = df.iloc[:50,:]
    df_out = exec_preprocessing(df_test, api)
    print(df_out)