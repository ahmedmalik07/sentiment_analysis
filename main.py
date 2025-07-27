from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Finviz base URL
finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'AMD', 'FB']
news_tables = {}

# Step 1: Scrape news tables from Finviz for each ticker
for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})  # Fake browser request
    response = urlopen(req)
    html = BeautifulSoup(response, 'html.parser')  # Fixed: use 'html.parser'
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

# Step 2: Parse data into a structured list
parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.find_all('tr'):
        title = row.a.get_text()
        date_data = row.td.text.strip().split(' ')

        if len(date_data) == 1:
            date = ''
            time = date_data[0].strip()
        else:
            date = date_data[0].strip()
            time = date_data[1].strip()

        parsed_data.append([ticker, date, time, title])

# Step 3: Create DataFrame and perform sentiment analysis
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

# Remove empty date rows (can happen for entries without date)
df = df[df['date'] != '']

# Convert date column to datetime object
df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

# Apply VADER sentiment analysis
vader = SentimentIntensityAnalyzer()
df['compound'] = df['title'].apply(lambda x: vader.polarity_scores(x)['compound'])

# Step 4: Group by date and ticker, then average sentiment
mean_df = df.groupby(['date', 'ticker'])['compound'].mean().unstack(level=1)

# Step 5: Plot the sentiment trend
mean_df.plot(kind='bar', figsize=(12, 8), title='Daily Average Sentiment per Ticker')
plt.xlabel("Date")
plt.ylabel("Average Compound Sentiment")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the final DataFrame
print(mean_df)
