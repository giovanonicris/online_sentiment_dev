import requests, random, re, os
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser
from newspaper import Article, Config
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googlenewsdecoder import new_decoderv1
from urllib.parse import urlparse
import datetime as dt
import nltk

# see if nltk 'punkt' is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Create a list of random user agents
user_agent_list = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...Firefox/77.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5)...Chrome/83.0.4103.97',
]
user_agent = random.choice(user_agent_list)
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 20
header = {'User-Agent': user_agent}

# encode-decode search terms
read_file = pd.read_csv('EnterpriseRisksListEncoded.csv', encoding='utf-8')
read_file['ENTERPRISE_RISK_ID'] = pd.to_numeric(read_file['ENTERPRISE_RISK_ID'], downcast='integer', errors='coerce')

def decode_term(encoded):
    try:
        return int(encoded).to_bytes((int(encoded).bit_length() + 7) // 8, 'little').decode('utf-8')
    except:
        return None

read_file['SEARCH_TERMS'] = read_file['ENCODED_TERMS'].apply(decode_term)
search_term = read_file['SEARCH_TERMS'].dropna().iloc[0]
print(f"First decoded search term: {search_term}")

# prep lists to store new entries
search_terms = []
title = []
published = []
link = []
domain = []
source = []
summary = []
keywords = []
sentiments = []
polarity = []

# load filter_out_sources.csv file
filtered_sources = set()
if os.path.exists('filter_out_sources.csv'):
    df = pd.read_csv('filter_out_sources.csv', encoding='utf-8')
    filtered_sources = set(df.iloc[:, 0].dropna().str.lower().str.strip())


# Grab Google links
url_start = 'https://news.google.com/rss/search?q={'
url_end = '}%20when%3A6h'

# limit article, get small sample for quick debugging
ARTICLE_LIMIT = 20
article_count = 0

try:
    # isolate rss url created
    rss_url = url_start + search_term + url_end
    print(f"Constructed URL: {rss_url}")
    req = requests.get(rss_url, headers=header)
    print("RSS request is completed.")

    soup = BeautifulSoup(req.text, 'xml')

    for item in soup.find_all("item"):
        if article_count >= ARTICLE_LIMIT:
            break

        title_text = item.title.text.strip()
        encoded_url = item.link.text.strip()
        source_text = item.source.text.strip().lower()
        decoded = new_decoderv1(encoded_url, interval=5)

        if not decoded.get("status"):
            print(f"Decoder failed for URL: {encoded_url}")
            continue

        decoded_url = decoded['decoded_url'].strip().lower()
        domain_name = urlparse(decoded_url).netloc.lower()

        if not decoded_url.endswith(('.com', '.edu', '.org', '.net')):
            print(f"Skipped non-standard domain: {decoded_url}")
            continue
        if source_text in filtered_sources:
            print(f"Filtered out source: {source_text}")
            continue
        if "/en/" in decoded_url:
            continue

        #log outcome of each step in Article
        article = Article(decoded_url, config=config)
        try:
            #download
            print(f"Attempt article download: {decoded_url}")
            article.download()
            if article.download_state != 2:
                print(f"Download failed for: {decoded_url}")
                continue
            #parse
            print(f"Downloaded successfully: {decoded_url}")
            article.parse()
            print(f"Parsed successfully: {decoded_url}")
            #summarize
            article.nlp()
            print(f"NLP completed: {decoded_url}")

            text = article.summary.strip() or article.text.strip()
            if not text:
                print(f"No content extracted from: {decoded_url}")
                continue

        except Exception as e:
            print(f"Newspaper3k error for {decoded_url}: {e}")
            continue

        score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
        sentiment = 'positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral'

        search_terms.append(search_term)
        title.append(title_text)
        source.append(source_text)
        link.append(decoded_url)
        summary.append(text)
        keywords.append(article.keywords)
        polarity.append(f"{score}")
        sentiments.append(sentiment)
        published.append(parser.parse(item.pubDate.text).date() if item.pubDate else None)
        domain.append(domain_name)
        article_count += 1

except requests.exceptions.RequestException as e:
    print(f"Request error for RSS: {e}")

# save results
alerts = pd.DataFrame({
    'SEARCH_TERMS': search_terms,
    'TITLE': title,
    'SUMMARY': summary,
    'KEYWORDS': keywords,
    'PUBLISHED_DATE': published,
    'LINK': link,
    'SOURCE': source,
    'SOURCE_URL': domain,
    'SENTIMENT': sentiments,
    'POLARITY': polarity
})

alerts['LAST_RUN_TIMESTAMP'] = dt.datetime.now().isoformat()
alerts.to_csv('DEBUG-validate_newspaper3k_results.csv', index=False, encoding='utf-8')
