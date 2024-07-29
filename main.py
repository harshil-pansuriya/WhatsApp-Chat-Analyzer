# helper.py

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import emoji
import os
from textblob import TextBlob
import argparse


f=open('stop_hinglish.txt', 'r')
stop_words=f.read()


def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity




def fetch_stats(user, df):
    if user != 'Overall':
        df = df[df['username'] == user]
    num_messages = df.shape[0]
    words = []
    for message in df['text']:
        words.extend(message.split())
    num_media_messages = df[df['text'] == '<Media omitted>'].shape[0]
    num_links = sum(df['text'].str.contains('http'))
    return num_messages, len(words), num_media_messages, num_links

def monthly_timeline(user, df):
    if user != 'Overall':
        df = df[df['username'] == user]
    timeline = df.groupby(['year', 'month']).size().reset_index(name='message')
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline

def daily_timeline(user, df):
    if user != 'Overall':
        df = df[df['username'] == user]
    return df.groupby(df['Timestamp'].dt.date).size().reset_index(name='message')

def week_activity_map(user, df):
    if user != 'Overall':
        df = df[df['username'] == user]
    return df['day'].value_counts()

def month_activity_map(user, df):
    if user != 'Overall':
        df = df[df['username'] == user]
    return df['month'].value_counts()

def activity_heatmap(user, df):
    if user != 'Overall':
        df = df[df['username'] == user]
    return df.pivot_table(index='day', columns='period', values='text', aggfunc='count').fillna(0)

def most_busy_users(df):
    x = df['username'].value_counts().head()
    new_df = df['username'].value_counts().reset_index()
    new_df.columns = ['username', 'message_count']
    return x, new_df

def create_wordcloud(user, df, font_path='NotoEmoji-VariableFont_wght.ttf'):
    if user != 'Overall':
        df = df[df['username'] == user]
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(df['text'].str.cat(sep=" "))

    temp = df[df['username'] != 'group_notification']
    temp = temp[temp['text'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['text'] = temp['text'].apply(remove_stop_words)
    df_wc = wc.generate(temp['text'].str.cat(sep=" "))
    return df_wc

def most_common_words(user, df):
    if user != 'Overall':
        df = df[df['username'] == user]
    words = []
    for message in df['text']:
        words.extend(message.split())

    temp = df[df['username'] != 'group_notification']
    temp = temp[temp['text'] != '<Media omitted>\n']

    words = []

    for message in temp['text']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df



def emoji_helper(user, df, font_path='NotoEmoji-VariableFont_wght.ttf'):
    if user != 'Overall':
        df = df[df['username'] == user]
    emojis = []
    for message in df['text']:
        emojis.extend([emoji.emojize(c) for c in message if c in emoji.EMOJI_DATA])
    return pd.DataFrame(Counter(emojis).most_common(20))



def plot_user_sentiment(sentiment_data, title):
    fig, ax = plt.subplots()
    sentiment_data.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title(title)
    ax.set_xlabel('User')
    ax.set_ylabel('Average Sentiment')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig
    
# Topic Modeling
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim as gensimvis

def perform_lda(df, num_topics=5, passes=5, chunk_size=10):
    texts = df['cleaned_text'].apply(lambda x: x.split()).tolist()
    
    # Process in chunks to reduce memory usage
    dictionary = corpora.Dictionary()
    corpus = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        dictionary.add_documents(chunk)
        corpus.extend([dictionary.doc2bow(text) for text in chunk])
    
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    
    return lda_model, corpus, dictionary

def save_lda_topics(lda_model, output_dir):
    topics = lda_model.print_topics(num_words=5)
    topics_file_path = os.path.join(output_dir, 'topics.txt')
    with open(topics_file_path, 'w', encoding='utf-8') as file:
        for topic_num, topic in topics:
            file.write(f"Topic #{topic_num}: {topic}\n")
    
    # Return topics for Streamlit display
    return topics

def create_lda_visualization(lda_model, corpus, dictionary, output_folder):
    vis = gensimvis.prepare(lda_model, corpus, dictionary)

    # Save the visualization
    vis_file_path = os.path.join(output_folder, 'lda_vis.html')
    pyLDAvis.save_html(vis, vis_file_path)
    
    # Return path for Streamlit display
    return vis_file_path
