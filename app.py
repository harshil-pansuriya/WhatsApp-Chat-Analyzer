import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from textblob import TextBlob
import preprocessor,main

# Streamlit UI
st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat file", type="txt")
stop_words_file = 'stop_hinglish.txt'

if uploaded_file is not None:
    # Load data
    data = uploaded_file.read().decode('utf-8')

    # Load Stopwords file
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())

    # Preprocess data
    df = preprocessor.preprocess(data)

    # Select user (change 'Overall' to a specific user if needed)
    user = st.sidebar.selectbox('Select User', ['Overall'] + df['username'].unique().tolist())

    # Create Output folder
    output_folder = 'Output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Fetch stats
    num_messages, words, num_media_messages, num_links = main.fetch_stats(user, df)
    st.title("Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.header("Total Messages")
        st.title(num_messages)
    with col2:
        st.header("Total Words")
        st.title(words)
    with col3:
        st.header("Media Shared")
        st.title(num_media_messages)
    with col4:
        st.header("Links Shared")
        st.title(num_links)

    # Monthly timeline
    st.title("Monthly Timline of chat")
    timeline = main.monthly_timeline(user, df)
    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(timeline['time'], timeline['message'], color='green')
    plt.xticks(rotation='vertical')
    plt.title('Monthly Timeline')
    st.pyplot(fig, clear_figure=True)

    # Daily timeline
    st.title("Daily Timline of chat")
    daily_timeline = main.daily_timeline(user, df)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(daily_timeline['Timestamp'], daily_timeline['message'], color='black')
    plt.xticks(rotation='vertical')
    plt.title('Daily Timeline')
    st.pyplot(fig, clear_figure=True)

    # Activity map
    st.title("Activity map")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
    ax1.bar(main.week_activity_map(user, df).index, main.week_activity_map(user, df).values, color='purple')
    ax1.set_title('Most Busy Day')
    ax2.bar(main.month_activity_map(user, df).index, main.month_activity_map(user, df).values, color='orange')
    ax2.set_title('Most Busy Month')
    st.pyplot(fig, clear_figure=True)

    # Weekly activity map
    st.title("Activity Heatmap Weekly")
    user_heatmap = main.activity_heatmap(user, df)
    fig, ax = plt.subplots()
    sns.heatmap(user_heatmap, ax=ax)
    plt.title('Weekly Activity Map')
    st.pyplot(fig, clear_figure=True)

    # Most busy users
    st.title("Most busy Users in the chat")
    if user == 'Overall':
        x, new_df = main.most_busy_users(df)
        fig, ax = plt.subplots(figsize=(15,15))
        ax.bar(x.index, x.values, color='red')
        plt.xticks(rotation='vertical')
        plt.title('Most Busy Users')
        st.pyplot(fig, clear_figure=True)
        st.write(new_df)

        new_df.to_csv('most_busy_users.csv', index=False)

    # WordCloud
    st.title("Wordcloud of chat ")
    df_wc = main.create_wordcloud(user, df)
    fig, ax = plt.subplots()
    ax.imshow(df_wc)
    plt.title('Wordcloud')
    plt.axis('off')
    st.pyplot(fig, clear_figure=True)

    # Most common words
    st.title("Most Common Words")
    most_common_df = main.most_common_words(user, df)
    fig, ax = plt.subplots()
    ax.barh(most_common_df[0], most_common_df[1])
    plt.xticks(rotation='vertical')
    plt.title('Most Common Words')
    st.pyplot(fig, clear_figure=True)

    # Emoji analysis
    st.title("Most Used Emojis")
    emoji_df = main.emoji_helper(user, df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
    ax1.set_title('Emoji Analysis')
    ax2.barh(emoji_df[0].head(), emoji_df[1].head())
    ax2.set_title('Top Emojis')
    st.pyplot(fig, clear_figure=True)

    # Sentiment distribution plot
    st.title("Sentiment Analysis ")
    
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_filtered = df[(df['sentiment'] < -0.1) | (df['sentiment'] > 0.1)]



    def compute_average_positive_sentiment(df):
        return df[df['sentiment'] > 0.1].groupby('username')['sentiment'].mean()

    # Function to compute average negative sentiment by user
    def compute_average_negative_sentiment(df):
        return df[df['sentiment'] < -0.1].groupby('username')['sentiment'].mean()  


    #Plot average positive sentiment
    average_positive_sentiment = compute_average_positive_sentiment(df_filtered)
    average_negative_sentiment = compute_average_negative_sentiment(df_filtered)
    
    col1, col2 = st.columns(2)

    with col1:
        fig_positive = main.plot_user_sentiment(average_positive_sentiment, 'Average Positive Sentiment by User')
        st.pyplot(fig_positive, clear_figure=True)

    with col2:
        fig_negative = main.plot_user_sentiment(average_negative_sentiment, 'Average Negative Sentiment by User')
        st.pyplot(fig_negative, clear_figure=True)



    st.title('Topic Modeling with LDA')

    num_topics = st.slider('Number of Topics', min_value=2, max_value=20, value=5)
    passes = st.slider('Number of Passes', min_value=1, max_value=20, value=5)

    # Assuming the DataFrame `df` is already loaded
    if st.button('Perform LDA'):
        lda_model, corpus, dictionary = main.perform_lda(df, num_topics=num_topics, passes=passes)

        output_folder = 'Output'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        topics = main.save_lda_topics(lda_model, output_folder)
        vis_path = main.create_lda_visualization(lda_model, corpus, dictionary, output_folder)
    
        # Display LDA topics
        st.subheader('LDA Topics')
        for topic_num, topic in topics:
            st.write(f"Topic #{topic_num}: {topic}")

        # Display LDA visualization
        st.subheader('LDA Visualization')
        with open(vis_path, 'r') as file:
            st.components.v1.html(file.read(), height=800,width=1000)