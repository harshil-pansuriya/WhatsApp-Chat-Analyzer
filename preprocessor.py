import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# load Stopwords file
with open('Data\stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
        
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)


def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s\w+\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    df = pd.DataFrame({"User_Message": messages, "Timestamp": dates})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%y, %H:%M %p - ', errors='coerce')

    df[['username', 'text']] = df['User_Message'].apply(lambda x: pd.Series(x.split(':', 1) if ':' in x else ['group_notification', x]))
    df['username'] = df['username'].str.strip()
    df['text'] = df['text'].str.strip()

    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month_name()
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    
    df['period'] = df['hour'].apply(lambda hour: f"{hour:02d}-{(hour+1)%24:02d}")

    df['cleaned_text'] = df['text'].apply(clean_text)
    
    return df