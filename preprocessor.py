import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# load Stopwords file
with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())


def preprocess(data):
    """Preprocess the raw WhatsApp chat data."""
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s\w+\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    # Construct DataFrame
    df = pd.DataFrame({"User_Message": messages, "Timestamp": dates})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%y, %H:%M %p - ', errors='coerce')

    # Extract username and text
    def separate_username_text(message):
        idx = message.find(':')
        if idx != -1:
            username = message[:idx].strip()
            text = message[idx + 1:].strip()
        else:
            username = ''
            text = message
        return username, text

    df[['username', 'text']] = df['User_Message'].apply(lambda x: pd.Series(separate_username_text(x)))

    # Additional preprocessing steps
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month_name()
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period


    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(cleaned_tokens)

    df['cleaned_text'] = df['text'].apply(clean_text)
    
    
    return df
