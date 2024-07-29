from flask import Flask, render_template, send_from_directory
import os
import main  # Import the main module to access fetch_stats
import preprocessor
app = Flask(__name__)

OUTPUT_FOLDER = 'Output'
USER = 'Overall'  # Change if you want to display stats for a specific user

@app.route('/')
def index():
    # Load chat data
    with open('WhatsApp Chat with BTD👑.txt', 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Preprocess data
    df = preprocessor.preprocess(data)

    # Fetch stats
    num_messages, words, num_media_messages, num_links = main.fetch_stats(USER, df)

    # List files
    files = [f for f in os.listdir(OUTPUT_FOLDER) if f != 'visualization.html']
    file_data = [{'name': f, 'title': f.replace('_', ' ').title()} for f in files]

    return render_template('index.html', 
                           total_words=words,
                           total_links=num_links,
                           num_messages=num_messages,
                           num_links=num_links,
                           files=file_data)

@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True,port=5000)
