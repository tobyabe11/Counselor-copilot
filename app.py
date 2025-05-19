from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import openai
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Load environment variables for API key
openai.api_key = os.getenv("OPENAI_API_KEY")
print("HERE IS YOUR OPENAI",os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r"C:\Users\thobi\mental_health_assistant\testapp\train.csv").fillna("")
df["Combined"] = df["Context"] + " " + df["Response"]

# Sentiment analysis for insights
df['Sentiment'] = df['Response'].apply(lambda x: TextBlob(x).sentiment.polarity)
avg_sentiment = df['Sentiment'].mean()
num_responses = df['Response'].apply(lambda x: len(x.strip()) > 0).sum()

# Vectorize for semantic matching
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Combined'])

# Logging files
LOG_FILE = "conversation_log.txt"


# Log interaction to both txt and Excel
def log_interaction(step, user_message, context, response, ai_summary, counselor_response=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a") as f:
        f.write(f"--- {timestamp} ({step}) ---\n")
        f.write(f"User Message: {user_message}\n")
        f.write(f"Matched Context: {context}\n")
        f.write(f"Reference Response: {response}\n")
        f.write(f"AI Copilot Summary: {ai_summary}\n")
        if counselor_response:
            f.write(f"Counselor Response: {counselor_response}\n")
        f.write("\n")

# Generate AI Summary from matched context

def get_ai_summary(user_message, matched_context, reference_response):
    prompt = f"""
You are a counselor's assistant AI. A patient has sent the following message:
"{user_message}"

Here is a similar past context and response from the dataset:
Context: "{matched_context}"
Response: "{reference_response}"

Summarize how this situation was handled and provide suggestions for deeper triaging or exploration before the counselor responds.
"""
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return completion.choices[0].message['content']
    except Exception as e:
        return f"AI summary unavailable: {str(e)}"

# Generate empathetic triage questions

def get_triage_questions(user_message):
    prompt = f"""
You are an empathetic mental health triage assistant. Based on the patient's message below, generate the next 1 most relevant and empathetic follow-up questions a counselor should ask:

"{user_message}"
"""
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8
        )
        return completion.choices[0].message['content']
    except Exception as e:
        return f"Triage questions unavailable: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html', num_responses=num_responses, avg_sentiment=avg_sentiment)

@app.route('/copilot_summary', methods=['POST'])
def copilot_summary():
    user_message = request.form['message']
    user_vec = vectorizer.transform([user_message])
    similarities = cosine_similarity(user_vec, X)
    top_index = similarities.argmax()
    context = df.iloc[top_index]['Context']
    response = df.iloc[top_index]['Response']

    if similarities[0][top_index] < 0.3 or response.strip() == "":
        ai_summary = "No strong context match found. Counselor may initiate AI triage."
        log_interaction("Step 2 - No Match", user_message, context, response, ai_summary)
        return jsonify({
            'matched_context': context,
            'reference_response': response,
            'copilot_summary': ai_summary,
            'no_match': True
        })

    ai_summary = get_ai_summary(user_message, context, response)
    log_interaction("Step 2 - Match", user_message, context, response, ai_summary)
    return jsonify({
        'matched_context': context,
        'reference_response': response,
        'copilot_summary': ai_summary,
        'no_match': False
    })

@app.route('/triage', methods=['POST'])
def triage():
    user_message = request.form['message']
    questions = get_triage_questions(user_message)
    log_interaction("Step 3 - AI Triage", user_message, "", "", questions)
    return jsonify({'triage_questions': questions})

@app.route('/send_response', methods=['POST'])
def send_response():
    user_message = request.form['message']
    counselor_response = request.form['counselor_response']
    log_interaction("Step 4 - Counselor Reply", user_message, "", "", "", counselor_response)
    return jsonify({'status': 'Message sent to patient'})

if __name__ == '__main__':
    app.run(debug=True)
