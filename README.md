**Mental Health Assistant – Counselor Copilot**

This is a Flask-based AI copilot application designed to assist mental health counselors by:
** Receiving and logging patient messages
** Matching them semantically with past context and responses
** Summarizing how similar cases were handled
** Suggesting AI-generated triage questions when no match is found
** Allowing counselors to respond manually or with AI assistance
** Logging all interactions to a text file for audit and analysis

**Features**

- Semantic search using sentence-transformers
- Sentiment analysis with TextBlob
- AI-powered summaries and triage via OpenAI GPT-4
- Simple HTML/CSS front-end
- Interaction logging to conversation_log.txt (no database)
- Deployable on AWS EC2 with NGINX + Gunicorn**

**Tech Stack**

- Python (Flask)
- OpenAI API
- Sentence Transformers (MiniLM)
- TextBlob
- HTML/CSS
- NGINX + Gunicorn (for server hosting)
- Amazon EC2 (deployment)

**Setup Instructions**

Clone the repo: git clone https://github.com/your-username/mental-health-assistant.git
cd mental-health-assistant

Create and activate virtual environment:

python3 -m venv .venv
source .venv/bin/activate

**Install dependencies:**

pip install -r requirements.txt
Create a .env file:
Add your OpenAI key: OPENAI_API_KEY=your_openai_key

Run the app:
python app.py

Deployment (Optional)
To deploy on AWS EC2 with NGINX and Gunicorn, follow the deployment guide here.

**📁 Logs**

All conversations are logged to:
- conversation_log.txt
This includes user messages, AI summaries, counselor responses, and triage suggestions.

**📌 Future Improvements**
Functional:
- Deeply personalized context with long term memory for individual patients
- Automated response generation using customized llm
- Inclusion of guardrails to protect counselors from liabilities
- Realtime respone reviewer system 

Non-Functional:
- Session management
- Data persistence
