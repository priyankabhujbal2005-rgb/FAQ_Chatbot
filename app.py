import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# NLTK DATA
# =====================================

nltk.download("stopwords", quiet=True)

stop_words = set(
    stopwords.words("english")
)


# =====================================
# TEXT PREPROCESSING
# =====================================

def preprocess(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =====================================
# LOAD FAQ DATA
# =====================================

df = pd.read_csv("faq_data.csv")


# =====================================
# PREPROCESS FAQ QUESTIONS
# =====================================

df["processed_question"] = (
    df["question"].apply(preprocess)
)


# =====================================
# TF-IDF
# =====================================

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    df["processed_question"]
)


# =====================================
# FIND BEST FAQ MATCH
# =====================================

def get_answer(user_question):

    processed_question = preprocess(
        user_question
    )

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarities.argmax()

    best_score = similarities[
        0
    ][best_match_index]

    return (
        df.iloc[best_match_index]["answer"],
        best_score
    )


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="FAQ Assistant",
    page_icon="🤖",
    layout="centered"
)


# =====================================
# CUSTOM CSS
# =====================================

st.markdown(
    """
    <style>

    /* Main page */

    .main {
        padding-top: 2rem;
    }


    /* Header */

    .chat-header {
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }

    .chat-header h1 {
        font-size: 36px;
        margin-bottom: 5px;
    }

    .chat-header p {
        font-size: 16px;
        opacity: 0.7;
    }


    /* Chat messages */

    .user-message {
        background-color: #dbeafe;
        color: #111827;
        padding: 12px 16px;
        border-radius: 15px;
        margin: 10px 0;
        margin-left: 20%;
    }

    .bot-message {
        background-color: #e5e7eb;
        color: #111827;
        padding: 12px 16px;
        border-radius: 15px;
        margin: 10px 0;
        margin-right: 20%;
    }


    /* Footer */

    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 13px;
        opacity: 0.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🤖 FAQ Assistant")

    st.write(
        "A smart FAQ chatbot that uses "
        "NLP to find the most relevant answer."
    )

    st.divider()

    st.subheader("📚 FAQ Topics")

    st.write("• 📦 Orders")
    st.write("• 🚚 Delivery")
    st.write("• 💳 Payments")
    st.write("• 🔄 Returns")
    st.write("• 👤 Account")
    st.write("• 📞 Customer Support")
    st.metric(
        "📚 Available FAQs",
        len(df)
    )

    st.divider()

    st.subheader("ℹ️ How it works")

    st.write(
        "The chatbot uses NLTK preprocessing, "
        "TF-IDF and cosine similarity to match "
        "your question with the FAQ database."
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! 👋 How can I help you today?"
            }
        ]

        st.rerun()

# =====================================
# HEADER
# =====================================

st.title("🤖 FAQ Assistant")

st.caption(
    "Your smart assistant for product and service questions"
)


# =====================================
# CHAT HISTORY
# =====================================

if "messages" not in st.session_state:

    st.session_state.messages = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content":
            "Hello! 👋 How can I help you today?"
        }
    )


# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">
                👤 <b>You</b><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="bot-message">
                🤖 <b>Assistant</b><br>
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )
# =====================================
# SUGGESTED QUESTIONS
# =====================================

# =====================================
# SUGGESTED QUESTIONS
# =====================================

st.subheader("💡 Try asking")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📦 Track my order"):
        st.session_state.selected_question = "How can I track my order?"

with col2:
    if st.button("🔄 Return a product"):
        st.session_state.selected_question = "How can I return a product?"

with col3:
    if st.button("💳 Payment methods"):
        st.session_state.selected_question = "What payment methods are available?"
# =====================================
# USER INPUT
# =====================================

# =====================================
# CHAT INPUT
# =====================================

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

user_question = st.chat_input(
    "Type your question here..."
)

if st.session_state.selected_question:
    user_question = st.session_state.selected_question
    st.session_state.selected_question = ""

# =====================================
# PROCESS QUESTION
# =====================================

if user_question:

    # Add user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # Get answer using existing NLP logic

    answer, score = get_answer(
        user_question
    )


    # Confidence threshold
    if score < 0.25:

        answer = (
            "🤔 I'm not sure I understand that question.\n\n"
            "Please try asking about **orders, delivery, "
            "payments, returns, accounts, or customer support**."
        )

    # Add bot response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # Refresh page

    st.rerun()


# =====================================
# FOOTER
# =====================================

st.title("🤖 FAQ Assistant")

st.caption(
    "Your smart assistant for product and service questions"
)