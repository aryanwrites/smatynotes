import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os

load_dotenv()

# ---- FIREBASE SETUP ----
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firebase()
collection = db.collection("chat_history")

def save_message(role, content):
    collection.add({
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    })

def load_history():
    messages = collection.order_by("timestamp").stream()
    history = []
    for msg in messages:
        data = msg.to_dict()
        if data["role"] == "human":
            history.append(HumanMessage(content=data["content"]))
        else:
            history.append(AIMessage(content=data["content"]))
    return history

def clear_history():
    docs = collection.stream()
    for doc in docs:
        doc.reference.delete()

# ---- MODEL ----
@st.cache_resource
def init_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

model = init_model()

# ---- PROMPT TEMPLATE ----
chat_template = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly and helpful AI assistant. 
     - If the user greets you, respond warmly and ask how you can help.
     - If the user gives you information to remember, store it and confirm.
     - If the user asks for previously given information, retrieve it from chat history.
     - If not found, reply: 'Sorry, I don't have that information stored.'
     - If user asks to "give notes" , u have to give all the  key values that are been stored 
    """),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{query}")
])

# ---- UI ----
st.title("🧠 SmartNotes")

# Input
user_input = st.chat_input("Type a message...")

if user_input:
    st.chat_message("user").write(user_input)
    save_message("human", user_input)

    chat_history = load_history()
    prompt = chat_template.invoke({
        'chat_history': chat_history,
        'query': user_input
    })

    with st.spinner("Thinking..."):
        aimessage = model.invoke(prompt)

    st.chat_message("assistant").write(aimessage.content)
    save_message("ai", aimessage.content)