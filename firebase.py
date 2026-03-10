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
cred = credentials.Certificate(os.getenv("FIREBASE_KEY_PATH"))
firebase_admin.initialize_app(cred)
db = firestore.client()
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
    print("🗑️ Chat history cleared!")

# ---- MODEL ----
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ---- PROMPT TEMPLATE ----
chat_template = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly and helpful AI assistant. 
     - If the user greets you, respond warmly and ask how you can help.
     - If the user gives you information to remember, store it and confirm.
     - If the user asks for previously given information, retrieve it from chat history.
     - If not found, reply: 'Sorry, I don't have that information stored.'
    """),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{query}")
])

# ---- MAIN LOOP ----
print("💬 Chat started! Type 'exit' to quit or 'clear' to clear history.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'clear':
        clear_history()
        continue

    chat_history = load_history()
    save_message("human", user_input)

    prompt = chat_template.invoke({
        'chat_history': chat_history,
        'query': user_input
    })

    aimessage = model.invoke(prompt)
    print("AI:", aimessage.content)

    save_message("ai", aimessage.content)