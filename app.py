from flask import Flask, render_template, jsonify, request, session
from src.GENAIBOTENDTOEND.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.GENAIBOTENDTOEND.prompt import *
import os
import uuid
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'medical-chatbot-secret-key-2024'

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
groq_api_key = os.environ['GROQ_API_KEY']
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_embeddings()
index_name = "medical-chatbot" 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Enhanced prompt with conversation history support
system_prompt = """You are MediBot, a medical information assistant having a conversation with a user.

PREVIOUS CONVERSATION:
{chat_history}

MEDICAL CONTEXT FROM DATABASE:
{context}

CURRENT QUESTION: {input}

INSTRUCTIONS:
1. Use both the conversation history and medical context to provide accurate answers
2. Maintain context from previous messages - refer back when relevant
3. If the medical context doesn't contain the answer, say "I don't have specific information about this in my medical resources"
4. Always include: "Remember to consult healthcare professionals for personal medical advice"
5. For follow-up questions, build upon previous information
6. Keep responses clear, educational, and conversational

IMPORTANT: You are an AI assistant providing general medical information, not a healthcare provider.

RESPONSE:
"""

prompt = ChatPromptTemplate.from_template(system_prompt)

# Persistent storage in JSON file
CHAT_STORAGE_FILE = "chat_sessions.json"

def load_chat_sessions():
    """Load chat sessions from JSON file"""
    try:
        if os.path.exists(CHAT_STORAGE_FILE):
            with open(CHAT_STORAGE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading chat sessions: {e}")
    return {}

def save_chat_sessions(chat_sessions):
    """Save chat sessions to JSON file"""
    try:
        with open(CHAT_STORAGE_FILE, 'w') as f:
            json.dump(chat_sessions, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving chat sessions: {e}")

def get_user_sessions():
    """Get or initialize user sessions with persistent storage"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        print(f"New user session created: {session['user_id']}")
    
    user_id = session['user_id']
    chat_sessions = load_chat_sessions()
    
    if user_id not in chat_sessions:
        chat_sessions[user_id] = {}
        save_chat_sessions(chat_sessions)
        print(f"New user added to storage: {user_id}")
    
    return chat_sessions

def create_new_chat():
    """Create a new chat session"""
    chat_sessions = get_user_sessions()
    user_id = session['user_id']
    chat_id = str(uuid.uuid4())
    
    chat_data = {
        'id': chat_id,
        'title': 'New Chat',
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'messages': [],
        'message_count': 0
    }
    
    chat_sessions[user_id][chat_id] = chat_data
    save_chat_sessions(chat_sessions)
    
    print(f"New chat created: {chat_id} for user: {user_id}")
    return chat_id

def get_current_chat():
    """Get current active chat from persistent storage"""
    chat_sessions = get_user_sessions()
    user_id = session['user_id']
    
    # Ensure user exists in sessions
    if user_id not in chat_sessions:
        chat_sessions[user_id] = {}
        save_chat_sessions(chat_sessions)
    
    # Check if current chat exists and is valid
    if 'current_chat_id' not in session or session['current_chat_id'] not in chat_sessions[user_id]:
        # Create first chat
        chat_id = create_new_chat()
        session['current_chat_id'] = chat_id
        print(f"Setting current chat to: {chat_id}")
        return chat_sessions[user_id][chat_id]
    
    current_chat_id = session['current_chat_id']
    print(f"Returning current chat: {current_chat_id}")
    return chat_sessions[user_id][current_chat_id]

def update_chat_data(chat_data):
    """Update chat data in persistent storage"""
    chat_sessions = get_user_sessions()
    user_id = session['user_id']
    chat_id = chat_data['id']
    
    if user_id not in chat_sessions:
        chat_sessions[user_id] = {}
    
    chat_sessions[user_id][chat_id] = chat_data
    save_chat_sessions(chat_sessions)

def update_chat_title(chat_data, new_message):
    """Update chat title based on first message"""
    if chat_data['title'] == 'New Chat' and len(new_message) > 10:
        title = new_message[:30] + '...' if len(new_message) > 30 else new_message
        chat_data['title'] = title
        update_chat_data(chat_data)

def format_chat_history(messages):
    """Format chat history for the prompt - last 6 messages for context"""
    if not messages:
        return "No previous conversation."
    
    formatted = []
    # Get last 6 messages (3 user + 3 assistant pairs)
    recent_messages = messages[-6:]
    
    for msg in recent_messages:
        role = "Human" if msg['type'] == 'user' else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)

@app.route("/")
def index():
    try:
        # Initialize user session and current chat
        get_user_sessions()
        get_current_chat()
        return render_template('chat.html')
    except Exception as e:
        print(f"Error in index route: {e}")
        return "Error initializing chat. Please refresh the page.", 500

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"].strip()
        
        # Input validation
        if not msg:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        if len(msg) > 500:
            return jsonify({"error": "Message too long. Maximum 500 characters allowed."}), 400
        
        print("User query:", msg)
        
        # Get current chat
        current_chat = get_current_chat()
        
        # Check message limit
        if current_chat['message_count'] >= 10:
            return jsonify({"error": "Chat limit reached (10 messages). Please start a new chat."}), 400
        
        # Format chat history for context
        chat_history = format_chat_history(current_chat['messages'])
        print(f"Chat history length: {len(current_chat['messages'])} messages")
        
        # Create the chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Invoke the chain with chat history
        response = rag_chain.invoke({
            "input": msg,
            "chat_history": chat_history
        })
        
        answer = response["answer"]
        print("Response:", answer)
        
        # Save to chat history
        current_chat['messages'].append({
            'type': 'user',
            'content': msg,
            'timestamp': datetime.now().strftime("%H:%M")
        })
        
        current_chat['messages'].append({
            'type': 'assistant',
            'content': answer,
            'timestamp': datetime.now().strftime("%H:%M")
        })
        
        current_chat['message_count'] += 1
        
        # Update chat title if it's the first message
        if current_chat['message_count'] == 1:
            update_chat_title(current_chat, msg)
        
        # Update persistent storage
        update_chat_data(current_chat)
        
        return jsonify({
            "answer": answer,
            "message_count": current_chat['message_count'],
            "remaining_messages": 10 - current_chat['message_count']
        })
        
    except Exception as e:
        print(f"Error in chat route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route("/chats", methods=["GET"])
def get_chats():
    """Get all chat sessions for the user"""
    try:
        chat_sessions = get_user_sessions()
        user_id = session['user_id']
        chats = []
        
        if user_id in chat_sessions:
            for chat_id, chat_data in chat_sessions[user_id].items():
                chats.append({
                    'id': chat_id,
                    'title': chat_data.get('title', 'New Chat'),
                    'created_at': chat_data.get('created_at', 'Unknown'),
                    'message_count': chat_data.get('message_count', 0),
                    'is_active': session.get('current_chat_id') == chat_id
                })
        
        # Sort by creation time (newest first)
        chats.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({"chats": chats})
    except Exception as e:
        print(f"Error in get_chats route: {e}")
        return jsonify({"chats": []})

@app.route("/chats/new", methods=["POST"])
def new_chat():
    """Create a new chat session"""
    try:
        chat_id = create_new_chat()
        session['current_chat_id'] = chat_id
        
        return jsonify({
            "success": True,
            "chat_id": chat_id,
            "message": "New chat created"
        })
    except Exception as e:
        print(f"Error in new_chat route: {e}")
        return jsonify({"error": "Failed to create new chat"}), 500

@app.route("/chats/<chat_id>", methods=["POST"])
def switch_chat(chat_id):
    """Switch to a different chat"""
    try:
        chat_sessions = get_user_sessions()
        user_id = session['user_id']
        
        if user_id not in chat_sessions or chat_id not in chat_sessions[user_id]:
            return jsonify({"error": "Chat not found"}), 404
        
        session['current_chat_id'] = chat_id
        
        # Return the chat messages
        chat_data = chat_sessions[user_id][chat_id]
        return jsonify({
            "success": True,
            "messages": chat_data.get('messages', []),
            "title": chat_data.get('title', 'New Chat'),
            "message_count": chat_data.get('message_count', 0)
        })
    except Exception as e:
        print(f"Error in switch_chat route: {e}")
        return jsonify({"error": "Failed to switch chat"}), 500

@app.route("/chats/<chat_id>/delete", methods=["POST"])
def delete_chat(chat_id):
    """Delete a chat session"""
    try:
        chat_sessions = get_user_sessions()
        user_id = session['user_id']
        
        if user_id in chat_sessions and chat_id in chat_sessions[user_id]:
            # If deleting current chat, switch to another or create new
            if session.get('current_chat_id') == chat_id:
                del chat_sessions[user_id][chat_id]
                # Create new chat or switch to existing one
                if chat_sessions[user_id]:
                    session['current_chat_id'] = next(iter(chat_sessions[user_id]))
                else:
                    create_new_chat()
            else:
                del chat_sessions[user_id][chat_id]
            
            save_chat_sessions(chat_sessions)
            return jsonify({"success": True, "message": "Chat deleted"})
        
        return jsonify({"error": "Chat not found"}), 404
    except Exception as e:
        print(f"Error in delete_chat route: {e}")
        return jsonify({"error": "Failed to delete chat"}), 500

@app.route("/current-chat", methods=["GET"])
def get_current_chat_messages():
    """Get messages for current chat"""
    try:
        current_chat = get_current_chat()
        return jsonify({
            "messages": current_chat.get('messages', []),
            "title": current_chat.get('title', 'New Chat'),
            "message_count": current_chat.get('message_count', 0),
            "remaining_messages": 10 - current_chat.get('message_count', 0)
        })
    except Exception as e:
        print(f"Error in get_current_chat_messages route: {e}")
        return jsonify({
            "messages": [],
            "title": "New Chat",
            "message_count": 0,
            "remaining_messages": 10
        })

@app.route("/clear-sessions", methods=["POST"])
def clear_sessions():
    """Clear all sessions (for testing)"""
    try:
        if os.path.exists(CHAT_STORAGE_FILE):
            os.remove(CHAT_STORAGE_FILE)
        session.clear()
        return jsonify({"success": True, "message": "All sessions cleared"})
    except Exception as e:
        print(f"Error clearing sessions: {e}")
        return jsonify({"error": "Failed to clear sessions"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)