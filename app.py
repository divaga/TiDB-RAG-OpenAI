import streamlit as st
import openai
import pymupdf
import os
import hashlib
import numpy as np
from typing import List, Dict, Any
import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime
import time
import io

# Configure page
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'documents_count' not in st.session_state:
    st.session_state.documents_count = 0
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = None
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

class RAGSystem:
    def __init__(self):
        self.openai_client = None
        self.db_connection = None
        self.embedding_model = "text-embedding-ada-002"
        self.chat_model = "gpt-4"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    def initialize_openai(self, api_key: str):
        """Initialize OpenAI client"""
        try:
            openai.api_key = api_key
            self.openai_client = openai
            # Test the connection
            self.openai_client.models.list()
            return True
        except Exception as e:
            st.error(f"OpenAI initialization failed: {str(e)}")
            return False
    
    def initialize_database(self, host: str, port: int, user: str, password: str, database: str):
        """Initialize TiDB connection and create table if needed"""
        try:
            self.db_connection = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                ssl_disabled=False,
                autocommit=True
            )
            
            # Create table if it doesn't exist
            cursor = self.db_connection.cursor()
            create_table_query = """
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(1536) NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_filename (filename),
                INDEX idx_file_hash (file_hash)
            )
            """
            cursor.execute(create_table_query)
            cursor.close()
            return True
        except Error as e:
            st.error(f"Database connection failed: {str(e)}")
            return False
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text content"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            st.error(f"PDF extraction failed: {str(e)}")
            return ""
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from text file"""
        try:
            # Reset file pointer to beginning
            txt_file.seek(0)
            text = txt_file.read().decode('utf-8')
            return text
        except Exception as e:
            st.error(f"Text extraction failed: {str(e)}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Embedding generation failed: {str(e)}")
            return []
    
    def check_document_exists(self, filename: str, file_hash: str) -> bool:
        """Check if document already exists in database"""
        try:
            cursor = self.db_connection.cursor()
            query = "SELECT COUNT(*) FROM documents WHERE filename = %s AND file_hash = %s"
            cursor.execute(query, (filename, file_hash))
            count = cursor.fetchone()[0]
            cursor.close()
            return count > 0
        except Error as e:
            st.error(f"Database query failed: {str(e)}")
            return False
    
    def store_document_chunks(self, filename: str, chunks: List[str], file_hash: str):
        """Store document chunks and embeddings in database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Delete existing chunks for this file (if re-uploading)
            delete_query = "DELETE FROM documents WHERE filename = %s"
            cursor.execute(delete_query, (filename,))
            
            # Insert new chunks
            insert_query = """
            INSERT INTO documents (filename, chunk_index, content, embedding, file_hash)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            for i, chunk in enumerate(chunks):
                embedding = self.generate_embedding(chunk)
                if embedding:
                    # Convert embedding to JSON string for storage
                    embedding_json = json.dumps(embedding)
                    cursor.execute(insert_query, (filename, i, chunk, embedding_json, file_hash))
            
            cursor.close()
            return True
        except Error as e:
            st.error(f"Database insertion failed: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity"""
        try:
            query_embedding = self.generate_embedding(query)
            if not query_embedding:
                return []
            
            cursor = self.db_connection.cursor()
            
            # Use TiDB's vector search with cosine similarity
            search_query = """
            SELECT filename, content, VEC_COSINE_DISTANCE(embedding, %s) as similarity
            FROM documents
            ORDER BY similarity ASC
            LIMIT %s
            """
            
            cursor.execute(search_query, (json.dumps(query_embedding), top_k))
            results = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    "filename": row[0],
                    "content": row[1],
                    "similarity": row[2]
                }
                for row in results
            ]
        except Error as e:
            st.error(f"Search failed: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using OpenAI GPT"""
        try:
            # Prepare context
            context = "\n\n".join([chunk["content"] for chunk in context_chunks])
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
            
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Answer generation failed: {str(e)}")
            return "Sorry, I couldn't generate an answer at this time."
    
    def get_document_count(self) -> int:
        """Get total number of documents in database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM documents")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Error as e:
            return 0

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    return RAGSystem()

def main():
    st.title("📚 RAG Document Q&A System")
    st.markdown("Upload documents and ask questions to get AI-powered answers!")
    
    # Sidebar for configuration and stats
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=os.getenv("OPENAI_API_KEY", "")
        )
        
        # Database configuration
        st.subheader("TiDB Configuration")
        db_host = st.text_input("Host", value=os.getenv("TIDB_HOST", ""))
        db_port = st.number_input("Port", value=int(os.getenv("TIDB_PORT", "4000")))
        db_user = st.text_input("User", value=os.getenv("TIDB_USER", ""))
        db_password = st.text_input("Password", type="password", value=os.getenv("TIDB_PASSWORD", ""))
        db_name = st.text_input("Database", value=os.getenv("TIDB_DATABASE", ""))
        
        # Initialize button
        if st.button("Initialize System"):
            rag = get_rag_system()
            
            if not openai_api_key:
                st.error("Please provide OpenAI API Key")
                return
            
            if not all([db_host, db_port, db_user, db_password, db_name]):
                st.error("Please provide all database credentials")
                return
            
            # Initialize OpenAI
            if rag.initialize_openai(openai_api_key):
                st.success("✅ OpenAI initialized")
                
                # Initialize database
                if rag.initialize_database(db_host, db_port, db_user, db_password, db_name):
                    st.success("✅ Database initialized")
                    st.session_state.db_initialized = True
                    st.session_state.documents_count = rag.get_document_count()
                else:
                    st.error("❌ Database initialization failed")
            else:
                st.error("❌ OpenAI initialization failed")
        
        # Stats
        st.header("📊 Statistics")
        st.metric("Documents Uploaded", st.session_state.documents_count)
        if st.session_state.last_query_time:
            st.metric("Last Query", st.session_state.last_query_time.strftime("%H:%M:%S"))
    
    # Check if system is initialized
    if not st.session_state.db_initialized:
        st.warning("Please initialize the system using the sidebar configuration.")
        return
    
    rag = get_rag_system()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt'],
            help="Upload PDF or text files to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    text = rag.extract_text_from_pdf(uploaded_file)
                else:
                    text = rag.extract_text_from_txt(uploaded_file)
                
                if text:
                    # Generate file hash
                    file_hash = rag.get_text_hash(text)
                    
                    # Check if document already exists
                    if rag.check_document_exists(uploaded_file.name, file_hash):
                        st.warning("Document already exists in the database!")
                    else:
                        # Chunk the text
                        chunks = rag.chunk_text(text)
                        
                        # Store chunks and embeddings
                        if rag.store_document_chunks(uploaded_file.name, chunks, file_hash):
                            st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                            st.info(f"Created {len(chunks)} chunks")
                            st.session_state.documents_count = rag.get_document_count()
                        else:
                            st.error("❌ Failed to store document")
                else:
                    st.error("❌ Failed to extract text from document")
    
    with col2:
        st.header("❓ Ask a Question")
        query = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?"
        )
        
        if st.button("Get Answer", type="primary"):
            if query:
                with st.spinner("Searching for relevant information..."):
                    # Search for similar chunks
                    similar_chunks = rag.search_similar_chunks(query, top_k=5)
                    
                    if similar_chunks:
                        with st.spinner("Generating answer..."):
                            # Generate answer
                            answer = rag.generate_answer(query, similar_chunks)
                            
                            # Display answer
                            st.subheader("Answer:")
                            st.write(answer)
                            
                            # Display sources
                            st.subheader("Sources:")
                            for i, chunk in enumerate(similar_chunks):
                                with st.expander(f"Source {i+1}: {chunk['filename']} (Similarity: {chunk['similarity']:.4f})"):
                                    st.write(chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'])
                            
                            st.session_state.last_query_time = datetime.now()
                    else:
                        st.warning("No relevant information found in the uploaded documents.")
            else:
                st.error("Please enter a question.")

if __name__ == "__main__":
    main()
