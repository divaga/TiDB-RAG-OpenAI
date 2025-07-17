# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) application built with Streamlit that allows users to upload documents, store them as vector embeddings in TiDB Serverless, and ask questions to get AI-powered answers using OpenAI's GPT models.

## Features

- **Document Upload**: Support for PDF and text files
- **Vector Storage**: Uses TiDB Serverless with vector embeddings
- **Semantic Search**: Cosine similarity search for relevant content
- **AI-Powered Answers**: OpenAI GPT-4 integration for natural language responses
- **Real-time Statistics**: Track uploaded documents and query history
- **Streamlit Cloud Ready**: Optimized for easy deployment

## Architecture

```
User Upload → Text Extraction → Chunking → Embedding → TiDB Storage
                                                          ↓
User Query → Query Embedding → Vector Search → Context Retrieval → GPT Answer
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- OpenAI API key
- TiDB Serverless account and database

### 2. TiDB Serverless Setup

1. Sign up for [TiDB Cloud](https://tidbcloud.com/)
2. Create a new TiDB Serverless cluster
3. Note down your connection details:
   - Host
   - Port (usually 4000)
   - Username
   - Password
   - Database name

### 3. Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-document-qa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

4. Run the application:
```bash
streamlit run app.py
```

### 4. Streamlit Cloud Deployment

1. Fork this repository to your GitHub account

2. Go to [Streamlit Cloud](https://share.streamlit.io/)

3. Connect your GitHub repository

4. Set up the following secrets in Streamlit Cloud:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `TIDB_HOST`: Your TiDB host
   - `TIDB_PORT`: Your TiDB port (usually 4000)
   - `TIDB_USER`: Your TiDB username
   - `TIDB_PASSWORD`: Your TiDB password
   - `TIDB_DATABASE`: Your TiDB database name

5. Deploy the app

## Usage

### 1. System Initialization

1. Open the application
2. In the sidebar, enter your OpenAI API key and TiDB credentials
3. Click "Initialize System"
4. Wait for successful initialization messages

### 2. Upload Documents

1. Use the "Upload Document" section
2. Select PDF or text files
3. Wait for processing completion
4. View statistics in the sidebar

### 3. Ask Questions

1. Enter your question in the "Ask a Question" section
2. Click "Get Answer"
3. View the AI-generated response and source documents

## Technical Details

### Database Schema

```sql
CREATE TABLE documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_filename (filename),
    INDEX idx_file_hash (file_hash)
);
```

### Key Components

- **Text Extraction**: PyMuPDF for PDF processing
- **Chunking**: Custom text chunking with overlap
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Storage**: TiDB VECTOR data type
- **Search**: Cosine similarity with TiDB's VEC_COSINE_DISTANCE
- **Answer Generation**: OpenAI GPT-4

### Configuration

- **Chunk Size**: 1000 tokens
- **Chunk Overlap**: 200 tokens
- **Top-K Retrieval**: 5 most similar chunks
- **Max Upload Size**: 200MB

## Error Handling

The application includes comprehensive error handling for:

- File upload failures
- Database connection issues
- OpenAI API rate limits
- Invalid file formats
- Duplicate document detection

## Performance Considerations

- Documents are deduplicated using SHA-256 hashing
- Embeddings are cached in the database
- Efficient vector search using TiDB's native vector operations
- Streamlit's caching for system initialization

## Limitations

- Maximum file size: 200MB
- Supported formats: PDF, TXT
- Vector dimension: 1536 (OpenAI embedding size)
- Database dependency: Requires TiDB Serverless

## Troubleshooting

### Common Issues

1. **OpenAI API Error**: Check your API key and rate limits
2. **Database Connection**: Verify TiDB credentials and network access
3. **File Upload Error**: Ensure file is not corrupted and under size limit
4. **No Results Found**: Upload more documents or refine your query

### Debug Mode

Set `debug=True` in the Streamlit configuration to see detailed error messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review TiDB and OpenAI documentation

## Acknowledgments

- OpenAI for the embedding and chat models
- TiDB for vector database capabilities
- Streamlit for the web application framework
- PyMuPDF for PDF processing
