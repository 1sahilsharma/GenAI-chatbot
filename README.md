# GenAI Project

AI-powered text processing and vector search with Pinecone, featuring both pre-embedded datasets and custom embedding generation.

## 🚀 Features

- **Vector Search**: Pinecone integration for similarity search
- **Multiple Embedding Models**: Support for various sentence transformer models
- **Pre-embedded Datasets**: Use existing datasets with pre-computed embeddings
- **Custom Embedding Generation**: Generate embeddings on-the-fly with custom models
- **LangChain Integration**: Seamless integration with LangChain ecosystem

## 📋 Prerequisites

- Python 3.12+
- UV package manager
- Pinecone API key
- OpenAI API key (optional)
- Groq API key (optional)

## 🛠️ Installation

### 1. Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Clone and Setup Project

```bash
git clone <your-repo-url>
cd genAI
uv init
uv sync
```

### 3. Install Dependencies

```bash
# Install production dependencies
make install

# Install development dependencies
make install-dev
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

### Pinecone Setup

1. Get your API key from [Pinecone Console](https://app.pinecone.io/)
2. Set the `PINECONE_API_KEY` environment variable
3. Choose your preferred cloud region (AWS us-east-1, us-west-2, etc.)

## 📁 Project Structure

```
genAI/
├── pinecone_pre_embedded.py      # Use pre-embedded datasets
├── pinecone_generate_embeddings.py # Generate embeddings with models
├── query.py                      # Query the vector database
├── requirements.txt              # Python dependencies
├── pyproject.toml               # UV project configuration
├── Makefile                     # Common commands
└── README.md                    # This file
```

## 🚀 Usage

### Using Pre-embedded Datasets

```bash
# Run with pre-computed embeddings
make run-pre-embedded

# Or directly
uv run python pinecone_pre_embedded.py
```

**Available Datasets:**
- `quora_all-MiniLM-L6-bm25` (384 dimensions)
- `wikipedia-simple-text-embedding-ada-002` (1536 dimensions)

### Generating Custom Embeddings

```bash
# Generate embeddings with custom model
make run-pinecone

# Or directly
uv run python pinecone_generate_embeddings.py
```

**Available Models:**
- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- `sentence-transformers/all-mpnet-base-v2` (768 dimensions)

### Querying the Database

```bash
# Query the vector database
make run-query

# Or directly
uv run python query.py
```

## 🛠️ Development

### Code Formatting

```bash
# Format code with black
make format

# Lint code with flake8
make lint
```

### Running Tests

```bash
# Run tests
make test
```

### Adding Dependencies

```bash
# Add production dependency
make add PKG=package_name

# Add development dependency
make add-dev PKG=package_name
```

### Cleaning Up

```bash
# Clean cache and temporary files
make clean
```

## 📊 Available Commands

Run `make help` to see all available commands:

```bash
make help
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're using the virtual environment
   ```bash
   source venv/bin/activate
   ```

2. **Pinecone Connection Issues**: Verify your API key and region settings

3. **Dimension Mismatch**: Ensure your index dimension matches your embedding model

4. **Memory Issues**: Reduce `SAMPLE_SIZE` in the scripts for large datasets

### Getting Help

- Check the Pinecone [documentation](https://docs.pinecone.io/)
- Review LangChain [guides](https://python.langchain.com/)
- Check the [sentence-transformers](https://www.sbert.net/) documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting section above
