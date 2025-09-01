import pinecone_datasets
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
# Load .env file
load_dotenv()

# Access environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")

def load_text_dataset(dataset_name, sample_size=None):
    """Load a dataset with text content (no pre-computed embeddings)"""
    print(f"Loading dataset: {dataset_name}")
    dataset = pinecone_datasets.load_dataset(dataset_name)
    print(f"Loaded {len(dataset.documents)} documents")

    # Sample if specified
    if sample_size and sample_size < len(dataset.documents):
        documents = dataset.documents.head(sample_size)
        print(f"Using {len(documents)} documents (sampled)")
        return documents
    else:
        return dataset.documents


def create_pinecone_index(pc, index_name, dimension, metric="cosine"):
    """Create a new Pinecone index"""
    # Check existing indexes
    print("Existing indexes:")
    for name in pc.list_indexes().names():
        print(f"  - {name}")

    # Delete existing index if it exists
    if index_name in pc.list_indexes().names():
        print(f"\nDeleting existing index: {index_name}")
        pc.delete_index(index_name)

    # Configure serverless spec
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    print("spec created.")

    print(f"\nCreating new index: {index_name}")
    pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
    print("index created.")


def generate_and_insert_embeddings(index, documents, model, batch_size=500):
    """Generate embeddings with a model and insert into Pinecone index"""
    print("Generating and inserting embeddings...")
    to_upsert = []

    for i, row in documents.iterrows():
        # Get the text content
        text = row["blob"]["text"]

        # Generate embedding
        vector = model.encode(text).tolist()

        to_upsert.append((str(row["id"]), vector, {"text": text}))

        if len(to_upsert) >= batch_size:
            index.upsert(vectors=to_upsert)
            print(f"Inserted batch of {len(to_upsert)} vectors")
            to_upsert = []

    # Insert remaining vectors
    if to_upsert:
        index.upsert(vectors=to_upsert)
        print(f"Inserted final batch of {len(to_upsert)} vectors")

    print("✅ Finished inserting embeddings")
    print(index.describe_index_stats())


def main():
    # Configuration    
    INDEX_NAME = "langchain-retrieval-augmentation-fast-wiki"

    # Available text datasets (no pre-computed embeddings):
    # - quora_all-MiniLM-L6-bm25 (has text content we can re-embed)
    # - wikipedia-simple-text-embedding-ada-002 (has text content we can re-embed)

    # DATASET_NAME = 'quora_all-MiniLM-L6-bm25'  # Change this to use different datasets
    DATASET_NAME = "wikipedia-simple-text-embedding-ada-002-100K"
    SAMPLE_SIZE = 100000  # Set to None to use all documents

    # Embedding model configuration
    EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
    # Alternative models:
    # - "sentence-transformers/all-MiniLM-L6-v2" (384 dimensions)
    # - "sentence-transformers/all-mpnet-base-v2" (768 dimensions)
    # - "sentence-transformers/all-mpnet-base-v2" (768 dimensions)

    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)

    # Load text dataset
    documents = load_text_dataset(DATASET_NAME, SAMPLE_SIZE)

    # Initialize the embedding model
    model = SentenceTransformer(EMBED_MODEL)
    print(f"Loaded embedding model: {EMBED_MODEL}")

    # Get model dimension
    sample_text = "This is a test sentence."
    sample_embedding = model.encode(sample_text)
    vector_dim = len(sample_embedding)
    print(f"Model produces {vector_dim}-dimensional vectors")

    # Create index
    create_pinecone_index(pc, INDEX_NAME, vector_dim)

    # Get the index
    index = pc.Index(INDEX_NAME)
    index.describe_index_stats()

    # Generate and insert embeddings
    generate_and_insert_embeddings(index, documents, model)


if __name__ == "__main__":
    main()
