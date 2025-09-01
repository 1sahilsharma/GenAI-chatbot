import pinecone_datasets
from pinecone import Pinecone, ServerlessSpec


def load_pre_embedded_dataset(dataset_name):
    """Load a dataset with pre-computed embeddings"""
    print(f"Loading dataset: {dataset_name}")
    dataset = pinecone_datasets.load_dataset(dataset_name)
    print(f"Loaded {len(dataset.documents)} documents")

    # Check vector dimensions
    if len(dataset.documents) > 0:
        vector_dim = len(dataset.documents.iloc[0]["values"])
        print(f"Vector dimension: {vector_dim}")
        return dataset, vector_dim
    else:
        print("Warning: Dataset is empty!")
        return dataset, None


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


def insert_pre_embedded_vectors(index, dataset, batch_size=100):
    """Insert pre-computed embeddings into Pinecone index"""
    print("Inserting pre-computed embeddings...")
    to_upsert = []

    for i, row in dataset.documents.iterrows():
        # Use the pre-computed embedding from the dataset
        vector = row["values"].tolist()  # Convert numpy array to list
        text = row["blob"]["text"]  # Get text from blob

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
    PINECONE_API_KEY = (
        "pcsk_5KUdhm_UYZX7QHCspnrXPPh8SyoFcXw76wds37WBo7XgJWMkj5wMzQwV3XA1DC5QUs1xYZ"
    )
    INDEX_NAME = "langchain-retrieval-augmentation-fast"

    # Available datasets with pre-computed embeddings:
    # - quora_all-MiniLM-L6-bm25 (384 dimensions)
    # - quora_all-MiniLM-L6-bm25-100K (384 dimensions)
    # - wikipedia-simple-text-embedding-ada-002 (1536 dimensions)
    # - wikipedia-simple-text-embedding-ada-002-50K (1536 dimensions)
    # - wikipedia-simple-text-embedding-ada-002-100K (1536 dimensions)

    DATASET_NAME = "quora_all-MiniLM-L6-bm25"  # Change this to use different datasets

    # Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Load dataset and get vector dimension
    dataset, vector_dim = load_pre_embedded_dataset(DATASET_NAME)

    if vector_dim is None:
        print("Cannot proceed with empty dataset")
        return

    # Create index
    create_pinecone_index(pc, INDEX_NAME, vector_dim)

    # Get the index
    index = pc.Index(INDEX_NAME)
    index.describe_index_stats()

    # Insert vectors
    insert_pre_embedded_vectors(index, dataset)


if __name__ == "__main__":
    main()
