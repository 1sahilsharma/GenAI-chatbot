import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# connect to Pinecone
pc = Pinecone(api_key=pinecone_api_key)

INDEX_NAME = "langchain-retrieval-augmentation-fast-wiki"
index = pc.Index(INDEX_NAME)

# embeddings (using all-mpnet-base-v2 which produces 768-dimensional vectors)
embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
)

# wrap Pinecone index as LangChain vectorstore
vectorstore = LangchainPinecone.from_existing_index(
    index_name=INDEX_NAME, embedding=embed, text_key="text"
)
# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
)

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

# Run a sample query
query = "What is Game Theory?"
print("🔍 Query:", query)
print("🤖 Answer:", qa.invoke(query))


# # now query
# query = "What is Schizophrenia?"
# print(f"\nQuery: {query}")

# try:
#     docs = vectorstore.similarity_search(query, k=3)
#     print(f"Found {len(docs)} documents")
#     for i, d in enumerate(docs):
#         print(f"\nDocument {i+1}:")
#         content = d.page_content[:300]
#         print(content + "..." if len(d.page_content) > 300 else content)
# except Exception as e:
#     print(f"Error during search: {e}")
#     print("This is expected if the index is empty. "
#           "Run pinecone_script.py first to populate it.")
