from document_utils import chunk_data, load_document, print_embedding_cost

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    if index_name in pc.list_indexes().names():
        print(f"index {index_name} already exists. Loading embeddings ...", end="")
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
    else:
        print(f"Creating index {index_name} and embeddings ...", end="")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter")
        )
        vector_store = Pinecone.from_documents(
            chunks, embeddings, index_name=index_name
        )

    print("Ok")
    return vector_store


def delete_pinecone_index(index_name="all"):
    import pinecone

    pc = pinecone.Pinecone()
    indexes = pc.list_indexes().names() if index_name == "all" else [index_name]
    print(
        f'Deleting {"all indexes" if index_name == "all" else index_name}...', end=""
    )
    [pc.delete_index(index) for index in indexes]
    print("Ok")

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    vector_store = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_directory
    )
    return vector_store

def load_embeddings_chroma(persist_directory="./chroma.db"):
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embeding-3-small", dimensions=1536)
    vectore_store = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    return vectore_store

def get_vector_store(input_resume):
    data = load_document(input_resume)
    chunks = chunk_data(data, chunk_size=256)
    chunks = chunks[:12]
    print_embedding_cost(chunks)

    vector_store = create_embeddings_chroma(chunks)
    return vector_store