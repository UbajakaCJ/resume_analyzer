def load_document(file):
    import os

    name, extension = os.path.splitext(file)

    print(f"\n\nLoading {file}")

    if extension == ".pdf":
        loader = get_pdf_loader(file)
    elif extension == ".docx":
        loader = get_docx_loader(file)
    else:
        print("Format is not supported")
        return None

    data = loader.load()
    return data


def get_pdf_loader(file):
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(file)

def get_docx_loader(file):
    from langchain_community.document_loaders import Docx2txtLoader

    return Docx2txtLoader(file)

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def print_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f"Total tokens: {total_tokens}")
    print(f"Embedding Cost in USD: {total_tokens / 10000 * 0.0004:.6f}")