import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
MODEL = "gpt-3.5-turbo"
open_api_key = "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"

# loading PDF, DOCX, and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None
    
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(chunks, embeddings)

# def ask_and_get_answer(vector_store, q, k=3):
#     from langchain.chains import RetrievalQA
#     from langchain_openai import ChatOpenAI


def ask_and_get_answer(vector_store, q, k=3):
    
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    
##



    system_template = r'''
    Use the following pieces of context to answer the user's question.
    Before answering, analyze the whole document for full context about the candidate's work history and skills.
    If you don't find the answer in the provided context, just respond "I don't know based on the provided data."
    ---------------
    Context: ```{context}```
    '''

    user_template = '''
    Question: ```{question}```
    '''

    messages= [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type='stuff',
        combine_docs_chain_kwargs={'prompt': qa_prompt },
        verbose=True
    )
##
  

    llm = ChatOpenAI(model=MODEL, temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,memory=memory,ConversationalRetrievalChain=crc)
  
    return chain.invoke(q)

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum(len(enc.encode(page.page_content)) for page in texts)
    return total_tokens, total_tokens / 1000 * 0.0004

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    st.image('img.png')
    st.subheader('Group 2 CV Analyzing Tool')
    with st.sidebar:
        if api_key := st.text_input('OpenAI API Key:', type='password'):
            os.environ['OPENAPI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a CV in PDF, docx or txt:', type= ['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding the CV ...'):
                bytes_data = uploaded_file.read()
                #file_name = os.path.join('/', uploaded_file.name)
                file_name = os.path.join('./files/', uploaded_file.name)
                with open(file_name,'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size:  {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success('File Uploaded, chunked and embedded successfully.')

    if q := st.text_input(
        'Ask a question about the contents of the uploaded document'
    ):
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store,q,k)
            st.text_area('LLM Answer: ', value=answer)

            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)