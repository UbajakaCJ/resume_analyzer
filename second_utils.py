from vector_store_utils import get_vector_store


def ask_and_get_answer(vector_store, query):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k":3}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    answer = chain.invoke(query)
    return answer

def answer_with_ai_one_question(vector_store, question):
    answer = ask_and_get_answer(vector_store, question)
    return answer['result']

def answer_with_ai_2(input_resume, questions):
    vector_store = get_vector_store(input_resume)
    answers = list(
        map(lambda q: answer_with_ai_one_question(vector_store, q), questions)
    )
    return answers

def validate_with_ai_1(input_resume, questions, answers):
    vector_store = get_vector_store(input_resume)
    queries = list(
        map(
            lambda x: f"Given this question ```{x[0]}```, validate the correctness of this answer ```{x[1]}```. The answer will be considered correct even if incomplete or just slightly off, as long as the provided information is true",
            zip(questions, answers),
        )
    )
    validation_results = list(
        map(lambda query: answer_with_ai_one_question(vector_store, query), queries)
    )
    return validation_results