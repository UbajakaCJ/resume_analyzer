import docx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import textract

def answer_with_ai_1(input_resume_file, questions):
    if input_resume_file.endswith('.pdf'):
        resume_text = textract.process(input_resume_file, method='pdfminer').decode()
    elif input_resume_file.endswith('.txt'):
        with open(input_resume_file, 'r') as file:
            resume_text =  file.read()
    elif input_resume_file.endswith('.docx'):
        doc = docx.Document(input_resume_file)
        resume_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

    else:
        raise ValueError("Unsupported file format. Use PDF, TXT, or DOCX")
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.9)
    answers = []
    for question in questions:
        template = '''You have been provided with the following resume. ANswer the question based on the information in the resume. Resume: {resume} Question: {question} Answer: '''
        prompt_template = PromptTemplate.from_template(template=template)
        chain = LLMChain(
            llm=llm, prompt=prompt_template
        )
    answer =  chain.invoke({'resume':resume_text, 'question':question})
    answers.append(answer)
    return answers