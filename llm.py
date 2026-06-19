from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings #임배딩
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

def get_ai_message(user_message):

    embedding = OpenAIEmbeddings(model='text-embedding-3-large') #임배딩 생성
    index_name = 'tax-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

    llm = ChatOpenAI(model='gpt-4o')
    prompt = ChatPromptTemplate.from_template("""
    [Identity]
    - 당신은 최고의 한국 소득세 전문가입니다.
    - Context를 참고하여 질문에 답변하세요.

    question:
    {question}

    Context:
    {context}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": itemgetter("query") | database.as_retriever() | format_docs,
            "question": itemgetter("query"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        사전: {dictionary}

        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})
    return ai_message