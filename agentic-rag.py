from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from  langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

chunks =[
    "je m'applle Ibtyssam Abdellaoui , je suis etudiante en 4eme annee a l'ecole marocaine des sciences de l'ingenieur"
    " ,"
    " j'ai 22 ans et je suis passionnee par le developpement web et l'intelligence artificielle .",
     " j'ai realiser plusieurs projets dans ces domaines et je suis toujours a la recherche de nouvelles opportunites pour apprendre et grandir dans ce domaine ."
     " je suis une personne dynamique , curieuse et motivée qui aime relever les défis et travailler en equipe .",
     " je suis convaincue que mes compétences et mon enthousiasme pour le developpement web et l'intelligence artificielle me permettront de contribuer de manière significative a tout projet ou entreprise dans ce domaine ."   
]
embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_texts(
    texts= chunks,
    collection_name="cv_profile",
    embedding=embedding_model)

retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(retriever, name="cv_retriever", description="get information abt me from cv")




@tool
def get_Employee_info(name: str):
    """
    Get employee information {name , salary , seniority}
    """
    print("get_Employee_info tool invoked ")
    return {"name": name, "salary": 12000, "seniority": 5}


@tool
def send_email(email: str, subject: str, content: str):
    """
    send email with subject nd content 
    """
    print(f"Email sent to {email} with subject {subject} and content {content}")
    return f"Email sent to {email} with subject {subject} and content {content}"


llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(
    model=llm,
    tools=[send_email, get_Employee_info, retriever_tool],
    system_prompt="Answer to user query using provided TOOLS "
)

#resp = agent.invoke(
 #   {"messages": [HumanMessage(content="quels est le salaire de yassine ?")]}
#)
#print(resp["messages"][-1].content)

