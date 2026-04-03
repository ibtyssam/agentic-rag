from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

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
    tools=[send_email, get_Employee_info],
    system_prompt="Answer to user query using provided TOOLS "
)

resp = agent.invoke(
    {"messages": [HumanMessage(content="quel est le salaire de yassine ?")]}
)
print(resp["messages"][-1].content)
