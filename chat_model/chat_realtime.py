from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
load_dotenv()



llm =ChatOpenAI(model="gpt-4o")

chathistory=[]

chathistory.append(SystemMessage(content="You are a helpful AI assistant"))

while True:
    query=input("You : ")
    message1=HumanMessage(content=query)
    if query.lower()=="exit":
        break
    chathistory.append(message1)
    result=llm.invoke(chathistory)
    print(result.content)
    chathistory.append(AIMessage(content=result.content))