from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
load_dotenv()



llm =ChatOpenAI(model="gpt-4o")

result=llm.invoke("what is square root of 49")
print(result.content)



# messages=[SystemMessage("You are a social media expert"),HumanMessage("give me technique for engaging content")]
# result=llm.invoke(messages)
# print(result.content)
