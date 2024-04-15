from langchain_openai import ChatOpenAI

if __name__ == "__main__":
    llm = ChatOpenAI()

    r = llm.invoke("What do you know about TextWorld games?")
    print(r)