import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_FILES_DIRECTORY = 'C:\\Users\\HP\\OneDrive\\Documents\\chatwithpdf\\documents'

def configure_retriever(directory):
    docs = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(directory, file_name)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 8})
    
    return retriever

class SimpleChatMessageHistory:
    def __init__(self):
        self.messages = []

    def add_human_message(self, content):
        self.messages.append({"type": "human", "content": content})

    def add_ai_message(self, content):
        self.messages.append({"type": "ai", "content": content})

    def clear(self):
        self.messages = []

    def get_last_message(self):
        return self.messages[-1] if self.messages else None



def main():
    if not os.listdir(PDF_FILES_DIRECTORY):
        print("No PDF documents found in the specified directory.")
        return

    retriever = configure_retriever(PDF_FILES_DIRECTORY)

    msgs = SimpleChatMessageHistory()  # Use the new simple chat history manager
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    openai_api_key = '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'
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=False
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    print("LangChain Terminal Chat")
    print("Type 'exit' to quit the conversation.")

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break

        msgs.add_human_message(user_query)

        # Since the example given is simplified and doesn't execute the actual LangChain logic in terminal mode,
        # you'd need to implement the logic for processing the user_query and generating a response
        # Here's a placeholder for where that logic would go
        response = "This is a placeholder response. Implement query processing logic here."
        print("AI:", response)
        msgs.add_ai_message(response)  # Store AI response in history


if __name__ == "__main__":
    main()
