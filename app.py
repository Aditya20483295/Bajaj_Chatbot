import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from bajaj_src.prompt import system_prompt

from flask import Flask, render_template, request
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from bajaj_src.prompt import system_prompt

import os

# === 🔐 API KEYS ===
os.environ["OPENAI_API_KEY"] = "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"  # Replace with yours
os.environ["PINECONE_API_KEY"] = "pcsk_3LMnvH_LX1QnbDvoB8CvTDdc3GyK3BdquLqdEFNaWDzAsNXaGHqta2W5WrBUCqTsyAsjwq"   # Replace with yours

# === 🔍 Vector DB Setup ===
index_name = "bajaj-fs-main-20250712-121632"  # or whichever exists

embedding_model = OpenAIEmbeddings()
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# === 🤖 LLM + Chain ===
llm = ChatOpenAI(temperature=0.3)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),   # this includes {context}
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# === 🚀 Flask Setup ===
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    result = rag_chain.invoke({"input": user_input})
    return str(result["answer"])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7862)
