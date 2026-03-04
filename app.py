import os
import gc
import markdown
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
from PyPDF2 import PdfReader

# Updated LangChain imports for 2026 standards
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Use the new langchain-huggingface package (Fixes Deprecation Warning)
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

import torch

# 🔥 CRITICAL: Reduce CPU & Memory usage for Render Free Tier
torch.set_num_threads(1)

load_dotenv()
app = Flask(__name__)

# -------------------------
# GLOBAL VARIABLES
# -------------------------
vectorstore = None
rag_chain = None
chat_history_data = []
rubric_text = ""

# -------------------------
# 🔥 LOAD EMBEDDING MODEL ONCE (GLOBAL)
# -------------------------
# Loading this globally ensures it stays in memory and isn't re-downloaded 
# or re-initialized every time a user uploads a PDF.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# -------------------------
# DATA PROCESSING
# -------------------------
def extract_text_from_pdfs(pdf_files):
    full_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            full_text += (page.extract_text() or "") + "\n"
    return full_text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def build_faiss_index(chunks):
    # USE GLOBAL EMBEDDINGS (Fixes Memory Overflow / 502 Error)
    global embeddings 
    return FAISS.from_texts(chunks, embeddings)


# -------------------------
# PROMPTS
# -------------------------
question_normalizer = ChatPromptTemplate.from_template("""
Rephrase the following question to be a standalone question by using the Chat History for context.
Do NOT add new meaning or external context outside of what is mentioned in the history.

Chat History:
{chat_history}

Question:
{question}

Standalone Question:
""")

answer_prompt = ChatPromptTemplate.from_template("""
You are a document-grounded assistant.

INSTRUCTIONS:
1. Carefully read the provided context.
2. Extract sentences or explanations relevant to the question.
3. Answer ONLY using information found in the context.
4. Do NOT use outside knowledge.
5. Do NOT assume or guess.
6. If the context contains details related to the topic but not a direct answer, provide the related details.
7. ONLY if the context contains no relevant information at all, reply exactly:
   "The document does not specify this information."
8. STRUCTURE: Use Markdown formatting.
   - Use ### for Section Headings.
   - Use **bold** for key terms.
   - Use bullet points for lists.

Context:
{context}

Question:
{question}

Answer:
""")

# -------------------------
# LLM INITIALIZATION
# -------------------------
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

# -------------------------
# RAG CHAIN
# -------------------------
def build_rag_chain(retriever):
    return (
        RunnablePassthrough.assign(
            standalone_question=question_normalizer | llm | StrOutputParser()
        )
        | {
            "context": lambda x: "\n\n".join(
                d.page_content for d in retriever.invoke(x["standalone_question"])
            ),
            "question": lambda x: x["standalone_question"]
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )


# -------------------------
# ESSAY GRADING
# -------------------------
def grade_essay_logic(essay_content):
    prompt = f"""
You are a university-level academic examiner.

TASK:
- Use ONLY the rubric provided.
- Stay within rubric score limits.
- Justify each score clearly.

OUTPUT FORMAT:
- Each criterion with score (e.g., 28/40)
- 2–3 sentence justification
- Final Total
- Letter Grade

RUBRIC:
{rubric_text}

ESSAY:
{essay_content}
"""

    raw_grade = llm.invoke(prompt).content
    return markdown.markdown(raw_grade)


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("new_home.html")


@app.route("/upload_pdf")
def upload_pdf():
    return render_template("new_pdf_chat.html")


@app.route("/process", methods=["POST"])
def process_documents():
    global vectorstore, rag_chain, chat_history_data

    chat_history_data = []
    pdf_files = request.files.getlist("pdf_docs")

    if not pdf_files or pdf_files[0].filename == "":
        return redirect(url_for("home"))

    # 1️⃣ Extract
    raw_text = extract_text_from_pdfs(pdf_files)

    # 2️⃣ Chunk
    chunks = chunk_text(raw_text)

    # 3️⃣ Free memory immediately (Garbage Collection)
    del raw_text
    gc.collect()

    # 4️⃣ Build FAISS
    vectorstore = build_faiss_index(chunks)

    # 5️⃣ Free chunks memory
    del chunks
    gc.collect()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    rag_chain = build_rag_chain(retriever)

    return redirect(url_for("chat"))


@app.route("/chat", methods=["GET", "POST"])
def chat():
    global chat_history_data, rag_chain

    if request.method == "POST":
        if rag_chain is None:
            return "Please upload a PDF first.", 400

        question = request.form["user_question"]

        history_str = "\n".join(
            [f"{m['role']}: {m['content']}" for m in chat_history_data]
        )

        response = rag_chain.invoke({
            "question": question,
            "chat_history": history_str
        })

        formatted_response = markdown.markdown(response)

        chat_history_data.append({"role": "User", "content": question})
        chat_history_data.append({"role": "Assistant", "content": formatted_response})

    return render_template("new_chat.html", chat_history=chat_history_data)


@app.route("/essay_grading", methods=["GET", "POST"])
def essay_grading():
    global rubric_text

    result = None
    input_text = ""

    if request.method == "POST":

        if "essay_rubric" in request.form and not request.form.get("essay_text"):
            rubric_text = request.form["essay_rubric"]
            result = "✅ Rubric updated!"

        else:
            if "file" in request.files and request.files["file"].filename != "":
                reader = PdfReader(request.files["file"])
                input_text = "".join(p.extract_text() or "" for p in reader.pages)

            elif request.form.get("essay_text"):
                input_text = request.form["essay_text"]

            if not rubric_text:
                result = "❌ Please set a rubric first."

            elif input_text:
                result = grade_essay_logic(input_text)

            else:
                result = "❌ No essay content found."

    return render_template(
        "new_essay_grading.html",
        result=result,
        input_text=input_text,
        rubric=rubric_text
    )


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    # Ensure the app binds to the PORT provided by Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)