
from flask import Flask, render_template, request, jsonify,send_file
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
import tempfile


app = Flask(__name__)
groq_api_key = "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"

vectorstore = None

def initialize_qa_system(pdf_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_file.save(temp_file.name)
    temp_file.close()
    global vectorstore
    
    loader=PyPDFLoader(temp_file.name)
    docs=loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents=text_splitter.split_documents(docs)
    
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(final_documents, embeddings)
    return vectorstore 
def question_and_answer(vectorstore,question):
    
    llm =ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """)
    
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=vectorstore.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({"input": question})
    processing_time=time.process_time() - start


    
    return  response,processing_time

@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        initialize_qa_system(file)
        # print(vectorstore)
        return jsonify({'message': 'System initialized successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global vectorstore
    if not vectorstore:
        return jsonify({'error': 'Please upload a PDF first'}), 400
    
    question = request.json.get('question')
    print(question)
    # response,processing_time=question_and_answer(vectorstore,question)

    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        response,processing_time=question_and_answer(vectorstore,question)
        print(response)


        return jsonify({
            'answer': response['answer'],
            'processing_time': processing_time
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8002)