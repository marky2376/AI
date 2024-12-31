from torch import cuda, bfloat16
import transformers
import torch
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import UnstructuredFileLoader, WebBaseLoader
import os
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain.document_loaders import YoutubeAudioLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import RedirectResponse
from langserve import add_routes
#from langchain_core.retrievers import BaseRetriever
from langchain.schema import BaseRetriever
from typing import List, Dict
import requests
import time
import numpy as np
import ssl
from pydantic import BaseModel
from pydantic import Field
from huggingface_hub import login
login(token='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')
import json
import re
from datetime import datetime

ssl._create_default_https_context = ssl._create_unverified_context

# Define model ID and access token
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#hf_auth = "<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>"
hf_auth = os.environ.get('HF_TOKEN')
os.environ['FFMPEG_BINARY'] = '/usr/bin/ffmpeg'
os.environ['FFPROBE_BINARY'] = '/usr/bin/ffprobe'

# Determine the device to use (CUDA or CPU)
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

# Configure model quantization
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

# Load model configuration and model
model_config = transformers.AutoConfig.from_pretrained(model_id, token=hf_auth)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_auth,
)

# Set model to evaluation mode
model.eval()
print(f"Model loaded on {device}")

# Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_auth)

# Define a text generation pipeline
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    task="text-generation",
    temperature=0.1,
    max_new_tokens=512,
    repetition_penalty=1.1,
)

# Integrate with LangChain
llm = HuggingFacePipeline(pipeline=generate_text)

# Define the directory where files are uploaded

upload_directory = "/usr/projects/langface/datafiles/"

syslog_file = "/usr/projects/langface/datafiles/network.log"
parsed_syslog_file = os.path.join(upload_directory, "parsed_syslog.json")

# Parse syslog and save it in a structured JSON format
def parse_and_save_syslog(syslog_file, output_file):
    logs = []
    with open(syslog_file, 'r') as file:
        for line in file:
            # Updated regex to match logs with additional IP and identifiers
            match = re.match(r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+([\d\.]+)\s+[^:]+:(.*)', line)
            if match:
                timestamp_str, ip_address, message = match.groups()
                try:
                    # Parse timestamp, assume current year as year info is missing
                    timestamp = datetime.strptime(timestamp_str, '%b %d %H:%M:%S')
                    timestamp = timestamp.replace(year=datetime.now().year)  # Add current year to timestamp
                except ValueError:
                    print(f"Could not parse timestamp for log entry: {line}")
                    continue

                # Add the parsed log to the list
                logs.append({
                    "timestamp": timestamp.isoformat(),
                    "ip_address": ip_address,
                    "message": message.strip()
                })

    # Save the parsed logs to JSON
    with open(output_file, 'w') as f:
        json.dump(logs, f, indent=2)
    return logs

# Run parsing and save to upload_directory
logs = parse_and_save_syslog(syslog_file, parsed_syslog_file)
print(f"Parsed syslog saved to {parsed_syslog_file}")


# Document loading, file splitting, and embeddings initialization
all_documents = []

if not os.path.exists(upload_directory):
    print(f"The directory {upload_directory} does not exist.")
else:
    # Iterate over all files in the directory
    for filename in os.listdir(upload_directory):
        file_path = os.path.join(upload_directory, filename)
        if os.path.isfile(file_path):
            # Create an instance of UnstructuredFileLoader for each file
            loader = UnstructuredFileLoader(file_path=file_path)

            # Load documents from the file
            documents = loader.load()
            all_documents.extend(documents)

# Path to the text file containing URLs
urls_file_path = "/usr/projects/langface/urls.txt"

# Function to read URLs from a text file
def read_urls_from_file(file_path):
    with open(file_path, "r") as file:
        urls = [line.strip() for line in file.readlines()]
    return urls

# Read URLs from the file
urls = read_urls_from_file(urls_file_path)

# Separate YouTube URLs from Website URLs
youtube_urls = [url for url in urls if "youtube.com" in url or "youtu.be" in url]
website_urls = [url for url in urls if "youtube.com" not in url and "youtu.be" not in url]

# Process YouTube URLs
if youtube_urls:
    save_dir = "/usr/projects/langface/YouTube"
    loader = GenericLoader(YoutubeAudioLoader(youtube_urls, save_dir), OpenAIWhisperParser())
    ytdocs = loader.load()
    all_documents.extend(ytdocs)

# Process Website URLs using WebBaseLoader
for url in website_urls:
    web_loader = WebBaseLoader(url)
    web_docs = web_loader.load()
    all_documents.extend(web_docs)

# Extract text content from documents
def extract_text(documents):
    return [doc.page_content for doc in documents if hasattr(doc, "page_content")]

# Extract text from all documents
all_texts = extract_text(all_documents)

# Split docs
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"], chunk_size=250, chunk_overlap=100, keep_separator=False
)

chunk_list = text_splitter.create_documents(all_texts)

# Initialize the SentenceTransformer model
try:
    embeddings_model = SentenceTransformer("BAAI/bge-base-en")
    print("Embeddings model initialized successfully.")
except Exception as e:
    print(f"Error initializing embeddings model: {e}")

# Convert embeddings to the required format for langchain
class LangchainEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return [self.model.encode(text) for text in texts]

    def embed_query(self, text):
        return self.model.encode(text)

langchain_embeddings_model = LangchainEmbeddings(embeddings_model)

# Function to chunk syslog data with IP address in text for better relevance
def chunk_logs_with_ip(logs, chunk_size=10):
    chunks = []
    for i in range(0, len(logs), chunk_size):
        chunk = logs[i:i + chunk_size]
        # Concatenate IP addresses and messages in each chunk for stronger embedding context
        chunk_text = ' '.join([f"{log['ip_address']} {log['message']}" for log in chunk])
        chunks.append({"text": chunk_text, "metadata": chunk})
    return chunks

# Load and parse syslog data into chunks with IP included in text
log_chunks = chunk_logs_with_ip(logs)

# Generate embeddings for each enhanced chunk
for chunk in log_chunks:
    chunk["embedding"] = langchain_embeddings_model.embed_query(chunk["text"])


# Compute embeddings for all chunks
all_chunks = log_chunks  # Combine with existing chunks
all_text_chunks = [chunk.page_content for chunk in chunk_list]
all_embeddings = [chunk["embedding"] for chunk in all_chunks]

document_embeddings = langchain_embeddings_model.embed_documents(all_text_chunks)

syslog_texts = [chunk["text"] for chunk in log_chunks]
syslog_embeddings = [langchain_embeddings_model.embed_query(text) for text in syslog_texts]

# Combine text and embeddings into a list of tuples

combined_texts = all_text_chunks + syslog_texts
combined_embeddings = document_embeddings + syslog_embeddings

text_embeddings = list(zip(combined_texts, combined_embeddings))

class HybridRetriever(BaseRetriever):
    faiss_retriever: BaseRetriever
    keyword_filter: callable

    def _get_relevant_documents(self, query: str) -> List[Dict]:
        # Use FAISS to retrieve semantic search results
        semantic_docs = self.faiss_retriever.get_relevant_documents(query)

        # Use keyword-based filtering
        filtered_docs = self.keyword_filter(query, semantic_docs)

        return filtered_docs

def keyword_filter(query: str, documents: List[Dict]) -> List[Dict]:
    # Split query into individual keywords dynamically
    keywords = query.split()

    # Filter documents based on keywords
    filtered_docs = [
        doc for doc in documents
        if any(keyword.lower() in doc.page_content.lower() for keyword in keywords)
    ]

    return filtered_docs


# Initialize FAISS index with the combined data
faiss_index = FAISS.from_embeddings(text_embeddings=text_embeddings, embedding=langchain_embeddings_model)
print("FAISS index updated with syslog data including IP address context.")

# Save the FAISS index to disk
faiss_index.save_local("faiss_index.index")
print("FAISS index saved to faiss_index.index")

# Load the FAISS index
faiss_index = FAISS.load_local(
    "faiss_index.index",
    embeddings=langchain_embeddings_model,
    allow_dangerous_deserialization=True,
)

# Define your custom prompt

template = """
Use your local pre loaded context, answer the question. If the answer is not in local context, answer the question by using your own knowledge about the topic from the LLM:\n\nQuestion: {question}\n\nContext:\n{context}",:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

#retriever = faiss_index.as_retriever()
faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": 250})
#print(retriever.search_kwargs)

# Initialize the hybrid retriever
hybrid_retriever = HybridRetriever(
    faiss_retriever=faiss_retriever,
    keyword_filter=keyword_filter
)

# Initialize the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=hybrid_retriever,
     chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    },
    return_source_documents=False
)

# Function to generate output based on a query
def make_output(query):
    # Query the QA chain and extract the result
    answer = qa_chain(query)
    result = answer["result"]
    return result
# Function to modify the output by adding spaces between each word with a delay
def modify_output(input):
    # Iterate over each word in the input string
    for text in input.split():
        # Yield the word with an added space
        yield text + " "
        # Introduce a small delay between each word
        time.sleep(0.05)
