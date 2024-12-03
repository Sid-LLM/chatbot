from huggingface_hub import login
login(token = "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")
from transformers import HfApiEngine, Tool, ReactJsonAgent
from langchain_core.vectorstores import VectorStore
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from datasets import load_dataset


class Data_processor():
    def __init__(self):
        self.db = load_dataset('csv', data_files='errors - Sheet1.csv')
        self.source_docs = [
            Document(
                page_content=f"error : {row['Error']} error_number: {row['Error No']} error_description: {row['Error Description']} reason: {row['Reasons']} points to check: {row['Points to check']} temporary correction step: {row['Temporary Correction steps']}",  # Combine question and answer
                metadata={"source" : "errors - Sheet1.csv"}
            )
            for row in self.db['train']
        ]
    
    def Text_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained("thenlper/gte-small"),
            chunk_size=200,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        
        docs_processed = []
        unique_texts = {}
        for doc in tqdm(self.source_docs):
            new_docs = text_splitter.split_documents([doc])
            for new_doc in new_docs:
                if new_doc.page_content not in unique_texts:
                    unique_texts[new_doc.page_content] = True
                    docs_processed.append(new_doc)
        return docs_processed
                    
                    
    def Embedder(self):
        docs_processed = self.Text_splitter()
        embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
        vectordb = FAISS.from_documents(
            documents=docs_processed,
            embedding=embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )
        return vectordb


class RAG_tool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=25,
        )

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)] 
        )
    


class Inference():
    def __init__(self):
        self.data_processor = Data_processor()
        self.__Model = "Qwen/Qwen2.5-72B-Instruct"
        self.vectordb = self.data_processor.Embedder()
        rag_tool = RAG_tool(self.vectordb)
        llm_engine = HfApiEngine(self.__Model)
        self.agent = ReactJsonAgent(tools=[rag_tool], llm_engine=llm_engine, max_iterations=4, verbose=2)
    
    
    def runner(self, query : str) -> str:
        return self.agent.run(query)
