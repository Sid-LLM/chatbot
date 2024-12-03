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


class Data_Loader():
    def __init__(self):
        self.metadata_path = 'metadata.pkl'
        self.faiss_index_path = 'faiss_index'
        self.embedding_model = "thenlper/gte-small"

    def load_data(self):
        vectordb = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization = True)
        with open(metadata_path, "rb") as f:
            docs_processed = pickle.load(f)

        return vectordb, docs_processed


class RAG_tool(Tool):
    name = "Solar Cleaner Troubleshooter"
    description = '''This tool specializes in troubleshooting issues for solar panel robotic cleaners. It uses semantic similarity to retrieve relevant documents from the knowledge base containing details about common errors, their reasons, points to check, and corrective measures. 
    Designed specifically for robotic cleaners equipped with track changers and cleaning mechanisms, this tool ensures precise solutions to user queries.'''

    inputs = {
    "query": {
        "type": "string",
        "description": "Describe the issue or error related to solar panel robotic cleaners. For example: 'Track changer malfunctioning', 'Error 101', 'Cleaner not starting', etc. Use affirmative statements."
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
        self.data_loader = Data_Loader()
        self.__Model = "Qwen/Qwen2.5-72B-Instruct"
        self.vectordb, self.metadata = self.data_loader.load_data()
        rag_tool = RAG_tool(self.vectordb)
        llm_engine = HfApiEngine(self.__Model)
        self.agent = ReactJsonAgent(tools=[rag_tool], llm_engine=llm_engine, max_iterations=4, verbose=2)
    
    
    def runner(self, query : str) -> str:
        return self.agent.run(query)
    
    
