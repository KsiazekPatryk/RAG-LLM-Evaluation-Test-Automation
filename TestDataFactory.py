from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangChainEmbeddingsWrapper
from ragas.testset import TestSetGenerator
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader

def test_dataCreation():

    llm=ChatOpenAI(model_name="gpt-4", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    embed = OpenAIEmbeddings()
    loader = DirectoryLoader(
        path= "YOUR_DOCUMENTS_PATH",
        glob= "**/*/.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    docs = loader.load()
    generate_embeddings = LangChainEmbeddingsWrapper(embed)
    generator = TestSetGenerator(llm=langchain_llm, embeddings_model = generate_embeddings)
    generator.generate_with_langchain_docs(docs,testset_size=20)


     
