class TestContainerDocumentService(DocumentService):
    """Document service with mock embeddings for testing"""

    def __init__(self):
        # Don't call super().__init__() to avoid Gemini API requirements
        self.embeddings = MockGeminiEmbeddings()


class TestContainerRAGService(RAGService):
    """RAG service with mock embeddings for testing"""

    def __init__(self, use_llm: bool = False):
        # Override to use mock document service and no LLM
        self.document_service = TestContainerDocumentService()
        self.use_llm = use_llm
        self.llm = None
        print("✅ RAG service initialized with mock embeddings (no LLM)")
