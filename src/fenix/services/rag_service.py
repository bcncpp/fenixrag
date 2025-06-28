from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from ..common.logger import LoggingMixin
from ..config import settings
from .document_service import DocumentService


class RAGService(LoggingMixin):
    """RAG (Retrieval-Augmented Generation) service using Gemini"""

    def __init__(self, use_llm: bool = True):
        """Constructor.

        Args:
            use_llm (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: Except when the key is not valid.
        """        
        self.document_service = DocumentService()
        self.use_llm = use_llm

        if use_llm:
            if not settings.validate_api_key():
                raise ValueError("Valid Google API key required for LLM functionality")

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=settings.google_api_key,
                temperature=0.1,
            )

            # Create prompt template
            self.prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.

Context from relevant documents:
{context}

Question: {question}

Instructions:
- Answer the question based only on the information provided in the context
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but comprehensive
- Cite specific information from the context when possible

Answer:
""")
            self.log.info("RAG service initialized with Gemini LLM")
        else:
            self.llm = None
            self.log.info("RAG service initialized without LLM (context-only mode)")

    async def query(
        self,
        question: str,
        max_docs: int = 5,
        similarity_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """QUery the RAG.

        Args:
            question (str): Question to ask.
            max_docs (int, optional): Upper bound on retrieval. Defaults to 5.
            similarity_threshold (float | None, optional): Cosine similarity threshold. Defaults to None.
            filters (dict[str, Any] | None, optional): _description_. Defaults to None.

        Returns:
            dict[str, Any]: _description_
        """        
        try:
            # Retrieve relevant documents
            self.log.debug(f"Searching for documents relevant to: {question}")

            docs_with_scores = await self.document_service.similarity_search(
                query=question,
                limit=max_docs,
                similarity_threshold=similarity_threshold,
                filters=filters,
            )

            if not docs_with_scores:
                return {
                    "answer": "I couldn't find any relevant documents to answer your question.",
                    "sources": [],
                    "context": "",
                    "confidence": 0.0,
                }

            # Prepare context
            context_parts = []
            sources = []
            total_confidence = 0.0

            for doc, score in docs_with_scores:
                context_parts.append(f"Document: {doc.title or 'Untitled'}\nContent: {doc.content}")
                sources.append(
                    {
                        "id": str(doc.id),
                        "title": doc.title,
                        "source": doc.source,
                        "confidence": score,
                        "content_preview": doc.content[:200] + "..."
                        if len(doc.content) > 200
                        else doc.content,
                    }
                )
                total_confidence += score

            context = "\n\n".join(context_parts)
            avg_confidence = total_confidence / len(docs_with_scores)

            # Generate answer
            if self.use_llm and self.llm:
                self.log.debug("Generating answer with Gemini LLM")

                # Create the chain
                chain = (
                    {
                        "context": lambda x: context,
                        "question": RunnablePassthrough(),
                    }
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
                )

                answer = await chain.ainvoke(question)

            else:
                # Context-only mode
                answer = f"Based on the retrieved documents for '{question}':\n\n{context}"

            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "confidence": avg_confidence,
                "num_sources": len(sources),
            }

        except Exception as e:
            self.log.error(f"Error in RAG query: {e}")
            raise

    async def multi_query(self, questions: list[str], **kwargs) -> list[dict[str, Any]]:
        """Process multiple questions."""
        results = []
        for question in questions:
            try:
                result = await self.query(question, **kwargs)
                results.append(result)
            except Exception as e:
                self.log.error(f"Error processing question '{question}': {e}")
                results.append(
                    {
                        "answer": f"Error processing question: {str(e)}",
                        "sources": [],
                        "context": "",
                        "confidence": 0.0,
                    }
                )

        return results