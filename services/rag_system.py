"""
RAG 시스템 서비스
"""
import streamlit as st
from typing import List, Tuple, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from kiwipiepy import Kiwi

from config.settings import AppConfig
from services.model_loader import ModelLoader


class RAGSystem:
    """RAG (Retrieval-Augmented Generation) 시스템 클래스"""
    
    def __init__(self):
        """RAG 시스템 초기화"""
        self.embedding_model = None
        self.ensemble_retriever = None
        self.kiwi = None
        self.documents = []
        self.config = AppConfig.RAG_CONFIG
    
    def initialize(self, documents: List[Document]) -> str:
        """
        RAG 시스템 초기화
        
        Args:
            documents: 검색에 사용할 문서 리스트
        
        Returns:
            str: 초기화 결과 메시지
        """
        try:
            self.documents = documents
            
            if not documents:
                return "문서가 없어 RAG 시스템을 초기화할 수 없습니다."
            
            # 임베딩 모델 로드
            self.embedding_model = ModelLoader.load_embedding_model()
            if not self.embedding_model:
                return "임베딩 모델 로드 실패"
            
            # Kiwi 토크나이저 초기화
            try:
                self.kiwi = Kiwi()
            except Exception as e:
                st.warning(f"Kiwi 토크나이저 초기화 실패: {e}")
                self.kiwi = None
            
            # 검색기 구축
            retriever_info = self._build_retrievers(documents)
            
            return retriever_info
            
        except Exception as e:
            st.error(f"RAG 시스템 초기화 실패: {e}")
            return "초기화 실패"
    
    def _build_retrievers(self, documents: List[Document]) -> str:
        """
        검색기들 구축
        
        Args:
            documents: 문서 리스트
        
        Returns:
            str: 구축된 검색기 정보
        """
        retrievers = []
        retriever_names = []
        
        # BM25 검색기 구축
        bm25_retriever = self._build_bm25_retriever(documents)
        if bm25_retriever:
            retrievers.append(bm25_retriever)
            retriever_names.append("BM25")
        
        # FAISS 벡터 검색기 구축
        faiss_retriever = self._build_faiss_retriever(documents)
        if faiss_retriever:
            retrievers.append(faiss_retriever)
            retriever_names.append("FAISS")
        
        # 앙상블 검색기 구축
        if len(retrievers) > 1:
            weights = [self.config['BM25_WEIGHT'], self.config['FAISS_WEIGHT']][:len(retrievers)]
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=weights
            )
            return f"앙상블 ({' + '.join(retriever_names)})"
        elif len(retrievers) == 1:
            self.ensemble_retriever = retrievers[0]
            return retriever_names[0]
        else:
            return "검색기 구축 실패"
    
    def _build_bm25_retriever(self, documents: List[Document]) -> Optional[BM25Retriever]:
        """
        BM25 검색기 구축
        
        Args:
            documents: 문서 리스트
        
        Returns:
            Optional[BM25Retriever]: BM25 검색기
        """
        try:
            bm25_retriever = BM25Retriever.from_documents(
                documents,
                preprocess_func=self._kiwi_tokenize if self.kiwi else None
            )
            bm25_retriever.k = self.config['BM25_K']
            return bm25_retriever
            
        except Exception as e:
            st.warning(f"BM25 검색기 구축 실패: {e}")
            return None
    
    def _build_faiss_retriever(self, documents: List[Document]) -> Optional:
        """
        FAISS 벡터 검색기 구축
        
        Args:
            documents: 문서 리스트
        
        Returns:
            Optional: FAISS 검색기
        """
        try:
            if not self.embedding_model:
                return None
            
            vectorstore = FAISS.from_documents(documents, self.embedding_model)
            faiss_retriever = vectorstore.as_retriever(
                search_kwargs={"k": self.config['FAISS_K']}
            )
            return faiss_retriever
            
        except Exception as e:
            st.error(f"FAISS 벡터 스토어 구축 실패: {e}")
            return None
    
    def _kiwi_tokenize(self, text: str) -> List[str]:
        """
        Kiwi 토크나이저를 사용한 텍스트 토큰화
        
        Args:
            text: 토큰화할 텍스트
        
        Returns:
            List[str]: 토큰 리스트
        """
        if not self.kiwi:
            return text.split()
        
        try:
            return [token.form for token in self.kiwi.tokenize(text)]
        except Exception:
            return text.split()
    
    def get_context(self, query: str, max_docs: int = None, max_length: int = None) -> Tuple[str, List[Document]]:
        """
        스마트 컨텍스트 생성
        
        Args:
            query: 검색 쿼리
            max_docs: 최대 문서 수 (기본값: 설정에서 가져옴)
            max_length: 최대 컨텍스트 길이 (기본값: 설정에서 가져옴)
        
        Returns:
            Tuple[str, List[Document]]: (컨텍스트, 사용된 문서들)
        """
        if not self.ensemble_retriever:
            return "검색기를 사용할 수 없습니다.", []
        
        if max_docs is None:
            max_docs = self.config['MAX_DOCS']
        if max_length is None:
            max_length = self.config['MAX_CONTEXT_LENGTH']
        
        try:
            # 문서 검색
            retrieved_docs = self.ensemble_retriever.invoke(query)
            
            # 문서 다양성 확보 및 선택
            selected_docs = self._select_diverse_documents(retrieved_docs, max_docs)
            
            # 컨텍스트 생성
            context, used_docs = self._build_context(selected_docs, max_length)
            
            return context, used_docs
            
        except Exception as e:
            st.error(f"컨텍스트 생성 오류: {e}")
            return "컨텍스트를 생성할 수 없습니다.", []
    
    def _select_diverse_documents(self, retrieved_docs: List[Document], max_docs: int) -> List[Document]:
        """
        문서 다양성을 고려한 문서 선택
        
        Args:
            retrieved_docs: 검색된 문서들
            max_docs: 최대 선택할 문서 수
        
        Returns:
            List[Document]: 선택된 문서들
        """
        if not retrieved_docs:
            return []
        
        # 문서 타입별 분류
        doc_types = {}
        selected_docs = []
        max_per_type = self.config['MAX_DOC_TYPES_PER_TYPE']
        
        # 검색된 문서를 타입별로 분류하면서 선택
        for doc in retrieved_docs[:max_docs * 2]:  # 여유있게 검색
            doc_type = doc.metadata.get('type', 'unknown')
            
            if doc_types.get(doc_type, 0) < max_per_type:
                selected_docs.append(doc)
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                if len(selected_docs) >= max_docs:
                    break
        
        # 최신성 고려하여 정렬
        selected_docs.sort(key=lambda x: x.metadata.get('date', ''), reverse=True)
        
        return selected_docs
    
    def _build_context(self, selected_docs: List[Document], max_length: int) -> Tuple[str, List[Document]]:
        """
        선택된 문서들로부터 컨텍스트 구축
        
        Args:
            selected_docs: 선택된 문서들
            max_length: 최대 컨텍스트 길이
        
        Returns:
            Tuple[str, List[Document]]: (컨텍스트, 실제 사용된 문서들)
        """
        context_parts = []
        used_docs = []
        current_length = 0
        
        for doc in selected_docs:
            content = doc.page_content
            content_length = len(content)
            
            # 길이 제한 확인
            if current_length + content_length <= max_length:
                context_parts.append(content)
                used_docs.append(doc)
                current_length += content_length + 2  # 구분자 고려
            else:
                # 남은 공간이 있으면 일부만 포함
                remaining_space = max_length - current_length
                if remaining_space > 100:  # 최소 100자는 남아야 의미있음
                    truncated_content = content[:remaining_space - 3] + "..."
                    context_parts.append(truncated_content)
                    used_docs.append(doc)
                break
        
        context = "\n\n".join(context_parts)
        return context.strip(), used_docs
    
    def get_retriever_statistics(self) -> dict:
        """
        검색기 통계 정보
        
        Returns:
            dict: 통계 정보
        """
        stats = {
            "total_documents": len(self.documents),
            "retriever_type": type(self.ensemble_retriever).__name__ if self.ensemble_retriever else "None",
            "embedding_model_loaded": self.embedding_model is not None,
            "kiwi_loaded": self.kiwi is not None
        }
        
        # 문서 타입별 분포
        if self.documents:
            doc_types = {}
            for doc in self.documents:
                doc_type = doc.metadata.get('type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            stats["document_types"] = doc_types
        
        return stats
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
        
        Returns:
            List[Document]: 검색된 문서들
        """
        if not self.ensemble_retriever:
            return []
        
        try:
            # k 값을 임시로 설정
            if hasattr(self.ensemble_retriever, 'k'):
                original_k = self.ensemble_retriever.k
                self.ensemble_retriever.k = k
            
            docs = self.ensemble_retriever.invoke(query)
            
            # 원래 k 값 복원
            if hasattr(self.ensemble_retriever, 'k'):
                self.ensemble_retriever.k = original_k
            
            return docs[:k]
            
        except Exception as e:
            st.error(f"문서 검색 오류: {e}")
            return []
    
    def update_documents(self, new_documents: List[Document]) -> str:
        """
        문서 업데이트
        
        Args:
            new_documents: 새로운 문서들
        
        Returns:
            str: 업데이트 결과
        """
        try:
            if not new_documents:
                return "업데이트할 문서가 없습니다."
            
            # 문서 업데이트
            self.documents = new_documents
            
            # 검색기 재구축
            retriever_info = self._build_retrievers(new_documents)
            
            return f"문서 업데이트 완료 ({len(new_documents)}개 문서, 검색기: {retriever_info})"
            
        except Exception as e:
            st.error(f"문서 업데이트 실패: {e}")
            return "업데이트 실패" 