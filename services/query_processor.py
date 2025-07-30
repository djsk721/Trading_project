"""
쿼리 처리 서비스
"""
import pandas as pd
from typing import Tuple, List, Dict
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from config.settings import AppConfig
from services.model_loader import ModelLoader
from services.rag_system import RAGSystem
from models.technical_indicators import TechnicalIndicators
from utils.date_utils import KoreanBusinessDay


class QueryProcessor:
    """쿼리 처리 및 분석 클래스"""
    
    def __init__(self):
        """쿼리 프로세서 초기화"""
        self.llm_model = None
        self.analysis_chain = None
        self.prediction_chain = None
        self.strategy_chain = None
        self.rag_system = RAGSystem()
        
    def initialize_llm(self) -> bool:
        """
        LLM 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.llm_model = ModelLoader.load_llm_model()
            
            if self.llm_model:
                self._create_prompt_chains()
                return True
            return False
            
        except Exception as e:
            return False
    
    def _create_prompt_chains(self):
        """프롬프트 체인들 생성"""
        # 설정에서 프롬프트 템플릿 가져오기
        templates = AppConfig.PROMPT_TEMPLATES
        
        # 기본 분석용 체인
        analysis_prompt_config = templates["기본 분석"]
        self.analysis_chain = LLMChain(
            prompt=PromptTemplate(
                input_variables=analysis_prompt_config["input_variables"],
                template=analysis_prompt_config["template"]
            ),
            llm=self.llm_model
        )
        
        # 20일 예측용 체인
        prediction_prompt_config = templates["20일 예측"]
        self.prediction_chain = LLMChain(
            prompt=PromptTemplate(
                input_variables=prediction_prompt_config["input_variables"],
                template=prediction_prompt_config["template"]
            ),
            llm=self.llm_model
        )
        
        # 투자전략용 체인
        strategy_prompt_config = templates["투자전략"]
        self.strategy_chain = LLMChain(
            prompt=PromptTemplate(
                input_variables=strategy_prompt_config["input_variables"],
                template=strategy_prompt_config["template"]
            ),
            llm=self.llm_model
        )
    
    def setup_rag(self, documents: List[Document]) -> str:
        """
        RAG 시스템 설정
        
        Args:
            documents: 문서 리스트
        
        Returns:
            str: 설정 결과
        """
        return self.rag_system.initialize(documents)
    
    def extract_answer_only(self, response):
        """LLM 응답에서 답변 부분만 추출"""
        if not response:
            return response

        answer_markers = ["답변:", "답변", "Answer:", "Answer"]
        
        for marker in answer_markers:
            if marker in response:
                parts = response.split(marker, 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    break
        else:
            # 전체 응답을 그대로 사용
            answer = response

        # 감사 인사 제거
        thank_you_markers = ["감사합니다.", "감사합니다", "Thank you.", "Thank you", "끝", "끝."]
        for marker in thank_you_markers:
            if marker in answer:
                answer = answer.split(marker, 1)[0].strip()
                break

        return answer
    
    def process_query(self, query: str, analysis_type: str, stock_name: str, 
                     current_date: str, current_price: float, 
                     stock_data: pd.DataFrame = None) -> Tuple[str, List[Document]]:
        """
        쿼리 처리
        
        Args:
            query: 사용자 질문
            analysis_type: 분석 유형
            stock_name: 종목명
            current_date: 현재 날짜
            current_price: 현재 가격
            stock_data: 주식 데이터 (예측용)
        
        Returns:
            Tuple[str, List[Document]]: (결과, 참고 문서들)
        """
        try:
            # RAG 컨텍스트 생성
            context, retrieved_docs = self.rag_system.get_context(query)
            
            if analysis_type == "기본 분석" and self.analysis_chain:
                result = self.analysis_chain.run({
                    "stock_name": stock_name,
                    "query": query,
                    "context": context,
                    "current_date": current_date,
                    "current_price": current_price
                })
            
            elif analysis_type == "20일 예측" and self.prediction_chain and stock_data is not None:
                indicators = TechnicalIndicators.get_current_indicators(stock_data)
                date_table = KoreanBusinessDay.generate_date_table(current_date, 20)
                
                # 날짜 테이블을 문자열로 변환
                date_table_text = "\n".join([
                    f"- {item['actual_date']}" 
                    for item in date_table
                ])
                
                result = self.prediction_chain.run({
                    "stock_name": stock_name,
                    "context": context,
                    "current_date": current_date,
                    "current_price": current_price,
                    "date_table": date_table_text,
                    **indicators
                })
            
            elif analysis_type == "투자전략" and self.strategy_chain:
                result = self.strategy_chain.run({
                    "stock_name": stock_name,
                    "query": query,
                    "context": context,
                    "current_date": current_date,
                    "current_price": current_price
                })
                
            result = self.extract_answer_only(result)
            
            return result, retrieved_docs
                            
        except Exception as e:
            return f"분석을 수행할 수 없습니다: {e}", []  
 