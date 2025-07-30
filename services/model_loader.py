"""
AI 모델 로딩 서비스
"""
import streamlit as st
import torch
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline

from config.settings import AppConfig


class ModelLoader:
    """AI 모델 로딩 관리 클래스"""
    
    @staticmethod
    @st.cache_resource
    def load_embedding_model() -> Optional[HuggingFaceEmbeddings]:
        """
        임베딩 모델 로드
        
        Returns:
            Optional[HuggingFaceEmbeddings]: 로드된 임베딩 모델
        """
        try:
            config = AppConfig.MODEL_CONFIG
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 한국어 특화 임베딩 모델 시도
            try:
                embedding_model = HuggingFaceEmbeddings(
                    model_name=config['EMBEDDING_MODEL'],
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': True},
                )
                st.success(f"한국어 임베딩 모델 로드 완료: {config['EMBEDDING_MODEL']}")
                return embedding_model
                
            except Exception as e:
                st.warning(f"한국어 임베딩 모델 로드 실패: {e}")
                
                # 폴백 모델 사용
                embedding_model = HuggingFaceEmbeddings(
                    model_name=config['FALLBACK_EMBEDDING'],
                    model_kwargs={'device': device},
                    encode_kwargs={'normalize_embeddings': True},
                )
                st.info(f"폴백 임베딩 모델 사용: {config['FALLBACK_EMBEDDING']}")
                return embedding_model
                
        except Exception as e:
            st.error(f"임베딩 모델 로드 실패: {e}")
            return None
    
    @staticmethod
    @st.cache_resource
    def load_llm_model() -> Optional[HuggingFacePipeline]:
        """
        LLM 모델 로드
        
        Returns:
            Optional[HuggingFacePipeline]: 로드된 LLM 모델
        """
        try:
            config = AppConfig.MODEL_CONFIG
            
            # 모델 이름
            model_name = config['LLM_MODEL']
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # 파이프라인 생성
            llm_pipeline = pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                max_new_tokens=config['MAX_NEW_TOKENS'],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=config['REPETITION_PENALTY'],
            )
            
            # HuggingFacePipeline 래퍼로 감싸기
            llm = HuggingFacePipeline(pipeline=llm_pipeline)
            
            st.success(f"LLM 모델 로드 완료: {model_name}")
            return llm
            
        except Exception as e:
            st.error(f"LLM 모델 로드 실패: {e}")
            return None
    
    @staticmethod
    def get_model_info() -> dict:
        """
        현재 로드된 모델 정보 반환
        
        Returns:
            dict: 모델 정보
        """
        config = AppConfig.MODEL_CONFIG
        
        # GPU 정보
        device_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        if torch.cuda.is_available():
            device_info["gpu_name"] = torch.cuda.get_device_name(0)
            device_info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        
        return {
            "embedding_model": config['EMBEDDING_MODEL'],
            "fallback_embedding": config['FALLBACK_EMBEDDING'],
            "llm_model": config['LLM_MODEL'],
            "max_new_tokens": config['MAX_NEW_TOKENS'],
            "repetition_penalty": config['REPETITION_PENALTY'],
            "device_info": device_info
        }
    
    @staticmethod
    def check_model_requirements() -> dict:
        """
        모델 요구사항 확인
        
        Returns:
            dict: 요구사항 확인 결과
        """
        results = {}
        
        # PyTorch 버전 확인
        results["pytorch_version"] = torch.__version__
        results["pytorch_ok"] = True
        
        # CUDA 확인
        if torch.cuda.is_available():
            results["cuda_version"] = torch.version.cuda
            results["cuda_ok"] = True
            
            # GPU 메모리 확인
            if torch.cuda.device_count() > 0:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                results["gpu_memory_gb"] = gpu_memory
                results["memory_sufficient"] = gpu_memory >= 8.0  # 최소 8GB 권장
            else:
                results["memory_sufficient"] = False
        else:
            results["cuda_ok"] = False
            results["memory_sufficient"] = False
        
        # Transformers 라이브러리 확인
        try:
            import transformers
            results["transformers_version"] = transformers.__version__
            results["transformers_ok"] = True
        except ImportError:
            results["transformers_ok"] = False
        
        # LangChain 확인
        try:
            import langchain
            results["langchain_version"] = langchain.__version__
            results["langchain_ok"] = True
        except ImportError:
            results["langchain_ok"] = False
        
        return results
    
    @staticmethod
    def optimize_model_settings() -> dict:
        """
        시스템 사양에 따른 모델 설정 최적화
        
        Returns:
            dict: 최적화된 설정
        """
        optimized_config = AppConfig.MODEL_CONFIG.copy()
        
        # GPU 메모리에 따른 최적화
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory < 8:
                # 메모리 부족 시 토큰 수 제한
                optimized_config['MAX_NEW_TOKENS'] = 1024
                optimized_config['recommended_batch_size'] = 1
            elif gpu_memory < 16:
                # 중간 사양
                optimized_config['MAX_NEW_TOKENS'] = 1536
                optimized_config['recommended_batch_size'] = 2
            else:
                # 고사양
                optimized_config['MAX_NEW_TOKENS'] = 2048
                optimized_config['recommended_batch_size'] = 4
        else:
            # CPU 전용
            optimized_config['MAX_NEW_TOKENS'] = 512
            optimized_config['recommended_batch_size'] = 1
            optimized_config['cpu_only'] = True
        
        return optimized_config
    
    @staticmethod
    def clear_model_cache():
        """모델 캐시 정리"""
        try:
            # Streamlit 캐시 정리
            st.cache_resource.clear()
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            st.success("모델 캐시가 정리되었습니다.")
            
        except Exception as e:
            st.error(f"캐시 정리 중 오류 발생: {e}")
    
    @staticmethod
    def get_memory_usage() -> dict:
        """
        현재 메모리 사용량 정보
        
        Returns:
            dict: 메모리 사용량 정보
        """
        memory_info = {}
        
        if torch.cuda.is_available():
            # GPU 메모리
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1e9
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1e9
            memory_info['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            memory_info['gpu_free'] = memory_info['gpu_total'] - memory_info['gpu_reserved']
        
        # 시스템 메모리 (가능한 경우)
        try:
            import psutil
            memory_info['system_memory_percent'] = psutil.virtual_memory().percent
            memory_info['system_memory_available'] = psutil.virtual_memory().available / 1e9
        except ImportError:
            pass
        
        return memory_info 