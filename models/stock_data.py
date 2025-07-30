"""
주식 데이터 관리 모듈
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay
from typing import Optional, Dict
from pykrx import stock

from config.settings import AppConfig
from models.technical_indicators import TechnicalIndicators


class StockDataManager:
    """주식 데이터 관리 클래스"""
    
    @staticmethod
    @st.cache_data(ttl=AppConfig.CACHE_TTL["ALL_STOCKS"])
    def get_all_stocks() -> Dict[str, str]:
        """
        전체 종목 리스트 가져오기 (캐싱 적용)
        
        Returns:
            Dict[str, str]: {종목명: 종목코드} 딕셔너리
        """
        try:
            # KOSPI 종목 리스트
            tickers = stock.get_market_ticker_list()
            stock_dict = {}
            
            # 배치로 종목명 가져오기 (성능상 제한)
            max_stocks = AppConfig.DATA_CONFIG["MAX_STOCKS_DISPLAY"]
            for ticker in tickers[:max_stocks]:
                try:
                    name = stock.get_market_ticker_name(ticker)
                    if name and name.strip():
                        stock_dict[f"{name} ({ticker})"] = ticker
                except Exception:
                    continue
            
            return stock_dict
            
        except Exception as e:
            st.error(f"종목 리스트 로드 오류: {e}")
            # 기본 종목들 반환
            return AppConfig.POPULAR_STOCKS.copy()
    
    @staticmethod
    @st.cache_data(ttl=AppConfig.CACHE_TTL["POPULAR_STOCKS"])
    def get_popular_stocks() -> Dict[str, str]:
        """
        인기 종목 리스트 (빠른 로딩용)
        
        Returns:
            Dict[str, str]: {종목명: 종목코드} 딕셔너리
        """
        return AppConfig.POPULAR_STOCKS.copy()
    
    @staticmethod
    def get_stock_data(ticker: str, days: int = None) -> Optional[pd.DataFrame]:
        """
        주식 데이터 수집
        
        Args:
            ticker: 종목 코드
            days: 수집할 데이터 일수 (기본값: 설정에서 가져옴)
        
        Returns:
            Optional[pd.DataFrame]: 주식 데이터 (기술적 지표 포함)
        """
        if days is None:
            days = AppConfig.DATA_CONFIG["DEFAULT_DAYS"]
            
        try:
            today = datetime.today().strftime("%Y%m%d")
            start_date = (datetime.today() - BDay(days)).strftime("%Y%m%d")
            
            # 주식 데이터 수집
            raw_data = stock.get_market_ohlcv(start_date, today, ticker, "d")
            
            if raw_data.empty:
                st.warning(f"종목 {ticker}의 데이터를 찾을 수 없습니다.")
                return None
            
            # 기술적 지표 계산
            stock_data = TechnicalIndicators.calculate_advanced_indicators(raw_data)
            
            return stock_data
            
        except Exception as e:
            st.error(f"데이터 수집 오류: {e}")
            return None
    
    @staticmethod
    def get_stock_name(ticker: str) -> str:
        """
        종목명 가져오기
        
        Args:
            ticker: 종목 코드
        
        Returns:
            str: 종목명
        """
        try:
            name = stock.get_market_ticker_name(ticker)
            return name if name else "알 수 없는 종목"
        except Exception:
            return "알 수 없는 종목"
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """
        종목 코드 유효성 검증
        
        Args:
            ticker: 종목 코드
        
        Returns:
            bool: 유효 여부
        """
        if not ticker or len(ticker) != 6:
            return False
        
        try:
            # 종목명을 가져올 수 있는지 확인
            name = stock.get_market_ticker_name(ticker)
            return bool(name and name.strip())
        except Exception:
            return False
    
    @staticmethod
    def get_market_summary() -> Dict[str, any]:
        """
        시장 개요 정보 가져오기
        
        Returns:
            Dict: 시장 개요 정보
        """
        try:
            today = datetime.today().strftime("%Y%m%d")
            
            # KOSPI 지수 정보
            kospi_data = stock.get_index_ohlcv(today, today, "1001")  # KOSPI
            kosdaq_data = stock.get_index_ohlcv(today, today, "2001")  # KOSDAQ
            
            summary = {
                "date": today,
                "kospi": {},
                "kosdaq": {}
            }
            
            if not kospi_data.empty:
                kospi_latest = kospi_data.iloc[-1]
                summary["kospi"] = {
                    "close": float(kospi_latest['종가']),
                    "change": float(kospi_latest['종가'] - kospi_latest['시가']),
                    "change_pct": float((kospi_latest['종가'] - kospi_latest['시가']) / kospi_latest['시가'] * 100)
                }
            
            if not kosdaq_data.empty:
                kosdaq_latest = kosdaq_data.iloc[-1]
                summary["kosdaq"] = {
                    "close": float(kosdaq_latest['종가']),
                    "change": float(kosdaq_latest['종가'] - kosdaq_latest['시가']),
                    "change_pct": float((kosdaq_latest['종가'] - kosdaq_latest['시가']) / kosdaq_latest['시가'] * 100)
                }
            
            return summary
            
        except Exception as e:
            st.warning(f"시장 개요 정보를 가져올 수 없습니다: {e}")
            return {}
    
    @staticmethod
    def get_stock_basic_info(ticker: str) -> Dict[str, any]:
        """
        종목 기본 정보 가져오기
        
        Args:
            ticker: 종목 코드
        
        Returns:
            Dict: 종목 기본 정보
        """
        try:
            today = datetime.today().strftime("%Y%m%d")
            
            # 기본 정보
            name = StockDataManager.get_stock_name(ticker)
            
            # 최근 데이터
            recent_data = stock.get_market_ohlcv(today, today, ticker, "d")
            
            basic_info = {
                "name": name,
                "ticker": ticker,
                "market": "KOSPI" if ticker.startswith(('0', '1', '2', '3', '4', '5')) else "KOSDAQ"
            }
            
            if not recent_data.empty:
                latest = recent_data.iloc[-1]
                basic_info.update({
                    "current_price": float(latest['종가']),
                    "volume": int(latest['거래량']),
                    "market_cap": float(latest['종가'] * latest['거래량']) if latest['거래량'] > 0 else 0
                })
            
            return basic_info
            
        except Exception as e:
            st.warning(f"종목 기본 정보를 가져올 수 없습니다: {e}")
            return {"name": "알 수 없는 종목", "ticker": ticker}
    
    @staticmethod
    def search_stocks(query: str, limit: int = 20) -> Dict[str, str]:
        """
        종목 검색
        
        Args:
            query: 검색어
            limit: 결과 제한 수
        
        Returns:
            Dict[str, str]: 검색 결과 {종목명: 종목코드}
        """
        if not query or len(query.strip()) < 2:
            return {}
        
        try:
            all_stocks = StockDataManager.get_all_stocks()
            query_lower = query.lower().strip()
            
            # 검색어로 필터링
            filtered_stocks = {}
            count = 0
            
            for display_name, ticker in all_stocks.items():
                if count >= limit:
                    break
                    
                # 종목명 또는 종목코드에서 검색
                if (query_lower in display_name.lower() or 
                    query_lower in ticker):
                    filtered_stocks[display_name] = ticker
                    count += 1
            
            return filtered_stocks
            
        except Exception as e:
            st.error(f"종목 검색 오류: {e}")
            return {}
    
    @staticmethod
    def get_data_statistics(df: pd.DataFrame) -> Dict[str, any]:
        """
        데이터 통계 정보 계산
        
        Args:
            df: 주식 데이터프레임
        
        Returns:
            Dict: 통계 정보
        """
        if df.empty:
            return {}
        
        try:
            stats = {
                "period_days": len(df),
                "start_date": df.index[0].strftime('%Y-%m-%d'),
                "end_date": df.index[-1].strftime('%Y-%m-%d'),
                "highest_price": float(df['고가'].max()),
                "lowest_price": float(df['저가'].min()),
                "average_price": float(df['종가'].mean()),
                "price_range_pct": float((df['고가'].max() - df['저가'].min()) / df['저가'].min() * 100),
                "average_volume": int(df['거래량'].mean()),
                "max_volume": int(df['거래량'].max()),
                "total_trading_value": float((df['종가'] * df['거래량']).sum()),
                "missing_values": int(df.isnull().sum().sum()),
                "volatility_avg": float(df['Price_Change'].std()) if 'Price_Change' in df.columns else 0
            }
            
            return stats
            
        except Exception as e:
            st.error(f"통계 계산 오류: {e}")
            return {} 