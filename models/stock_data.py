"""
주식 데이터 관리 모듈
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from typing import Optional, Dict, Tuple
from pykrx import stock
import yfinance as yf

from config.settings import AppConfig
from models.technical_indicators import TechnicalIndicators


class CurrencyConverter:
    """환율 변환 클래스"""
    
    @staticmethod
    @st.cache_data(ttl=3600)  # 1시간 캐싱
    def get_exchange_rate(from_currency: str = "USD", to_currency: str = "KRW") -> float:
        """
        환율 정보 가져오기
        
        Args:
            from_currency: 원본 통화 (기본값: USD)
            to_currency: 대상 통화 (기본값: KRW)
        
        Returns:
            float: 환율 (1 USD = ? KRW)
        """
        try:
            # yfinance를 사용해 환율 정보 가져오기
            ticker_symbol = f"{from_currency}{to_currency}=X"
            currency_data = yf.Ticker(ticker_symbol)
            hist = currency_data.history(period="1d")
            
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            else:
                # 기본 환율 (설정에서 가져옴)
                return AppConfig.CURRENCY_CONFIG["DEFAULT_EXCHANGE_RATE"]
                
        except Exception:
            # 오류 시 기본 환율 반환
            return AppConfig.CURRENCY_CONFIG["DEFAULT_EXCHANGE_RATE"]
    
    @staticmethod
    def convert_price(price: float, from_currency: str, to_currency: str) -> float:
        """
        가격 환율 변환
        
        Args:
            price: 변환할 가격
            from_currency: 원본 통화
            to_currency: 대상 통화
        
        Returns:
            float: 변환된 가격
        """
        if from_currency == to_currency:
            return price
        if from_currency == "USD" and to_currency == "KRW":
            exchange_rate = CurrencyConverter.get_exchange_rate("USD", "KRW")
            return price * exchange_rate
        elif from_currency == "KRW" and to_currency == "USD":
            exchange_rate = CurrencyConverter.get_exchange_rate("USD", "KRW")
            return price / exchange_rate
        else:
            return price
    
    @staticmethod
    def convert_dataframe(df: pd.DataFrame, from_currency: str, to_currency: str) -> pd.DataFrame:
        """
        데이터프레임의 가격 컬럼들을 환율 변환
        
        Args:
            df: 주식 데이터프레임
            from_currency: 원본 통화
            to_currency: 대상 통화
        
        Returns:
            pd.DataFrame: 환율 변환된 데이터프레임
        """
        if from_currency == to_currency:
            return df.copy()
        df_converted = df.copy()
        price_columns = ['시가', '고가', '저가', '종가']

        # 기본 가격 컬럼들 변환
        for col in price_columns:
            if col in df_converted.columns:
                df_converted[col] = df_converted[col].apply(
                    lambda x: CurrencyConverter.convert_price(x, from_currency, to_currency)
                )
        
        return df_converted


class StockDataManager:
    """주식 데이터 관리 클래스"""
    
    @staticmethod
    def is_korean_stock(ticker: str) -> bool:
        """
        한국 주식인지 확인
        
        Args:
            ticker: 종목 코드
            
        Returns:
            bool: 한국 주식 여부
        """
        # 한국 주식은 6자리 숫자
        if len(ticker) == 6 and ticker.isdigit():
            return True
        # yfinance 형태의 한국 주식 (.KS, .KQ 등)
        if ticker.endswith(('.KS', '.KQ')):
            return True
        return False
    
    @staticmethod
    def normalize_ticker(ticker: str) -> Tuple[str, bool]:
        """
        종목 코드 정규화 및 한국/해외 구분
        
        Args:
            ticker: 원본 종목 코드
            
        Returns:
            Tuple[str, bool]: (정규화된 코드, 한국주식 여부)
        """
        ticker = ticker.strip().upper()
        is_korean = StockDataManager.is_korean_stock(ticker)
        
        if is_korean:
            # .KS, .KQ 제거하여 6자리 숫자만 반환
            if '.' in ticker:
                ticker = ticker.split('.')[0]
        
        return ticker, is_korean
    
    @staticmethod
    @st.cache_data(ttl=AppConfig.CACHE_TTL["ALL_STOCKS"])
    def get_all_stocks(include_international: bool = False) -> Dict[str, str]:
        """
        전체 종목 리스트 가져오기 (캐싱 적용)
        
        Args:
            include_international: 해외 주식 포함 여부
        
        Returns:
            Dict[str, str]: {종목명: 종목코드} 딕셔너리
        """
        stock_dict = {}
        
        try:
            # 한국 주식 (pykrx)
            tickers = stock.get_market_ticker_list()
            max_stocks = AppConfig.DATA_CONFIG["MAX_STOCKS_DISPLAY"]
            
            for ticker in tickers[:max_stocks]:
                try:
                    name = stock.get_market_ticker_name(ticker)
                    if name and name.strip():
                        stock_dict[f"{name} ({ticker})"] = ticker
                except Exception:
                    continue
                    
        except Exception as e:
            st.error(f"한국 종목 리스트 로드 오류: {e}")
        
        # 해외 주식 추가
        if include_international and AppConfig.DATA_SOURCE_CONFIG["SUPPORT_INTERNATIONAL"]:
            stock_dict.update(AppConfig.POPULAR_INTERNATIONAL_STOCKS)
        
        return stock_dict if stock_dict else AppConfig.POPULAR_STOCKS.copy()
    
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
    def get_stock_data(ticker: str, days: int = None, target_currency: str = None) -> Optional[pd.DataFrame]:
        """
        주식 데이터 수집 (한국: pykrx, 해외: yfinance)
        
        Args:
            ticker: 종목 코드
            days: 수집할 데이터 일수 (기본값: 설정에서 가져옴)
            target_currency: 목표 통화 ("KRW" 또는 "USD", None이면 원본 통화 유지)
        
        Returns:
            Optional[pd.DataFrame]: 주식 데이터 (기술적 지표 포함)
        """
        if days is None:
            days = AppConfig.DATA_CONFIG["DEFAULT_DAYS"]
            
        normalized_ticker, is_korean = StockDataManager.normalize_ticker(ticker)
        
        try:
            if is_korean:
                # 한국 주식 - pykrx 사용
                today = datetime.today().strftime("%Y%m%d")
                start_date = (datetime.today() - BDay(days)).strftime("%Y%m%d")
                
                raw_data = stock.get_market_ohlcv(start_date, today, normalized_ticker, "d")
                
                if raw_data.empty:
                    st.warning(f"종목 {ticker}의 데이터를 찾을 수 없습니다.")
                    return None
                
                original_currency = "KRW"
                    
            else:
                # 해외 주식 - yfinance 사용
                end_date = datetime.today()
                start_date = end_date - timedelta(days=days)
                
                yf_ticker = yf.Ticker(normalized_ticker)
                raw_data = yf_ticker.history(start=start_date, end=end_date)
                
                if raw_data.empty:
                    st.warning(f"종목 {ticker}의 데이터를 찾을 수 없습니다.")
                    return None
                
                # yfinance 컬럼명을 pykrx 형식으로 변환
                raw_data = raw_data.rename(columns={
                    'Open': '시가',
                    'High': '고가', 
                    'Low': '저가',
                    'Close': '종가',
                    'Volume': '거래량'
                })
                
                # 필요한 컬럼만 선택
                raw_data = raw_data[['시가', '고가', '저가', '종가', '거래량']]
                original_currency = "USD"
            # 기술적 지표 계산
            stock_data = TechnicalIndicators.calculate_advanced_indicators(raw_data)
            # 환율 변환 (필요한 경우)
            if target_currency and target_currency != original_currency:
                stock_data = CurrencyConverter.convert_dataframe(
                    stock_data, original_currency, target_currency
                )
                # 메타데이터에 통화 정보 추가
                stock_data.attrs['currency'] = target_currency
                stock_data.attrs['original_currency'] = original_currency
            else:
                stock_data.attrs['currency'] = original_currency
                stock_data.attrs['original_currency'] = original_currency
            
            return stock_data
            
        except Exception as e:
            st.error(f"데이터 수집 오류: {e}")
            return None
    
    @staticmethod
    def get_stock_name(ticker: str) -> str:
        """
        종목명 가져오기 (한국: pykrx, 해외: yfinance)
        
        Args:
            ticker: 종목 코드
        
        Returns:
            str: 종목명
        """
        normalized_ticker, is_korean = StockDataManager.normalize_ticker(ticker)
        
        try:
            if is_korean:
                name = stock.get_market_ticker_name(normalized_ticker)
                return name if name else "알 수 없는 종목"
            else:
                yf_ticker = yf.Ticker(normalized_ticker)
                info = yf_ticker.info
                return info.get('shortName', info.get('longName', normalized_ticker))
                
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
        if not ticker:
            return False
            
        normalized_ticker, is_korean = StockDataManager.normalize_ticker(ticker)
        
        try:
            if is_korean:
                if len(normalized_ticker) != 6:
                    return False
                name = stock.get_market_ticker_name(normalized_ticker)
                return bool(name and name.strip())
            else:
                # 해외 주식 검증
                yf_ticker = yf.Ticker(normalized_ticker)
                info = yf_ticker.info
                return bool(info and 'symbol' in info)
                
        except Exception:
            return False
    
    @staticmethod
    def get_market_summary() -> Dict[str, any]:
        """
        시장 개요 정보 가져오기 (한국 시장만)
        
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
    def get_stock_basic_info(ticker: str, target_currency: str = None) -> Dict[str, any]:
        """
        종목 기본 정보 가져오기
        
        Args:
            ticker: 종목 코드
            target_currency: 목표 통화
        
        Returns:
            Dict: 종목 기본 정보
        """
        normalized_ticker, is_korean = StockDataManager.normalize_ticker(ticker)
        
        try:
            name = StockDataManager.get_stock_name(ticker)
            
            basic_info = {
                "name": name,
                "ticker": normalized_ticker,
                "is_korean": is_korean,
                "market": "Korean" if is_korean else "International"
            }
            
            if is_korean:
                # 한국 주식
                today = datetime.today().strftime("%Y%m%d")
                recent_data = stock.get_market_ohlcv(today, today, normalized_ticker, "d")
                original_currency = "KRW"
                
                if not recent_data.empty:
                    latest = recent_data.iloc[-1]
                    current_price = float(latest['종가'])
                    volume = int(latest['거래량'])
                    
                    # 환율 변환 (필요한 경우)
                    if target_currency and target_currency != original_currency:
                        current_price = CurrencyConverter.convert_price(
                            current_price, original_currency, target_currency
                        )
                    
                    basic_info.update({
                        "current_price": current_price,
                        "volume": volume,
                        "market_cap": current_price * volume if volume > 0 else 0,
                        "currency": target_currency if target_currency else original_currency
                    })
            else:
                # 해외 주식
                yf_ticker = yf.Ticker(normalized_ticker)
                recent_data = yf_ticker.history(period="1d")
                original_currency = "USD"
                
                if not recent_data.empty:
                    latest = recent_data.iloc[-1]
                    current_price = float(latest['Close'])
                    volume = int(latest['Volume'])
                    
                    # 환율 변환 (필요한 경우)
                    if target_currency and target_currency != original_currency:
                        current_price = CurrencyConverter.convert_price(
                            current_price, original_currency, target_currency
                        )
                    
                    basic_info.update({
                        "current_price": current_price,
                        "volume": volume,
                        "market_cap": 0,  # yfinance에서 별도 조회 필요
                        "currency": target_currency if target_currency else original_currency
                    })
            
            return basic_info
            
        except Exception as e:
            st.warning(f"종목 기본 정보를 가져올 수 없습니다: {e}")
            return {"name": "알 수 없는 종목", "ticker": ticker}
    
    @staticmethod
    def search_stocks(query: str, limit: int = 20, include_international: bool = False) -> Dict[str, str]:
        """
        종목 검색
        
        Args:
            query: 검색어
            limit: 결과 제한 수
            include_international: 해외 주식 포함 여부
        
        Returns:
            Dict[str, str]: 검색 결과 {종목명: 종목코드}
        """
        if not query or len(query.strip()) < 2:
            return {}
        
        try:
            all_stocks = StockDataManager.get_all_stocks(include_international)
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
            # 통화 정보 가져오기
            currency = getattr(df, 'attrs', {}).get('currency', 'KRW')
            currency_symbol = '원' if currency == 'KRW' else '$'
            
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
                "volatility_avg": float(df['Price_Change'].std()) if 'Price_Change' in df.columns else 0,
                "currency": currency,
                "currency_symbol": currency_symbol
            }
            
            return stats
            
        except Exception as e:
            st.error(f"통계 계산 오류: {e}")
            return {} 