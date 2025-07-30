"""
한국 영업일 계산 유틸리티
"""
from datetime import datetime, timedelta
from typing import List, Dict
import pytz


class KoreanBusinessDay:
    """한국 영업일 계산 유틸리티"""
    
    @staticmethod
    def get_korea_timezone():
        """한국 시간대 반환"""
        return pytz.timezone('Asia/Seoul')
    
    @staticmethod
    def get_business_days(start_date, num_days: int = 20) -> List[str]:
        """
        시작 날짜 이후의 영업일 리스트 반환 (한국 시간 기준)
        주말만 제외하고 계산 (공휴일은 별도 고려 필요시 추가)
        
        Args:
            start_date: 시작 날짜 (str 또는 datetime)
            num_days: 반환할 영업일 수
        
        Returns:
            List[str]: 영업일 날짜 리스트 (YYYY-MM-DD 형식)
        """
        korea_tz = KoreanBusinessDay.get_korea_timezone()
        
        # 시작 날짜를 한국 시간으로 변환
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # 한국 시간대 적용
        if start_date.tzinfo is None:
            start_date = korea_tz.localize(start_date)
        else:
            start_date = start_date.astimezone(korea_tz)
        
        business_days = []
        current_date = start_date + timedelta(days=1)  # 다음 날부터 시작
        
        while len(business_days) < num_days:
            # 주말 제외 (월요일=0, 일요일=6)
            if current_date.weekday() < 5:  # 월~금
                business_days.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        return business_days
    
    @staticmethod
    def generate_date_table(start_date, num_days: int = 20) -> List[Dict[str, str]]:
        """
        예측용 날짜 테이블 생성
        D+1, D+2 형태와 실제 날짜를 매핑
        
        Args:
            start_date: 시작 날짜
            num_days: 예측 일수
        
        Returns:
            List[Dict]: 날짜 매핑 정보
        """
        business_days = KoreanBusinessDay.get_business_days(start_date, num_days)
        
        date_mapping = []
        for i, date_str in enumerate(business_days, 1):
            date_mapping.append({
                'actual_date': date_str,
            })
        
        return date_mapping
    
    @staticmethod
    def is_business_day(date) -> bool:
        """
        주어진 날짜가 영업일인지 확인
        
        Args:
            date: 확인할 날짜 (str 또는 datetime)
        
        Returns:
            bool: 영업일 여부
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        # 주말 체크 (월요일=0, 일요일=6)
        return date.weekday() < 5
    
    @staticmethod
    def get_previous_business_day(date) -> str:
        """
        주어진 날짜의 이전 영업일 반환
        
        Args:
            date: 기준 날짜 (str 또는 datetime)
        
        Returns:
            str: 이전 영업일 (YYYY-MM-DD 형식)
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        current_date = date - timedelta(days=1)
        
        while not KoreanBusinessDay.is_business_day(current_date):
            current_date -= timedelta(days=1)
        
        return current_date.strftime('%Y-%m-%d')
    
    @staticmethod
    def get_next_business_day(date) -> str:
        """
        주어진 날짜의 다음 영업일 반환
        
        Args:
            date: 기준 날짜 (str 또는 datetime)
        
        Returns:
            str: 다음 영업일 (YYYY-MM-DD 형식)
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        current_date = date + timedelta(days=1)
        
        while not KoreanBusinessDay.is_business_day(current_date):
            current_date += timedelta(days=1)
        
        return current_date.strftime('%Y-%m-%d') 