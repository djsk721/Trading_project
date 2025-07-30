"""
채팅 세션 관리 서비스
"""
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional


class ChatManager:
    """채팅 세션 관리 클래스"""
    
    @staticmethod
    def get_chat_list() -> List[str]:
        """
        채팅 목록 가져오기
        
        Returns:
            List[str]: 채팅 세션 이름 리스트
        """
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
        return list(st.session_state.chat_sessions.keys())
    
    @staticmethod
    def add_new_chat() -> str:
        """
        새로운 채팅 세션 생성
        
        Returns:
            str: 생성된 채팅 세션 이름
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_name = f"주식분석_{timestamp}"
        
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
        
        st.session_state.chat_sessions[chat_name] = {
            'messages': [],
            'created_at': datetime.now(),
            'stock_data': None,
            'analysis_docs': [],
            'last_updated': datetime.now(),
            'analysis_count': 0
        }
        
        return chat_name
    
    @staticmethod
    def delete_chat(chat_name: str) -> bool:
        """
        채팅 세션 삭제
        
        Args:
            chat_name: 삭제할 채팅 세션 이름
        
        Returns:
            bool: 삭제 성공 여부
        """
        if chat_name not in st.session_state.get('chat_sessions', {}):
            return False
        
        try:
            del st.session_state.chat_sessions[chat_name]
            
            # 현재 채팅이 삭제된 경우 초기화
            if st.session_state.get('current_chat') == chat_name:
                remaining_chats = ChatManager.get_chat_list()
                st.session_state.current_chat = remaining_chats[0] if remaining_chats else None
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_current_chat() -> Optional[str]:
        """
        현재 활성 채팅 세션 가져오기
        
        Returns:
            Optional[str]: 현재 채팅 세션 이름
        """
        return st.session_state.get('current_chat')
    
    @staticmethod
    def set_current_chat(chat_name: str) -> bool:
        """
        현재 채팅 세션 설정
        
        Args:
            chat_name: 설정할 채팅 세션 이름
        
        Returns:
            bool: 설정 성공 여부
        """
        if chat_name not in st.session_state.get('chat_sessions', {}):
            return False
        
        st.session_state.current_chat = chat_name
        return True
    
    @staticmethod
    def get_chat_info(chat_name: str) -> Optional[Dict]:
        """
        채팅 세션 정보 가져오기
        
        Args:
            chat_name: 채팅 세션 이름
        
        Returns:
            Optional[Dict]: 채팅 세션 정보
        """
        chat_sessions = st.session_state.get('chat_sessions', {})
        return chat_sessions.get(chat_name)
    
    @staticmethod
    def update_chat_data(chat_name: str, stock_data=None, analysis_docs=None) -> bool:
        """
        채팅 세션 데이터 업데이트
        
        Args:
            chat_name: 채팅 세션 이름
            stock_data: 주식 데이터
            analysis_docs: 분석 문서들
        
        Returns:
            bool: 업데이트 성공 여부
        """
        if chat_name not in st.session_state.get('chat_sessions', {}):
            return False
        
        try:
            chat_session = st.session_state.chat_sessions[chat_name]
            
            if stock_data is not None:
                chat_session['stock_data'] = stock_data
            
            if analysis_docs is not None:
                chat_session['analysis_docs'] = analysis_docs
            
            chat_session['last_updated'] = datetime.now()
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def add_message(chat_name: str, message_type: str, content: str, metadata: Dict = None) -> bool:
        """
        채팅 메시지 추가
        
        Args:
            chat_name: 채팅 세션 이름
            message_type: 메시지 타입 ('user', 'assistant', 'system')
            content: 메시지 내용
            metadata: 추가 메타데이터
        
        Returns:
            bool: 추가 성공 여부
        """
        if chat_name not in st.session_state.get('chat_sessions', {}):
            return False
        
        try:
            message = {
                'type': message_type,
                'content': content,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            }
            
            st.session_state.chat_sessions[chat_name]['messages'].append(message)
            st.session_state.chat_sessions[chat_name]['last_updated'] = datetime.now()
            
            if message_type == 'assistant':
                st.session_state.chat_sessions[chat_name]['analysis_count'] += 1
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_messages(chat_name: str) -> List[Dict]:
        """
        채팅 메시지 목록 가져오기
        
        Args:
            chat_name: 채팅 세션 이름
        
        Returns:
            List[Dict]: 메시지 리스트
        """
        chat_info = ChatManager.get_chat_info(chat_name)
        return chat_info.get('messages', []) if chat_info else []
    
    @staticmethod
    def clear_messages(chat_name: str) -> bool:
        """
        채팅 메시지 지우기
        
        Args:
            chat_name: 채팅 세션 이름
        
        Returns:
            bool: 삭제 성공 여부
        """
        if chat_name not in st.session_state.get('chat_sessions', {}):
            return False
        
        try:
            st.session_state.chat_sessions[chat_name]['messages'] = []
            st.session_state.chat_sessions[chat_name]['analysis_count'] = 0
            st.session_state.chat_sessions[chat_name]['last_updated'] = datetime.now()
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_chat_statistics() -> Dict:
        """
        전체 채팅 통계 정보
        
        Returns:
            Dict: 통계 정보
        """
        chat_sessions = st.session_state.get('chat_sessions', {})
        
        total_chats = len(chat_sessions)
        total_messages = sum(len(chat.get('messages', [])) for chat in chat_sessions.values())
        total_analyses = sum(chat.get('analysis_count', 0) for chat in chat_sessions.values())
        
        # 가장 활성화된 채팅
        most_active_chat = None
        max_messages = 0
        
        for chat_name, chat_info in chat_sessions.items():
            message_count = len(chat_info.get('messages', []))
            if message_count > max_messages:
                max_messages = message_count
                most_active_chat = chat_name
        
        return {
            'total_chats': total_chats,
            'total_messages': total_messages,
            'total_analyses': total_analyses,
            'most_active_chat': most_active_chat,
            'max_messages': max_messages
        }
    
    @staticmethod
    def cleanup_old_chats(max_chats: int = 10, max_age_days: int = 7) -> int:
        """
        오래된 채팅 세션 정리
        
        Args:
            max_chats: 최대 유지할 채팅 수
            max_age_days: 최대 보관 일수
        
        Returns:
            int: 삭제된 채팅 수
        """
        chat_sessions = st.session_state.get('chat_sessions', {})
        
        if len(chat_sessions) <= max_chats:
            return 0
        
        # 날짜별로 정렬
        sorted_chats = sorted(
            chat_sessions.items(),
            key=lambda x: x[1].get('last_updated', x[1].get('created_at', datetime.min))
        )
        
        # 오래된 채팅 삭제
        deleted_count = 0
        current_time = datetime.now()
        
        for chat_name, chat_info in sorted_chats:
            # 최대 수 초과 시 삭제
            if len(chat_sessions) > max_chats:
                ChatManager.delete_chat(chat_name)
                deleted_count += 1
                continue
            
            # 오래된 채팅 삭제
            last_updated = chat_info.get('last_updated', chat_info.get('created_at'))
            if last_updated and (current_time - last_updated).days > max_age_days:
                ChatManager.delete_chat(chat_name)
                deleted_count += 1
        
        return deleted_count
    
    @staticmethod
    def export_chat_history(chat_name: str) -> Optional[str]:
        """
        채팅 히스토리 내보내기
        
        Args:
            chat_name: 채팅 세션 이름
        
        Returns:
            Optional[str]: 내보낸 채팅 히스토리 (텍스트)
        """
        chat_info = ChatManager.get_chat_info(chat_name)
        if not chat_info:
            return None
        
        try:
            lines = [
                f"=== {chat_name} 채팅 히스토리 ===",
                f"생성일: {chat_info.get('created_at', '알 수 없음')}",
                f"마지막 업데이트: {chat_info.get('last_updated', '알 수 없음')}",
                f"총 메시지 수: {len(chat_info.get('messages', []))}",
                f"분석 횟수: {chat_info.get('analysis_count', 0)}",
                "",
                "=== 메시지 내역 ==="
            ]
            
            for message in chat_info.get('messages', []):
                timestamp = message.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if message.get('timestamp') else '알 수 없음'
                msg_type = message.get('type', 'unknown')
                content = message.get('content', '')
                
                lines.append(f"\n[{timestamp}] {msg_type.upper()}")
                lines.append(content)
                lines.append("-" * 50)
            
            return "\n".join(lines)
            
        except Exception:
            return None 