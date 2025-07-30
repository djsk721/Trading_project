"""
차트 관리 UI 컴포넌트
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ChartManager:
    """차트 생성 및 관리 클래스"""
    
    @staticmethod
    def create_technical_dashboard(stock_data: pd.DataFrame, stock_name: str):
        """
        기술적 지표 대시보드 생성
        
        Args:
            stock_data: 주식 데이터
            stock_name: 종목명
        
        Returns:
            plotly.graph_objects.Figure: 생성된 차트
        """
        # Plotly 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['가격 및 이동평균', 'RSI(14)', 'MACD', '거래량'],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. 가격 및 이동평균
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['종가'], 
                      name='종가', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['SMA_5'], 
                      name='SMA(5)', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['SMA_20'], 
                      name='SMA(20)', line=dict(color='green', dash='dash')),
            row=1, col=1
        )
        
        # 2. RSI
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['RSI_14'], 
                      name='RSI(14)', line=dict(color='purple', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
        
        # 3. MACD
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MACD'], 
                      name='MACD', line=dict(color='blue', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['Signal'], 
                      name='Signal', line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        # 4. 거래량
        colors = ['red' if vol > avg else 'blue' for vol, avg in 
                  zip(stock_data['거래량'], stock_data['Volume_SMA_20'])]
        fig.add_trace(
            go.Bar(x=stock_data.index, y=stock_data['거래량'], 
                   name='거래량', marker_color=colors, opacity=0.7),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=f'{stock_name} 기술적 분석 대시보드',
            height=700,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_candlestick_chart(stock_data: pd.DataFrame, stock_name: str):
        """
        캔들스틱 차트 생성
        
        Args:
            stock_data: 주식 데이터
            stock_name: 종목명
        
        Returns:
            plotly.graph_objects.Figure: 생성된 차트
        """
        fig = go.Figure()
        
        # 캔들스틱 추가
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['시가'],
            high=stock_data['고가'],
            low=stock_data['저가'],
            close=stock_data['종가'],
            name='캔들스틱'
        ))
        
        # 볼린저 밴드 추가
        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['BB_Upper'],
            name='볼린저 상단', line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['BB_Lower'],
            name='볼린저 하단', line=dict(color='blue', dash='dash'),
            fill='tonexty', fillcolor='rgba(0,0,255,0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['BB_Middle'],
            name='볼린저 중앙', line=dict(color='orange', dash='dot')
        ))
        
        fig.update_layout(
            title=f'{stock_name} 캔들스틱 차트 (볼린저 밴드)',
            xaxis_title='날짜',
            yaxis_title='가격 (원)',
            height=600
        )
        
        return fig 