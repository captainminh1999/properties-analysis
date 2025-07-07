"""
Visualization utilities for property analytics.
Creates interactive charts, maps, and dashboards using Plotly and Folium.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PropertyVisualizationSuite:
    """Comprehensive visualization suite for property analytics"""
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize visualization suite
        
        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
        self.color_palette = px.colors.qualitative.Set3
    
    def create_price_trend_chart(self, df: pd.DataFrame, 
                                group_by: str = 'suburb',
                                date_col: str = 'date_listed',
                                price_col: str = 'price') -> go.Figure:
        """
        Create price trend line chart
        
        Args:
            df: DataFrame with property data
            group_by: Column to group by
            date_col: Date column name
            price_col: Price column name
            
        Returns:
            Plotly figure
        """
        # Prepare data
        df_trend = df.copy()
        df_trend[date_col] = pd.to_datetime(df_trend[date_col])
        df_trend = df_trend.dropna(subset=[date_col, price_col])
        
        # Group by date and calculate median prices
        df_monthly = df_trend.groupby([
            df_trend[date_col].dt.to_period('M'),
            group_by
        ])[price_col].median().reset_index()
        df_monthly[date_col] = df_monthly[date_col].dt.to_timestamp()
        
        # Create figure
        fig = px.line(
            df_monthly,
            x=date_col,
            y=price_col,
            color=group_by,
            title=f'Property Price Trends by {group_by.title()}',
            labels={price_col: 'Median Price ($)', date_col: 'Date'}
        )
        
        fig.update_layout(
            template=self.theme,
            hovermode='x unified',
            xaxis_title='Date',
            yaxis_title='Median Price ($)'
        )
        
        return fig
    
    def create_price_distribution_chart(self, df: pd.DataFrame,
                                      group_by: str = 'property_type',
                                      price_col: str = 'price') -> go.Figure:
        """
        Create price distribution box plot
        
        Args:
            df: DataFrame with property data
            group_by: Column to group by
            price_col: Price column name
            
        Returns:
            Plotly figure
        """
        fig = px.box(
            df,
            x=group_by,
            y=price_col,
            title=f'Price Distribution by {group_by.title()}',
            labels={price_col: 'Price ($)', group_by: group_by.title()}
        )
        
        fig.update_layout(
            template=self.theme,
            xaxis_title=group_by.title(),
            yaxis_title='Price ($)'
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame,
                           x_col: str,
                           y_col: str,
                           color_col: str = None,
                           size_col: str = None) -> go.Figure:
        """
        Create scatter plot for property relationships
        
        Args:
            df: DataFrame with property data
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Color grouping column
            size_col: Size column
            
        Returns:
            Plotly figure
        """
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=f'{y_col.title()} vs {x_col.title()}',
            labels={x_col: x_col.title(), y_col: y_col.title()}
        )
        
        fig.update_layout(
            template=self.theme,
            xaxis_title=x_col.title(),
            yaxis_title=y_col.title()
        )
        
        return fig
    
    def create_valuation_gauge(self, actual_price: float,
                              predicted_price: float,
                              property_address: str = "Property") -> go.Figure:
        """
        Create valuation gauge chart
        
        Args:
            actual_price: Actual property price
            predicted_price: Model predicted price
            property_address: Property identifier
            
        Returns:
            Plotly figure
        """
        # Calculate valuation percentage
        valuation_pct = ((actual_price / predicted_price) - 1) * 100
        
        # Determine color based on valuation
        if valuation_pct > 15:
            color = "red"
            status = "Overvalued"
        elif valuation_pct < -15:
            color = "green"
            status = "Undervalued"
        else:
            color = "yellow"
            status = "Fair Value"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = valuation_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Valuation Analysis<br>{property_address}"},
            delta = {'reference': 0, 'suffix': "%"},
            gauge = {
                'axis': {'range': [-50, 50]},
                'bar': {'color': color},
                'steps': [
                    {'range': [-50, -15], 'color': "lightgreen"},
                    {'range': [-15, 15], 'color': "lightyellow"},
                    {'range': [15, 50], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        
        fig.update_layout(
            template=self.theme,
            annotations=[
                dict(
                    x=0.5, y=0.1,
                    text=f"Status: {status}<br>Actual: ${actual_price:,.0f}<br>Predicted: ${predicted_price:,.0f}",
                    showarrow=False,
                    xanchor="center"
                )
            ]
        )
        
        return fig
    
    def create_kpi_dashboard(self, kpis: Dict) -> go.Figure:
        """
        Create KPI dashboard with multiple indicators
        
        Args:
            kpis: Dictionary with KPI values
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Median Price", "Median Rent", "Rental Yield", "Price Growth"),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Add indicators
        fig.add_trace(go.Indicator(
            mode = "number",
            value = kpis.get('median_price', 0),
            title = {"text": "Median Price"},
            number = {'prefix': "$", 'valueformat': ",.0f"}
        ), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode = "number",
            value = kpis.get('median_rent', 0),
            title = {"text": "Median Rent (weekly)"},
            number = {'prefix': "$", 'valueformat': ",.0f"}
        ), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = kpis.get('rental_yield', 0),
            title = {'text': "Rental Yield"},
            number = {'suffix': "%"},
            gauge = {'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [{'range': [0, 4], 'color': "lightgray"},
                             {'range': [4, 7], 'color': "gray"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75, 'value': 6}}
        ), row=2, col=1)
        
        fig.add_trace(go.Indicator(
            mode = "number+delta",
            value = kpis.get('price_growth_yoy', 0),
            title = {"text": "YoY Price Growth"},
            number = {'suffix': "%"},
            delta = {"reference": 5, "relative": False, "suffix": "% vs target"}
        ), row=2, col=2)
        
        fig.update_layout(
            template=self.theme,
            title="Property Market KPIs",
            height=600
        )
        
        return fig
    
    def create_property_map(self, df: pd.DataFrame,
                           lat_col: str = 'latitude',
                           lng_col: str = 'longitude',
                           price_col: str = 'price',
                           center_lat: float = -33.8688,
                           center_lng: float = 151.2093,
                           zoom: int = 10) -> folium.Map:
        """
        Create interactive property map using Folium
        
        Args:
            df: DataFrame with property data
            lat_col: Latitude column name
            lng_col: Longitude column name
            price_col: Price column name
            center_lat: Map center latitude
            center_lng: Map center longitude
            zoom: Initial zoom level
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        # Filter valid coordinates
        df_map = df.dropna(subset=[lat_col, lng_col, price_col])
        
        # Create color mapping based on price
        min_price = df_map[price_col].min()
        max_price = df_map[price_col].max()
        
        def get_color(price):
            normalized = (price - min_price) / (max_price - min_price)
            if normalized < 0.33:
                return 'green'
            elif normalized < 0.67:
                return 'orange'
            else:
                return 'red'
        
        # Add markers
        for idx, row in df_map.iterrows():
            popup_text = f"""
            <b>Price:</b> ${row[price_col]:,.0f}<br>
            <b>Type:</b> {row.get('property_type', 'N/A')}<br>
            <b>Bedrooms:</b> {row.get('bedrooms', 'N/A')}<br>
            <b>Suburb:</b> {row.get('suburb', 'N/A')}
            """
            
            folium.CircleMarker(
                location=[row[lat_col], row[lng_col]],
                radius=6,
                popup=popup_text,
                color=get_color(row[price_col]),
                fill=True,
                fillColor=get_color(row[price_col]),
                fillOpacity=0.6
            ).add_to(m)
        
        # Add a color legend
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Price Range</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> Low</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Medium</p>
        <p><i class="fa fa-circle" style="color:red"></i> High</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_heatmap(self, df: pd.DataFrame,
                      lat_col: str = 'latitude',
                      lng_col: str = 'longitude',
                      weight_col: str = 'price') -> folium.Map:
        """
        Create heat map of property density/prices
        
        Args:
            df: DataFrame with property data
            lat_col: Latitude column name
            lng_col: Longitude column name
            weight_col: Column to use for heatmap weights
            
        Returns:
            Folium map with heatmap
        """
        # Create base map
        center_lat = df[lat_col].mean()
        center_lng = df[lng_col].mean()
        
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Prepare heat data
        heat_data = []
        for idx, row in df.iterrows():
            if pd.notna(row[lat_col]) and pd.notna(row[lng_col]):
                weight = row[weight_col] if pd.notna(row[weight_col]) else 1
                heat_data.append([row[lat_col], row[lng_col], weight])
        
        # Add heatmap
        plugins.HeatMap(heat_data).add_to(m)
        
        return m


def create_comparison_chart(df_before: pd.DataFrame,
                           df_after: pd.DataFrame,
                           metric_col: str,
                           group_col: str = 'suburb') -> go.Figure:
    """
    Create before/after comparison chart
    
    Args:
        df_before: DataFrame with before data
        df_after: DataFrame with after data
        metric_col: Metric to compare
        group_col: Grouping column
        
    Returns:
        Plotly figure
    """
    # Calculate group metrics
    before_metrics = df_before.groupby(group_col)[metric_col].mean().reset_index()
    after_metrics = df_after.groupby(group_col)[metric_col].mean().reset_index()
    
    # Merge data
    comparison = before_metrics.merge(
        after_metrics, 
        on=group_col, 
        suffixes=('_before', '_after')
    )
    
    # Create figure
    fig = go.Figure()
    
    # Add before bars
    fig.add_trace(go.Bar(
        name='Before',
        x=comparison[group_col],
        y=comparison[f'{metric_col}_before'],
        marker_color='lightblue'
    ))
    
    # Add after bars
    fig.add_trace(go.Bar(
        name='After',
        x=comparison[group_col],
        y=comparison[f'{metric_col}_after'],
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title=f'{metric_col.title()} Comparison by {group_col.title()}',
        xaxis_title=group_col.title(),
        yaxis_title=metric_col.title(),
        barmode='group',
        template='plotly_white'
    )
    
    return fig


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
