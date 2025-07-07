"""
Monte Carlo simulation for property price forecasting.
Simulates future property prices using economic and demographic variables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MonteCarloPropertySimulation:
    """Monte Carlo simulation for property price forecasting"""
    
    def __init__(self, base_price: float, simulation_years: int = 5, num_simulations: int = 1000):
        """
        Initialize Monte Carlo simulation
        
        Args:
            base_price: Current property price
            simulation_years: Number of years to simulate
            num_simulations: Number of simulation runs
        """
        self.base_price = base_price
        self.simulation_years = simulation_years
        self.num_simulations = num_simulations
        self.simulation_results = None
        
        # Default parameter distributions (can be customized)
        self.parameter_distributions = {
            'interest_rate': {'type': 'normal', 'mean': 0.035, 'std': 0.015, 'min': 0.005, 'max': 0.10},
            'inflation': {'type': 'normal', 'mean': 0.025, 'std': 0.008, 'min': 0.005, 'max': 0.06},
            'population_growth': {'type': 'normal', 'mean': 0.015, 'std': 0.005, 'min': 0.005, 'max': 0.04},
            'income_growth': {'type': 'normal', 'mean': 0.03, 'std': 0.01, 'min': 0.01, 'max': 0.06},
            'unemployment_change': {'type': 'normal', 'mean': 0.0, 'std': 0.01, 'min': -0.02, 'max': 0.03},
            'construction_cost_growth': {'type': 'normal', 'mean': 0.035, 'std': 0.015, 'min': 0.01, 'max': 0.08}
        }
    
    def update_parameters(self, new_distributions: Dict):
        """Update parameter distributions for simulation"""
        self.parameter_distributions.update(new_distributions)
    
    def generate_parameter_values(self) -> Dict[str, np.ndarray]:
        """Generate random parameter values for all simulations"""
        parameter_values = {}
        
        for param_name, dist_config in self.parameter_distributions.items():
            if dist_config['type'] == 'normal':
                # Generate normal distribution with clipping
                values = np.random.normal(
                    dist_config['mean'], 
                    dist_config['std'], 
                    (self.num_simulations, self.simulation_years)
                )
                values = np.clip(values, dist_config['min'], dist_config['max'])
            elif dist_config['type'] == 'uniform':
                # Generate uniform distribution
                values = np.random.uniform(
                    dist_config['min'], 
                    dist_config['max'], 
                    (self.num_simulations, self.simulation_years)
                )
            else:
                # Default to normal if type not recognized
                values = np.random.normal(
                    0.02, 0.01, 
                    (self.num_simulations, self.simulation_years)
                )
            
            parameter_values[param_name] = values
        
        return parameter_values
    
    def calculate_price_impact(self, parameters: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate price impact based on economic parameters
        
        Args:
            parameters: Dictionary of parameter arrays
            
        Returns:
            Array of price multipliers for each simulation and year
        """
        # Base growth rate
        base_growth = 0.02  # 2% baseline appreciation
        
        # Calculate weighted impact of each parameter
        interest_impact = -0.8 * (parameters['interest_rate'] - 0.035)  # Negative correlation
        inflation_impact = 0.6 * parameters['inflation']  # Positive correlation
        population_impact = 1.2 * parameters['population_growth']  # Strong positive correlation
        income_impact = 0.7 * parameters['income_growth']  # Positive correlation
        unemployment_impact = -0.3 * parameters['unemployment_change']  # Negative correlation
        construction_impact = 0.4 * parameters['construction_cost_growth']  # Positive correlation
        
        # Combine impacts
        total_impact = (base_growth + 
                       interest_impact + 
                       inflation_impact + 
                       population_impact + 
                       income_impact + 
                       unemployment_impact + 
                       construction_impact)
        
        # Add some random market volatility
        market_volatility = np.random.normal(0, 0.05, total_impact.shape)
        total_impact += market_volatility
        
        # Convert to multipliers (1 + growth_rate)
        price_multipliers = 1 + total_impact
        
        return price_multipliers
    
    def run_simulation(self) -> pd.DataFrame:
        """
        Run Monte Carlo simulation
        
        Returns:
            DataFrame with simulation results
        """
        # Generate parameter values
        parameters = self.generate_parameter_values()
        
        # Calculate price impacts
        price_multipliers = self.calculate_price_impact(parameters)
        
        # Initialize results array
        prices = np.zeros((self.num_simulations, self.simulation_years + 1))
        prices[:, 0] = self.base_price  # Starting price
        
        # Calculate cumulative price changes
        for year in range(self.simulation_years):
            prices[:, year + 1] = prices[:, year] * price_multipliers[:, year]
        
        # Convert to DataFrame
        columns = ['Year_0'] + [f'Year_{i+1}' for i in range(self.simulation_years)]
        results_df = pd.DataFrame(prices, columns=columns)
        
        # Add simulation metadata
        results_df['simulation_id'] = range(self.num_simulations)
        results_df['base_price'] = self.base_price
        
        self.simulation_results = results_df
        return results_df
    
    def get_simulation_statistics(self) -> Dict:
        """Get statistical summary of simulation results"""
        if self.simulation_results is None:
            raise ValueError("Must run simulation first")
        
        # Get final year prices
        final_year_col = f'Year_{self.simulation_years}'
        final_prices = self.simulation_results[final_year_col]
        
        statistics = {
            'base_price': self.base_price,
            'final_price_mean': final_prices.mean(),
            'final_price_median': final_prices.median(),
            'final_price_std': final_prices.std(),
            'final_price_min': final_prices.min(),
            'final_price_max': final_prices.max(),
            'percentile_5': final_prices.quantile(0.05),
            'percentile_25': final_prices.quantile(0.25),
            'percentile_75': final_prices.quantile(0.75),
            'percentile_95': final_prices.quantile(0.95),
            'probability_gain': (final_prices > self.base_price).mean(),
            'probability_loss': (final_prices < self.base_price).mean(),
            'expected_return_pct': ((final_prices.mean() / self.base_price) - 1) * 100,
            'annual_return_pct': (((final_prices.mean() / self.base_price) ** (1/self.simulation_years)) - 1) * 100
        }
        
        return statistics
    
    def plot_simulation_results(self) -> go.Figure:
        """Create interactive plot of simulation results"""
        if self.simulation_results is None:
            raise ValueError("Must run simulation first")
        
        # Create percentile bands
        years = list(range(self.simulation_years + 1))
        percentiles = {}
        
        for year in years:
            col = f'Year_{year}'
            year_prices = self.simulation_results[col]
            percentiles[year] = {
                'p5': year_prices.quantile(0.05),
                'p25': year_prices.quantile(0.25),
                'p50': year_prices.quantile(0.50),
                'p75': year_prices.quantile(0.75),
                'p95': year_prices.quantile(0.95),
                'mean': year_prices.mean()
            }
        
        # Create figure
        fig = go.Figure()
        
        # Add percentile bands
        p95_values = [percentiles[year]['p95'] for year in years]
        p5_values = [percentiles[year]['p5'] for year in years]
        p75_values = [percentiles[year]['p75'] for year in years]
        p25_values = [percentiles[year]['p25'] for year in years]
        
        # 90% confidence band
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=p95_values + p5_values[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='90% Confidence Band',
            hoverinfo='skip'
        ))
        
        # 50% confidence band
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=p75_values + p25_values[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.4)',
            line=dict(color='rgba(255,255,255,0)'),
            name='50% Confidence Band',
            hoverinfo='skip'
        ))
        
        # Median line
        median_values = [percentiles[year]['p50'] for year in years]
        fig.add_trace(go.Scatter(
            x=years,
            y=median_values,
            mode='lines',
            name='Median Forecast',
            line=dict(color='blue', width=3)
        ))
        
        # Mean line
        mean_values = [percentiles[year]['mean'] for year in years]
        fig.add_trace(go.Scatter(
            x=years,
            y=mean_values,
            mode='lines',
            name='Mean Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Property Price Forecast - {self.simulation_years} Year Monte Carlo Simulation',
            xaxis_title='Years',
            yaxis_title='Property Price ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_final_price_distribution(self) -> go.Figure:
        """Plot distribution of final year prices"""
        if self.simulation_results is None:
            raise ValueError("Must run simulation first")
        
        final_year_col = f'Year_{self.simulation_years}'
        final_prices = self.simulation_results[final_year_col]
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=final_prices,
            nbinsx=50,
            name='Price Distribution',
            opacity=0.7
        ))
        
        # Add vertical lines for key statistics
        stats = self.get_simulation_statistics()
        
        # Mean line
        fig.add_vline(
            x=stats['final_price_mean'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${stats['final_price_mean']:,.0f}"
        )
        
        # Median line
        fig.add_vline(
            x=stats['final_price_median'],
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Median: ${stats['final_price_median']:,.0f}"
        )
        
        # Base price line
        fig.add_vline(
            x=self.base_price,
            line_dash="solid",
            line_color="green",
            annotation_text=f"Current: ${self.base_price:,.0f}"
        )
        
        fig.update_layout(
            title=f'Final Price Distribution After {self.simulation_years} Years',
            xaxis_title='Property Price ($)',
            yaxis_title='Frequency',
            template='plotly_white'
        )
        
        return fig


def run_portfolio_simulation(property_prices: List[float], 
                           simulation_years: int = 5,
                           num_simulations: int = 1000) -> Dict:
    """
    Run Monte Carlo simulation for a portfolio of properties
    
    Args:
        property_prices: List of current property prices
        simulation_years: Number of years to simulate
        num_simulations: Number of simulation runs
        
    Returns:
        Dictionary with portfolio simulation results
    """
    portfolio_results = []
    individual_results = {}
    
    for i, price in enumerate(property_prices):
        simulator = MonteCarloPropertySimulation(price, simulation_years, num_simulations)
        results = simulator.run_simulation()
        portfolio_results.append(results)
        individual_results[f'property_{i+1}'] = simulator.get_simulation_statistics()
    
    # Calculate portfolio totals
    portfolio_totals = np.zeros((num_simulations, simulation_years + 1))
    
    for results in portfolio_results:
        year_cols = [f'Year_{i}' for i in range(simulation_years + 1)]
        portfolio_totals += results[year_cols].values
    
    # Portfolio statistics
    final_values = portfolio_totals[:, -1]
    initial_value = sum(property_prices)
    
    portfolio_stats = {
        'initial_portfolio_value': initial_value,
        'final_portfolio_mean': final_values.mean(),
        'final_portfolio_median': final_values.median(),
        'portfolio_return_pct': ((final_values.mean() / initial_value) - 1) * 100,
        'annual_return_pct': (((final_values.mean() / initial_value) ** (1/simulation_years)) - 1) * 100,
        'value_at_risk_5pct': final_values.quantile(0.05),
        'individual_properties': individual_results
    }
    
    return portfolio_stats


if __name__ == "__main__":
    print("Monte Carlo simulation utilities loaded successfully!")
