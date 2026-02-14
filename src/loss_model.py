"""
Generic loss calculation model.

Works with any HazardModel and VulnerabilityModel to calculate portfolio losses.
"""

import pandas as pd
import numpy as np
from hazards.base import HazardModel
from vulnerability.base import VulnerabilityModel


class LossCalculator:
    """
    Generic loss calculator using composition pattern.
    
    Takes any hazard model and vulnerability function to compute losses.
    """
    
    def __init__(self, hazard: HazardModel, vulnerability: VulnerabilityModel):
        """
        Initialize with hazard and vulnerability models.
        
        Args:
            hazard: Any HazardModel subclass (TropicalCycloneHazard, etc.)
            vulnerability: Any VulnerabilityModel subclass (WindVulnerability, etc.)
        """
        self.hazard = hazard
        self.vulnerability = vulnerability
        self.portfolio_results = None 

    def calculate_portfolio_loss(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate losses for entire portfolio.
        
        Args:
            portfolio: DataFrame with 'latitude', 'longitude', 'tiv' columns.
                      Optionally includes 'construction' column for construction types.
        
        Returns:
            Portfolio DataFrame with added columns:
            - 'intensity': Hazard intensity at each location
            - 'damage_ratio': Damage ratio (0 to 1)
            - 'loss': Monetary loss (damage_ratio * tiv)
        """
      
        coordinates = portfolio[['latitude', 'longitude']].to_numpy()


        intensity = self.hazard.compute_intensity(coordinates)
        
        # Check if portfolio has construction type information
        if 'construction' in portfolio.columns:
            construction_types = portfolio['construction'].to_numpy()
            damage_ratio = self.vulnerability.get_damage_ratio(intensity, construction_types)
        else:
            damage_ratio = self.vulnerability.get_damage_ratio(intensity)
        
        tiv = portfolio['tiv'].to_numpy()
        loss = damage_ratio * tiv
        portfolio = portfolio.copy()
       
        portfolio['intensity'] = intensity
        portfolio['damage_ratio'] = damage_ratio
        portfolio['loss'] = loss

        self.portfolio_results = portfolio
        return portfolio
        
    
    @property
    def total_loss(self) -> float:
        """Calculate total loss across portfolio."""
        
        if 'loss' not in self.portfolio_results.columns:
            raise ValueError("Portfolio DataFrame must contain 'loss' column. Run calculate_portfolio_loss() first.")
        
        total_loss = self.portfolio_results['loss'].sum()
        return total_loss
        
    # Time evolution portfolio loss calculation
    def calculate_portfolio_loss_over_time(self, portfolio: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
        """
        Calculate portfolio losses over a series of timestamps.
        
        Args:
            portfolio: DataFrame with 'latitude', 'longitude', 'tiv' columns.
            timestamps: Series of timestamps to evaluate. Minimum one timestamp required.
        
        Returns:
            DataFrame with losses for each timestamp.
        """

        # Check if just one timestamp or multiple timestamps are provided
        if len(timestamps) == 0:
            raise ValueError("At least one timestamp must be provided.")
        
        # check if timestamps are sorted, if not sort them
        if not timestamps.is_monotonic_increasing:
            print("Timestamps are not sorted. Sorting timestamps for loss calculation.")
            timestamps = timestamps.sort_values()
        
        coordinates = portfolio[['latitude', 'longitude']].to_numpy()
        construction_types = portfolio['construction'].to_numpy() if 'construction' in portfolio.columns else None
        tiv = portfolio['tiv'].to_numpy()
        
        results = []
        for timestamp in timestamps:
            # Get intensity up to this timestamp (considers all track points <= timestamp)
            intensity = self.hazard.compute_intensity_at_timestep(coordinates, timestamp)
            
            # Calculate damage ratio
            if construction_types is not None:
                damage_ratio = self.vulnerability.get_damage_ratio(intensity, construction_types)
            else:
                damage_ratio = self.vulnerability.get_damage_ratio(intensity)
            
            # Calculate losses
            loss = damage_ratio * tiv
            
            # Create result dataframe for this timestamp
            result_df = portfolio.copy()
            result_df['timestamp'] = timestamp
            result_df['intensity'] = intensity
            result_df['damage_ratio'] = damage_ratio
            result_df['loss'] = loss
            
            results.append(result_df)
        
        return pd.concat(results, ignore_index=True)