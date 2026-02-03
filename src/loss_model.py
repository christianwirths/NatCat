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
        