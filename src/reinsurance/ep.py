"""
Exceedence probability calculations. 
"""

import numpy as np
from typing import List, Dict, Any


class ExceedenceProbabilityCalculator:
    """
    Class to calculate exceedence probabilities for different hazard intensities.
    
    This can be used to compute the probability that a certain intensity level is exceeded at a given location and time, based on the hazard model outputs.
    """
    
    def __init__(self, simulation) -> None:
        """
        Initialize with a simulation object that contains the hazard model outputs.
        """
        
        self.simulation = simulation
        
    
    def calculate_eOEP(self, loss_threshold = 0) -> float:
        """
        Calculate the emperical exceedence probability for a given location and loss threshold.
        
        Args:
            loss_threshold: The loss level for which to calculate the exceedence probability.
        
        Returns:
            Exceedence probability (between 0 and 1).
        """
        
        # Extract strongest event for every year 
        strongest_events = [max(yr['losses'], default=0) for yr in self.simulation._results]

        # Calculate exceedence probability
        exceedences = sum(1 for loss in strongest_events if loss >= loss_threshold)
        total_years = len(strongest_events)
        oep = exceedences / total_years if total_years > 0 else 0
        return oep
    
    def calculate_eAEP(self, loss_threshold = 0) -> float:
        """
        Calculate the emperical annual exceedence probability for a given location and loss threshold.
        
        Args:
            loss_threshold: The loss level for which to calculate the annual exceedence probability.

        Returns:
            Annual exceedence probability (between 0 and 1).
        """
        # Extract all events across all years
        all_events = self.simulation.losses_per_year

        # Calculate annual exceedence probability
        exceedences = sum(1 for loss in all_events if loss >= loss_threshold)
        total_events = len(all_events)
        aep = exceedences / total_events if total_events > 0 else 0
        return aep
    

    def emperical_EP_curve(self, loss_thresholds= np.arange(0, 10) * 1e10, method='OEP') -> Dict[float, float]:
        """
        Calculate the exceedence probability curve for a range of loss thresholds.
        
        Args:
            loss_thresholds: List of loss thresholds to calculate exceedence probabilities for.
            method: 'OEP' for Occurrence Exceedence Probability, 'AEP' for Annual Exceedence Probability.

        Returns:
            Dictionary mapping each loss threshold to its exceedence probability.
        """
        if method == 'OEP':
            return {threshold: self.calculate_eOEP(threshold) for threshold in loss_thresholds}
        elif method == 'AEP':
            return {threshold: self.calculate_eAEP(threshold) for threshold in loss_thresholds}
        else:
            raise ValueError("Method must be 'OEP' or 'AEP'.")
        

    def plot_EP_curve(self, loss_thresholds= np.arange(0, 10) * 1e10, method='OEP'):
        """
        Plot the exceedence probability curve for a range of loss thresholds.
        
        Args:
            loss_thresholds: List of loss thresholds to calculate exceedence probabilities for.
            method: 'OEP' for Occurrence Exceedence Probability, 'AEP' for Annual Exceedence Probability.
        """
        import matplotlib.pyplot as plt
        
        ep_values = self.emperical_EP_curve(loss_thresholds, method)
        fig = plt.figure(figsize=(10, 6))
        plt.plot(list(ep_values.keys()), list(ep_values.values()), marker='o')
        plt.xscale('log')
        plt.xlabel('Loss Threshold')
        plt.ylabel(f'{method} Exceedence Probability')
        plt.title(f'{method} Curve')
        plt.grid(True)
        plt.show()

    def return_period(self, loss_threshold = 0, method='OEP') -> float:
        """
        Calculate the return period for a given loss threshold based on the exceedence probability.
        
        Args:
            loss_threshold: The loss level for which to calculate the return period.
            method: 'OEP' for Occurrence Exceedence Probability, 'AEP' for Annual Exceedence Probability.
        Returns:
            Return period in years (float). Returns np.inf if exceedence probability is 0.
        """
        if method == 'OEP':
            ep = self.calculate_eOEP(loss_threshold)
        elif method == 'AEP':
            ep = self.calculate_eAEP(loss_threshold)
        else:
            raise ValueError("Method must be 'OEP' or 'AEP'.")
        
        return 1 / ep if ep > 0 else np.inf

    def calculate_AAL(self) -> float:
        """
        Calculate the Average Annual Loss (AAL) across all simulated years.
        
        Returns:
            Average Annual Loss (float).
        """
        return self.simulation.aal
    

