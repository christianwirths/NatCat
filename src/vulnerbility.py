import numpy as np
import matplotlib.pyplot as plt

def get_damage_ratio(wind_speed_knots, construction='Frame'):
    """
    Returns the Mean Damage Ratio (0.0 to 1.0) for a given wind speed.
    """
    # Define vulnerability parameters based on construction
    # Frame (Wood) is weaker -> Lower v50 (fails earlier)
    # Masonry is stronger -> Higher v50 (survives longer)
    
    if construction == 'Frame':
        # 50% damage at 100 knots, steep curve
        v_50 = 100
        k = 0.12 
    elif construction == 'Masonry':
        # 50% damage at 120 knots, slightly flatter
        v_50 = 120
        k = 0.12
    else:
        # Default fallback
        v_50 = 110
        k = 0.12
        
    # Logistic Function (Sigmoid)
    # We clip at 0 because negative wind doesn't heal houses
    # We clip at 1.0 because you can't be more than 100% destroyed
    
    # Only apply damage if wind > 40 knots (minor threshold)
    mdr = np.zeros_like(wind_speed_knots, dtype=float)
    no_impact = wind_speed_knots < 40
    impact = wind_speed_knots >= 40    
    mdr[no_impact] = 0.0 
    mdr[impact] = 1 / (1 + np.exp(-k * (wind_speed_knots[impact] - v_50)))
    
    return mdr

if __name__ == "__main__":
    # Let's plot our curves to visualize them (Resume artifact!)
    winds = np.arange(0, 180, 1)
    
    frame_damage = [get_damage_ratio(w, 'Frame') for w in winds]
    masonry_damage = [get_damage_ratio(w, 'Masonry') for w in winds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(winds, frame_damage, label='Wood Frame', color='brown', linewidth=2)
    plt.plot(winds, masonry_damage, label='Masonry', color='gray', linestyle='--')
    
    plt.title('Vulnerability Curves: Wind Speed vs. Mean Damage Ratio')
    plt.xlabel('Wind Speed (Knots)')
    plt.ylabel('Damage Ratio (0% to 100%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axvline(64, color='orange', alpha=0.5, label='Cat 1 Hurricane') # Cat 1 threshold
    plt.axvline(137, color='red', alpha=0.5, label='Cat 5 Hurricane')   # Cat 5 threshold
    
    plt.savefig('vulnerability_curves.png')
    print("Plot saved to vulnerability_curves.png")