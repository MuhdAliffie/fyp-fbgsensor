import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PHYSICAL PARAMETERS (THEORETICAL FEA)
# ==========================================
E_MODULUS = 0.85e9       # Effective Young's Modulus for 5% Infill PLA (0.85 GPa)
WIDTH = 0.02             # Beam width (20 mm = 0.02 m)
HEIGHT = 0.005           # Beam height (5 mm = 0.005 m)
M_TIP = 0.008            # Tip mass (8.0 grams = 0.008 kg)
DENSITY = 682.0          # Effective density of 5% infill PLA (kg/m^3)
DAMPING_RATIO = 0.15     # Estimated damping ratio for thermoplastic PLA
G_FORCE = 9.81           # 1 g acceleration (m/s^2)

# Calculate Area Moment of Inertia (I) for rectangular beam
I_MOMENT = (WIDTH * (HEIGHT ** 3)) / 12.0

def calculate_theoretical_sensitivity(L_mm, freqs):
    """Calculates the theoretical FEA sensitivity based on dynamic beam theory."""
    L = L_mm / 1000.0  # Convert to meters
    
    # Calculate mass of the beam itself
    m_beam = DENSITY * WIDTH * HEIGHT * L
    # Effective mass for cantilever (tip mass + 24% of beam mass)
    m_eff = M_TIP + (0.24 * m_beam)
    
    # 1. Calculate Theoretical Natural Frequency (Resonance)
    f_n = (1 / (2 * np.pi)) * np.sqrt((3 * E_MODULUS * I_MOMENT) / (m_eff * (L ** 3)))
    
    # 2. Calculate Static Strain at the root (where FBG is attached) for 1g
    F_static = M_TIP * G_FORCE
    Moment = F_static * L
    y = HEIGHT / 2.0  # Distance to neutral axis
    strain_static = (Moment * y) / (E_MODULUS * I_MOMENT)
    
    # Convert static strain to static optical shift (1 microstrain = 1.2 pm)
    shift_static_pm = strain_static * 1.2 * 1e6
    
    simulated_sensitivities = []
    
    for f in freqs:
        # 3. Apply Dynamic Amplification Factor (Harmonic Response)
        freq_ratio = f / f_n
        amplification = 1.0 / np.sqrt((1 - freq_ratio**2)**2 + (2 * DAMPING_RATIO * freq_ratio)**2)
        
        dynamic_shift_pm = shift_static_pm * amplification
        simulated_sensitivities.append(dynamic_shift_pm)
        
    return simulated_sensitivities, f_n

# ==========================================
# 2. EXPERIMENTAL DATA INPUT
# ==========================================
# Frequencies tested
freqs = [2.0, 3.0, 4.0, 5.0]

# Experimental Sensitivity Data (pm/g) from your prompt
exp_data = {
    30: [0.1480, 0.0538, 0.0786, 0.0755],
    40: [0.1606, 0.0814, 0.0698, 0.0543],
    50: [0.0667, 0.0876, 0.0296, 0.1315],
    60: [0.0636, 0.0649, 0.0408, 0.0244], # Corrected from duplicate 50
    70: [0.0134, 0.0836, 0.0885, 0.0548]
}

# ==========================================
# 3. RUN SIMULATION & COMPARE
# ==========================================
def main():
    results = []
    
    print("--- THEORETICAL FEA VS. EXPERIMENTAL SENSITIVITY ---")
    print(f"{'Beam':<6} | {'Freq (Hz)':<10} | {'Exp (pm/g)':<12} | {'Sim (pm/g)':<12} | {'Error (%)':<10}")
    print("-" * 60)
    
    for beam_length, exp_sensitivities in exp_data.items():
        sim_sensitivities, theoretical_fn = calculate_theoretical_sensitivity(beam_length, freqs)
        
        for i, freq in enumerate(freqs):
            exp_val = exp_sensitivities[i]
            sim_val = sim_sensitivities[i]
            
            # Calculate Percentage Error
            if exp_val > 0:
                error = abs(sim_val - exp_val) / exp_val * 100
            else:
                error = 0.0
                
            results.append({
                'Beam (mm)': beam_length,
                'Frequency (Hz)': freq,
                'Experimental (pm/g)': exp_val,
                'Simulated (pm/g)': sim_val,
                'Error (%)': error
            })
            
            print(f"{beam_length:<6} | {freq:<10.1f} | {exp_val:<12.4f} | {sim_val:<12.4f} | {error:<10.1f}")
        print("-" * 60)

    # Convert to DataFrame for easy export
    df = pd.DataFrame(results)
    df.to_csv("FEA_vs_Experimental_Comparison.csv", index=False)
    
    # --- GENERATE COMPARISON GRAPH ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('FEA Simulation vs. Experimental Absolute Sensitivity', fontsize=16)
    
    axes = axes.flatten()
    
    for idx, beam in enumerate([30, 40, 50, 60, 70]):
        ax = axes[idx]
        subset = df[df['Beam (mm)'] == beam]
        
        ax.plot(subset['Frequency (Hz)'], subset['Experimental (pm/g)'], 
                marker='o', color='blue', label='Experimental', linewidth=2)
        ax.plot(subset['Frequency (Hz)'], subset['Simulated (pm/g)'], 
                marker='x', color='red', linestyle='--', label='Theoretical FEA', linewidth=2)
        
        ax.set_title(f'{beam} mm Cantilever Beam')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Sensitivity (pm/g)')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()

    # Hide the 6th empty subplot
    axes[5].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('FEA_Comparison_Plot.png', dpi=300)
    print("\nComparison complete! Graph saved as 'FEA_Comparison_Plot.png'")
    print("Full data table saved as 'FEA_vs_Experimental_Comparison.csv'")
    plt.show()

if __name__ == "__main__":
    main()