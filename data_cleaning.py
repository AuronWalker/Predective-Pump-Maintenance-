import pandas as pd
import numpy as np

def convert_ac_to_dc(input_voltage_csv, input_current_csv, input_vibration_csv, output_csv):
    df_voltage = pd.read_csv(input_voltage_csv)
    df_current = pd.read_csv(input_current_csv)
    df_vibration = pd.read_csv(input_vibration_csv)

    # Cuts out values to match vibration data
    df_voltage = df_voltage[df_voltage["time"] <= 12]
    df_current = df_current[df_current["time"] <= 12]

    time = df_voltage.iloc[:, 0].values
    voltage_ac = df_voltage.iloc[:, 1:].values
    current_ac = df_current.iloc[:, 1:].values
    vibrations = df_vibration.iloc[:, 1:].values
    
    voltage_rms = np.sqrt(np.mean(voltage_ac**2, axis=1))
    current_rms = np.sqrt(np.mean(current_ac**2, axis=1))
    vibration = np.mean(vibrations, axis=1)

    df_dc = pd.DataFrame({
        "time": time,
        "voltage_rms": voltage_rms,
        "current_rms": current_rms,
        "vibration": vibration  
    })

    df_dc.to_csv(output_csv, index=False)
    # print(f"DC-equivalent single channel file saved as '{df_dc}'")

ac_files = ['impeller', 'healthy', 'unbalanced_motor', 'unbalanced_pump', 'cavitation_discharge', 'cavitation_suction']

for i in range(len(ac_files)):
    convert_ac_to_dc('./Dataset\Electric AC\{}_voltage.csv'.format(ac_files[i]), './Dataset\Electric AC\{}_current.csv'.format(ac_files[i]), './Dataset\Vibration\{}.csv'.format(ac_files[i]), './Cleaned Data\{}.csv'.format(ac_files[i]))
