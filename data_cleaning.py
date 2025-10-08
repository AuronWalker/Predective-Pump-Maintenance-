import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

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


    voltage_series = pd.Series(voltage_rms)
    current_series = pd.Series(current_rms)
    vibration_series = pd.Series(vibration)

    window = 1000
    voltage_mean = voltage_series.rolling(window).mean()
    voltage_var = voltage_series.rolling(window).var()
    # df['voltage_rms_skew'] = df['voltage_rms'].rolling(window).apply(lambda x: skew(x, bias=False))
    # df['voltage_rms_kurt'] = df['voltage_rms'].rolling(window).apply(lambda x: kurtosis(x, bias=False))

    current_mean = current_series.rolling(window).mean()
    current_var = current_series.rolling(window).var()
    # df['current_rms_skew'] = df['current_rms'].rolling(window).apply(lambda x: skew(x, bias=False))
    # df['current_rms_kurt'] = df['current_rms'].rolling(window).apply(lambda x: kurtosis(x, bias=False))

    vibration_mean = vibration_series.rolling(window).mean()
    vibration_var = vibration_series.rolling(window).var()
    # df['vibration_skew'] = df['vibration'].rolling(window).apply(lambda x: skew(x, bias=False))
    # df['vibration_kurt'] = df['vibration'].rolling(window).apply(lambda x: kurtosis(x, bias=False))

    df_dc = pd.DataFrame({
        "time": time,
        "voltage_rms": voltage_rms,
        "current_rms": current_rms,
        "vibration": vibration,
        "voltage_mean" :  voltage_mean,
        "current_mean": current_mean,
        "vibration_mean": vibration_mean,
        "voltage_var": voltage_var,
        "current_var": current_var,
        "vibration_var": vibration_var
    })

    df_dc = df_dc.iloc[window:] #removes rows without extra stats
    df_dc.to_csv(output_csv, index=False)
    print(f"DC-equivalent single channel file saved as '{df_dc}'")

ac_files = ['impeller', 'healthy', 'unbalanced_motor', 'unbalanced_pump', 'cavitation_discharge', 'cavitation_suction']

for i in range(len(ac_files)):
    convert_ac_to_dc('./Dataset\Electric AC\{}_voltage.csv'.format(ac_files[i]), './Dataset\Electric AC\{}_current.csv'.format(ac_files[i]), './Dataset\Vibration\{}.csv'.format(ac_files[i]), './Cleaned Data\{}.csv'.format(ac_files[i]))
