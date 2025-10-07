import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Creates trainning data to pass into the random forest
# Not a specfic csv file as it would just add unneeded size
def create_training_data():
    data_files = ['impeller', 'healthy', 'unbalanced_motor', 'unbalanced_pump', 'cavitation_discharge', 'cavitation_suction']
    df_list = []

    for i in range(len(data_files)):
        df = pd.read_csv('./Cleaned Data\{}.csv'.format(data_files[i]))

        # Add this to data_cleaning.py at home
        # Adds uneeded time
        window = 1000
        df['voltage_rms_mean'] = df['voltage_rms'].rolling(window).mean()
        df['voltage_rms_var'] = df['voltage_rms'].rolling(window).var()
        df['voltage_rms_skew'] = df['voltage_rms'].rolling(window).apply(lambda x: skew(x, bias=False))
        df['voltage_rms_kurt'] = df['voltage_rms'].rolling(window).apply(lambda x: kurtosis(x, bias=False))

        df['current_rms_mean'] = df['current_rms'].rolling(window).mean()
        df['current_rms_var'] = df['current_rms'].rolling(window).var()
        df['current_rms_skew'] = df['current_rms'].rolling(window).apply(lambda x: skew(x, bias=False))
        df['current_rms_kurt'] = df['current_rms'].rolling(window).apply(lambda x: kurtosis(x, bias=False))

        df['vibration_mean'] = df['vibration'].rolling(window).mean()
        df['vibration_var'] = df['vibration'].rolling(window).var()
        df['vibration_skew'] = df['vibration'].rolling(window).apply(lambda x: skew(x, bias=False))
        df['vibration_kurt'] = df['vibration'].rolling(window).apply(lambda x: kurtosis(x, bias=False))

        df['fault'] = data_files[i]

        #This could be useful later 
        df = df.drop(columns = ['time'])
        df = df.drop(columns = ['voltage_rms'])
        df = df.drop(columns = ['current_rms'])

        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def create_random_forest(training_data):
    X = training_data.iloc[:, :-1]
    y = training_data.iloc[:, -1]
    print(X.head())
    print(y.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=69, test_size=0.2)

    # rf = RandomForestClassifier(n_estimators = 1000,
    #                             criterion = 'entropy',
    #                             min_samples_split = 10,
    #                             max_depth = 14,
    #                             random_state = 42)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    #shows what columns had the biggest impact
    features = pd.DataFrame(rf.feature_importances_, index=X.columns)
    print(features.head())

    print(classification_report(y_test, y_pred))


training_data = create_training_data()
create_random_forest(training_data)
