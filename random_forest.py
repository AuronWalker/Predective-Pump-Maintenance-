import pandas as pd
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
        df['fault'] = data_files[i]

        #This could be useful later
        df = df.drop(columns = ['time'])

        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def create_random_forest(training_data):
    X = training_data.iloc[:, 0:3]
    y = training_data.iloc[:, 3]
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
