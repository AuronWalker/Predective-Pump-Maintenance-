import pandas as pd
import numpy as np
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
    X = training_data.iloc[:, :-1]
    y = training_data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=69, test_size=0.2)

    rf = RandomForestClassifier(n_estimators = 30,
                                criterion = 'gini',
                                min_samples_split = 5,
                                max_depth = 8,
                                random_state = 42)
    
    #No hyper parameters
    # rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    #shows what columns had the biggest impact
    features = pd.DataFrame(rf.feature_importances_, index=X.columns)
    print(features.head(9))

    print(classification_report(y_test, y_pred))

    n_nodes = [estimator.tree_.node_count for estimator in rf.estimators_]
    max_depths = [estimator.tree_.max_depth for estimator in rf.estimators_]

    print(f"Number of trees: {len(rf.estimators_)}")
    print(f"Average nodes per tree: {np.mean(n_nodes):.1f}")
    print(f"Average depth per tree: {np.mean(max_depths):.1f}")
    print(f"Total nodes: {np.sum(n_nodes)}")

    total_nodes = np.sum([est.tree_.node_count for est in rf.estimators_])
    model_size_bytes = total_nodes * 28
    print(f"Estimated model size: {model_size_bytes/1024:.2f} KB")


training_data = create_training_data()
create_random_forest(training_data)
