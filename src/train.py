from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
def train(nEsimator, randomState, nJobs, xTrain, yTrain, xTest, yTest):
    rf = RandomForestRegressor(n_estimators=nEsimator, random_state=randomState, n_jobs=nJobs)
    rf.fit(xTrain, yTrain)
    y_pred_test = rf.predict(xTest)
    print("R² on test:", r2_score(yTest, y_pred_test))
    print("MAE on test:", mean_absolute_error(yTest, y_pred_test))
    return rf

def clean(df, group_cols, agg_dict):
    # make a smaller data frame to hold data above cols
    cols_to_keep = list(set(group_cols + list(agg_dict.keys())))
    df_small = df[cols_to_keep].copy()
    
    # groups smaller data set into counties and states
    county_df = df_small.groupby(group_cols).agg(agg_dict).reset_index()
    
    # aggregates the data according to previous aggregation rules
    county_df = county_df.rename(columns={"ID": "Total_Accidents"})
    county_df = county_df.dropna(subset=["Total_Accidents"])
    
    # creates a column features
    feature_cols = [c for c in county_df.columns if c not in ["State", "County", "Total_Accidents"]]
    
    # removes anything not containing a feature
    county_df = county_df.dropna(subset=feature_cols)
    print("County data ready:", county_df.shape)
    
    return county_df, feature_cols
