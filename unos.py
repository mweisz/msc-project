import pandas as pd
import numpy as np

class UNOS_data:
    def __init__(self, csv_path):
        self.csv_path = csv_path
    
    def __normalise_dataframe(self, df):
        normalised_df = df.copy()

        for col_name in df.columns:

            # Check if it is a binary column
            if len(df[col_name].unique()) == 2 and df[col_name].min() == 0 and df[col_name].max() == 1:
                # Binary. Do nothing
                pass
            else:
                # Multivariate. Perform z1 normalisation
                normalised_df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()

        return normalised_df
    
    def draw_sample(self, enable_feature_scaling=True):
        unos_df = pd.read_csv(self.csv_path)

        # Feature Scaling
        if enable_feature_scaling:
            unos_df = self.__normalise_dataframe(unos_df)

        # Features
        X = np.array(unos_df.drop('LVAD', axis=1))

        # Number of samples
        N = X.shape[0]

        # Number of Features
        D = X.shape[1]

        # Treatment
        W = np.array(unos_df['LVAD'])

        BetaB    = np.random.choice([0, 0.1, 0.2, 0.3, 0.4], size=D, replace=True, p=[0.6, 0.1, 0.1, 0.1,0.1])
        Y_0      = np.random.normal(size=N) + np.exp(np.dot(X+0.5,BetaB))
        Y_1      = np.random.normal(size=N) + np.dot(X,BetaB)
        AVG      = np.mean(Y_1[W==1]-Y_0[W==1])
        Y_1      = Y_1-AVG+4  
        TE       = np.dot(X,BetaB)-AVG+4-np.exp(np.dot(X+0.5,BetaB))
        Y        = np.transpose(np.array([W,(1-W)*Y_0+W*Y_1,TE]))
        DatasetX = pd.DataFrame(X, columns=list(filter(lambda x: x!= 'LVAD', unos_df.columns)))
        DatasetY = pd.DataFrame(Y, columns='Treatment Response TE'.split())
        Dataset  = DatasetX.join(DatasetY)                        
        return Dataset
    
    
        