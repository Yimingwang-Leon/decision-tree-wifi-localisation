import torch
import pickle
import numpy as np
import pandas as pd
from part1_nn_lib import MultiLayerNetwork, MSELossLayer 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

class Regressor:

    def __init__(self, x, nb_epoch = 1000, learning_rate=0.001, hidden=[64, 32], activations=['relu', 'relu', 'identity']):
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - learning_rate {float} -- learning_rate, by default is 1e-3
            - hiden {list} -- A list of output dims for each hidden layer, 
                by default set is two hidden layers with 64 and 32 output dims respectively
            - activations {list} -- A list of activation functions for each layer, 
            by default is 'relu', 'relu' for hidden layer, and final output function is 'identity'
        """

        self.nb_epoch = nb_epoch 
        self.learning_rate = learning_rate

        self.num_cols = [col for col in x.columns if col != 'ocean_proximity']
        self.cat_col = 'ocean_proximity'

        # Z-score 
        self.num_median = None 
        self.num_mean = None
        self.num_std = None 
        self.cat_fill = None 

        self.y_mean = None 
        self.y_std = None 

        self.lb = LabelBinarizer()

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1

        # Construct neutral networks
        neurons = hidden + [self.output_size]
        self.network = MultiLayerNetwork(self.input_size, neurons, activations)
        self.loss_layer = MSELossLayer() 

    
    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network using Z-score standardization for numerical columns
        and one-hot encoding for categorical columns.
        """

        X = x.copy()
        y_numpy = None 

        # 1. Deal with NA
        if training:
            self.num_mean = X[self.num_cols].mean(axis=0)
            self.num_median = X[self.num_cols].median(axis=0)
        X[self.num_cols] = X[self.num_cols].fillna(self.num_median)

        if training:
            self.cat_fill = X[self.cat_col].mode()[0]
        X[self.cat_col] = X[self.cat_col].fillna(self.cat_fill)

        # 2. One-Hot Encoding
        if training:
            X_cat_encoded = self.lb.fit_transform(X[self.cat_col].values)
        else:
            X_cat_encoded = self.lb.transform(X[self.cat_col].values)

        # 3. Z-score 
        if training:
            self.num_std = X[self.num_cols].std(axis=0)
            
            self.num_std = self.num_std.where(self.num_std != 0.0, 1.0) 

        # Z-score formula
        X[self.num_cols] = (X[self.num_cols] - self.num_mean) / self.num_std

        X_num_numpy = X[self.num_cols].to_numpy(dtype=float)

       
        X_numpy = np.hstack((X_num_numpy, X_cat_encoded))

        # 4. Standardize y
        if y is not None:
            y_arr = y.values.reshape(-1, 1) if isinstance(y, pd.DataFrame) else np.asarray(y).reshape(-1, 1)
            if training:
                self.y_mean = y_arr.mean(axis=0)
                self.y_std = y_arr.std(axis=0)
                self.y_std[self.y_std == 0.0] = 1.0
            y_numpy = (y_arr - self.y_mean) / self.y_std
        
        return X_numpy, y_numpy

        
    def fit(self, x, y):
        """
        Regressor training function
        """
        X, Y = self._preprocessor(x, y = y, training = True)
        
        for _ in range(self.nb_epoch):
            Y_pred = self.network.forward(X)
            loss = self.loss_layer(Y_pred, Y)
            grad = self.loss_layer.backward()
            grad = self.network.backward(grad)
            self.network.update_params(self.learning_rate)

        return self

            
    def predict(self, x):
        """
        Output the value corresponding to an input x (De-standardized).
        """
        X, _ = self._preprocessor(x, training = False)
        y_pred_scaled = self.network.forward(X).reshape(-1, 1)

        
        return y_pred_scaled * self.y_std + self.y_mean
        

    def score(self, x, y):
        """
        Function to evaluate the model efficiency (Returns RMSE).
        """
        
        y_pred = self.predict(x) 

        
        y_true = y.to_numpy(dtype=float).reshape(-1, 1) 
        
       
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        return rmse


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def perform_hyperparameter_search(): 
    """
    Performs a hyper-parameter search for fine-tuning the regressor.
    """

    data = pd.read_csv('housing.csv')
    output_label = 'median_house_value'

    X = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        'learning_rate' : [1e-3, 5e-4, 1e-4],
        'nb_epoch' : [300, 400, 500],
        'hidden' : [
            [64, 32],
            [128, 64],
            [128, 64, 32],
            [256, 128, 64]
        ]
    }

    best_params = {}
    best_rmse = float('inf')
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for lr in param_grid['learning_rate']:
        for epochs in param_grid['nb_epoch']:
            for layer in param_grid['hidden']:

                num_hidden_layers = len(layer) 
                current_activations = ['relu'] * num_hidden_layers + ['identity'] 

                fold_rmses = []
                fold_r2s = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
                    X_train = X_train_val.iloc[train_idx]
                    X_val = X_train_val.iloc[val_idx]
                    y_train = y_train_val.iloc[train_idx]
                    y_val = y_train_val.iloc[val_idx]
                    
                    y_true_val = y_val.to_numpy(dtype=float).reshape(-1, 1)

                    try:
                        reg = Regressor(
                            X_train,
                            nb_epoch=epochs,
                            learning_rate=lr,
                            hidden=layer,
                            activations=current_activations
                        )
                        reg.fit(X_train, y_train)
                        y_pred = reg.predict(X_val)
                        
                        rmse = np.sqrt(mean_squared_error(y_true_val, y_pred))
                        r2 = r2_score(y_true_val, y_pred)
                        
                        fold_rmses.append(rmse)
                        fold_r2s.append(r2)
                        
                    except Exception as e:
                        
                        print(f'Error! Fold {fold+1} failed with {e}')
                        fold_rmses.append(float('inf')) 
                        fold_r2s.append(-9999.0) 

                
                avg_rmse = np.mean(fold_rmses)
                valid_r2s = [r for r in fold_r2s if r != -9999.0]
                avg_r2 = np.mean(valid_r2s) if valid_r2s else -float('inf')
                    
                print(f"Params: LR={lr}, Epochs={epochs}, Hidden={layer}, Avg CV RMSE: {avg_rmse:.4f}, Avg CV R2: {avg_r2:.4f}")

                
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params = {
                        'learning_rate': lr,
                        'nb_epoch': epochs,
                        'hidden': layer,
                        'activations': current_activations
                    }

    print("Hyperparameter Search Complete")
    print(f"Optimal Hyperparameters: {best_params}")
    print(f"Best CV RMSE: {best_rmse:.4f}")

    print("\n--- Final Model Training on Full Train/Val Set ---")
    
    if not best_params:
        print("Warning: No optimal parameters found (all attempts failed). Cannot train final model.")
        return None

    final_regressor = Regressor(
        X_train_val, 
        **best_params 
    )
    final_regressor.fit(X_train_val, y_train_val)
    
    # Evaluation on final test
    y_true_test = y_test.to_numpy(dtype=float).reshape(-1, 1)
    y_pred_test = final_regressor.predict(X_test)
    
    final_test_rmse = float(np.sqrt(mean_squared_error(y_true_test, y_pred_test)))
    final_test_r2 = r2_score(y_true_test, y_pred_test)
    
    print(f"Final Model Test Set RMSE (Generalization Error): {final_test_rmse:.4f}")
    print(f"Final Model Test Set R-squared (R2): {final_test_r2:.4f}")

    return final_regressor


def example_main():

    best_model = perform_hyperparameter_search()

    if best_model is not None:
        save_regressor(best_model)
        print("Final model saved.")
    else:
        print("Skipping model save due to errors in hyperparameter search.")


if __name__ == "__main__":
    example_main()