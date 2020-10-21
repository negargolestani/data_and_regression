from utils import*

# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
def get_data(setting, target, features='vind'):    
    if setting == 'single':
        train_data_p, test_data_p = get_data('parallel', target, features='vind_1')
        train_data_o, test_data_o = get_data('orthogonal', target, features='vind_1')
        train_data = train_data_p.append(train_data_o)
        train_data = train_data.iloc[1::2,:]
        test_data = [*test_data_p, *test_data_o]                
    else:
        train_data_raw = load_data('synth_'+setting, 'synth_'+features, target)
        test_data_raw = load_data('arduino_'+setting, 'meas_'+features, target)
        train_data = data_processing(train_data_raw).get_df(merge=True)
        test_data = data_processing(test_data_raw).get_df(merge=False)
    return train_data, test_data


# -----------------------------------------------------
# Process Data
# -----------------------------------------------------
def data_processing(data): 
    return data.segment(win_size=20, step=1)


# -----------------------------------------------------
# Load Data
# -----------------------------------------------------
def load_data(dataset_name, features, target):
    dataset_df_list = load_dataset(dataset_name, as_dict=False)
    X, Y = list(), list()
    for data in dataset_df_list:                    
        x = data.filter(regex=features).to_numpy()  # V_ind
        c1 = np.array(data.center_1.to_list())      # center_1
        d1 = np.linalg.norm(c1, axis=1)                       
        X.append(x)
        if target == 'r': Y.append(d1)
        elif target == 'x': Y.append(c1[:,0])
        elif target == 'y': Y.append(c1[:,1])
        elif target == 'z': Y.append(c1[:,2])
    return DATA(X, Y)



# -----------------------------------------------------
# Predict
# -----------------------------------------------------
def predict_data(model, test_data):
    predictions = list()        
    for data in test_data:
        prediction = predict_model(model, data=data)
        y_true = prediction.target.to_numpy()
        y_pred = prediction.Label.to_numpy()   
        y_pred = signal.savgol_filter(y_pred, window_length=5, polyorder=1, axis=0)  
        y_pred = (y_pred - np.nanmean(y_pred))/np.nanstd(y_pred) *np.nanstd(y_true) + np.nanmean(y_true)         
        predictions.append( pd.DataFrame(dict(y_true=y_true, y_pred=y_pred)) )
    return predictions              




# -----------------------------------------------------
# MAPE
# -----------------------------------------------------
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true))





# -----------------------------------------------------
# Evaluate
# -----------------------------------------------------
def evaluate(predictions): 

    rmse = [np.sqrt(mean_squared_error(prediction.y_true.values, prediction.y_pred.values)) for prediction in predictions]
    r2 = [r2_score(prediction.y_true.values, prediction.y_pred.values) for prediction in predictions]
    mape = [mean_absolute_percentage_error(prediction.y_true.values, prediction.y_pred.values) for prediction in predictions]

    return dict(
        RMSE = np.array(rmse), 
        R2 = np.array(r2), 
        MAPE = np.array(mape)
        )