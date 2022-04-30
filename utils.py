import os
from pathlib import Path
import pandas as pd 
import numpy as np
from itertools import chain, combinations
from scipy import signal, interpolate, stats
from math import*
from collections import defaultdict
from datetime import datetime, date, time, timedelta
import copy
import glob
from ast import literal_eval
import pywt
import pickle

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, LSTM, SimpleRNN, Dropout, MaxPooling2D, BatchNormalization, MaxPooling1D, Flatten, Conv1D, Conv2D, TimeDistributed
import tensorflow as tf
import tcn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from pycaret.regression import*

main_directory = str( Path(__file__).parents[1] )
dataset_folder_name = 'datasets'
datime_format = '%H:%M:%S.%f'
eps = 1e-12

####################################################################################################################################################
def get_markers_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/markers/' + file_name + '.csv'
####################################################################################################################################################
def get_rfid_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/rfid/' + file_name + '.csv'
####################################################################################################################################################
def get_arduino_file_path(dataset_name, file_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/arduino/' + file_name + '.csv'
####################################################################################################################################################
def get_sys_info(dataset_name):
    file_path = main_directory + '/' + dataset_folder_name +  '/' + dataset_name  + '/calibration_setting/sys_info.txt'
    sys_info = pd.read_csv(file_path, delimiter='\t').replace({'None': None})
    return sys_info
####################################################################################################################################################
def get_dataset_folder_path(dataset_name):
    return main_directory + '/' + dataset_folder_name + '/' + dataset_name + '/data' 
####################################################################################################################################################
####################################################################################################################################################
def create_folder(file_path):
    # Create folder if it does not exist
    folder_path = str( Path(file_path).parents[0] )
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
####################################################################################################################################################



####################################################################################################################################################
class SYSTEM(object):
    ################################################################################################################################################
    def __init__(self, dataset_name=None):
        self.reader = None
        self.tags = list()
        self.dataset_name = None
        
        if dataset_name is not None:
            self.dataset_name = dataset_name
            system_info = get_sys_info(dataset_name)    
            for idx in system_info.index:                
                node_info = system_info.loc[idx, system_info.columns!='name'].to_dict()                                
                node = NODE( **node_info )
                if system_info.loc[idx, 'name'][:3] == 'tag': self.tags.append(node)
                else: self.reader = node
        return
    ################################################################################################################################################
    def add_reader(self, color):
        self.reader = NODE( color )
        return
    ################################################################################################################################################
    def add_tag(self, color, IDD=None, port=None):
        self.tags.append( NODE(color, IDD=IDD, port=port) )   
        return               
    ################################################################################################################################################
    def get_motion(self, file_name, D=None, th_c=3, th_n=1, window_length=5, sampler_kind='linear'):

        # Reader
        reader_markers_df = self.reader.load_markers(self.dataset_name, file_name)
        reader_markers = np.array(reader_markers_df.markers.to_list()).reshape( [ len(reader_markers_df), -1, 3])
        reader_norm = np.nanmedian( get_norm( reader_markers[:10]), axis=0)
        reader_center = np.nanmedian( np.nanmean( reader_markers[:10], axis=1), axis=0)

        # Tags
        tags_markers_df = [tag.load_markers(self.dataset_name, file_name) for tag in self.tags if tag.color is not None]
        time_merged = tags_markers_df[0].time
        for df in tags_markers_df[1:]: time_merged = pd.merge(time_merged, df.time, how='inner')
        tags_markers_df = [pd.merge(tag_markers_df, time_merged, how='inner') for tag_markers_df in tags_markers_df]
        tags_markers = [np.array(tag_markers_df.markers.to_list()).reshape([len(tags_markers_df[0]), -1, 3]) for tag_markers_df in tags_markers_df]

        M = np.concatenate(tags_markers, axis=1) 
        Nt, Nm, _ = np.shape(M)

        time = tags_markers_df[0].time.to_numpy()
        dt = np.tile( np.diff(time).reshape(-1,1), (1 , 3))

        # Norm
        tags_norm =  np.array([ get_norm(M[:,[i,j,k],:]) for (i,j,k) in list(combinations(np.arange(Nm), 3))])
        # Remove jumps
        d_tags_norm = np.zeros_like(tags_norm)
        d_tags_norm = np.array([np.abs(np.diff(n, axis=0))/dt for n in tags_norm])        
        x, y, z = np.where(d_tags_norm > th_c)
        for shift in range(window_length): tags_norm[ x[y<Nt-shift], y[y<Nt-shift]+shift, z[y<Nt-shift] ] = np.nan
        tags_norm = signal.medfilt(tags_norm, [1, window_length,1])
        tags_norm = np.nanmedian(tags_norm, axis=0)
        tags_norm = signal.medfilt(tags_norm, [window_length,1])

        finite_idxs = np.where(np.all(np.isfinite(tags_norm), axis=1))[0]
        for n in range(3):
            resampler = interpolate.interp1d(time[finite_idxs], tags_norm[finite_idxs,n], kind=sampler_kind)
            # tags_norm[5:-5,n] = np.nan_to_num( resampler(time[5:-5]) )  
            tags_norm[finite_idxs[0]:finite_idxs[-1],n] = np.nan_to_num( resampler(time[finite_idxs[0]:finite_idxs[-1]]) )         
        tags_norm /= ( np.reshape(np.linalg.norm(tags_norm, axis=1), (-1,1)) * np.ones((1,3)) + eps)

        # Tags Center
        plane_point = np.nanmean(M, axis=1)
        for tag_marker in tags_markers:
            for j in range(tag_marker.shape[1]): 
                dist = np.sum(np.multiply( tags_norm, tag_marker[:,j,:]-plane_point), axis=1).reshape(-1,1) * np.ones([1,3])
                tag_marker[:,j,:] -= np.multiply( dist, tags_norm )
        tags_centers = np.array([ signal.medfilt(np.nanmean(tag_marker, axis=1), [window_length,1]) for tag_marker in tags_markers ])
        
        # Center
        center = np.nanmean(tags_centers,axis=0)
        d_center = np.abs(np.diff(center, axis=0))/dt 
        x, y = np.where(d_center > th_n)
        for shift in range(window_length): center[x[x<Nt-shift]+shift, y[x<Nt-shift] ] = np.nan
        center = signal.medfilt(center, [window_length,1])
        finite_idxs = np.where(np.all(np.isfinite(center), axis=1))[0]
        for n in range(3):
            resampler = interpolate.interp1d(time[finite_idxs], center[finite_idxs,n], kind=sampler_kind)
            center[finite_idxs[0]:finite_idxs[-1],n] = np.nan_to_num( resampler(time[finite_idxs[0]:finite_idxs[-1]]) )         
        # Tags centers
        for i, tag_center in enumerate(tags_centers): 
            dc = tag_center - center
            d = np.linalg.norm(dc, axis=1)
            if D is None: D = np.nanmedian(d)
            nc = dc / ( d.reshape([-1,1]) * np.ones((1,3)) + eps)
            tags_centers[i] = center + D*nc

        # Relative motion data
        tags_centers -= reader_center
        if reader_norm[1] == 0: thetaX = 0
        else: thetaX = atan( reader_norm[1]/reader_norm[2] )    
        thetaY = atan( -reader_norm[0] / sqrt(reader_norm[1]**2 + reader_norm[2]**2 + eps) )
        R_rot = get_rotationMatrix(thetaX, thetaY, 0)

        for n in range(len(tags_markers)): tags_centers[n] = np.matmul(R_rot, tags_centers[n].transpose()).transpose()
        tags_norm = np.matmul(R_rot, tags_norm.transpose()).transpose()

        clean_idx = np.all(np.all(~np.isnan(tags_centers),axis=2), axis=0) * np.all(~np.isnan(tags_norm), axis=1)
        data = dict( time = time[clean_idx], norm = list(tags_norm[clean_idx]) )
        for n in range(len(tags_markers)): data.update({'center_'+str(n+1):list(tags_centers[n,clean_idx])})
        
        return pd.DataFrame(data).dropna()  
    ################################################################################################################################################    
    def get_data(self, file_name, resample_dt=None, save=False, **motion_params):
        if 'sampler_kind' in motion_params: sampler_kind = motion_params['sampler_kind']
        else: sampler_kind = 'linear'

        data = self.get_motion(file_name, sampler_kind=sampler_kind, **motion_params)
        for i, tag in enumerate(self.tags):
            tag_data = tag.load_measurements(self.dataset_name, file_name)
            tag_data.columns = [ '{}{}'.format(c,'' if c in ['time'] else '_'+str(i+1)) for c in tag_data.columns]
            data = pd.merge( data, tag_data, on='time', how='outer', suffixes=('', ''), sort=True)

        data.dropna(inplace=True)

        if resample_dt is not None:
            resampling_time = np.arange(data.time.iloc[0], data.time.iloc[-1], resample_dt)
            resampled_data = pd.DataFrame({'time':resampling_time})

            for column in data.columns:
                if column =='time':continue
                val_ts = data[column].dropna()
                val = np.array(val_ts.to_list())
                time = data.time[val_ts.index].to_numpy()
                time_new = copy.deepcopy(resampling_time)
                time_new = time_new[ (time_new > time[0]) * ( time_new < time[-1]) ]

                if val.ndim > 1:
                    val_new = np.zeros((len(time_new), val.shape[1]))
                    for n in range(val.shape[1]):
                        resampler = interpolate.interp1d(time, val[:,n], kind=sampler_kind)
                        val_new[:,n] = resampler(time_new)         
                        # val_new[:,n] = np.nan_to_num( val_new[:,n]  )          
                else:
                    resampler = interpolate.interp1d(time, val, kind=sampler_kind)
                    val_new = resampler(time_new)         
                    # val_new = np.nan_to_num( val_new ) 

                resampled_column = pd.DataFrame({'time':time_new, column:list(val_new)})
                resampled_data = pd.merge( resampled_data, resampled_column, on='time', how='inner', suffixes=('', ''), sort=True)
            data = resampled_data


       # Processing
        data.interpolate(method='nearest', axis=0, inplace=True)      
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.time -= data.time.iloc[0]

        if save: 
            file_path = get_dataset_folder_path(self.dataset_name) + '/' + file_name + '.csv'
            create_folder(file_path)
            data.to_csv(file_path, index=False, sep=",") 

        return data
    ################################################################################################################################################    
    def get_dataset(self, as_dict=False, **params):
        file_name_list = [file_path.replace("\\","/").split('/')[-1][:-4] for file_path in glob.glob(get_markers_file_path(self.dataset_name, '')[:-4]+'*.csv')]            
        dataset = [ self.get_data(file_name, **params) for file_name in file_name_list]

        if as_dict:
            dataset_dict = defaultdict(list)
            for data in dataset:
                for key, value in data.to_dict('list').items():
                    dataset_dict[key].append(value)
            return dataset_dict

        return dataset
####################################################################################################################################################
class NODE(object):
    ################################################################################################################################################
    def __init__(self, color, IDD=None, port=None):
        self.color = color
        self.IDD = IDD
        self.port = port
    ################################################################################################################################################
    def load_markers(self, dataset_name, file_name):   
        markers_file_path = get_markers_file_path(dataset_name, file_name)  
        raw_df  = pd.read_csv(
            markers_file_path,                                            # relative python path to subdirectory
            usecols = ['time', self.color],                               # Only load the three columns specified.
            parse_dates = ['time'] )         

        # Time
        date_time = pd.to_datetime( raw_df['time'] , format=datime_format)
        time = np.array([np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time])
        
        # Markers
        markers = np.array([list(map(float, l.replace(']','').replace('[','').replace('\n','').split(", "))) for l in raw_df[self.color].values]) 
        
        # markers = raw_df[self.color].apply(literal_eval).to_list()
        # markers_npy = markers.reshape(len(time), -1, 3)        
        clean_idx = np.where(np.all(~np.isnan(markers), axis=1))[0]
        return pd.DataFrame({
            'time': time[clean_idx],
            'markers': list(markers[clean_idx,:])
            })                
    ################################################################################################################################################
    def load_measurements(self, dataset_name, file_name):        
        
        data = pd.DataFrame({'time':list()})
        if self.IDD is not None:  
            data = pd.merge( data, self.load_rssi(dataset_name, file_name), on='time', how='outer', suffixes=('', ''), sort=True)
        if self.port is not None: 
            data = pd.merge( data, self.load_vind(dataset_name, file_name), on='time', how='outer', suffixes=('', ''), sort=True)

        # data.interpolate(method='nearest', axis=0, inplace=True)      
        # data.dropna(inplace=True)
        # data.reset_index(drop=True, inplace=True)
        # data.reset_index(inplace=True)
        return data.dropna()
    ################################################################################################################################################
    def load_rssi(self, dataset_name, file_name):
        # Load data 
        rfid_file_path = get_rfid_file_path(dataset_name, file_name)       
        raw_df = pd.read_csv(
            rfid_file_path,                                                     # relative python path to subdirectory
            delimiter  = ';',                                                   # Tab-separated value file.
            usecols = ['IDD', 'Time', 'Ant/RSSI'],                              # Only load the three columns specified.
            parse_dates = ['Time'] )                                            # Intepret the birth_date column as a date      
        raw_df = raw_df.loc[ raw_df['IDD'] == self.IDD, :]

        date_time = pd.to_datetime( raw_df['Time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]
        rssi_df = raw_df['Ant/RSSI'].str.replace('Ant.No 1 - RSSI: ', '').astype(float) 

        return pd.DataFrame({
            'time':time,
            'rssi':rssi_df.tolist() 
            })
    ################################################################################################################################################
    def load_vind(self, dataset_name, file_name):
        # Load data 
        arduino_file_path = get_arduino_file_path(dataset_name, file_name)               
        raw_df = pd.read_csv(arduino_file_path)
        raw_df = raw_df.loc[ raw_df['port'] == self.port, :]

        date_time = pd.to_datetime( raw_df['time'] , format=datime_format)
        time = [ np.round( (datetime.combine(date.min, t.time())-datetime.min).total_seconds(), 2) for t in date_time]        
        vind_df = raw_df['vind'].astype(float) 
        
        return pd.DataFrame({
            'time':time,
            'meas_vind':vind_df.tolist() 
            })
####################################################################################################################################################
####################################################################################################################################################
def get_norm(markers):
    # Markers: Nt*Nm*3
    # Nm>=3
    norm = np.cross( markers[:,1,:] - markers[:,0,:], markers[:,2,:] - markers[:,0,:])
    idx = norm[:,2]<0
    for i in range(3): norm[idx,i] *= -1
    norm /= ( np.reshape(np.linalg.norm(norm, axis=1), (-1,1)) * np.ones((1,3)) + eps)
    return norm
####################################################################################################################################################
def get_rotationMatrix(XrotAngle, YrotAngle, ZrotAngle):
    Rx = np.array([ [1, 0,0], [0, cos(XrotAngle), -sin(XrotAngle)], [0, sin(XrotAngle), cos(XrotAngle)] ])
    Ry = np.array([ [cos(YrotAngle), 0, sin(YrotAngle)], [0, 1, 0], [-sin(YrotAngle), 0, cos(YrotAngle)] ])
    Rz = np.array([ [cos(ZrotAngle), -sin(ZrotAngle), 0], [sin(ZrotAngle), cos(ZrotAngle), 0], [0, 0, 1] ])
    Rtotal =  np.matmul(np.matmul(Rz,Ry),Rx)
    return Rtotal
####################################################################################################################################################



####################################################################################################################################################
class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)
    ################################################################################################################################################
    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) -  K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs
####################################################################################################################################################
class SYNTHESIZER(object):
    ################################################################################################################################################
    def __init__(self):
        pass
    ################################################################################################################################################
    def build(self, Nt, hiddendim=300, latentdim=300):
        epsilon_std = 1.0

        x = Input(shape=(Nt,))
        h = Dense(hiddendim, activation='tanh')(x)
        z_mu = Dense(latentdim)(h)
        z_log_var = Dense(latentdim)(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
        eps = Input(tensor=K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0], latentdim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        self.encoder = Model(x, z_mu)        
        self.decoder = Sequential([
            Dense(hiddendim, input_dim=latentdim, activation='tanh'),
            Dense(Nt, activation='sigmoid') ])
        x_pred = self.decoder(z)

        self.vae = Model(inputs=[x, eps], outputs=x_pred)

        def nll(y_true, y_pred): return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
        self.vae.compile(optimizer='rmsprop', loss=nll)
        
        return 
    ################################################################################################################################################
    def synthesize(self, train_data_, N=2000, window_length=15, epochs=500, hiddendim=300, latentdim=300):                
        
        train_data = np.array(train_data_)        
        if train_data.ndim == 2: train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], 1])
        _, Nt, Nc = np.shape(train_data)
        synth_data = np.zeros((N,Nt,Nc))

        for c in range(Nc):
            self.build(Nt, hiddendim=hiddendim, latentdim=latentdim)

            train_data_c = train_data[:,:,c]
            MIN, MAX = np.nanmin(train_data_c), np.nanmax(train_data_c)     
            train_data_c = (train_data_c - MIN) / (MAX-MIN)
            self.vae.fit(train_data_c, train_data_c, epochs=epochs, validation_data=(train_data_c, train_data_c), verbose=0)

            synth_data_c = self.decoder.predict(np.random.multivariate_normal( [0]*latentdim, np.eye(latentdim), N))
            synth_data_c = signal.savgol_filter( synth_data_c, window_length=window_length, polyorder=1, axis=1) 
            synth_data_c = signal.savgol_filter( synth_data_c, window_length=window_length, polyorder=1, axis=1) 
            synth_data_c = (synth_data_c - np.nanmin(synth_data_c)) / (np.nanmax(synth_data_c) - np.nanmin(synth_data_c) )
            synth_data[:,:,c] = synth_data_c * (MAX-MIN) + MIN
        
        return np.array(synth_data)
    ################################################################################################################################################
    def generate(self, train_data_, cond=False, **params):
        train_data = np.array(train_data_)
        if train_data.ndim == 2: train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1], 1])

        if cond:
            d = np.linalg.norm(train_data, axis=2)
            n = train_data / (np.reshape( d, [train_data.shape[0], train_data.shape[1], 1]) * np.ones([1,3]) + eps)
            D = np.nanmedian( np.nanmean(d, axis=1)) 

            theta = np.arccos( n[:,:,2] )
            phi = np.mod(np.arctan2(n[:,:,1], n[:,:,0]), 2*np.pi)

            phi_synth = self.synthesize(phi,  **params)
            theta_synth = self.synthesize(theta, **params)
            phi_synth, theta_synth = phi_synth[:,:,0], theta_synth[:,:,0]

            data_synth = np.zeros([phi_synth.shape[0], phi_synth.shape[1], 3])
            data_synth[:,:,0] = np.multiply(np.sin(theta_synth), np.cos(phi_synth))
            data_synth[:,:,1] = np.multiply(np.sin(theta_synth), np.sin(phi_synth))
            data_synth[:,:,2] = np.cos(theta_synth)    
            return data_synth * D
        
        return self.synthesize(train_data, **params)
####################################################################################################################################################
####################################################################################################################################################   
def generate_synth_motion_data_(train_dataset_name, save_dataset_name=None, resample_dt=.1, Ncoils=1, D=None, **params):

    dataset = load_dataset(train_dataset_name, resample_dt=resample_dt)
    Nt = np.inf
    for data in dataset: Nt = min(Nt, len(data)) 
    norm = np.array([ np.array(data[:Nt].norm.to_list()) for data in dataset ])

    centers = np.array([np.array([ list(center_i) for center_i in data[:Nt].filter(regex='center').values.transpose() ]) for data in dataset])
    centers = [centers[:,i,:,:] for i in range(np.shape(centers)[1])]
    center = np.nanmean(centers, axis=0)
    
    synthesizer = SYNTHESIZER()
    norm_synth = synthesizer.generate( norm, cond=True, **params)
    center_synth = synthesizer.generate( center, **params)
    centers_synth = np.zeros((Ncoils, center_synth.shape[0], center_synth.shape[1], center_synth.shape[2]))


    if Ncoils==1:
        centers_synth[0,:,:,:] = center_synth

    else:
        d_centers = np.array([center_n-center for center_n in centers])
        if D is None: D = np.nanmedian(np.linalg.norm(d_centers, axis=-1)) 

        phi = np.arctan2(d_centers[0,:,:,1], d_centers[0,:,:,0])
        phi_synth = synthesizer.generate( phi, **params)
        phi_synth = phi_synth[:,:,0]

        for i in range(centers_synth.shape[1]):
            for j in range(centers_synth.shape[2]):
                R_rot = get_rotationMatrix( 
                    np.arctan(norm_synth[i,j,1]/norm_synth[i,j,2] ), 
                    -np.arctan(norm_synth[i,j,0]/np.sqrt(norm_synth[i,j,1]**2+norm_synth[i,j,2]**2)), 0).transpose()
                for n in range(centers_synth.shape[0]): 
                    phi_n = phi_synth[i,j] + n * 2*pi/Ncoils
                    c_n = np.array([ D*np.cos(phi_n), D*np.sin(phi_n), 0 ])             
                    centers_synth[n,i,j,:] = np.matmul(R_rot, c_n.reshape(3,1)).reshape(1,3)[0] + center_synth[i,j]

    time = list(np.arange(Nt)*resample_dt)

    if save_dataset_name is not None:
        folder_path = get_dataset_folder_path(save_dataset_name) 
        create_folder(folder_path + '/tst.csv')
        
        for m in range(norm_synth.shape[0]):
            file_path = folder_path + '/record_' + "{0:0=4d}".format(m) + '.csv'
            data =  pd.DataFrame({ 'time': time, 'norm': list(norm_synth[m]) }) 
            for n in range(Ncoils): data['center_'+str(n+1)] = list(centers_synth[n, m]) 
            data.to_csv(file_path, index=None) 
    else:
        return dict(
            time = time,
            norm = norm_synth,
            centers = centers_synth
        )
####################################################################################################################################################   
def generate_synth_motion_data(train_dataset_name, D=None, resample_dt=.1, **params):
    dataset = load_dataset(train_dataset_name, resample_dt=resample_dt)
    Nt = np.inf
    for data in dataset: Nt = min(Nt, len(data))
    Ncoils = data.filter(regex='center').values.shape[1]
    
    # Only uses norm_1 and returns one norm
    norm = np.array([ np.array(data[:Nt].norm_1.to_list()) for data in dataset ])

    centers = np.array([np.array([ list(center_i) for center_i in data[:Nt].filter(regex='center').values.transpose() ]) for data in dataset])
    centers = [centers[:,i,:,:] for i in range(np.shape(centers)[1])]
    center = np.nanmean(centers, axis=0)
    
    synthesizer = SYNTHESIZER()
    norm_synth = synthesizer.generate( norm, cond=True, **params)
    center_synth = synthesizer.generate( center, **params)
    centers_synth = np.zeros((Ncoils, center_synth.shape[0], center_synth.shape[1], center_synth.shape[2]))


    if Ncoils==1:
        centers_synth[0,:,:,:] = center_synth

    else:
        d_centers = np.array([center_n-center for center_n in centers])
        if D is None: D = np.nanmedian(np.linalg.norm(d_centers, axis=-1)) 

        phi = np.arctan2(d_centers[0,:,:,1], d_centers[0,:,:,0])
        phi_synth = synthesizer.generate( phi, **params)
        phi_synth = phi_synth[:,:,0]

        for i in range(centers_synth.shape[1]):
            for j in range(centers_synth.shape[2]):
                R_rot = get_rotationMatrix( 
                    np.arctan(norm_synth[i,j,1]/norm_synth[i,j,2] ), 
                    -np.arctan(norm_synth[i,j,0]/np.sqrt(norm_synth[i,j,1]**2+norm_synth[i,j,2]**2)), 0).transpose()
                for n in range(centers_synth.shape[0]): 
                    phi_n = phi_synth[i,j] + n * 2*pi/Ncoils
                    c_n = np.array([ D*np.cos(phi_n), D*np.sin(phi_n), 0 ])             
                    centers_synth[n,i,j,:] = np.matmul(R_rot, c_n.reshape(3,1)).reshape(1,3)[0] + center_synth[i,j]

    time = list(np.arange(Nt)*resample_dt)

    return dict(
        time = time,
        norm = norm_synth,
        centers = centers_synth
    )
####################################################################################################################################################   



####################################################################################################################################################
def load_dataset(dataset_name, resample_dt=None, as_dict=True):
    file_path_list = glob.glob(get_dataset_folder_path(dataset_name) +'/*.csv')
    data = pd.read_csv(file_path_list[0])
    columns = list(set(['time', *data.filter(regex='norm').columns, *data.filter(regex='center_').columns, *data.filter(regex='vind').columns]))
    
    converters = dict()
    for column in columns:
        val = data.loc[0,column]
        if type(val) == str:
            if ',' in val: converters.update({ column: literal_eval })
            else: converters.update({ column:lambda x: list(map(float, x.strip('[]').split())) })
    
    # converters = {key:lambda x: list(map(float, x.strip('[]').split())) for key in ['norm', 'center_1', 'center_2']}
    # converters = {key:literal_eval for key in ['norm', 'center_1', 'center_2']}
    
    dataset = list()
    for file_path in file_path_list:        
        data = pd.read_csv(file_path, converters=converters)

        if resample_dt is not None:
            # resampled_data = pd.DataFrame({'time':np.arange(data.time.iloc[0], data.time.iloc[-1], resample_dt)})
            # for column in data.columns:
            #     if column =='time':continue
            #     resampler = interpolate.interp1d(data.time.values, data[column].values, kind='linear')
            #     resampled_data[column] = np.nan_to_num( resampler(resampled_data.time.values) )
            #     # resampled_data[column] = signal.savgol_filter( resampled_data[column], window_length=window_length, polyorder=1, axis=0)             
            # data = resampled_data  
            time_new = np.arange(data.time.iloc[0], data.time.iloc[-1], resample_dt) 
            time_old = data.time.values
            resampled_data_dict = dict(time=time_new)
            for column in data.columns:    
                if column =='time':continue
                column_val = np.array(data[column].to_list()).reshape((len(time_old),-1))
                resampled_column_val = np.zeros((len(time_new), np.shape(column_val)[1]))    
                for i in range(np.shape(column_val)[1]):
                    resampler = interpolate.interp1d(time_old, column_val[:,i], kind='linear')
                    resampled_column_val[:,i] = np.nan_to_num( resampler(time_new) )
                resampled_data_dict.update({column:list(resampled_column_val)})
            data = pd.DataFrame(resampled_data_dict)              
        dataset.append(data)

    return dataset
####################################################################################################################################################
####################################################################################################################################################
class DATA(object):    
    ######################################################################################################
    def __init__(self, X=[], Y=[], dataset_name=None, **params): 
        if dataset_name is not None:
            self.load( dataset_name, **params)   
        else:
            self.X = np.array(X)  
            self.Y = np.array(Y)        
        return        
    ######################################################################################################
    def segment(self, win_size, step=None):
        # returns Nsample list of n * win_size * Nf   where n is number of segments extracted from Nt samples 
        if step is None: step = win_size                
        X, Y = list(), list()
        for (x,y) in zip(self.X, self.Y):
            Nt = np.shape(x)[0]
            # x_s = [ x[t:t+win_size,:] for t in range(0, Nt-win_size, step) ]
            # y_s = [ y[t+win_size] for t in range(0, Nt-win_size, step) ]                
            x_s = np.array([ x[t:t+win_size,:] for t in range(0, Nt-win_size+1, step) ])
            y_s = np.array([ y[t+win_size-1] for t in range(0, Nt-win_size+1, step) ])
            X.append(x_s)
            Y.append(y_s)
        return DATA(X, Y)         
    ######################################################################################################
    def get_features(self):
        features = np.array([ get_features_sample(x) for x in self.X ])
        return DATA(X=features, Y=self.Y)
    ######################################################################################################
    def get_df(self, merge=False):
        if merge:
            X = np.concatenate( self.X, axis=0)
            Nf = X.shape[1]*X.shape[2]
            X = X.reshape(-1, Nf)
            data_df = pd.DataFrame( X, columns=['feature_'+str(i) for i in range(Nf)] )
            data_df['target'] = list(np.concatenate( self.Y, axis=0))
            return data_df

        else:
            data_df_list= list()
            for (x,y) in zip(self.X, self.Y):
                
                Nf = x.shape[1]*x.shape[2]
                x = x.reshape(-1, Nf)
                data_df = pd.DataFrame( x, columns=['feature_'+str(i) for i in range(Nf)] )
                data_df['target'] = list(y)
                data_df_list.append(data_df)

            return data_df_list
    ######################################################################################################
    def merge(self, new_dataset):
        merged_dataset = copy.deepcopy(self)
        merged_dataset.X = np.array([*self.X, *new_dataset.X])
        merged_dataset.Y = np.array([*self.Y, *new_dataset.Y])
        return merged_dataset
    ######################################################################################################
    def select(self, idx_list):
        selected_dataset = copy.deepcopy(self)
        selected_dataset.X = self.X[idx_list]
        selected_dataset.Y = self.Y[idx_list]
        return selected_dataset
    ######################################################################################################
    def split(self, ratio):
        N = len(self.X)
        idxs = np.arange(N)
        # random.shuffle(idxs)
        
        Ntrain = int(N*ratio)
        data_p1 = self.select(idxs[:Ntrain])
        data_p2 = self.select(idxs[Ntrain:])
        
        return data_p1, data_p2
    ######################################################################################################
    def mtx( self, Nt_mtx='max' ):  
        # This function padds or cuts all input data (X) to make them same length and generate matrix data(X_mtx)
        # it also nomalize data X-mean(X)
        data_mtx = copy.deepcopy(self)
        if len(np.shape(data_mtx.X))>1:  return data_mtx    

        Nd, Nf = len(self.X),  np.shape(self.X[0])[1]
        Nt_list = list()
        for x in self.X: Nt_list.append( np.shape(x)[0] )
        if type(Nt_mtx) is str: Nt = int( eval('np.' + Nt_mtx)(Nt_list) )
        else:  Nt = Nt_mtx
        data_mtx.X = np.zeros( (Nd,Nt,Nf) )
        data_mtx.Y = np.zeros( (Nd,Nt) )
        for idx, (x,y) in enumerate(zip(self.X, self.Y)): 
            # x = np.subtract(x,np.mean(x,axis=0))        
            nt = np.shape(x)[0]
            if Nt >= nt:
                # data_mtx.X[idx,:,:] = np.pad( x, ((0,Nt-nt),(0,0)),'constant')
                data_mtx.X[idx,:nt,:] = x
                data_mtx.Y[idx,:nt] = y
            else:
                data_mtx.X[idx,:] = x[:Nt,:]
                data_mtx.Y[idx,:] = y[:Nt]
        return data_mtx
    ######################################################################################################
    def bound(self, min_value=None, max_value=None):
        # This function limits the amplitude value 
        
        bounded_data = copy.deepcopy(self)
        if min_value is not None:
            for x in bounded_data.X: x[ x<min_value ] = min_value
        if max_value is not None:                
            for x in bounded_data.X: x[ x>max_value ] = max_value
        
        return bounded_data
    ######################################################################################################
    def trim(self, keep_ratio=None):
        trimmed_data = copy.deepcopy(self)
        trimmed_data.X = list()
        
        if keep_ratio is None:
            dt = 20   
            for x in self.X:     
                N = len(x)
                n1, n2 = dt, N-dt 
                xx = abs( np.diff(x))
                xx = np.sum(xx, axis=1)    
                xx = abs(np.diff(xx))
                xx /= ( np.nanmax(xx) + eps )                 
                idxs = np.where( xx > 0.5 )[0]    
                idxs1 = idxs[idxs < 0.5*N] 
                idxs2 = idxs[idxs > 0.5*N]      
                if np.any(idxs1): n1 = np.min(idxs1) + dt
                if np.any(idxs2): n2 = np.max(idxs2) - dt   
                if (n2-n1) < 0.5*N: n1, n2 = 0, N            
                trimmed_data.X.append( x[n1:n2,:] )
        else:   
            for x in self.X:
                L = int( len(x) * keep_ratio)
                trimmed_data.X.append( x[:L,:] ) 

        trimmed_data.X = np.array(trimmed_data.X)    
        return trimmed_data    
    ######################################################################################################
    def quantize(self, Qstep):        
        quantized_data = copy.deepcopy(self)
        for idx, x in enumerate(quantized_data.X): 
            quantized_data.X[idx] = Qstep * np.floor(x/Qstep)
        return quantized_data   
    ######################################################################################################
    def clean(self):
        # cleans data from NANs ! 
        cleaned_data = copy.deepcopy(self)
        for idx, x in enumerate(cleaned_data.X):
            if np.any(np.isnan(x)):
                df = pd.DataFrame(x)
                df = df.fillna(method='ffill', axis=0).bfill(axis=0)      
                cleaned_data.X[idx] = df.as_matrix()

        return cleaned_data                
    ######################################################################################################
    def filter_noise(self, window_length=5, polyorder=2):
        filtered_data = copy.deepcopy(self)
        for n, x in enumerate(self.X):
            for i in range(np.shape(x)[1]):
                filtered_data.X[n][:,i] = signal.savgol_filter(x[:,i], window_length, polyorder)        
        return filtered_data
    ######################################################################################################
    def MinMax(self):
        # Rescale data value to (0,1)
        normalized_data = copy.deepcopy(self)
        for idx, x in enumerate(normalized_data.X): 
            MIN = np.nanmin(x,axis=0)
            MAX = np.nanmax(x,axis=0)
            normalized_data.X[idx] = np.subtract(x,MIN) / ( np.subtract(MAX,MIN) + eps )
        return normalized_data    
    ######################################################################################################
    def standardize(self, scale=True):
        normalized_data = copy.deepcopy(self)
        STD = 1
        for idx, x in enumerate(normalized_data.X): 
            MEAN = np.mean(x,axis=0)
            if scale: STD = np.std(x,axis=0) + eps
            normalized_data.X[idx] = np.subtract(x,MEAN) / STD    
        return normalized_data         
####################################################################################################################################################
def get_features_sample(x_t):
    features = list()   
    axis = -2
    x_f = np.real( np.fft.fft(x_t, axis=axis) )
    x_wA, x_wD = pywt.dwt(x_t, 'db1', axis=axis)
    dx_t = np.diff( x_t, axis=axis )
    for x_ in [x_t, x_f, x_wA, x_wD, dx_t]:                                    
        features.append( np.mean( x_, axis=axis)) 
        features.append( np.std( x_, axis=axis ))                               
        features.append( np.median( x_, axis=axis ))               
        features.append( np.min( x_, axis=axis ))               
        features.append( np.max( x_, axis=axis ))               
        features.append( np.var( x_, axis=axis ))               
        features.append( np.percentile( x_, 25, axis=axis ))               
        features.append( np.percentile( x_, 75, axis=axis ))               
        features.append( stats.skew( x_, axis=axis))               
        features.append( stats.kurtosis( x_, axis=axis))               
        features.append( stats.iqr( x_, axis=axis))               
        features.append( np.sqrt(np.mean(np.power(x_,2), axis=axis)))   
    features = np.concatenate(np.array(features), axis=-1)

    if np.ndim(features) == 1: return features.reshape(60,-1)
    return features.reshape(np.shape(x_t)[0], 60, -1)
####################################################################################################################################################



