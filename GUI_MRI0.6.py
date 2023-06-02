import numpy as np
import scipy.fft as fft
import scipy.io
import matplotlib.pyplot as plt 
import keaDataProcessing as keaProc
import cv2
import math
import cmath
import datetime
import interactiveAxesPlot_SE as pltaxes_SE
import interactiveAxesPlot_GE as pltaxes_GE
import interactiveAxesPlot_IN as pltaxes_IN
import interactiveAxesPlot_DIN as pltaxes_DIN
import interactiveAxesPlot_FLAIR as pltaxes_FLAIR
import interactiveAxesPlot_SSFP as pltaxes_SSFP
import interactiveAxesPlot_Dif as pltaxes_Dif
import interactiveAxesPlot_TSE as pltaxes_TSE
import interactiveAxesPlot_Lowpass as pltaxes_low
import interactiveAxesPlot_Highpass as pltaxes_high
import interactiveAxesPlot_Gausspass as pltaxes_gauss
import interactiveAxesPlot_Nonlocalpass as pltaxes_nonlocal
import interactivePlot_SNR as pltsnr
from Sequences import *
from Grad_distortion_functions import *
from scipy import signal
from scipy import constants
from scipy.ndimage import zoom
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
from scipy import ndimage
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy.interpolate import RectBivariateSpline
from numpy import inf
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma # estimate_sigma --> wavelet-based estimator of the (Gaussian) noise standard deviation.
from skimage.metrics import peak_signal_noise_ratio

# For matplotlib plots on a window:
# First, we need to create the figure object using the Figure() class. 
# Then, a Tkinter canvas(containing the figure) is created using FigureCanvasTkAgg() class. Matplotlib charts by default have a toolbar at the bottom. 
# When working with Tkinter, however, this toolbar needs to be embedded in the canvas separately using the NavigationToolbar2Tk() class.

root = Tk()
root.title("Low field MRI simulator")
root.state('zoomed') # To have a full screen display

##### FUNCTIONS #####
# Reset function
def reset():
    """
    Reset all the parameters to their inital value and plot the initial proton density images
    
    Input: None 
    Output: None
    """
    J = [1, 2, 3]
    num = [128, 150, 135]
    
    for i in range(3):
        if i == 0:
            t = np.rot90(M0_3D[num[i],:,:])
        elif i == 1:
            t = np.rot90(M0_3D[:,num[i],:])
        elif i == 2:
            t = np.rot90(M0_3D[:,:,num[i]])
        
        fig = plt.figure(figsize=(imag_size,imag_size))     # Create plot
        plt.imshow(t, cmap='gray')                          # Create image plot 
        plt.axis('off')
        canvas = FigureCanvasTkAgg(fig, root)               # Tkinter canvas which contains matplotlib figure
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = J[i], rowspan = 4) # Placing canvas on Tkinter window        
        plt.close()

    TR_entry.delete(0, END); TR_entry.insert(0, '500')
    TE_entry.delete(0, END); TE_entry.insert(0, '20')
    TI_entry.delete(0, END); TI_entry.insert(0, '250')
    FOV1_entry.delete(0, END); FOV1_entry.insert(0, '250')
    FOV2_entry.delete(0, END); FOV2_entry.insert(0, '300')
    FOV3_entry.delete(0, END); FOV3_entry.insert(0, '275')
    Res1_entry.delete(0, END); Res1_entry.insert(0, '1.95')
    Res2_entry.delete(0, END); Res2_entry.insert(0, '2.34')
    Res3_entry.delete(0, END); Res3_entry.insert(0, '2.14')
    Bandwidth_entry.delete(0, END); Bandwidth_entry.insert(0, '50000')
    Alpha_entry.delete(0, END); Alpha_entry.insert(0, '45')
    G_entry.delete(0,END); G_entry.insert(0, '10')
    smalldelta_entry.delete(0,END); smalldelta_entry.insert(0, '1')
    bigdelta_entry.delete(0,END);bigdelta_entry.insert(0, '2')

    Time_scan_num_label.grid_forget()
    Data_mat1_label.grid_forget()
    Bd_by_pixel_label.grid_forget()
    SNR_num_label.grid_forget()
        
###### Functions for estimating the noise in image, computing the noise, and compute the SNR
def estimate_noise_var(I):
    """
    from paper : 
    NOTE
    Fast Noise Variance Estimation
    JOHN IMMERKÆR
    COMPUTER VISION AND IMAGE UNDERSTANDING
    Vol. 64, No. 2, September, pp. 300–302, 1996
    
    Input: I --> image
    Output: std --> standard deviation of the noise in image I
    
    """
    H, W = I.shape
    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    std = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    std = std * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return std

# Noise computation (NEW WAY)
def noise_generation(final_seq, Data_mat, Resolution, Bandwidth):
    """
    Creates a random noise tensor of same size as 'Data_mat' (data shape of simulated data) 
    The std of the noise to add is based on the resolution, bandwidth and number of voxels in the final data
    
    Input: final_seq  --> simulated data
           Data_mat   --> shape of the simulated data
           Resolution --> resolution of the simulated data
           Bandwidth  --> bandwidth
                      
    Output: n --> noise tensor
    """
    S = [int(Data_mat[0]/2),int(Data_mat[1]/2),int(Data_mat[2]/2)]                 # S is a vector with the center of the Data matrix
    length = 20                                                                    # size of the boxes used to compute the SNR
    half_length = length/2
    mean_box = [int(S[0]-half_length), int(S[0]+half_length), int(40-half_length), int(40+half_length)]
    mean = np.nanmean(final_seq[mean_box[0]:mean_box[1],mean_box[2]:mean_box[3],S[2]]) # Computes the mean of the signal without noise
    tot_res = Resolution[0]*Resolution[1]
    tot_data = Data_mat[0]*Data_mat[1]

    # Equation below links the SNR with the resolution, bandwidth and FOV
    # noise_relation = 8.688078756400369 * tot_res / np.sqrt(np.divide(Bandwidth,tot_data)) # 8.688... is a constant computed from real a lowfield image
    noise_relation = 2 * tot_res / np.sqrt(np.divide(Bandwidth,tot_data))
    std = mean / noise_relation
    n = np.abs(np.random.normal(0,std,Data_mat))  
    return n

# Function computing SNR based on 2 boxes defined by the parameters (box "s" is in a noisy region of the image (to compute the std of noise) and box "m" is in a image region where their is signal (to compute the mean of the signal))
def snr_homemade(im,s1,s2,s3,s4,m1,m2,m3,m4):
    """
    Computes the SNR of an image as the ration of the mean in the box defined by points {m1,m2,m3,m4} by the standard deviation in the bx defined by the points {s1,s2,s3,s4}
    
    Input: im --> image
           s1,s2 --> top and bottum of "noise box" (height)
           s3,s4 --> left and rigth of "noise box" (width)
           m1,m2 --> top and bottum of "signal box" (height)
           m3,m4 --> left and rigth of "signal box" (width)
           
    Output: snr --> signal to noise ration of im
    """
    m = np.mean(im[m1:m2,m3:m4])
    std = np.std(im[s1:s2,s3:s4])
    
    if std == 0:
        snr = 1000
    else:
        snr = m/std    
    return snr

# Interactive function used to visualize and define the boxes used in the computation of the SNR
def SNR_vis():
    """
    Creates an interactive plot to visualise and define the boxes used in the computation of the SNR
    Input: None       
    Output: None 
    """
    global SNR_length
    global SNR_noise_box_center
    global SNR_mean_box_center
    global SNR_define
    
    TR = TR_entry.get(); TR = int(TR); TR = np.divide(TR,1000) # Divide by 1000 to have the values in milli
    TE = TE_entry.get(); TE = int(TE); TE = np.divide(TE,1000)
    fov1 = int(FOV1_entry.get()); fov2 = int(FOV2_entry.get()); fov3 = int(FOV3_entry.get()); FOV = [fov1, fov2, fov3]
    res1 = float(Res1_entry.get()); res2 = float(Res2_entry.get()); res3 = float(Res3_entry.get()); Resolution = [res1, res2, res3]
    Bandwidth = int(Bandwidth_entry.get())
    Data_mat = np.divide(FOV, Resolution)
    Data_mat = [int(Data_mat[0]), int(Data_mat[1]), int(Data_mat[2])]  
    
    # loading the 3D distorted data
    T1_3D_grad = np.load('T1_3D_gradientdistortion02.npy')
    T2_3D_grad = np.load('T2_3D_gradientdistortion02.npy')
    M0_3D_grad = np.load('M0_3D_gradientdistortion02.npy')
    B1map_3D_grad = np.load('B1_3D_gradientdistortion02.npy')
    # Resizing
    n_seq = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_seq = np.zeros((Data_mat))
    SE_3D = spin_echo_seq(TR, TE, T1_3D_grad, T2_3D_grad, M0_3D_grad); SE_3D = np.multiply(SE_3D, B1map_3D_grad)
    # Resizing
    for x in range(T1_3D.shape[0]):
        n_seq[x,:,:] = cv2.resize(SE_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
    for x in range(Data_mat[1]):
        final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
    # Noise
    n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
    
    fig, ax = plt.subplots() # figsize=(4, 4)
    figSNR = pltsnr.interactivePlot(fig, ax, final_seq + n, Data_mat, snr_homemade, plotAxis = 2, fov = FOV)        
    plt.show() 
    
    # REMARK the coordinates in the boxes center vectors are defined on the data rotated 90 degree!
    SNR_length = figSNR.length
    SNR_noise_box_center = [int(figSNR.noise_box_center[0]), int(figSNR.noise_box_center[1])]
    SNR_mean_box_center = [int(figSNR.mean_box_center[0]), int(figSNR.mean_box_center[1])]
    
    if np.sum(SNR_noise_box_center) != 0 and np.sum(SNR_mean_box_center) != 0:
        SNR_define = 'yes'

#/////////////////////// FUNCTIONS FOR THE SEQUENCES ///////////////////////#

# Function computing the b coef and the TEmin related to the diffusion seq
def b_TEmin():
    """
    Computes and prints the minimum TE and the b coefficient for the diffusion sequence

    Input : None
    Output: None
    """    
    global Bval_label
    global TEminval_label
    global B
    global dif_TEmin

    gamma =  42.58*(10**6)#*2*constants.pi      # gyromagnetic ratio for hydrogen 42.58 [MHz/T] 
    
    G = G_entry.get(); G = float(G); G = np.divide(G,1000)
    smalldelta = smalldelta_entry.get(); smalldelta = float(smalldelta); smalldelta = np.divide(smalldelta,1000)
    bigdelta = bigdelta_entry.get(); bigdelta = float(bigdelta); bigdelta = np.divide(bigdelta,1000)
    
    B = (bigdelta - smalldelta/3)*(gamma*G*smalldelta)**2; 
    dif_TEmin = smalldelta + bigdelta; 

    Bval_label.grid_forget()
    Bval_label = Label(frame3, text = np.round(B,2), font=("Helvetica", 12));              Bval_label.grid(row = 6, column = 4)
    TEminval_label.grid_forget()
    TEminval_label = Label(frame3, text = np.round(dif_TEmin*1000,2), font=("Helvetica", 12)); TEminval_label.grid(row = 7, column = 4)
    
def TETRmin():
    """
    Computes and prints the minimum TE (or TR) for a sequence to be simulated
    
    Input: None
    Output: None
    """
    global TE
    global FOV
    global Data_mat
    global Resolution
    global Bandwidth
    global minimumTE_num_label
    
    TE = TE_entry.get(); TE = int(TE); TE = np.divide(TE,1000) # Divide by 1000 to have the values in milli
    fov1 = FOV1_entry.get(); fov1 = int(fov1)
    fov2 = FOV2_entry.get(); fov2 = int(fov2)
    fov3 = FOV3_entry.get(); fov3 = int(fov3)
    res1 = Res1_entry.get(); res1 = float(res1) 
    res2 = Res2_entry.get(); res2 = float(res2)
    res3 = Res3_entry.get(); res3 = float(res3)
    bd = Bandwidth_entry.get(); Bandwidth = int(bd)
    FOV = [fov1, fov2, fov3]
    Resolution = [res1, res2, res3]
    Data_mat = np.divide(FOV, Resolution)
    Data_mat = [int(Data_mat[0]), int(Data_mat[1]), int(Data_mat[2])]
    
    minimumTE = Data_mat[0]/(2*Bandwidth)
    minimumTR = 2*minimumTE
    
    minimumTE_num_label.grid_forget()
    
    if Pre_def_seq == 'SSFP':
        minimumTE_label = Label(frame3, text = "TRmin (ms) ", font=("Helvetica", 12)).grid(row = 5, column = 6)
        minimumTE_num_label = Label(frame3, text = np.round(minimumTR * 1000,2), font=("Helvetica", 12))
        minimumTE_num_label.grid(row = 5, column = 7)
    else:
        minimumTE_label = Label(frame3, text = "TEmin (ms)", font=("Helvetica", 12)).grid(row = 5, column = 6)
        minimumTE_num_label = Label(frame3, text = np.round(minimumTE * 1000,2), font=("Helvetica", 12))
        minimumTE_num_label.grid(row = 5, column = 7)
    
def showTEeff(*args):
    """
    Computes and prints the effective TE for the TSE sequence

    Input : None
    Output: None
    """
    global ETL
    global TE
    global TEeff_val_label
    ETL = ETL_entry.get(); ETL = int(ETL) 
    TE = TE_entry.get(); TE = int(TE)
    trajectory = traj.get()
    eff = 0
    
    if trajectory == 'Linear':
        eff = 0.5 * TE * ETL
    elif trajectory == 'In-out':
        eff = TE
    elif trajectory == 'Out-in':
        eff = TE * ETL   
        
    TEeff_val_label.grid_forget()
    TEeff_val_label = Label(frame3, text = np.round(eff,2), font = ("Helvetica", 12)); TEeff_val_label.grid(row = 7, column = 3) 

def remove_widgets(seq):
    """
    Removes the widgets (labels and entries) that were needed for the sequence
    
    Input : seq --> string of Pre_def_seq
    Output: None
    """
    if seq == 'SE':
        TE_entry.grid_forget();         TE_label.grid_forget()
    elif seq == 'GE':
        TE_entry.grid_forget();         TE_label.grid_forget()
        Alpha_entry.grid_forget();      Alpha_label.grid_forget()
    elif seq == 'IN':
        TE_entry.grid_forget();         TE_label.grid_forget()
        TI_entry.grid_forget();         TI_label.grid_forget()
    elif seq == 'Double IN':
        TE_entry.grid_forget();         TE_label.grid_forget()
        TI_entry.grid_forget();         TI_label.grid_forget()
        TI2_entry.grid_forget();        TI2_label.grid_forget()
    elif seq == 'FLAIR':
        TE_entry.grid_forget();         TE_label.grid_forget()
        TI_entry.grid_forget();         TI_label.grid_forget()
    elif seq == 'Dif':
        TE_entry.grid_forget();         TE_label.grid_forget()
        G_label.grid_forget();          G_entry.grid_forget();         bcomput_button.grid_forget()
        smalldelta_label.grid_forget(); smalldelta_entry.grid_forget()
        bigdelta_label.grid_forget();   bigdelta_entry.grid_forget()
        B_label.grid_forget()
        Bval_label.grid_forget()
        TEmin_label.grid_forget()   
        TEminval_label.grid_forget()      
    elif seq == 'TSE':
        TE_entry.grid_forget();         TE_label.grid_forget()
        ETL_entry.grid_forget();        ETL_label.grid_forget()
        Kspace_traj_label.grid_forget();tse_drop.grid_forget()
        TEeff_label.grid_forget()
        TEeff_val_label.grid_forget()
    elif seq == 'SSFP':
        Alpha_entry.grid_forget();      Alpha_label.grid_forget()
        
def show_widgets(seq):
    """
    show the widgets (labels and entries) that are needed for the sequence and the minimum TE (or TR)
    
    Input : seq --> string of Pre_def_seq
    Output: None
    """    
    global TR
    global TE
    global FOV
    global Data_mat
    global Resolution
    global Bandwidth
    global minimumTE_num_label
    
    TR = TR_entry.get(); TR = int(TR); TR = np.divide(TR,1000) # Divide by 1000 to have the values in milli
    TE = TE_entry.get(); TE = int(TE); TE = np.divide(TE,1000)
    fov1 = FOV1_entry.get(); fov1 = int(fov1)
    fov2 = FOV2_entry.get(); fov2 = int(fov2)
    fov3 = FOV3_entry.get(); fov3 = int(fov3)
    res1 = Res1_entry.get(); res1 = float(res1) 
    res2 = Res2_entry.get(); res2 = float(res2)
    res3 = Res3_entry.get(); res3 = float(res3)
    bd = Bandwidth_entry.get(); Bandwidth = int(bd)
    FOV = [fov1, fov2, fov3]
    Resolution = [res1, res2, res3]
    Data_mat = np.divide(FOV, Resolution)
    Data_mat = [int(Data_mat[0]), int(Data_mat[1]), int(Data_mat[2])]
    
    minimumTE = Data_mat[0]/(2*Bandwidth)
    minimumTR = 2*minimumTE
    
    minimumTE_num_label.grid_forget()
    
    if seq == 'SE':
        TE_label.grid(row = 5, column = 0);         TE_entry.grid(row = 5, column = 1);
    elif seq == 'GE':
        TE_label.grid(row = 5, column = 0);         TE_entry.grid(row = 5, column = 1);
        Alpha_label.grid(row = 6, column = 0);      Alpha_entry.grid(row = 6, column = 1)
    elif seq == 'IN':
        TE_label.grid(row = 5, column = 0);         TE_entry.grid(row = 5, column = 1);
        TI_label.grid(row = 6, column = 0);         TI_entry.grid(row = 6, column = 1); TI_entry.delete(0,END); TI_entry.insert(0, '250')  
    elif seq == 'Double IN':
        TE_label.grid(row = 5, column = 0);         TE_entry.grid(row = 5, column = 1);
        TI_label.grid(row = 6, column = 0);         TI_entry.grid(row = 6, column = 1); TI_entry.delete(0,END); TI_entry.insert(0, '250')
        TI2_label.grid(row = 7, column = 0);        TI2_entry.grid(row = 7, column = 1)
    elif seq == 'FLAIR':
        TE_label.grid(row = 5, column = 0);         TE_entry.grid(row = 5, column = 1);
        TI_label.grid(row = 6, column = 0);         TI_entry.grid(row = 6, column = 1); TI_entry.delete(0,END); TI_entry.insert(0, '2561')         
    elif seq == 'Dif':
        TE_label.grid(row = 5, column = 0);         TE_entry.grid(row = 5, column = 1);
        G_label.grid(row = 6, column = 0);          G_entry.grid(row = 6, column = 1);  bcomput_button.grid(row = 7, column = 2)
        smalldelta_label.grid(row = 7, column = 0); smalldelta_entry.grid(row = 7, column = 1)
        bigdelta_label.grid(row = 8, column = 0);   bigdelta_entry.grid(row = 8, column = 1)
        B_label.grid(row = 7, column = 3)
        Bval_label.grid(row = 7, column = 4)
        TEmin_label.grid(row = 8, column = 3)     
        TEminval_label.grid(row = 8, column = 4)
    elif seq == 'TSE':
        TE_label.grid(row = 5, column = 0);         TE_entry.grid(row = 5, column = 1)
        ETL_label.grid(row = 6, column = 0);        ETL_entry.grid(row = 6, column = 1)
        Kspace_traj_label.grid(row = 7, column = 0);tse_drop.grid(row = 7, column = 1)
        TEeff_label.grid(row = 7, column = 2)
        TEeff_val_label.grid(row = 7, column = 5)
    elif seq == 'SSFP':
        Alpha_label.grid(row = 5, column = 0);      Alpha_entry.grid(row = 5, column = 1)      
        
    if seq == 'SSFP':
        minimumTE_label = Label(frame3, text = "TRmin (ms) ", font=("Helvetica", 12)).grid(row = 5, column = 6)
        minimumTE_num_label = Label(frame3, text = np.round(minimumTR * 1000,2), font=("Helvetica", 12))
        minimumTE_num_label.grid(row = 5, column = 7)
    else:
        minimumTE_label = Label(frame3, text = "TEmin (ms)", font=("Helvetica", 12)).grid(row = 5, column = 6)
        minimumTE_num_label = Label(frame3, text = np.round(minimumTE * 1000,2), font=("Helvetica", 12))
        minimumTE_num_label.grid(row = 5, column = 7)


# Functions informing the user of predefined sequence selected
def SE():
    global Pre_def_seq
    global seq_label2
    remove_widgets(Pre_def_seq)
    seq_label2.grid_forget()
    Pre_def_seq = "SE"
    seq_label2 = Label(frame1, text = 'Spin echo', font=("Helvetica", 18)); seq_label2.grid(row = 10, column = 0, rowspan = 2)
    show_widgets(Pre_def_seq)
    
def GE():
    global Pre_def_seq
    global seq_label2
    remove_widgets(Pre_def_seq)
    seq_label2.grid_forget()
    Pre_def_seq = "GE"
    seq_label2 = Label(frame1, text = 'Gradient echo', font=("Helvetica", 18)); seq_label2.grid(row = 10, column = 0, rowspan = 2)
    show_widgets(Pre_def_seq)

def IN():
    global Pre_def_seq
    global seq_label2
    remove_widgets(Pre_def_seq)
    seq_label2.grid_forget()
    Pre_def_seq = "IN"
    seq_label2 = Label(frame1, text = 'Inversion recovery', font=("Helvetica", 18)); seq_label2.grid(row = 10, column = 0, rowspan = 2)
    show_widgets(Pre_def_seq)
    
def double_IN():
    global Pre_def_seq
    global seq_label2
    remove_widgets(Pre_def_seq)
    seq_label2.grid_forget()
    Pre_def_seq = "Double IN"
    seq_label2 = Label(frame1, text = 'Double inversion recovery', font=("Helvetica", 18)); seq_label2.grid(row = 10, column = 0, rowspan = 2)
    show_widgets(Pre_def_seq)
    
def FLAIR():
    global Pre_def_seq
    global seq_label2
    remove_widgets(Pre_def_seq)
    seq_label2.grid_forget()
    Pre_def_seq = "FLAIR"
    seq_label2 = Label(frame1, text = 'FLAIR', font=("Helvetica", 18)); seq_label2.grid(row = 10, column = 0, rowspan = 2)
    show_widgets(Pre_def_seq)
    
def SSFP():
    global Pre_def_seq
    global seq_label2
    remove_widgets(Pre_def_seq)
    seq_label2.grid_forget()
    Pre_def_seq = "SSFP"
    seq_label2 = Label(frame1, text = 'Steady-state free precession', font=("Helvetica", 18)); seq_label2.grid(row = 10, column = 0, rowspan = 2)
    show_widgets(Pre_def_seq)
    
def Diffusion():
    global Pre_def_seq
    global seq_label2
    remove_widgets(Pre_def_seq)
    seq_label2.grid_forget()
    Pre_def_seq = "Dif"
    seq_label2 = Label(frame1, text = 'Diffusion', font=("Helvetica", 18)); seq_label2.grid(row = 10, column = 0, rowspan = 2)
    show_widgets(Pre_def_seq)
    
def TSE():
    global Pre_def_seq
    global seq_label2
    remove_widgets(Pre_def_seq)
    seq_label2.grid_forget()
    Pre_def_seq = "TSE"
    seq_label2 = Label(frame1, text = 'Turbo spin echo', font=("Helvetica", 18)); seq_label2.grid(row = 10, column = 0, rowspan = 2)
    show_widgets(Pre_def_seq)

##### The sequence to run #####       
def run():   
    global imag_size
    global Pre_def_seq
    global TR
    global TE
    global TI
    global TI2
    global FOV
    global Data_mat
    global Resolution
    global Bandwidth
    global Alpha
    global final_seq
    global S
    global B
    global dif_TEmin
    global final_im_sag
    global final_im_cor
    global final_im_ax
    global minimumTE
    global minimumTR
    
    global Time_scan_num_label
    global Data_mat1_label
    global Bd_by_pixel_label
    global SNR_num_label
    global minimumTE_num_label
    global minimumTR_num_label
    global warning_label1
    global warning_label2
    
    TR = TR_entry.get(); TR = int(TR); TR = np.divide(TR,1000) # Divide by 1000 to have the values in milli
    TE = TE_entry.get(); TE = int(TE); TE = np.divide(TE,1000)
    TI = TI_entry.get(); TI = int(TI); TI = np.divide(TI,1000)
    TI2 = TI2_entry.get(); TI2 = int(TI2); TI2 = np.divide(TI2,1000)
    fov1 = FOV1_entry.get(); fov1 = int(fov1)
    fov2 = FOV2_entry.get(); fov2 = int(fov2)
    fov3 = FOV3_entry.get(); fov3 = int(fov3)
    res1 = Res1_entry.get(); res1 = float(res1) 
    res2 = Res2_entry.get(); res2 = float(res2)
    res3 = Res3_entry.get(); res3 = float(res3)
    bd = Bandwidth_entry.get(); Bandwidth = int(bd)
    Alpha = Alpha_entry.get(); Alpha = int(Alpha)
    ETL = ETL_entry.get(); ETL = int(ETL)
    FOV = [fov1, fov2, fov3]
    Resolution = [res1, res2, res3]
    Data_mat = np.divide(FOV, Resolution)
    Data_mat = [int(Data_mat[0]), int(Data_mat[1]), int(Data_mat[2])]
    S = [int(Data_mat[0]/2),int(Data_mat[1]/2),int(Data_mat[2]/2)]   # S is a vector with the center of the Data matrix
    
    Time_scan_num_label.grid_forget()
    Data_mat1_label.grid_forget()
    Bd_by_pixel_label.grid_forget()
    SNR_num_label.grid_forget()
    minimumTE_num_label.grid_forget()
    minimumTR_num_label.grid_forget()
    
    minimumTE = Data_mat[0]/(2*Bandwidth)
    minimumTR = 2*minimumTE
    
    if Pre_def_seq == 'SSFP':
        minimumTE_label = Label(frame3, text = "TRmin (ms) ", font=("Helvetica", 12)).grid(row = 5, column = 6)
        minimumTE_num_label = Label(frame3, text = np.round(minimumTR * 1000,2), font=("Helvetica", 12))
        minimumTE_num_label.grid(row = 5, column = 7)
    else:
        minimumTE_label = Label(frame3, text = "TEmin (ms)", font=("Helvetica", 12)).grid(row = 5, column = 6)
        minimumTE_num_label = Label(frame3, text = np.round(minimumTE * 1000,2), font=("Helvetica", 12))
        minimumTE_num_label.grid(row = 5, column = 7)
        
    final_zeros = np.zeros((Data_mat))
    
    # If ok --> 'yes' the sequence will be simulated, if ok --> 'no', there is a problem with the parameters and the images will be completly black
    ok = 'yes'
    
    # Checking if the parameters make physical sense
    if Pre_def_seq == 'SSFP':
        TE = 0 # Because the original (inserted) value of TE can be higher than a desired TR, but TE isn't used in the SSFP seq
     
        if (TR) < minimumTR:  
            print('TR can not be smaller than the minimum TR!!!')
            ok = 'no' 
            w1 = "Warning!!"
            w2 = "Error, TR < minTR!!!"
            
    if Pre_def_seq == 'IN':
        if TR < (TE + TI):
            print('TR can not be smaller than TE + TI!!!')
            ok = 'no'
            w1 = "Warning!!"
            w2 = "TR smaller than TE + TI!!!"

    elif Pre_def_seq == 'Double IN':
        if TR < (TE + TI + TI2):
            print('TR can not be smaller than TE + TI + TI2!!!')
            ok = 'no'
            w1 = "Warning!!"
            w2 = "TR smaller than TE + TI + TI2!!!"
            
    if TR < TE:
        print('TE can not be larger than TR!!!')
        ok = 'no'
        w1 = "Warning!!"
        w2 = "Error, TR < TE!!!"
        
    if (TE) < minimumTE:
        print('TE can not be smaller than the minimum TE!!!')
        ok = 'no'
        w1 = "Warning!!"
        w2 = "Error, TE < TEmin!!!"

    if B != 0:
        if Pre_def_seq == 'Dif':
            if dif_TEmin > TE:
                print('Error, TE must be grater than diffusion TEmin!!!')
                ok = 'no'
                w1 = "Warning!!"
                w2 = "Error, TE < diffusion TEmin !!!"
                
    if ok == 'no': # There is a problem with the parameters and the images will be completly black
        
        warning_label1.grid_forget()
        warning_label2.grid_forget()
        
        warning_label1 = Label(root, text = w1, font=("Helvetica", 15), fg='#f00'); warning_label1.grid(row = 3, column = 0)
        warning_label2 = Label(root, text = w2, font=("Helvetica", 15), fg='#f00'); warning_label2.grid(row = 4, column = 0)
        
        # Create axial plot
        fig = plt.figure(figsize=(imag_size,imag_size))
        
        plt.imshow(np.rot90(final_zeros[:,:,0], k = 1), cmap='gray')  # Create image plot
        plt.axis('off')
        canvas = FigureCanvasTkAgg(fig, root)                         # Tkinter canvas which contains matplotlib figure
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = 3, rowspan = 4) # Placing canvas on Tkinter window
        plt.close()
        
        # Create coronal plot
        fig = plt.figure(figsize=(imag_size,imag_size))
        plt.imshow(np.rot90(final_zeros[:,0,:], k = 1), cmap='gray')   
        plt.axis('off')
        canvas = FigureCanvasTkAgg(fig, root)         
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = 2, rowspan = 4)
        plt.close()
        
        # Create sagittal plot
        fig = plt.figure(figsize=(imag_size,imag_size))
        plt.imshow(np.rot90(final_zeros[0,:,:], k = 1), cmap='gray')   
        plt.axis('off')
        canvas = FigureCanvasTkAgg(fig, root)         
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = 1, rowspan = 4) 
        plt.close()
                
    else: # The sequence will be simulated
        
        warning_label1.grid_forget()
        warning_label2.grid_forget()
        
        time_scan = TR * Data_mat[1] * Data_mat[2]
        
        if Pre_def_seq == 'TSE':
            ETL = ETL_entry.get(); ETL = int(ETL)
            time_scan = time_scan/ETL

        time = datetime.timedelta(seconds=time_scan) # Converts amount of seconds to hours:minutes:seconds
        Bd_by_pixel = np.divide(Bandwidth,Data_mat[0])
        
        # Plotting the parameters
        Time_scan_num_label.grid_forget()
        Time_scan_num_label = Label(frame3, text = str(time), font=("Helvetica", 12));               Time_scan_num_label.grid(row = 1, column = 7)
        s = str(Data_mat[0]) + "x" + str(Data_mat[1]) + "x" + str(Data_mat[2])
        Data_mat1_label = Label(frame3, text = s, font=("Helvetica", 12));                           Data_mat1_label.grid(row = 2, column = 7)
        Bd_by_pixel_label = Label(frame3, text = str(round(Bd_by_pixel,2)), font=("Helvetica", 12)); Bd_by_pixel_label.grid(row = 3, column = 7)

        # loading the 3D distorted data
        T1_3D_grad = np.load('T1_3D_gradientdistortion02.npy')
        T2_3D_grad = np.load('T2_3D_gradientdistortion02.npy')
        M0_3D_grad = np.load('M0_3D_gradientdistortion02.npy')
        B1map_3D_grad = np.load('B1_3D_gradientdistortion02.npy')
        flipAngleMaprescale_3D_grad = np.load('flipAngleMaprescale_3D_gradientdistortion02.npy')
        t2_star_3D_grad = np.load('t2_star_tensor_3D_gradientdistortion02.npy')
        ADC_3D_grad = np.load('ADC_3D_gradientdistortion02.npy')

        # Resizing
        n_t1 = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_t1 = np.zeros((Data_mat))
        n_t2 = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_t2 = np.zeros((Data_mat))
        n_m0 = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_m0 = np.zeros((Data_mat))
        n_B1 = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_B1 = np.zeros((Data_mat))
        n_t2star = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2]));  final_t2star = np.zeros((Data_mat))
        n_flipmap = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_flipmap = np.zeros((Data_mat))
        n_adc = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_adc = np.zeros((Data_mat))
        n_seq = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_seq = np.zeros((Data_mat))
        n_phi = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_phi = np.zeros((Data_mat))
        n_offset = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_offset = np.zeros((Data_mat))

        for x in range(T1_3D.shape[0]):
            n_t1[x,:,:] = cv2.resize(T1_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            n_t2[x,:,:] = cv2.resize(T2_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            n_m0[x,:,:] = cv2.resize(M0_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            n_B1[x,:,:] = cv2.resize(B1map_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            n_t2star[x,:,:] = cv2.resize(t2_star_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            n_flipmap[x,:,:] = cv2.resize(flipAngleMaprescale_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            n_adc[x,:,:] = cv2.resize(ADC_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))

        for x in range(Data_mat[1]):
            final_t1[:,x,:] = cv2.resize(n_t1[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            final_t2[:,x,:] = cv2.resize(n_t2[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            final_m0[:,x,:] = cv2.resize(n_m0[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            final_B1[:,x,:] = cv2.resize(n_B1[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            final_t2star[:,x,:] = cv2.resize(n_t2star[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            final_flipmap[:,x,:] = cv2.resize(n_flipmap[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            final_adc[:,x,:] = cv2.resize(n_adc[:,x,:], dsize=(Data_mat[2], Data_mat[0]))

        if Pre_def_seq == 'SE':

            SE_3D = spin_echo_seq(TR, TE, T1_3D_grad, T2_3D_grad, M0_3D_grad); SE_3D = np.multiply(SE_3D, B1map_3D_grad)  

            # Resizing
            for x in range(T1_3D.shape[0]):
                n_seq[x,:,:] = cv2.resize(SE_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            for x in range(Data_mat[1]):
                final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))

        elif Pre_def_seq == 'GE':         
            angle = flipAngleMaprescale_3D/Alpha
            GE_3D = Gradient_seq(TR, TE, T1_3D_grad, t2_star_3D_grad, M0_3D_grad, angle); GE_3D = np.multiply(GE_3D, B1map_3D_grad)

            # Resizing
            for x in range(T1_3D.shape[0]):
                n_seq[x,:,:] = cv2.resize(GE_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))        
            for x in range(Data_mat[1]):
                final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))

        elif Pre_def_seq == 'IN':
            IN_3D = IN_seq(TR, TE, TI, T1_3D_grad, T2_3D_grad, M0_3D_grad); IN_3D = np.multiply(IN_3D, B1map_3D_grad)

            # Resizing
            for x in range(T1_3D.shape[0]):
                n_seq[x,:,:] = cv2.resize(IN_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            for x in range(Data_mat[1]):
                final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))

        elif Pre_def_seq == 'Double IN':
            DIN_3D = DoubleInversion_seq(TR, TE, TI, TI2, T1_3D_grad, T2_3D_grad, M0_3D_grad); DIN_3D = np.multiply(DIN_3D, B1map_3D_grad)

            # Resizing
            for x in range(T1_3D.shape[0]):
                n_seq[x,:,:] = cv2.resize(DIN_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            for x in range(Data_mat[1]):
                final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))

        elif Pre_def_seq == 'FLAIR':
            TI = np.log(2) * 3.695 
            FLAIR_3D = IN_seq(TR, TE, TI, T1_3D_grad, T2_3D_grad, M0_3D_grad); FLAIR_3D = np.multiply(FLAIR_3D, B1map_3D_grad)

            # Resizing
            for x in range(T1_3D.shape[0]):
                n_seq[x,:,:] = cv2.resize(FLAIR_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            for x in range(Data_mat[1]):
                final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))

        elif Pre_def_seq == 'Dif':
            DIF_3D = Diffusion_seq(TR, TE, T1_3D_grad, T2_3D_grad, M0_3D_grad, B, ADC_3D_grad); DIF_3D = np.multiply(DIF_3D, B1map_3D_grad)

            # Resizing
            for x in range(T1_3D.shape[0]):
                n_seq[x,:,:] = cv2.resize(DIF_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            for x in range(Data_mat[1]):
                final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))

        elif Pre_def_seq == 'TSE':
            trajectory = traj.get()
            c = 10
            TSE_3D = TSE_seq(TR, TE, ETL, M0_3D_grad, T1_3D_grad, T2_3D_grad, c, trajectory); TSE_3D = np.multiply(TSE_3D, B1map_3D_grad)

            # Resizing
            for x in range(T1_3D.shape[0]):
                n_seq[x,:,:] = cv2.resize(TSE_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            for x in range(Data_mat[1]):
                final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))

        elif Pre_def_seq == 'SSFP':        
            gamma = 42.58*(10**6)*2*constants.pi # 267538030.37970677 [rad/sT]
            omega = B0_3D * gamma
            center = np.divide(omega.shape,2).astype(int)
            center_freq_value = omega[center[0],center[1],center[2]]

            offset = omega - center_freq_value
            phi = offset * TR

            SSFP_3D = SSFP_Echo_seq(T1_3D_grad, T2_3D_grad, M0_3D_grad, Alpha, phi); 
            SSFP_3D[SSFP_3D > 1] = 1
            SSFP_3D = np.multiply(SSFP_3D, B1map_3D_grad)

            # Resizing
            for x in range(T1_3D.shape[0]):
                n_seq[x,:,:] = cv2.resize(SSFP_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            for x in range(Data_mat[1]):
                final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))           
                
        # Noise computation     
        n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
        final_seq = final_seq + n
        
        final_im_sag = final_seq[S[0],:,:]
        final_im_cor = final_seq[:,S[1],:]
        final_im_ax = final_seq[:,:,S[2]]

        # To change the axis value to mm (the fov)
        ax_pos = [[0, Data_mat[0]/2, Data_mat[0]-1], [0, Data_mat[1]/2, Data_mat[1]-1], [0, Data_mat[2]/2, Data_mat[2]-1]]
        ax_label = [[0, FOV[0]/2, FOV[0]], [0, FOV[1]/2, FOV[1]], [0, FOV[2]/2, FOV[2]]]
        Labels = ['Left/Right [mm]', 'Anterior/Posterior [mm]', 'Foot/Head [mm]']

        # Create axial plot
        fig = plt.figure(figsize=(imag_size,imag_size))
        plt.imshow(np.rot90(final_im_ax, k = 1), cmap='gray')  # Create image plot
        plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
        plt.ylabel(Labels[1]); plt.yticks(ax_pos[1], ax_label[1])
        canvas = FigureCanvasTkAgg(fig, root)                  # Tkinter canvas which contains matplotlib figure
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = 3, rowspan = 4)       # Placing canvas on Tkinter window
        plt.close()
        
        # Create coronal plot
        fig = plt.figure(figsize=(imag_size,imag_size))
        plt.imshow(np.rot90(final_im_cor, k = 1), cmap='gray')   
        plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
        plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        canvas = FigureCanvasTkAgg(fig, root)         
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = 2, rowspan = 4)
        plt.close()
        
        # Create sagittal plot
        fig = plt.figure(figsize=(imag_size,imag_size))
        plt.imshow(np.rot90(final_im_sag, k = 1), cmap='gray')   
        plt.xlabel(Labels[1]); plt.xticks(ax_pos[1], ax_label[1])
        plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        canvas = FigureCanvasTkAgg(fig, root)         
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = 1, rowspan = 4) 
        plt.close()
                     
        if SNR_define == 'yes':    
            noise_col = SNR_noise_box_center[0]
            noise_row = SNR_noise_box_center[1]
            mean_col = SNR_mean_box_center[0]
            mean_row = SNR_mean_box_center[1]

            n1 = noise_row - SNR_length
            n2 = noise_row + SNR_length
            n3 = noise_col - SNR_length
            n4 = noise_col + SNR_length

            m1 = mean_row - SNR_length
            m2 = mean_row + SNR_length
            m3 = mean_col - SNR_length
            m4 = mean_col + SNR_length

            snr = snr_homemade(np.rot90(final_im_ax),n1, n2, n3, n4, m1, m2, m3, m4)
            SNR_num_label.grid_forget()
            SNR_num_label = Label(frame3, text = str(np.abs(round(snr,2))), font=("Helvetica", 12))
            SNR_num_label.grid(row = 4, column = 7) 
            
        elif SNR_define == 'no':
            SNR_num_label.grid_forget()
            SNR_num_label = Label(frame3, text = 'SNR not yet define!', font=("Helvetica", 12))
            SNR_num_label.grid(row = 4, column = 7)
    
#//////////////// 3D Parameters visulization //////////////// 

def param_vis():
    global Pre_def_seq
    global TR
    global TE
    global TI
    global TI2
    global FOV
    global Data_mat
    global Resolution
    global Bandwidth
    global flipAngleMaprescale_3D
    global alp_3D
    global delta_B0_3D
    global B0_3D
    global b0_3D
    global B1map_3D
    global T1_3D
    global T2_3D
    global t2_star_tensor_3D
    global M0_3D
    global Xgrad_3D 
    global Zgrad_3D 
    global ADC_3D
    global B
    global c
    global ETL
    
    global zgrad_ax 
    global xgrad_cor
    global xgrad_sag
    
    TR = TR_entry.get(); TR = int(TR); TR = np.divide(TR,1000) # Divide by 1000 to have the values in milli
    TE = TE_entry.get(); TE = int(TE); TE = np.divide(TE,1000)
    TI = TI_entry.get(); TI = int(TI); TI = np.divide(TI,1000)
    TI2 = TI2_entry.get(); TI2 = int(TI2); TI2 = np.divide(TI2,1000)
    fov1 = FOV1_entry.get(); fov1 = int(fov1)
    fov2 = FOV2_entry.get(); fov2 = int(fov2)
    fov3 = FOV3_entry.get(); fov3 = int(fov3)
    res1 = Res1_entry.get(); res1 = float(res1) 
    res2 = Res2_entry.get(); res2 = float(res2)
    res3 = Res3_entry.get(); res3 = float(res3)
    bd = Bandwidth_entry.get(); Bandwidth = int(bd)
    Alpha = Alpha_entry.get(); Alpha = int(Alpha)
    ETL = ETL_entry.get(); ETL = int(ETL) 
    FOV = [fov1, fov2, fov3]
    Resolution = [res1, res2, res3]
    Data_mat = np.divide(FOV, Resolution)
    Data_mat = [int(Data_mat[0]), int(Data_mat[1]), int(Data_mat[2])]    

    # loading the 3D distorted data
    T1_3D_grad = np.load('T1_3D_gradientdistortion02.npy')
    T2_3D_grad = np.load('T2_3D_gradientdistortion02.npy')
    M0_3D_grad = np.load('M0_3D_gradientdistortion02.npy')
    B1map_3D_grad = np.load('B1_3D_gradientdistortion02.npy')
    flipAngleMaprescale_3D_grad = np.load('flipAngleMaprescale_3D_gradientdistortion02.npy')
    t2_star_3D_grad = np.load('t2_star_tensor_3D_gradientdistortion02.npy')
    ADC_3D_grad = np.load('ADC_3D_gradientdistortion02.npy')
    
    # Resizing
    n_t1 = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_t1 = np.zeros((Data_mat))
    n_t2 = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_t2 = np.zeros((Data_mat))
    n_m0 = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_m0 = np.zeros((Data_mat))
    n_B1 = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_B1 = np.zeros((Data_mat))
    n_t2star = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2]));  final_t2star = np.zeros((Data_mat))
    n_flipmap = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_flipmap = np.zeros((Data_mat))
    n_adc = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_adc = np.zeros((Data_mat))
    n_seq = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_seq = np.zeros((Data_mat))
    n_phi = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_phi = np.zeros((Data_mat))
    n_offset = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2])); final_offset = np.zeros((Data_mat))

    for x in range(T1_3D.shape[0]):
        n_t1[x,:,:] = cv2.resize(T1_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        n_t2[x,:,:] = cv2.resize(T2_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        n_m0[x,:,:] = cv2.resize(M0_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        n_B1[x,:,:] = cv2.resize(B1map_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        n_t2star[x,:,:] = cv2.resize(t2_star_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        n_flipmap[x,:,:] = cv2.resize(flipAngleMaprescale_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        n_adc[x,:,:] = cv2.resize(ADC_3D_grad[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
    
    for x in range(Data_mat[1]):
        final_t1[:,x,:] = cv2.resize(n_t1[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        final_t2[:,x,:] = cv2.resize(n_t2[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        final_m0[:,x,:] = cv2.resize(n_m0[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        final_B1[:,x,:] = cv2.resize(n_B1[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        final_t2star[:,x,:] = cv2.resize(n_t2star[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        final_flipmap[:,x,:] = cv2.resize(n_flipmap[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        final_adc[:,x,:] = cv2.resize(n_adc[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
 
    if Pre_def_seq == 'SE':

        SE_3D = spin_echo_seq(TR, TE, T1_3D_grad, T2_3D_grad, M0_3D_grad); SE_3D = np.multiply(SE_3D, B1map_3D_grad)
        
        # Resizing
        for x in range(T1_3D.shape[0]):
            n_seq[x,:,:] = cv2.resize(SE_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        for x in range(Data_mat[1]):
            final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            
        n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
        
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_SE.interactivePlot(fig, ax, final_seq + n, TR, TE, final_t1, final_t2, final_m0, final_B1, n, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
        
    elif Pre_def_seq == 'GE':         
        angle = flipAngleMaprescale_3D/Alpha
        GE_3D = Gradient_seq(TR, TE, T1_3D_grad, t2_star_3D_grad, M0_3D_grad, angle); GE_3D = np.multiply(GE_3D, B1map_3D_grad)
    
        # Resizing
        for x in range(T1_3D.shape[0]):
            n_seq[x,:,:] = cv2.resize(GE_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))        
        for x in range(Data_mat[1]):
            final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        
        n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
        
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_GE.interactivePlot(fig, ax, final_seq + n, TR, TE, final_t1, final_t2star, final_m0, final_flipmap, Alpha, final_B1, n, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
        
    elif Pre_def_seq == 'IN':
        IN_3D = IN_seq(TR, TE, TI, T1_3D_grad, T2_3D_grad, M0_3D_grad); IN_3D = np.multiply(IN_3D, B1map_3D_grad)

        # Resizing
        for x in range(T1_3D.shape[0]):
            n_seq[x,:,:] = cv2.resize(IN_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        for x in range(Data_mat[1]):
            final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            
        n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
        
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_IN.interactivePlot(fig, ax, final_seq + n, TR, TE, TI, final_t1, final_t2, final_m0, final_B1, n, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
    
    elif Pre_def_seq == 'Double IN':
        
        DIN_3D = DoubleInversion_seq(TR, TE, TI, TI2, T1_3D_grad, T2_3D_grad, M0_3D_grad); DIN_3D = np.multiply(DIN_3D, B1map_3D_grad)
        
        # Resizing
        for x in range(T1_3D.shape[0]):
            n_seq[x,:,:] = cv2.resize(DIN_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        for x in range(Data_mat[1]):
            final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            
        # This code is because in double inversion recovery their is an exp with a positive value (usualy negative)
        # and the resizing code use a linear interpolation that can give some very small values for T1 --> TR/T1 quite big
        # resulting in masive (overflow) values. So the code bellow puts all values of resized T1 in the correct T1 bins (0 / 0.157 / 0.272 / 0.33 / 4)
        t = final_t1
        t[(0.4 < t)] = 4
        t[t <= 0.157/2] = 0
        t[((0.157/2 < t) & (t <= (0.272-0.157)/2 + 0.157))] = 0.157
        t[(((0.272-0.157)/2 + 0.157 < t) & (t <= (0.33-0.272)/2 + 0.272))] = 0.272
        t[(((0.33-0.272)/2 + 0.272 < t) & (t <= 0.4))] = 0.33
        
        n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
        
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_DIN.interactivePlot(fig, ax, final_seq + n, TR, TE, TI, TI2, t, final_t2, final_m0, final_B1, n, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
        
    elif Pre_def_seq == 'FLAIR':
        TI = np.log(2) * 3.695 
        FLAIR_3D = IN_seq(TR, TE, TI, T1_3D_grad, T2_3D_grad, M0_3D_grad); FLAIR_3D = np.multiply(FLAIR_3D, B1map_3D_grad)
        
        # Resizing
        for x in range(T1_3D.shape[0]):
            n_seq[x,:,:] = cv2.resize(FLAIR_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        for x in range(Data_mat[1]):
            final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        
        n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
        
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_FLAIR.interactivePlot(fig, ax, final_seq + n, TR, TE, TI, final_t1, final_t2, final_m0, final_B1, n, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
        
    elif Pre_def_seq == 'Dif':
        DIF_3D = Diffusion_seq(TR, TE, T1_3D_grad, T2_3D_grad, M0_3D_grad, B, ADC_3D_grad); DIF_3D = np.multiply(DIF_3D, B1map_3D_grad)
             
        # Resizing
        for x in range(T1_3D.shape[0]):
            n_seq[x,:,:] = cv2.resize(DIF_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        for x in range(Data_mat[1]):
            final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        
        n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
        
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_Dif.interactivePlot(fig, ax, final_seq + n, TR, TE, final_t1, final_t2, final_m0, B, final_adc, final_B1, n, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()

    elif Pre_def_seq == 'TSE':
        trajectory = traj.get()
        c = 10
        TSE_3D = TSE_seq(TR, TE, ETL, M0_3D_grad, T1_3D_grad, T2_3D_grad, c, trajectory); TSE_3D = np.multiply(TSE_3D, B1map_3D_grad)
          
        # Resizing
        for x in range(T1_3D.shape[0]):
            n_seq[x,:,:] = cv2.resize(TSE_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        for x in range(Data_mat[1]):
            final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            
        n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
        
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_TSE.interactivePlot(fig, ax, final_seq + n, TR, TE, final_t1, final_t2, c, ETL, trajectory, final_m0, final_B1, n, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
        
    elif Pre_def_seq == 'SSFP': 
        gamma = 42.58*(10**6)*2*constants.pi # 267538030.37970677 [rad/sT]
        omega = B0_3D * gamma
        center = np.divide(omega.shape,2).astype(int)
        center_freq_value = omega[center[0],center[1],center[2]]

        offset = omega - center_freq_value
        phi = offset * TR
        
        SSFP_3D = SSFP_Echo_seq(T1_3D_grad, T2_3D_grad, M0_3D_grad, Alpha, phi); 
        SSFP_3D[SSFP_3D > 1] = 1
        SSFP_3D = np.multiply(SSFP_3D, B1map_3D_grad)
                
        # Resizing
        for x in range(T1_3D.shape[0]):
            n_seq[x,:,:] = cv2.resize(SSFP_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            n_phi[x,:,:] = cv2.resize(phi[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
            n_offset[x,:,:] = cv2.resize(offset[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
        for x in range(Data_mat[1]):
            final_seq[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            final_phi[:,x,:] = cv2.resize(n_phi[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            final_offset[:,x,:] = cv2.resize(n_offset[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
            
        n = noise_generation(final_seq, Data_mat, Resolution, Bandwidth)
        
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_SSFP.interactivePlot(fig, ax, final_seq + n, TR, final_t1, final_t2, final_m0, Alpha, final_phi, final_offset, final_B1, n, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
        
# Filter 3D visualisation
def filter_vis(*args):
    
    global final_seq
    global low_pass_entry
    global high_pass_entry
    global Gauss_entry
    global Non_local_h_entry
    global Non_local_psize_entry
    global Non_local_pdist_entry
    
    fil = fi.get()
    
    if fil == 'low':
        ks = low_pass_entry.get(); ks = int(ks)
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_low.interactivePlot(fig, ax, final_seq, ks, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
        
    elif fil == 'high':
        ks = high_pass_entry.get(); ks = int(ks)
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_high.interactivePlot(fig, ax, final_seq, ks, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
        
    elif fil == 'gauss':
        sig = Gauss_entry.get()
        fig, ax = plt.subplots(1,3)
        fig3D = pltaxes_gauss.interactivePlot(fig, ax, final_seq, sig, Data_mat, plotAxis = 2, fov = FOV)
        plt.show()
        
    elif fil == 'non local':
        H = Non_local_h_entry.get(); H = float(H)
        s = Non_local_psize_entry.get(); s = int(s)
        d = Non_local_pdist_entry.get(); d = int(d)
        fig, ax = plt.subplots(1,2)
        fig3D = pltaxes_nonlocal.interactivePlot(fig, ax, final_seq, H, s, d, Data_mat, plotAxis = 2, fov = FOV)        
        plt.show()        
        
#/////////////////// Functions for post-processing ///////////////////
def lowpass(s):
    global final_im_sag
    global final_im_cor
    global final_im_ax
    global SNR_num_label
    global Data_mat
    global FOV
    
    # To change the axis value to mm (the fov)
    ax_pos = [[0, Data_mat[0]/2, Data_mat[0]-1], [0, Data_mat[1]/2, Data_mat[1]-1], [0, Data_mat[2]/2, Data_mat[2]-1]]
    ax_label = [[0, FOV[0]/2, FOV[0]], [0, FOV[1]/2, FOV[1]], [0, FOV[2]/2, FOV[2]]]
    Labels = ['Left/Right [mm]', 'Anterior/Posterior [mm]', 'Foot/Head [mm]']
    
    s = int(s)
    final_im_sag = cv2.blur(final_im_sag,(s,s))
    final_im_cor = cv2.blur(final_im_cor,(s,s))
    final_im_ax = cv2.blur(final_im_ax,(s,s))
    
    I = [final_im_sag, final_im_cor, final_im_ax]
    J = [1, 2, 3]
    
    for i in range(3):
        fig = plt.figure(figsize=(imag_size,imag_size)) # Create plot
        plt.imshow(np.rot90(I[i], k = 1), cmap='gray')  # Create image plot
        
        if i == 0:   # Sagittal
            plt.xlabel(Labels[1]); plt.xticks(ax_pos[1], ax_label[1])
            plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        elif i == 1: # Coronal
            plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
            plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        elif i == 2: # Axial
            plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
            plt.ylabel(Labels[1]); plt.yticks(ax_pos[1], ax_label[1])
        
        canvas = FigureCanvasTkAgg(fig, root)         # Tkinter canvas which contains matplotlib figure
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = J[i], rowspan = 4)    # Placing canvas on Tkinter window
        plt.close()
    
    if SNR_define == 'yes':    
        noise_col = SNR_noise_box_center[0]
        noise_row = SNR_noise_box_center[1]
        mean_col = SNR_mean_box_center[0]
        mean_row = SNR_mean_box_center[1]

        n1 = noise_row - SNR_length
        n2 = noise_row + SNR_length
        n3 = noise_col - SNR_length
        n4 = noise_col + SNR_length

        m1 = mean_row - SNR_length
        m2 = mean_row + SNR_length
        m3 = mean_col - SNR_length
        m4 = mean_col + SNR_length

        snr = snr_homemade(np.rot90(final_im_ax),n1, n2, n3, n4, m1, m2, m3, m4)
        SNR_num_label.grid_forget()
        SNR_num_label = Label(frame3, text = str(np.abs(round(snr,2))), font=("Helvetica", 12))
        SNR_num_label.grid(row = 4, column = 7) 

    elif SNR_define == 'no':
        SNR_num_label.grid_forget()
        SNR_num_label = Label(frame3, text = 'SNR not yet define!', font=("Helvetica", 12))
        SNR_num_label.grid(row = 4, column = 7)
    
def highpass(s):
    global final_im_sag
    global final_im_cor
    global final_im_ax
    global SNR_num_label
    global Data_mat
    global FOV
    
    # To change the axis value to mm (the fov)
    ax_pos = [[0, Data_mat[0]/2, Data_mat[0]-1], [0, Data_mat[1]/2, Data_mat[1]-1], [0, Data_mat[2]/2, Data_mat[2]-1]]
    ax_label = [[0, FOV[0]/2, FOV[0]], [0, FOV[1]/2, FOV[1]], [0, FOV[2]/2, FOV[2]]]
    Labels = ['Left/Right [mm]', 'Anterior/Posterior [mm]', 'Foot/Head [mm]']
    
    s = int(s)  
    final_im_sag = cv2.Laplacian(final_im_sag, -1, ksize = s) # The -1 is so that the final image as the same depth as the original one
    final_im_cor = cv2.Laplacian(final_im_cor, -1, ksize = s)
    final_im_ax = cv2.Laplacian(final_im_ax, -1, ksize = s)
    
    I = [final_im_sag, final_im_cor, final_im_ax]
    J = [1, 2, 3]
    
    for i in range(3):
        fig = plt.figure(figsize=(imag_size,imag_size)) # Create plot
        plt.imshow(np.rot90(I[i], k = 1), cmap='gray')  # Create image plot
        
        if i == 0:   # Sagittal
            plt.xlabel(Labels[1]); plt.xticks(ax_pos[1], ax_label[1])
            plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        elif i == 1: # Coronal
            plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
            plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        elif i == 2: # Axial
            plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
            plt.ylabel(Labels[1]); plt.yticks(ax_pos[1], ax_label[1])
        
        canvas = FigureCanvasTkAgg(fig, root)         # Tkinter canvas which contains matplotlib figure
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = J[i], rowspan = 4)    # Placing canvas on Tkinter window
        plt.close()
    
    if SNR_define == 'yes':    
        noise_col = SNR_noise_box_center[0]
        noise_row = SNR_noise_box_center[1]
        mean_col = SNR_mean_box_center[0]
        mean_row = SNR_mean_box_center[1]

        n1 = noise_row - SNR_length
        n2 = noise_row + SNR_length
        n3 = noise_col - SNR_length
        n4 = noise_col + SNR_length

        m1 = mean_row - SNR_length
        m2 = mean_row + SNR_length
        m3 = mean_col - SNR_length
        m4 = mean_col + SNR_length

        snr = snr_homemade(np.rot90(final_im_ax),n1, n2, n3, n4, m1, m2, m3, m4)
        SNR_num_label.grid_forget()
        SNR_num_label = Label(frame3, text = str(np.abs(round(snr,2))), font=("Helvetica", 12))
        SNR_num_label.grid(row = 4, column = 7) 

    elif SNR_define == 'no':
        SNR_num_label.grid_forget()
        SNR_num_label = Label(frame3, text = 'SNR not yet define!', font=("Helvetica", 12))
        SNR_num_label.grid(row = 4, column = 7)

def gauss(sig):
    global final_im_sag
    global final_im_cor
    global final_im_ax
    global SNR_num_label
    global Data_mat
    global FOV
    
    # To change the axis value to mm (the fov)
    ax_pos = [[0, Data_mat[0]/2, Data_mat[0]-1], [0, Data_mat[1]/2, Data_mat[1]-1], [0, Data_mat[2]/2, Data_mat[2]-1]]
    ax_label = [[0, FOV[0]/2, FOV[0]], [0, FOV[1]/2, FOV[1]], [0, FOV[2]/2, FOV[2]]]
    Labels = ['Left/Right [mm]', 'Anterior/Posterior [mm]', 'Foot/Head [mm]']

    final_im_sag = gaussian_filter(final_im_sag, sigma=float(sig))
    final_im_cor = gaussian_filter(final_im_cor, sigma=float(sig))
    final_im_ax = gaussian_filter(final_im_ax, sigma=float(sig))
    
    I = [final_im_sag, final_im_cor, final_im_ax]
    J = [1, 2, 3]
    
    for i in range(3):
        fig = plt.figure(figsize=(imag_size,imag_size)) # Create plot
        plt.imshow(np.rot90(I[i], k = 1), cmap='gray')  # Create image plot
        
        if i == 0:   # Sagittal
            plt.xlabel(Labels[1]); plt.xticks(ax_pos[1], ax_label[1])
            plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        elif i == 1: # Coronal
            plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
            plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        elif i == 2: # Axial
            plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
            plt.ylabel(Labels[1]); plt.yticks(ax_pos[1], ax_label[1])
        
        canvas = FigureCanvasTkAgg(fig, root)         # Tkinter canvas which contains matplotlib figure
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = J[i], rowspan = 4)    # Placing canvas on Tkinter window
        plt.close()
    
    if SNR_define == 'yes':    
        noise_col = SNR_noise_box_center[0]
        noise_row = SNR_noise_box_center[1]
        mean_col = SNR_mean_box_center[0]
        mean_row = SNR_mean_box_center[1]

        n1 = noise_row - SNR_length
        n2 = noise_row + SNR_length
        n3 = noise_col - SNR_length
        n4 = noise_col + SNR_length

        m1 = mean_row - SNR_length
        m2 = mean_row + SNR_length
        m3 = mean_col - SNR_length
        m4 = mean_col + SNR_length

        snr = snr_homemade(np.rot90(final_im_ax),n1, n2, n3, n4, m1, m2, m3, m4)
        SNR_num_label.grid_forget()
        SNR_num_label = Label(frame3, text = str(np.abs(round(snr,2))), font=("Helvetica", 12))
        SNR_num_label.grid(row = 4, column = 7) 

    elif SNR_define == 'no':
        SNR_num_label.grid_forget()
        SNR_num_label = Label(frame3, text = 'SNR not yet define!', font=("Helvetica", 12))
        SNR_num_label.grid(row = 4, column = 7)
    
def non_local(H, s, d):
    global final_im_sag
    global final_im_cor
    global final_im_ax
    global SNR_num_label
    global Data_mat
    global FOV
    
    # To change the axis value to mm (the fov)
    ax_pos = [[0, Data_mat[0]/2, Data_mat[0]-1], [0, Data_mat[1]/2, Data_mat[1]-1], [0, Data_mat[2]/2, Data_mat[2]-1]]
    ax_label = [[0, FOV[0]/2, FOV[0]], [0, FOV[1]/2, FOV[1]], [0, FOV[2]/2, FOV[2]]]
    Labels = ['Left/Right [mm]', 'Anterior/Posterior [mm]', 'Foot/Head [mm]']
    
    H = float(H); s = int(s); d = int(d)

    # estimate the noise standard deviation from the noisy image
    sigma_est_sag = estimate_sigma(final_im_sag, channel_axis=None)
    sigma_est_cor = estimate_sigma(final_im_cor, channel_axis=None)
    sigma_est_ax = estimate_sigma(final_im_ax, channel_axis=None)
    
    final_im_sag = denoise_nl_means(final_im_sag, h=H * sigma_est_sag, sigma=sigma_est_sag, fast_mode=False, patch_size=s, patch_distance=d)
    final_im_cor = denoise_nl_means(final_im_cor, h=H * sigma_est_cor, sigma=sigma_est_sag, fast_mode=False, patch_size=s, patch_distance=d)
    final_im_ax = denoise_nl_means(final_im_ax, h=H * sigma_est_ax, sigma=sigma_est_sag, fast_mode=False, patch_size=s, patch_distance=d)
    
    I = [final_im_sag, final_im_cor, final_im_ax]
    J = [1, 2, 3]
    
    for i in range(3):
        fig = plt.figure(figsize=(imag_size,imag_size)) # Create plot
        plt.imshow(np.rot90(I[i], k = 1), cmap='gray')  # Create image plot  
        
        if i == 0:   # Sagittal
            plt.xlabel(Labels[1]); plt.xticks(ax_pos[1], ax_label[1])
            plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        elif i == 1: # Coronal
            plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
            plt.ylabel(Labels[2]); plt.yticks(ax_pos[2], ax_label[2])
        elif i == 2: # Axial
            plt.xlabel(Labels[0]); plt.xticks(ax_pos[0], ax_label[0])
            plt.ylabel(Labels[1]); plt.yticks(ax_pos[1], ax_label[1])
        
        canvas = FigureCanvasTkAgg(fig, root)         # Tkinter canvas which contains matplotlib figure
        canvas.draw()
        canvas.get_tk_widget().grid(row = 1, column = J[i], rowspan = 4)    # Placing canvas on Tkinter window
        plt.close()
    
    if SNR_define == 'yes':    
        noise_col = SNR_noise_box_center[0]
        noise_row = SNR_noise_box_center[1]
        mean_col = SNR_mean_box_center[0]
        mean_row = SNR_mean_box_center[1]

        n1 = noise_row - SNR_length
        n2 = noise_row + SNR_length
        n3 = noise_col - SNR_length
        n4 = noise_col + SNR_length

        m1 = mean_row - SNR_length
        m2 = mean_row + SNR_length
        m3 = mean_col - SNR_length
        m4 = mean_col + SNR_length

        snr = snr_homemade(np.rot90(final_im_ax),n1, n2, n3, n4, m1, m2, m3, m4)
        SNR_num_label.grid_forget()
        SNR_num_label = Label(frame3, text = str(np.abs(round(snr,2))), font=("Helvetica", 12))
        SNR_num_label.grid(row = 4, column = 7) 

    elif SNR_define == 'no':
        SNR_num_label.grid_forget()
        SNR_num_label = Label(frame3, text = 'SNR not yet define!', font=("Helvetica", 12))
        SNR_num_label.grid(row = 4, column = 7)
    
#/////////////////////// FRAMES ///////////////////////#

# Font size parameter
f = 12 # labels
e = 10 # entries

##### Frame regarding sequences #####
frame1 = LabelFrame(root, text = "Possible sequences to simulate", font=("Helvetica", 15))
frame1.grid(row = 0, column = 0, rowspan = 2)

# Label informing user of predefined sequence choosen
lab = Label(frame1, text = "Select one sequence: ", font=("Helvetica", f)).grid(row = 0, column = 0)

# Spin Echo button
TSE_button = Button(frame1, text = "Spin echo", font=("Helvetica", f), command = SE).grid(row = 1, column = 0)
# Gradient Echo Sequence button
GES_button = Button(frame1, text = "Gradient echo", font=("Helvetica", f), command = GE).grid(row = 2, column = 0)
# Inversion recovery button
IN_button = Button(frame1, text = "Inversion recovery", font=("Helvetica", f), command = IN).grid(row = 3, column = 0)
# Double inversion recovery button
double_IN_button = Button(frame1, text = "Double inversion recovery", font=("Helvetica", f), command = double_IN).grid(row = 4, column = 0)
# FLAIR button
FLAIR_button = Button(frame1, text = "Fluid-attenuated inversion recovery", font=("Helvetica", f), command = FLAIR).grid(row = 5, column = 0)
# TSE button
TSE_button = Button(frame1, text = "Turbo spin echo", font=("Helvetica", f), command = TSE).grid(row = 6, column = 0)
# Diffusion button
Dif_button = Button(frame1, text = "Diffusion", font=("Helvetica", f), command = Diffusion).grid(row = 7, column = 0)
# SSFP button
SSFP_button = Button(frame1, text = "Stateady-state free precession", font=("Helvetica", f), command = SSFP).grid(row = 8, column = 0)

global Pre_def_seq
Pre_def_seq = " "
seq_label1 = Label(frame1, text = "You have chosen: ", font=("Helvetica", f)).grid(row = 9, column = 0)
seq_label2 = Label(frame1, text = Pre_def_seq); seq_label2.grid(row = 10, column = 0); seq_label2.grid_forget()

# Reset button
reset_button = Button(root, text = "Reset images and parameters", font=("Helvetica", f), command = reset).grid(row = 2, column = 0)

# //////////////////////////////////////////////////////////////////// #
#################### Frame regarding the parameters ####################
# //////////////////////////////////////////////////////////////////// #

frame3 = LabelFrame(root, text = "Parameters", font=("Helvetica", 15))
frame3.grid(row = 7, column = 1, columnspan = 2)

# Labels of the parameters
# Parameters that will always be shown
readout_label = Label(frame3, text = "Readout gradient ", font=("Helvetica", f));          readout_label.grid(row = 0, column = 1)
phase1_label = Label(frame3, text = "Phase gradient 1", font=("Helvetica", f));            phase1_label.grid(row = 0, column = 3)
phase2_label = Label(frame3, text = "Phase gradient 2", font=("Helvetica", f));            phase2_label.grid(row = 0, column = 5)
FOV1_label = Label(frame3, text = "Field of view (mm) ", font=("Helvetica", f));           FOV1_label.grid(row = 1, column = 0)
FOV2_label = Label(frame3, text = "x", font=("Helvetica", f));                             FOV2_label.grid(row = 1, column = 2)
FOV3_label = Label(frame3, text = "x", font=("Helvetica", f));                             FOV3_label.grid(row = 1, column = 4)
Resolution1_label = Label(frame3, text = "Voxel resolution (mm) ", font=("Helvetica", f)); Resolution1_label.grid(row = 2, column = 0)
Resolution2_label = Label(frame3, text = "x", font=("Helvetica", f));                      Resolution2_label.grid(row = 2, column = 2)
Resolution3_label = Label(frame3, text = "x", font=("Helvetica", f));                      Resolution3_label.grid(row = 2, column = 4)
Bandwidth_label = Label(frame3, text = "Bandwidth (Hz) ", font=("Helvetica", f));          Bandwidth_label.grid(row = 3, column = 0, pady = 10)
TR_label = Label(frame3, text = "TR (ms) ", font=("Helvetica", f));                        TR_label.grid(row = 4, column = 0)

# Parameters dependent on the sequence
TE_label = Label(frame3, text = "TE (ms) ", font=("Helvetica", f));                        TE_label.grid(row = 5, column = 0)
TI_label = Label(frame3, text = "TI (ms) ", font=("Helvetica", f));                        TI_label.grid(row = 6, column = 0)
TI2_label = Label(frame3, text = "TI2 (ms) ", font=("Helvetica", f));                      TI2_label.grid(row = 6, column = 2)
Alpha_label = Label(frame3, text = "Alpha ", font=("Helvetica", f));                       Alpha_label.grid(row = 7, column = 0)
ETL_label = Label(frame3, text = "ETL ", font=("Helvetica", f));                           ETL_label.grid(row = 7, column = 0)
Kspace_traj_label = Label(frame3, text = "Kspace trajectory ", font = ("Helvetica", f));   Kspace_traj_label.grid(row = 7, column = 1)    
TEeff_label = Label(frame3, text = "TEeff (ms) ", font = ("Helvetica", f));                TEeff_label.grid(row = 7, column = 2) 
TEeff_val_label = Label(frame3, text = "  ", font = ("Helvetica", f));                     TEeff_val_label.grid(row = 7, column = 3)  
G_label = Label(frame3, text = "G (mT/mm) ", font=("Helvetica", f));                       G_label.grid(row = 9, column = 0)
smalldelta_label = Label(frame3, text = "small delta (ms) ", font=("Helvetica", f));       smalldelta_label.grid(row = 10, column = 0)
bigdelta_label = Label(frame3, text = "big delta (ms) ", font=("Helvetica", f));           bigdelta_label.grid(row = 11, column = 0)
B_label = Label(frame3, text = "b (s/mm2) ", font=("Helvetica", f));                       B_label.grid(row = 9, column = 3)
TEmin_label = Label(frame3, text = "TEmin (ms) ", font=("Helvetica", f));                  TEmin_label.grid(row = 10, column = 3)
Bval_label = Label(frame3, text = "302.18", font=("Helvetica", f));                        Bval_label.grid(row = 9, column = 4)
TEminval_label = Label(frame3, text = "3.0", font=("Helvetica", f));                       TEminval_label.grid(row = 10, column = 4)

# Entries of the parameters
# Parameters that will always be shown
FOV1_entry = Entry(frame3, font=("Helvetica", e));      FOV1_entry.grid(row = 1, column = 1);       FOV1_entry.insert(0, '250')
FOV2_entry = Entry(frame3, font=("Helvetica", e));      FOV2_entry.grid(row = 1, column = 3);       FOV2_entry.insert(0, '300')
FOV3_entry = Entry(frame3, font=("Helvetica", e));      FOV3_entry.grid(row = 1, column = 5);       FOV3_entry.insert(0, '275')
Res1_entry = Entry(frame3, font=("Helvetica", e));      Res1_entry.grid(row = 2, column = 1);       Res1_entry.insert(0, '1.95')
Res2_entry = Entry(frame3, font=("Helvetica", e));      Res2_entry.grid(row = 2, column = 3);       Res2_entry.insert(0, '2.34')
Res3_entry = Entry(frame3, font=("Helvetica", e));      Res3_entry.grid(row = 2, column = 5);       Res3_entry.insert(0, '2.14')
Bandwidth_entry = Entry(frame3, font=("Helvetica", e)); Bandwidth_entry.grid(row = 3, column = 1);  Bandwidth_entry.insert(0, '50000')
TR_entry = Entry(frame3, font=("Helvetica", e));        TR_entry.grid(row = 4, column = 1);         TR_entry.insert(0, '500')
# Parameters dependent on the sequence
TE_entry = Entry(frame3, font=("Helvetica", e));        TE_entry.grid(row = 5, column = 1);         TE_entry.insert(0, '20')
TI_entry = Entry(frame3, font=("Helvetica", e));        TI_entry.grid(row = 6, column = 1);         TI_entry.insert(0, '250')
TI2_entry = Entry(frame3, font=("Helvetica", e));       TI2_entry.grid(row = 6, column = 3);        TI2_entry.insert(0, '250')
Alpha_entry = Entry(frame3, font=("Helvetica", e));     Alpha_entry.grid(row = 7, column = 1);      Alpha_entry.insert(0, '45')
ETL_entry = Entry(frame3, font=("Helvetica", e));       ETL_entry.grid(row = 8, column = 1);        ETL_entry.insert(0, '8')
G_entry = Entry(frame3, font=("Helvetica", e));         G_entry.grid(row = 9, column = 1);          G_entry.insert(0, '10')
smalldelta_entry = Entry(frame3, font=("Helvetica",e)); smalldelta_entry.grid(row = 10, column = 1);smalldelta_entry.insert(0, '1')
bigdelta_entry = Entry(frame3, font=("Helvetica", e));  bigdelta_entry.grid(row = 11, column = 1);  bigdelta_entry.insert(0, '2')

# Button to compute the b parameter in diffusion sequence
bcomput_button = Button(frame3, text = "Compute b", font=("Helvetica", f), command = b_TEmin)
bcomput_button.grid(row = 10, column = 2);  
# Dropdown menu to select kspace trajectory in TSE sequence   
options = [
    "Linear",
    "In-out",
    "Out-in"
]
traj = StringVar()
traj.set("No selection")                                           # Default value, or could use; options[0]
tse_drop = OptionMenu(frame3, traj, *options, command = showTEeff) # Using a list, NEEDS a star in front
tse_drop.grid(row = 8, column = 1)

# Hide the labels and the entries of specific sequences, they will appear only if the sequence where they are involed is selected
# (Label ---- Entry ---- Button ---- Dropdown)
TE_label.grid_forget();         TE_entry.grid_forget();                                          tse_drop.grid_forget()
TI_label.grid_forget();         TI_entry.grid_forget();         bcomput_button.grid_forget()
TI2_label.grid_forget();        TI2_entry.grid_forget()
Alpha_label.grid_forget();      Alpha_entry.grid_forget()
ETL_label.grid_forget();        ETL_entry.grid_forget()
Kspace_traj_label.grid_forget()
TEeff_label.grid_forget()
TEeff_val_label.grid_forget()
G_label.grid_forget();          G_entry.grid_forget()
smalldelta_label.grid_forget(); smalldelta_entry.grid_forget()
bigdelta_label.grid_forget();   bigdelta_entry.grid_forget()
B_label.grid_forget()
Bval_label.grid_forget()
TEmin_label.grid_forget()       
TEminval_label.grid_forget()

# Those values are computed from the G/small delta/big delta values
B = 302.18
dif_TEmin = np.divide(3.0,1000)
# This value is computed from the initial FOV, resolution, bandwidth values ( 128/(50000*2) = 0.00128)
minimumTE = 0.00128 * 1000 
minimumTR = 2 * minimumTE

# Labels of the parameters computed from input ones
Timescan_label = Label(frame3, text = "Total scan duration ", font=("Helvetica", f)); Timescan_label.grid(row = 1, column = 6)
Data_matrix_label = Label(frame3, text = "Data matrix ", font=("Helvetica", f));      Data_matrix_label.grid(row = 2, column = 6)
Bd_pix_label = Label(frame3, text = "Bandwidth / pixel ", font=("Helvetica", f));     Bd_pix_label.grid(row = 3, column = 6)
SNR_label = Label(frame3, text = "SNR ", font=("Helvetica", f));                      SNR_label.grid(row = 4, column = 6)
minimumTE_label = Label(frame3, text = "TEmin (ms) ", font=("Helvetica", f));         minimumTE_label.grid(row = 5, column = 5)
minimumTR_label = Label(frame3, text = "TRmin (ms) ", font=("Helvetica", f));         minimumTE_label.grid(row = 5, column = 6);  minimumTR_label.grid_forget()

Time_scan_num_label = Label(frame3, text = "2:16:32", font=("Helvetica", f));         Time_scan_num_label.grid(row = 1, column = 7)
Data_mat1_label = Label(frame3, text = "390.62", font=("Helvetica", f));              Data_mat1_label.grid(row = 2, column = 7)
Bd_by_pixel_label = Label(frame3, text = "128x128x128", font=("Helvetica", f));       Bd_by_pixel_label.grid(row = 3, column = 7)
SNR_num_label = Label(frame3, text = "      ", font=("Helvetica", f));                SNR_num_label.grid(row = 4, column = 7)
minimumTE_num_label = Label(frame3, text = minimumTE, font=("Helvetica", f));         minimumTE_num_label.grid(row = 5, column = 7)
minimumTR_num_label = Label(frame3, text = minimumTR, font=("Helvetica", f));         minimumTR_num_label.grid(row = 5, column = 8); minimumTR_num_label.grid_forget()

# Button to compute the minimum TE (or TR)
min_TETR_button = Button(frame3, text = "Compute TE/TR min", font=("Helvetica", f), command = TETRmin).grid(row = 6, column = 6)
# Button to define the boxes used in the computation of the SNR
SNR_vis_button = Button(frame3, text = "Setting & visualize SNR", font=("Helvetica", f), command = SNR_vis).grid(row = 7, column = 6, columnspan = 2)
SNR_length = 0
SNR_noise_box_center = [0,0]
SNR_mean_box_center = [0,0]
SNR_define = 'no' # String keeping track if the boxes to compute the SNR have be define or not

# Button for applying the parameters and simulate the chosen sequence
Run_seq_button = Button(frame3, text = "Run sequence", font=("Helvetica", 18), command = run).grid(row = 15, column = 5)


# function to be called when
# keyboard buttons are pressed
def update(event):
    
    global Pre_def_seq
    global TR
    global TE
    global TI
    global TI2
    global FOV
    global Data_mat
    global Resolution
    global Bandwidth
    global dif_TEmin
    global minimumTE
    global minimumTR
    
    global Time_scan_num_label
    global Data_mat1_label
    global Bd_by_pixel_label
    global SNR_num_label
    global minimumTE_num_label
    global minimumTR_num_label
    
    TR = TR_entry.get();        TR = int(TR);   TR = np.divide(TR,1000) # Divide by 1000 to have the values in milli
    TE = TE_entry.get();        TE = int(TE);   TE = np.divide(TE,1000)
    TI = TI_entry.get();        TI = int(TI);   TI = np.divide(TI,1000)
    TI2 = TI2_entry.get();      TI2 = int(TI2); TI2 = np.divide(TI2,1000)
    fov1 = FOV1_entry.get();    fov1 = int(fov1)
    fov2 = FOV2_entry.get();    fov2 = int(fov2)
    fov3 = FOV3_entry.get();    fov3 = int(fov3)
    res1 = Res1_entry.get();    res1 = float(res1) 
    res2 = Res2_entry.get();    res2 = float(res2)
    res3 = Res3_entry.get();    res3 = float(res3)
    bd = Bandwidth_entry.get(); Bandwidth = int(bd)
    Alpha = Alpha_entry.get();  Alpha = int(Alpha)
    ETL = ETL_entry.get();      ETL = int(ETL)
    FOV = [fov1, fov2, fov3]
    Resolution = [res1, res2, res3]
    Data_mat = np.divide(FOV, Resolution)
    Data_mat = [int(Data_mat[0]), int(Data_mat[1]), int(Data_mat[2])]
    
    time_scan = TR * Data_mat[1] * Data_mat[2]
    
    if Pre_def_seq == 'TSE':
        ETL = ETL_entry.get(); ETL = int(ETL)
        time_scan = time_scan/ETL

    time = datetime.timedelta(seconds=time_scan) # Converts amount of seconds to hours:minutes:seconds
    Bd_by_pixel = np.divide(Bandwidth,Data_mat[0])
    
    Time_scan_num_label.grid_forget()
    Data_mat1_label.grid_forget()
    Bd_by_pixel_label.grid_forget()
    
    Time_scan_num_label = Label(frame3, text = str(time), font=("Helvetica", 12));               Time_scan_num_label.grid(row = 1, column = 7)    
    s = str(Data_mat[0]) + "x" + str(Data_mat[1]) + "x" + str(Data_mat[2])
    Data_mat1_label = Label(frame3, text = s, font=("Helvetica", 12));                           Data_mat1_label.grid(row = 2, column = 7)
    Bd_by_pixel_label = Label(frame3, text = str(round(Bd_by_pixel,2)), font=("Helvetica", 12)); Bd_by_pixel_label.grid(row = 3, column = 7)

# Binding the keyboard with the main window, to be able to change the scan duration, bandwidth/pixel, etc... in real time
root.bind('<Key>', update)


# ///////////////////////////////////////////////////////////////////////// #
#################### Frame regarding the post-processing ####################
# ///////////////////////////////////////////////////////////////////////// #

frame4 = LabelFrame(root, text = "Post-processing & visualization", font=("Helvetica", 15))
frame4.grid(row = 7, column = 3)

### Labels and buttons of post-processing options
Postprocess_label = Label(frame4, text = "Post-processing options", font=("Helvetica", f)).grid(row = 0, column = 0)
### Labels
low_pass_label = Label(frame4, text = "Kernel size --> ", font=("Helvetica", f)).grid(row = 1, column = 0)
high_pass_label = Label(frame4, text = "Kernel size (must be odd)--> ", font=("Helvetica", f)).grid(row = 2, column = 0)
Gauss_label = Label(frame4, text = "Std of kernel filter --> ", font=("Helvetica", f)).grid(row = 3, column = 0)
non_local_label = Label(frame4, text = "h, patch size and distance --> ", font=("Helvetica", f)).grid(row = 4, column = 0)
Filter_vis_button = Label(frame4, text = "Filter visualization", font=("Helvetica", f)).grid(row = 5, column = 1, columnspan = 2)
### Entries
low_pass_entry = Entry(frame4, font=("Helvetica", e));                   low_pass_entry.grid(row = 1, column = 1, columnspan = 3); low_pass_entry.insert(0,'3')
high_pass_entry = Entry(frame4, font=("Helvetica", e));                  high_pass_entry.grid(row = 2, column = 1, columnspan =3); high_pass_entry.insert(0,'3')
Gauss_entry = Entry(frame4, font=("Helvetica", e));                      Gauss_entry.grid(row = 3, column = 1, columnspan = 3); Gauss_entry.insert(0, '0.5')
Non_local_h_entry = Entry(frame4, font=("Helvetica", e), width = 5);     Non_local_h_entry.grid(row = 4, column = 1); Non_local_h_entry.insert(0, '0.8')
Non_local_psize_entry = Entry(frame4, font=("Helvetica", e), width = 5); Non_local_psize_entry.grid(row = 4, column = 2); Non_local_psize_entry.insert(0, '7')
Non_local_pdist_entry = Entry(frame4, font=("Helvetica", e), width = 5); Non_local_pdist_entry.grid(row = 4, column = 3); Non_local_pdist_entry.insert(0, '11')
### Buttons (Lambda is to use the parentheses and arguments)
Low_pass_button = Button(frame4, text = "Low-pass filter", font=("Helvetica", f), command = lambda: lowpass(low_pass_entry.get())).grid(row = 1, column = 4)
High_pass_button = Button(frame4, text = "High-pass filter", font=("Helvetica", f), command = lambda: highpass(high_pass_entry.get())).grid(row = 2, column = 4)
Gaussian_button = Button(frame4, text = "Gaussian filter", font=("Helvetica", f), command = lambda: gauss(Gauss_entry.get())).grid(row = 3, column = 4) 
Non_local_button = Button(frame4, text = "Non local filter", font=("Helvetica", f), command = lambda: non_local(Non_local_h_entry.get(), Non_local_psize_entry.get(), Non_local_pdist_entry.get())).grid(row = 4, column = 4) 

Param_vis_button = Button(frame4, text = "Parameters visualization", font=("Helvetica", f), command = param_vis).grid(row = 6, column = 1, columnspan = 3)

# Dropdown menu to select filter to visualize   
options = [
    "low",
    "high",
    "gauss",
    "non local"
]
fi = StringVar()
fi.set("No selection")                                               # Default value, or could use; options[0]
filter_drop = OptionMenu(frame4, fi, *options, command = filter_vis) # Using a list, NEEDS a star in front
filter_drop.grid(row = 5, column = 3)

# //////////////////////////////////////////////////////////////// #
#################### Frame regarding the saving ####################
# //////////////////////////////////////////////////////////////// #

frame2 = LabelFrame(frame3, text = "Saving images", font=("Helvetica", f))
frame2.grid(row = 15, column = 0, columnspan = 2)

def save_img(axe):
    global final_im_sag
    global final_im_cor
    global final_im_ax
    global Pre_def_seq
    
    er_label.grid_forget(); 
    all_axes = ['sagittal', 'coronal', 'axial']
    
    if Pre_def_seq == " ":
        er_label.grid(row = 2, column = 0)
    else:
        
        if axe in all_axes:
            #files = [('JPEG', '*.jpg'), ('TIFF', '*.tif')]
            #filename = filedialog.asksaveasfile(mode='w', filetypes = files, defaultextension=files)
            #filename = filedialog.asksaveasfile(mode='w', defaultextension=".tif")
            filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg")
            if not filename:
                return
            if axe == 'sagittal':
                img = Image.fromarray(np.rot90(final_im_sag*200))
            elif axe == 'coronal':
                img = Image.fromarray(np.rot90(final_im_cor*200))
            elif axe == 'axial':
                img = Image.fromarray(np.rot90(final_im_ax*200))

            img = img.convert("L")
            img.save(filename)

        else:
            er_label.grid(row = 2, column = 0)

Axe_save_label = Label(frame2, text = "sagittal, coronal, or axial ", font=("Helvetica", f)); Axe_save_label.grid(row = 0, column = 0)
Axe_save_entry = Entry(frame2, font=("Helvetica", e)); Axe_save_entry.grid(row = 1, column = 0); Axe_save_entry.insert(0,'sagittal')

Sag_save_button = Button(frame2, text="Save", command = lambda: save_img(Axe_save_entry.get()))
Sag_save_button.grid(row = 1, column = 1)

er_label = Label(frame2, text = "Misspelled axe or no sequence selected! ", font=("Helvetica", 10)); er_label.grid(row = 2, column = 0); 
er_label.grid_forget();

# Warning labels if there is a paradox in the parameters
warning_label1 = Label(root, text = "Warning!! ", font=("Helvetica", 15)); warning_label1.grid(row = 3, column = 0); warning_label1.grid_forget()
warning_label2 = Label(root, text = "  ", font=("Helvetica", 15)); warning_label2.grid(row = 4, column = 0); warning_label2.grid_forget()

# Exit button
Exit_button = Button(root, text = "Exit program", font=("Helvetica", f), command = root.destroy)
Exit_button.grid(row = 10, column = 0) 

#/////////////////////// DATA ///////////////////////#
##### ////////////////// 3D //////////////////// #####
##### Data to compute the reconstruction images #####
global alpha2_3D
global alpha_3D
global alp_3D
global delta_B0_3D
global B0_3D
global T1_3D
global T2_3D
global t2_star_tensor_3D
global M0_3D
global Xgrad_3D 
global Zgrad_3D 
global ADC_3D

# The scipy.io.loadmat() function creates pyhton dictionnaries, 250 x 300 x 275
mat1 = scipy.io.loadmat('2alpha_webb.mat')
mat2 = scipy.io.loadmat('alpha_webb.mat')
mat3 = scipy.io.loadmat('T1file.mat')
mat4 = scipy.io.loadmat('T2file.mat')
mat5 = scipy.io.loadmat('PDfile.mat')
B0_Tom_3D = np.load('Tom-B0.npy')

alpha2_3D = mat1['image_2alpha']
alpha_3D = mat2['image_alpha']
T1_3D = mat3['T1']; T1_3D = T1_3D[6:256,:,:]
T2_3D = mat4['T2']; T2_3D = T2_3D[6:256,:,:]
M0_3D = np.load('M0.npy')      
del mat1; del mat2; del mat3; del mat4; del mat5 

# shifting M0 to match it to T1 and T2 maps
M0_3D = np.delete(M0_3D, [0,1,2], axis = 0)
M0_3D = np.append(M0_3D, np.zeros((3,300,275)), axis = 0)

# M0, T1, and T2 shift
d = 50                      # number of columns to delete and add
add = np.zeros((250,300,d))
delete = np.arange(d)
T1_shifted_3D = np.delete(T1_3D, delete, 2);
T1_3D = np.append(T1_shifted_3D, add, axis=2)
T2_shifted_3D = np.delete(T2_3D, delete, 2); T2_3D = np.append(T2_shifted_3D, add, axis=2)
m0_shifted_3D = np.delete(M0_3D, delete, 2); M0_3D = np.append(m0_shifted_3D, add, axis=2)

# ADC (for diffusion)
ADC_3D = np.copy(M0_3D)
ADC_3D[ADC_3D == 0.7] = 700
ADC_3D[ADC_3D == 0.82] = 850
ADC_3D[ADC_3D == 1] = 3200
ADC_3D = ADC_3D * 10**(-6)

# B0
"""
(51,51,51)
first axis: up-down --> SAGITTAL
second axis: left-right --> CORONAL
third axis: down the bore --> AXIAL
250/51 = 4.90196078431, 300/51 = 5.88235294118, 275/51 = 5.39215686275
"""
B0_3D = B0_Tom_3D
B0_3D= np.nan_to_num(B0_3D)
B0_3D = zoom(B0_3D, (4.9, 5.88, 5.39), order=0)
B0_3D[B0_3D<1] = np.nan
B0_3D = np.divide(B0_3D,1000)
mean_3D = np.nanmean(B0_3D) 
B0_3D = np.nan_to_num(B0_3D)
B0_mean_3D = B0_3D; B0_mean_3D = np.nan_to_num(B0_mean_3D); B0_mean_3D[B0_mean_3D > 0] = mean_3D
delta_B0_3D = np.abs(B0_3D - B0_mean_3D)
delta_B0_3D[np.isnan(delta_B0_3D)] = 0

# Flipmap
fa60dataFile = r'C:\Users\mathieu\Desktop\Leiden\Code\20220805 - B1 head phantom\GRE_3D\2\data.3d'
fa120dataFile = r'C:\Users\mathieu\Desktop\Leiden\Code\20220805 - B1 head phantom\GRE_3D\3\data.3d'
scanParams = keaProc.readPar(r'C:\Users\mathieu\Desktop\Leiden\Code\20220805 - B1 head phantom\GRE_3D\2\acqu.par')
fa60kSpace = keaProc.readKSpace(fa60dataFile)
fa120kSpace = keaProc.readKSpace(fa120dataFile)
fa60kSpace = keaProc.sineBellSquaredFilter(fa60kSpace, filterStrength=0.5)
fa120kSpace = keaProc.sineBellSquaredFilter(fa120kSpace, filterStrength=0.5)
fa60Image = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(fa60kSpace)))
fa120Image = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(fa120kSpace)))
mask_3D = (np.abs(fa120Image)>250).astype(np.double)

d_3D = np.abs(fa120Image/(2*fa60Image))
d_3D = d_3D*mask_3D
d_3D[ d_3D > 1] = 1
d_3D[ d_3D < -1] = -1
flipAngleMap_3D = np.arccos(d_3D)
flipAngleMap_3D = np.nan_to_num(flipAngleMap_3D)

result_3D = ndimage.zoom(flipAngleMap_3D, (3.1875, 3.5, 4)) # (204, 224, 256)

a = np.divide(250 - result_3D.shape[0],2); a = int(a)
b = np.divide(300 - result_3D.shape[1],2); b = int(b)
c = np.divide(275 - result_3D.shape[2],2); c = int(c)
aa = 250 - 2*a - result_3D.shape[0]
bb = 300 - 2*b - result_3D.shape[1]
cc = 275 - 2*c - result_3D.shape[2]
result_pad_3D = np.pad(result_3D, ((a,a+aa), (b,b+bb), (c,c+cc)))

d = 40 # number of columns to delete and add
add = np.zeros((250,300,d))
delete = np.arange((275-d),275)
t_3D = np.delete(result_pad_3D, delete, 2)
flipAngleMap_3D = np.concatenate((add,t_3D), axis = 2)

# B1 map
gyro = 42.58
tau = 0.0001 # 100 micro
B1map_3D = np.divide(flipAngleMap_3D,gyro*tau)

# Rescale flip angle map
flipAngleMaprescale_3D = (flipAngleMap_3D/flipAngleMap_3D.max())*90

Alpha = 45
alp_3D = flipAngleMaprescale_3D/Alpha

# Gradient dostortions
# The data are in Telsa (V*s/m^2)
# 50 x 50 x 50
matx_3D = scipy.io.loadmat('X_gradient50.mat') 
matz_3D = scipy.io.loadmat('Z_gradient50.mat') 
mapx_3D = matx_3D['Grad']
mapz_3D = matz_3D['Grad']
xgrad_3D = np.reshape(mapx_3D[:,3], (50, 50, 50))
zgrad_3D = np.reshape(mapz_3D[:,3], (50, 50, 50))
Zgrad_3D = ndimage.zoom(zgrad_3D, (5, 6, 5.5)) 
Xgrad_3D = ndimage.zoom(xgrad_3D, (5, 6, 5.5)) 

# 3D t2* 
T1_3D = np.divide(T1_3D,1000)
T2_3D = np.divide(T2_3D,1000)
gamma =  42.58*(10**6)*2*constants.pi # gyromagnetic ratio for hydrogen 42.58 [MHz/T] * 2pi
t2_inverse_3D = np.divide(1, T2_3D, out=np.zeros_like(T2_3D), where=T2_3D!=0)
D_gam_3D = delta_B0_3D * np.divide(gamma,1000)
inv_t2star_3D = t2_inverse_3D + D_gam_3D
t2_star_tensor_3D = np.divide(1, inv_t2star_3D, out=np.zeros_like(inv_t2star_3D), where=t2_inverse_3D!=0)

############ Figures that will be there when programm is started
imag_size = 5.5
fig = plt.figure(figsize=(imag_size,imag_size))                # Create plot
plt.imshow(np.rot90(M0_3D[128,:,:]), cmap='gray')              # Create image plot 128
plt.axis('off')
canvas = FigureCanvasTkAgg(fig, root)                          # Tkinter canvas which contains matplotlib figure
canvas.draw()
canvas.get_tk_widget().grid(row = 1, column = 1, rowspan = 4)  # Placing canvas on Tkinter window        
plt.close()

fig = plt.figure(figsize=(imag_size,imag_size))
plt.imshow(np.rot90(M0_3D[:,150,:]), cmap='gray')   
plt.axis('off')
canvas = FigureCanvasTkAgg(fig, root)         
canvas.draw()
canvas.get_tk_widget().grid(row = 1, column = 2, rowspan = 4)        
plt.close()

fig = plt.figure(figsize=(imag_size,imag_size))
plt.imshow(np.rot90(M0_3D[:,:,135]), cmap='gray')   
plt.axis('off')
canvas = FigureCanvasTkAgg(fig, root)         
canvas.draw()
canvas.get_tk_widget().grid(row = 1, column = 3, rowspan = 4)          
plt.close()

Sagittal_label = Label(root, text = "Sagittal", font=("Helvetica", 15)). grid(row = 0, column = 1)
Coronal_label = Label(root, text = "Coronal", font=("Helvetica", 15)). grid(row = 0, column = 2)
Axial_label = Label(root, text = "Axial", font=("Helvetica", 15)). grid(row = 0, column = 3)

root.mainloop()






