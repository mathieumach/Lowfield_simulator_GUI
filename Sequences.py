import numpy as np
from scipy import constants

# Functions of the sequences
def spin_echo_seq(TR, TE, t1, t2, m0):
    # Spin echo formula
    TR_div = np.divide(TR, t1, out=np.zeros_like(t1), where=t1!=0)
    TE_div = np.divide(TE, t2, out=np.zeros_like(t2), where=t2!=0)
    return np.abs(m0 * (1 - np.exp(-TR_div)) * np.exp(-TE_div))   

def Gradient_seq(TR, TE, t1, t2_star, m0, alp):
    # Gradient echo formula
    TR_div = np.divide(-TR, t1, out=np.zeros_like(t1), where=t1!=0)
    TE_div = np.divide(-TE, t2_star, out=np.zeros_like(t2_star), where=t2_star!=0)
    a = np.sin(alp) * (1 - np.exp(TR_div)) * np.exp(TE_div)
    b = 1 - np.cos(alp) * np.exp(TR_div)
    im = np.divide(a, b, out=np.zeros_like(a), where=a!=0)
    return np.abs( m0 * im )

def IN_seq(TR, TE, TI, t1, t2, m0):
    # Inversion recovery formula
    a = 2 * np.exp(-np.divide(TI,t1, out=np.zeros_like(t1), where=t1!=0))
    b = np.exp(-np.divide(TR, t1, out=np.zeros_like(t1), where=t1!=0))
    c = 1-a+b
    return np.abs(m0 * c * np.exp(-np.divide(TE,t2, out=np.zeros_like(t2), where=t2!=0)))

def DoubleInversion_seq(TR, TE, TI1, TI2, t1, t2, m0):
    # Double inversion recovery formula
    E1 = np.exp(-np.divide(TI1, t1, out=np.zeros_like(t1), where=t1!=0))
    E2 = np.exp(-np.divide(TI2, t1, out=np.zeros_like(t1), where=t1!=0))
    Ec = np.exp(-np.divide(TR, t1, out=np.zeros_like(t1), where=t1!=0))
    Etau = np.exp(np.divide((TE/2), t1, out=np.zeros_like(t1), where=t1!=0))
    Ee = np.exp(-np.divide(TE,t2, out=np.zeros_like(t2), where=t2!=0))
    
    Mz = 1 - 2*E2 + 2*E1*E2 - Ec*(2*Etau - 1)
    return np.abs(m0 * Mz * Ee)

def Diffusion_seq(TR, TE, t1, t2, m0, b, D):
    # Diffusion formula with predefine b
    TR_div = np.divide(TR, t1, out=np.zeros_like(t1), where=t1!=0)
    TE_div = np.divide(TE, t2, out=np.zeros_like(t2), where=t2!=0)
    spin = np.abs(m0 * (1 - np.exp(-TR_div)) * np.exp(-TE_div))
    return np.abs(spin * np.exp(-b*D))    

def T2_ETL_decay(c,t2,TE,ETL):
    # function computing the T2 decay signal  with respect to the ETL
    # Use broadcasting to speed up computation
    
    t = TE * ETL    
    time = np.linspace(0,t,c)[:, None]
    sig = np.zeros(c)
    
    t2_flat = t2.flatten()[None]
    ma = t2_flat[t2_flat != 0]
    
    T2decay = np.exp((-time)/ma)
    sig = np.sum(T2decay, 1)
    return sig

def TSE_seq(TR, TE, ETL, m0, t1, t2, c, met):
    # Turbo spin echo formula
    TEeff = 0
    if met == 'Out-in':    
        TEeff = TE * ETL    
        spin = spin_echo_seq(TR, TEeff, t1, t2, m0) # Computing a spin echo
    elif met == 'In-out':
        TEeff = TE
        TE_div = np.divide(TEeff, t2, out=np.zeros_like(t2), where=t2!=0)
        spin = np.abs(m0 * np.exp(-TE_div))         # Computing a spin echo with the (1 - exp(-TR/T1)) set to 1
    elif met == 'Linear':
        TEeff = 0.5 * TE * ETL
        spin = spin_echo_seq(TR, TEeff, t1, t2, m0) # Computing a spin echo
        
    # computing the T2 decay signal with respect to the ETL
    # Use broadcasting to speed up computation
    T = TE * ETL
    time = np.linspace(0,T,c)[:, None]
    sig = np.zeros(c)
    
    t2_flat = t2.flatten()[None]
    ma = t2_flat[t2_flat != 0]
    
    T2decay = np.exp((-time)/ma)
    sig = np.sum(T2decay, 1)  # Normal t2 decay

    # kspace trajectories
    linear = np.zeros((sig.shape))
    inout = np.zeros((sig.shape))
    outin = np.zeros((sig.shape))
    
    even = np.arange(0,sig.shape[0],2)
    odd = np.arange(1,sig.shape[0],2)
    flip = np.flip(even)
    
    arr = np.concatenate((flip,odd))
    arr2 = np.concatenate((odd,flip))
    
    for i in range(len(sig)):
        linear[i] = sig[i]
        inout[i] = sig[arr[i]]
        outin[i] = sig[arr2[i]]
        
    # Fourier transform of trajectory
    if met == 'Out-in':    
        FT = np.fft.fftshift(np.fft.fft(outin))     
    elif met == 'In-out':
        FT = np.fft.fftshift(np.fft.fft(inout))
    elif met == 'Linear':
        FT = np.fft.fftshift(np.fft.fft(linear))
    FT = FT/np.max(FT)
        
    im = np.zeros((spin.shape))
    
    if len(spin.shape) == 2:
        for i in range(spin.shape[1]):           # Convolution of each row with the lorentzian
            im[:,i] = np.convolve(FT.real, spin[:,i], mode='same')
            
    elif len(im.shape) == 3:
        for j in range(spin.shape[2]):
            for i in range(spin.shape[1]):
                im[:,i,j] = np.convolve(FT.real, spin[:,i,j], mode='same')

    return np.abs(im)

def TSE_post_seq(TE, ETL, t2, c, met, data):
    
    # Turbo spin echo formula
    TEeff = 0
    if met == 'Out-in':    
        TEeff = TE * ETL    
    elif met == 'In-out':
        TEeff = TE
    elif met == 'Linear':
        TEeff = 0.5 * TE * ETL
        
    # computing the T2 decay signal with respect to the ETL
    # Use broadcasting to speed up computation
    T = TE * ETL
    time = np.linspace(0,T,c)[:, None]
    sig = np.zeros(c)
    
    t2_flat = t2.flatten()[None]
    ma = t2_flat[t2_flat != 0]
    
    T2decay = np.exp((-time)/ma)
    sig = np.sum(T2decay, 1)  # Normal t2 decay

    # kspace trajectories
    linear = np.zeros((sig.shape))
    inout = np.zeros((sig.shape))
    outin = np.zeros((sig.shape))
    
    even = np.arange(0,sig.shape[0],2)
    odd = np.arange(1,sig.shape[0],2)
    flip = np.flip(even)
    
    arr = np.concatenate((flip,odd))
    arr2 = np.concatenate((odd,flip))
    
    for i in range(len(sig)):
        linear[i] = sig[i]
        inout[i] = sig[arr[i]]
        outin[i] = sig[arr2[i]]
        
    # Fourier transform of trajectory
    if met == 'Out-in':    
        FT = np.fft.fftshift(np.fft.fft(outin))     
    elif met == 'In-out':
        FT = np.fft.fftshift(np.fft.fft(inout))
    elif met == 'Linear':
        FT = np.fft.fftshift(np.fft.fft(linear))
    FT = FT/np.max(FT)
        
    im = np.zeros((data.shape))
    
    if len(data.shape) == 2:
        for i in range(data.shape[1]):           # Convolution of each row with the lorentzian
            im[:,i] = np.convolve(FT.real, data[:,i], mode='same')
            
    elif len(im.shape) == 3:
        for j in range(data.shape[2]):
            for i in range(data.shape[1]):
                im[:,i,j] = np.convolve(FT.real, data[:,i,j], mode='same')

    return np.abs(im)

def SSFP_Echo_seq(t1, t2, M0, alpha, phi):
    # SSFP formula
    a = np.abs(np.sin(alpha))*np.sqrt(2 - 2*np.cos(phi))
    b = (1+np.cos(alpha))*(1-np.cos(alpha)) + 2*(1-np.cos(alpha))*np.divide(t1, t2, out=np.zeros_like(t2), where=t2!=0)
    return M0 * (a/b)
