
import scipy
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy.ndimage as ndi
from scipy.signal import find_peaks
from numpy.linalg import norm
import math
from scipy.io.wavfile import write
#define function
def HPSS(
    x,
    fs,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    kernel_size=(31, 17),  
    margin=(1.0, 2.0),      
    power=2.0,
    return_masks=False,
):
    x = x.astype(np.float32, copy=False)
    S = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,  
        window=window,
        center=True,
        pad_mode="reflect",
    )
    mag = np.abs(S)
    Mh, Mp = librosa.decompose.hpss(
        mag,
        kernel_size=kernel_size,
        power=power,
        mask=True,
        margin=margin,
    )
    Sh = Mh * S
    Sp = Mp * S
    y_h = librosa.istft(Sh, hop_length=hop_length, win_length=win_length, window=window, length=len(x))
    y_p = librosa.istft(Sp, hop_length=hop_length, win_length=win_length, window=window, length=len(x))

    return (y_h, y_p, Mh, Mp) if return_masks else (y_h, y_p)

f_window = {
    "hann": scipy.signal.windows.hann,
    "hamming": scipy.signal.windows.hamming,
    "blackman": scipy.signal.windows.blackman,
    "bartlett": scipy.signal.windows.bartlett
}
def plot_signal(signal,sr):
    fft_result = scipy.fft.fft(signal)
    f, t_spec, Sxx = scipy.signal.spectrogram(signal, sr, nperseg=256, noverlap=128)
    t_axis = np.linspace(0,len(signal)/sr,len(signal))
    freq_axis = np.linspace(0,sr//2,len(signal)//2)
    plt.figure(figsize=(16,8))
    plt.subplot(4,1,1)
    plt.plot(t_axis,signal)
    plt.subplot(4,1,2)
    plt.plot(freq_axis,np.abs(fft_result)[:len(signal)//2])
    plt.subplot(4,1,3)
    plt.plot(freq_axis,np.unwrap(np.angle(fft_result))[:len(signal)//2])
    plt.subplot(4,1,4)
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Intensity [dB]')
def butter_filter(signal,order,freq,sr,type='highpass'):
    b,a = scipy.signal.butter(order,freq/(sr/2),btype=type)
    x = scipy.signal.lfilter(b,a,signal)
    return x

def spectral_flux(x, sr, n_fft=1024, hop_length=256, win_length=None,
                  pre_emph=0.97,median_win=0.15, thresh_delta=0.02,):
    x = np.append(x[0], x[1:] - pre_emph * x[:-1])

    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann'))
    S = S / (np.maximum(np.median(S, axis=1, keepdims=True), 1e-8))

    dS = np.diff(S, axis=1)
    dS[dS < 0] = 0
    flux = dS.sum(axis=0)
    times = librosa.frames_to_time(np.arange(flux.size), sr=sr, hop_length=hop_length)

    med_len = max(3, int(median_win * sr / hop_length))
    baseline = ndi.median_filter(flux, size=med_len)
    odf = np.clip(flux - (baseline + thresh_delta), a_min=0, a_max=None)

    return  odf, times
def segment_by_range(signal,sr,peak_inx,head,tail):
    out = []
    start = head*sr*10**-3
    end = tail*sr*10**-3
    for p in peak_inx:
        target = signal[int(p-start):int(p+end)]
        out.append(target)
    return out

def plot_signal_HP(x_h,x_p,sr):
    fft_result_h = scipy.fft.fft(x_h)
    fft_result_p = scipy.fft.fft(x_p)
    f_h, t_spec_h, Sxx_h = scipy.signal.spectrogram(x_h, sr, nperseg=256, noverlap=128)
    f_p, t_spec_p, Sxx_p = scipy.signal.spectrogram(x_p, sr, nperseg=256, noverlap=128)
    t_axis = np.linspace(0,len(x_h)/sr,len(x_h))
    freq_axis = np.linspace(0,sr//2,len(x_h)//2)
    plt.figure(figsize=(16,8))
    plt.subplot(4,2,1)
    plt.plot(t_axis,x_h)
    plt.subplot(4,2,2)
    plt.plot(t_axis,x_p)
    plt.subplot(4,2,3)
    plt.plot(freq_axis,np.abs(fft_result_h)[:len(x_h)//2])
    plt.subplot(4,2,4)
    plt.plot(freq_axis,np.abs(fft_result_p)[:len(x_p)//2])
    plt.subplot(4,2,5)
    plt.plot(freq_axis,np.unwrap(np.angle(fft_result_h)[:len(x_h)//2]))
    plt.subplot(4,2,6)
    plt.plot(freq_axis,np.unwrap(np.angle(fft_result_p)[:len(x_p)//2]))
    plt.subplot(4,2,7)
    plt.pcolormesh(t_spec_h, f_h, 10 * np.log10(Sxx_h), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Intensity [dB]')
    plt.subplot(4,2,8)
    plt.pcolormesh(t_spec_p, f_p, 10 * np.log10(Sxx_p), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Intensity [dB]')
def freq2sampling(freq,s_length,sr):
    return freq*s_length//sr
def sampling2freq(x,s_length,sr):
    return int(round(sr*x//s_length))
def sp_coeff_range(x,sr,segment):
    x_len = len(x)
    fft_signal = np.abs(scipy.fft.fft(x))**2
    fft_signal = fft_signal/np.max(fft_signal)
    coeff = []
    parts =[[freq2sampling(start,x_len,sr),freq2sampling(stop,x_len,sr)] for start,stop in segment]
    for head,tail in parts:
        target = fft_signal[head:tail]
        coeff.append(np.sum(target)//len(target))
    return coeff
def pick_onsets_from_flux(odf, sr, hop,
                          thr_win_ms=200, delta_k=0.8,
                          distance_ms=90, prominence_k=0.4, width_frames=(1, None)):

    odf_s = odf

    win = max(3, int((thr_win_ms/1000) * sr / hop))
    baseline = np.convolve(odf_s, np.ones(win)/win, mode='same')  
    dev = np.convolve(np.abs(odf_s-baseline), np.ones(win)/win, mode='same')  
    dev = np.max(dev)
    thr = baseline +delta_k*dev

    excess = np.maximum(0, odf_s - thr)

    distance = int((distance_ms/1000) * sr / hop)

    prom = prominence_k * np.median(excess[excess>0]) if np.any(excess>0) else 0.0

    peaks, props = find_peaks(excess,
                              distance=distance,
                              prominence=prom,
                              width=width_frames)

    return peaks, props
def group_peaks(peaks, sr, hop, group_ms=300):
    if len(peaks)==0: return peaks
    gap = int((group_ms/1000)*sr/hop)
    grouped = [peaks[0]]
    for p in peaks[1:]:
        if p - grouped[-1] > gap:
            grouped.append(p)
    return np.array(grouped)
def segment_from_onsets(y, sr, onset_frames, hop, floor_db=30, max_len_ms=900):
    hits = []
    max_len = int(sr*max_len_ms/1000)
    for f in onset_frames:
        s = librosa.frames_to_samples(f, hop_length=hop)
        e_prov = min(len(y), s+max_len)

        seg = y[s:e_prov]
        rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=256).ravel()
        if len(rms) < 3: continue
        pk = rms.max()+1e-12
        rms_db = 20*np.log10(rms/pk + 1e-12)

        below = np.where(rms_db < -abs(floor_db))[0]
        if len(below):
            e = s + min(e_prov - s, below[0]*256 + 1024)
        else:
            e = e_prov

        if e - s > int(sr*0.02):   # >20 ms
            hits.append((s,e))
    return hits
def band_edges_to_bins(edges_hz, sr, n_fft):
    #convert frequencies edge to index in fft
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    nyq = sr * 0.5
    edges = np.clip(np.asarray(edges_hz, dtype=float), 0.0, nyq)
    idx = [int(np.searchsorted(freqs, f, side='left')) for f in edges]
    return idx, freqs

def avg_power_spectrum(mag, frame_sel=None, use_power=True):
    #compute power spectrum from magnitude
    S = mag[:, frame_sel] if frame_sel is not None else mag
    if S.size == 0:
        return np.zeros(mag.shape[0], dtype=float)
    if use_power:
        S = S ** 2
    return S.mean(axis=1)

def coin_featurs(mag, sr, frame_sel=None):
    #compute coin featurs from magnitude spectral
    S = mag[:, frame_sel] if frame_sel is not None else mag
    if S.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    c  = librosa.feature.spectral_centroid(S=S, sr=sr).ravel()
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr).ravel()
    fl = librosa.feature.spectral_flatness(S=S).ravel()
    return (float(np.nanmean(c)),
            float(np.nanstd(c)),
            float(np.nanmean(bw)),
            float(np.nanmedian(fl)))
def getPath(prefix,type,endNum):
    path= []
    for i in range(1,endNum+1,1):
        path.append(f"{prefix}{type}.{i}.mp3")
    return path
def compute_coin_features(x, sr, n_fft=1024, hop=256,
                          band_edges_hz=( 5000,10000,15000)
                          ,window='hann',kernal_size=(31,17)):
  
    # STFT
   
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop, window=window, center=True)
    
    mag = np.abs(X)
    
    # HPSS on magnitudes with differenct masks for each components
    Mh, Mp = librosa.decompose.hpss(mag, kernel_size=kernal_size, power=2.0, mask=True, margin=(1, 2))
    Sh = Mh * X
    Sp = Mp * X
    magH = np.abs(Sh)
    magP = np.abs(Sp)
    
    
    # frame selection
    n_frames = magP.shape[1]
    ms2f = lambda t_ms: int(round((t_ms / 1000.0) * sr / hop))
    early_start, early_end = 0, min(ms2f(100), n_frames)
    late_start, late_end   = min(ms2f(80), n_frames), min(ms2f(300), n_frames)
    early_frames = slice(early_start, early_end)
    late_frames  = slice(late_start, late_end)

    # Harmonic features 
    c_mean, c_std, bw_mean, flat_med = coin_featurs(magH, sr, early_frames)

    # Percussive spectrum 
    Ppow_late = avg_power_spectrum(magP, frame_sel=late_frames, use_power=True)

    # compute power spectrum ratio
    idx, freqs = band_edges_to_bins(band_edges_hz, sr, n_fft)
    band_sums = []
    for b in range(0,len(idx),2):
        i0, i1 = idx[b], idx[b + 1]
        band_sums.append(Ppow_late[i0:i1].sum() if i1 > i0 else 0.0)
    band_sums = np.asarray(band_sums, dtype=float)
    total = float(band_sums.sum() + 1e-12)
    ratios = band_sums / total

    feats = np.array([
        c_mean, c_std, bw_mean, flat_med, 
        *ratios.tolist(),                                                         
    ], dtype=float)

    return feats
def add_awgn(signal, snr_db):
    signal_power = np.mean(signal**2) 
    snr_linear = 10**(snr_db / 10)    
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)  


    noise = np.random.normal(0, noise_std, signal.shape)

    noisy_signal = signal + noise
    return noisy_signal
def get_coin_features(paths,band_edges_hz,kernal_size,n_fft,hop,snr_db=None):
    if isinstance(paths,str):
        paths = [paths]
    test_feature = None
    for path in paths:
        y, sr = librosa.load( path,sr=None) 
        if snr_db != None:
            y = add_awgn(y,snr_db=snr_db)
        filtered_signal = butter_filter(y,6,5000,sr,type="highpass")
        HOP=256
        odf,_ = spectral_flux(filtered_signal,sr=sr,hop_length=HOP)
        peaks,_ = pick_onsets_from_flux(odf,sr,HOP)
        peaks = group_peaks(peaks,sr,HOP,group_ms=1000)
        segmented_signal=segment_from_onsets(filtered_signal,sr,peaks,HOP)
        coin_feature = np.array([compute_coin_features(filtered_signal[s:e],sr,band_edges_hz=band_edges_hz,kernal_size=kernal_size,n_fft=n_fft,hop=hop) for s,e in segmented_signal])
        if np.any(test_feature) ==None:
            test_feature = np.asarray(coin_feature.copy())
            continue
        test_feature =np.concatenate([test_feature,coin_feature])
    return test_feature
def get_coin_train_zscore(band_edges_hz,kernal_size,n_fft,hop,c1_path,c5_path,c10_path):
    eps = 1e-8
    fea_small= get_coin_features(c1_path,band_edges_hz,kernal_size=kernal_size,n_fft=n_fft,hop=hop)
    fea_med =get_coin_features(c5_path,band_edges_hz,kernal_size=kernal_size,n_fft=n_fft,hop=hop)
    fea_large =get_coin_features(c10_path,band_edges_hz,kernal_size=kernal_size,n_fft=n_fft,hop=hop)
    
    fea_all = np.vstack([fea_small, fea_med, fea_large])

    mu_pool = fea_all.mean(axis=0)                      
    sd_pool = fea_all.std(axis=0, ddof=1) + eps          


    Z_small = (fea_small - mu_pool) / sd_pool          
    Z_med   = (fea_med   - mu_pool) / sd_pool
    Z_large = (fea_large - mu_pool) / sd_pool

    m_small = Z_small.mean(axis=0)                    
    m_med   = Z_med.mean(axis=0)
    m_large = Z_large.mean(axis=0)

  
    m_small = np.clip(m_small, -6, 6)
    m_med   = np.clip(m_med,   -6, 6)
    m_large = np.clip(m_large, -6, 6)
    return (m_small,m_med,m_large,mu_pool,sd_pool)
def find_z_dist(fea_test,z_small,z_med,z_large,all_mean,all_sd):
    def D2_mean_to_proto(xz, m):
            d = xz - m
            return float(np.mean(d*d))
    xz = (fea_test - all_mean) / all_sd
    xz = np.clip(xz, -6, 6)
    d0 = D2_mean_to_proto(xz, z_small)  # class 0 (small)
    d1 = D2_mean_to_proto(xz, z_med)    # class 1 (medium)
    d2 = D2_mean_to_proto(xz, z_large)  # class 2 (large)
    c = int(np.argmin([d0, d1, d2]))
    return c,d0,d1,d2
def test_coin(test_paths,band_edges_hz,answers,kernal_size,n_fft,hop,snr_db):
    eps = 1e-8
    all_value=0
    correct_rate = 0
    vary_rate = 0
    for inx,path in enumerate(test_paths):
        fea_test = get_coin_features(path,band_edges_hz,kernal_size=kernal_size,n_fft=n_fft,hop=hop,snr_db=snr_db) 
        
        m_small ,m_med  ,m_large,mu_pool,sd_pool= get_coin_train_zscore(band_edges_hz=band_edges_hz,kernal_size=kernal_size,n_fft=n_fft,hop=hop)
        
        preds = []
        all_value+=fea_test.shape[0]
        for i in range(fea_test.shape[0]):
            # z-normalize test 
            c,d0,d1,d2 = find_z_dist(fea_test[i],m_small,m_med,m_large,mu_pool,sd_pool)
            vary_rate+=np.sort(np.array([d0,d1,d2])/[d0,d1,d2][c])[1]
            if c ==answers[inx][i]:
                correct_rate+=1
            preds.append(c)
    correct_rate/=(all_value+eps)
    correct_rate*=100
    vary_rate/=(all_value+eps)
    return (correct_rate,vary_rate)
def get_record_path(path,duration):

    fs = 44100 
    duration = duration
    filename = path

    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float64')
    sd.wait() 
    print("Recording finished.")

    write(filename, fs, recording)  # Save file
    print(f"Audio saved to {filename}")
    return path