import numpy as np
from scipy import signal
from BaselineRemoval import BaselineRemoval

class ArgAccessor:
    def __init__(self, kwargs: dict):
        self.kwargs = kwargs
    def get(self, name, defval):
        return self.kwargs[name] if name in self.kwargs else defval

def filter(x: np.ndarray, fs: float, fmin: float, fmax: float, order: int) -> np.ndarray:
    x = signal.detrend(x)
    sos = signal.butter(order, [fmin, fmax], 'band', fs=fs, output='sos')
    x = signal.sosfilt(sos, x)
    return x

def correct_baseline(x: np.ndarray, porder: int) -> np.ndarray:
    return BaselineRemoval(x).ZhangFit(porder=porder)

def split(x: np.ndarray, fs: float, min_dist: float, window_length: float) -> np.ndarray:
    dist = int(min_dist*fs)
    hwindow = int(0.5*window_length*fs)

    deriv1 = np.diff(x)
    deriv1sq = np.square(deriv1) * np.sign(deriv1)
    deriv2 = np.diff(deriv1sq)
    
    peaks1 = np.where(np.diff(np.sign(deriv2)) < 0)[0]+1
    ths = np.array([2 * np.std(deriv1sq[max(peak - hwindow, 0) : min(peak + hwindow, len(deriv1sq))]) for peak in peaks1])
    peaks1 = peaks1[deriv1sq[peaks1] >= ths]
    peaks0 = np.where(np.diff(np.sign(deriv1)) != 0)[0]+1
    idxs = np.searchsorted(peaks0, peaks1, side='right')-1
    idxs = idxs[idxs >= 0]
    bounds = peaks0[idxs]
    bounds = bounds[np.diff(bounds, prepend=0) >= dist]
    bounds = np.concatenate(([0], bounds, [len(x)-1]))

    return bounds

def find_outliers(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x)
    std = np.std(x)
    return (x <= mean + 2*std) & (x >= mean - 2*std)

def threshold_duration(bounds: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = np.diff(bounds)
    mask = np.logical_and(mask, find_outliers(diff))
    return mask

def threshold_amplitude(x: np.ndarray, bounds: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ampDiff = np.empty(len(bounds)-1)
    for i in range(1, len(bounds)):
        waveform = x[bounds[i-1]:bounds[i]]
        ampDiff[i-1] = np.max(waveform) - waveform[0]
    mask = np.logical_and(mask, find_outliers(ampDiff))
    return mask

def filter_wavelet_spectrum(x: np.ndarray, bounds: np.ndarray, mask: np.ndarray, args: ArgAccessor) -> np.ndarray:
    x = x - np.mean(x)
    x = x / np.max(np.abs(x))

    period = np.median(np.diff(bounds)[mask])
    ranges = np.array([\
        [args.get('wlm_hfc_min', 0.006), args.get('wlm_hfc_max', 0.028)],\
        [args.get('wlm_ac_min', 0.068), args.get('wlm_ac_max', 0.125)],\
        [args.get('wlm_mc_min', 0.199), args.get('wlm_mc_max', 0.369)],\
        [args.get('wlm_lfc_min', 0.710), args.get('wlm_lfc_max', 1.705)]\
            ])
    ranges = period * ranges

    scales_per_channel = args.get('wlm_spc', 10)
    widths = np.empty(len(ranges) * scales_per_channel)
    for i in range(len(ranges)):
        widths[i * scales_per_channel : (i+1) * scales_per_channel] = np.arange(ranges[i][0], ranges[i][1], (ranges[i][1] - ranges[i][0]) / float(scales_per_channel))

    coefficients = signal.cwt(x, signal.ricker, widths)

    window_size = int(args.get('wlm_sm_window', 0.2) * period)
    window = np.ones(window_size) / float(window_size)

    channels = np.empty((len(ranges), len(x)))
    for i in range(len(channels)):
        normalizers = np.array(np.sqrt(widths[i * scales_per_channel : (i+1) * scales_per_channel]))
        channel = coefficients[i * scales_per_channel : (i+1) * scales_per_channel] / normalizers[:, np.newaxis]
        channel = np.max(channel, axis=0)
        channel = np.convolve(channel, window, mode='same')
        channels[i] = channel

    std = np.mean(channels[2]) + np.std(channels[2])

    def check_conditions(wavelet: np.ndarray):
        
        def get_peaks(chn, th):
            deriv = np.diff(wavelet[chn])
            zero_crossings = np.where(np.diff(np.sign(deriv)) <= -1)[0]+1
            peaks = wavelet[chn][zero_crossings] > th
            return zero_crossings[peaks]
        
        peaks2 = get_peaks(2, 0)
        if len(peaks2) != 1 or wavelet[2][peaks2[0]] <= max(args.get('wlm_mcp', 0.5) * np.max(wavelet), std):
            return False
            
        peaks0 = get_peaks(0, args.get('wlm_hfcp_min', 0.2) * wavelet[2][peaks2[0]])
        if len(peaks0) > 1 or (len(peaks0) == 1 and not (np.abs(peaks2[0] - peaks0[0]) <= args.get('wlm_hfcp_max_dist', 0.15) * len(wavelet[0]) and wavelet[0][peaks0[0]] <= args.get('wlm_hfcp_max', 0.7) * wavelet[2][peaks2[0]])):
            return False
        
        peaks1 = get_peaks(1, args.get('wlm_acp', 0.3) * wavelet[2][peaks2[0]])
        if len(peaks1) < 1 or len(peaks1) > 2:
            return False
        
        if len(peaks1) == 2 and not (wavelet[1][peaks1[0]] > args.get('wlm_acps_min', 0.4) * wavelet[2][peaks2[0]] and wavelet[1][peaks1[0]] < args.get('wlm_acps_max', 0.8) * wavelet[2][peaks2[0]] and wavelet[1][peaks1[1]] < wavelet[1][peaks1[0]] and peaks1[1] < peaks2[0] + np.argmin(wavelet[2][peaks2[0]:])):
            return False
        
        if not args.get('blw_drop', True):
            return True
        
        if np.max(np.abs(wavelet[3])) > args.get('wlm_lwcp', 0.7) * wavelet[2][peaks2[0]]:
            return False
        
        return True

    for i in range(1, len(bounds)):
        if mask[i-1]:
            wavelet = channels[:, bounds[i-1]:bounds[i]]
            mask[i-1] = check_conditions(wavelet)

    return mask

def merge_defective(bounds: np.ndarray, mask: np.ndarray) -> tuple:
    to_remove = np.full(len(bounds), True)
    to_remove[1:-1] = mask[1:] | np.roll(mask, 1)[1:]
    mask = mask[to_remove[:-1]]
    bounds = bounds[to_remove]
    return (bounds, mask)

def preprocess(x: np.ndarray, fs: float, **kwargs) -> tuple:
    args = ArgAccessor(kwargs)
    x = filter(x, fs, args.get('fmin', 0.05), args.get('fmax', 15.0), args.get('forder', 6))
    x = correct_baseline(x, args.get('porder', 10))
    bounds = split(x, fs, args.get('min_dur', 0.2), args.get('split_window', 1.0))
    mask = np.full(len(bounds)-1, True)
    mask = threshold_duration(bounds, mask)
    mask = threshold_amplitude(x, bounds, mask)
    mask = filter_wavelet_spectrum(x, bounds, mask, args)
    bounds, mask = merge_defective(bounds, mask)
    return (x, bounds, mask)