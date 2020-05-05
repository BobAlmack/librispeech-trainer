import sys
from tqdm import tqdm
import numpy as np
from pathlib import Path
from python_speech_features.base import mfcc, logfbank
import soundfile as sf
import multiprocessing as mp

labels = ['G', 'Z', 'CH', 'spn', 'B', 'HH', 'F', 'IH', 'IY', 'T', 'sil', 'ER', 'AY', 'R', 'P', 'AH', 'K', 'L', 'JH', 'V', 'OW', 'AA', 'DH', 'ZH', 'NG', 'M', 'SH', 'AW', 'EY', 'OY', 'S', 'UH', 'TH', 'AE', 'N', 'UW', 'AO', 'W', 'D', 'EH', 'Y']
global_labels = labels

def load_phn_file(f):
    content = [x for x in open(f).read().split("name = \"phones\"")[1].split("\n") if "text = " in x]
    content = [x.replace("\t\t\t\ttext = \"", "")[:-1] for x in content]
    content = [x for x in content if x not in ["", "sp"]]
    content = [x[:-1] if x[-1] in ['0', '1', '2'] else x for x in content]
    return [(0, 0, x) for x in content]

def load_phn_files(files):
    return [load_phn_file(x) for x in tqdm(files)]

def get_files(label):
    arpafiles = [str(x) for x in list(Path(label).rglob('*.TextGrid'))]
    flacfiles = [x.replace("TextGrid", "flac") for x in arpafiles]
    return flacfiles, arpafiles

def load_flac_files(files):
    return [sf.read(x)[0] for x in tqdm(files)]

def get_onehot(label):
    global labels
    y = [0] * len(labels)
    y[labels.index(label)] = 1
    return y

def create_phn_labels(wp):
    wav, phns = wp
    wav_len, wav_windows, wav_noisy_windows = get_mfcc(wav)
    _labels = []
    for i in phns:
        _labels.append(get_onehot(i[2]))
    _labels = np.array(_labels)
    ll = len(_labels)
    pad = np.zeros((400 - ll, len(global_labels)))
    _labels = np.concatenate([_labels, pad], axis=0)
    return wav_len, wav_windows, wav_noisy_windows, ll, _labels

def get_mfcc(x):
    y = np.concatenate([mfcc(x, numcep=12, winlen=0.01, winstep=0.005), logfbank(x, nfilt=1, winlen=0.01, winstep=0.005)], axis=-1)
    derivatives = []
    previousf = np.zeros((13,))
    for i in range(len(y)):
        if (i + 1) == len(y):
            nextf = np.zeros((13,))
        else:
            nextf = y[i + 1]
        derivatives.append(((nextf - previousf) / 2).reshape((1, 13)))
        previousf = y[i]
    derivatives = np.concatenate(derivatives, axis=0)
    y = np.concatenate([y, derivatives], axis=1)
    ynoise = np.random.normal(0, 0.6, y.shape)
    orig_len = len(y)
    pad = [np.zeros((1, 26))] * (3150 - y.shape[0])
    ypad = np.concatenate([y] + pad, axis=0)
    noisepad = np.concatenate([ynoise] + pad, axis=0)
    return orig_len, ypad, ypad + noisepad

def data_pipeline(label):
    wav_files, phn_files = get_files(label)
    wav_content = load_flac_files(wav_files)
    phn_content = load_phn_files(phn_files)
    zipped = list(zip(wav_content, phn_content))
    zipped = [x for x in zipped if len(x[0]) < 250000]
    pool = mp.Pool()
    pair_data = list(tqdm(pool.imap(create_phn_labels, zipped), total=len(zipped)))
    xlen = np.array([z[0] for z in pair_data])
    xclean = np.array([z[1] for z in pair_data])
    if label == "train-clean-100":
        all_inputs = np.concatenate([t[0][:int(t[1])].tolist() for t in zip(xclean, xlen)], axis=0)
        standardize_mean = all_inputs.mean(axis=0)
        standardize_std = all_inputs.std(axis=0)
        np.save("standardization.npy", [standardize_mean, standardize_std])
    x = np.array([z[2] for z in pair_data])
    if label == "test-clean":
        x = xclean
    ylen = np.array([z[3] for z in pair_data])
    y = np.array([z[4] for z in pair_data])
    print(xlen.shape)
    print(x.shape)
    print(ylen.shape)
    print(y.shape)
    return [xlen, x, ylen, y]

#np.save("labels.npy", global_labels)
#xlen, x, ylen, y = data_pipeline("train-clean-100")
#np.save("trainxlen.npy", xlen)
#np.save("trainx.npy", x)
#np.save("trainylen.npy", ylen)
#np.save("trainy.npy", y)
xlen, x, ylen, y = data_pipeline("test-clean")
np.save("testxlen.npy", xlen)
np.save("testx.npy", x)
np.save("testylen.npy", ylen)
np.save("testy.npy", y)
