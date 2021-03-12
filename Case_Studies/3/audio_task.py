import window
import sklearn.model_selection
import scipy.io.wavfile
import numpy as np

import scipy.fftpack, scipy.signal
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io, scipy.io.wavfile, scipy.stats
import sklearn.metrics

# possible feature transforms
# you don't know what these do, but you can try applying them
# and see how they affect the visualisations
feature_fns = {
    "dct": lambda x: np.abs(scipy.fftpack.dct(x)),
    "fft": lambda x: np.abs(scipy.fftpack.fft(x)),
    "fft_phase": lambda x: np.angle(scipy.fftpack.fft(x)),
    "dct_phase": lambda x: np.angle(scipy.fftpack.dct(x)),
    "cepstrum": lambda x: np.abs(
        scipy.fftpack.ifft(np.log(np.abs(scipy.fftpack.fft(x)) ** 2 + 1e-4))
    )
    ** 2,
    "raw": lambda x: x,
}

# possible windowing functions
window_fns = {
    "hamming": scipy.signal.hamming,
    "hann": scipy.signal.hann,
    "boxcar": scipy.signal.boxcar,
    "blackmanharris": scipy.signal.blackmanharris,
}


def load_wav(fname):
    sr, wave = scipy.io.wavfile.read(fname)
    return wave / 32768.0


def load_features_window(data, size, step, window_fn, feature_fn, label, feature_range, decimate):

    features = window.window_data(data, size=size, step=step)
    labels = np.full(len(features), label)
    print(f"Loading into {len(features)} windows of length {size}")

    fn = feature_fns[feature_fn]
    start_range = int(feature_range[0] * features.shape[1])
    end_range = int(feature_range[1] * features.shape[1])
    win = window_fns[window_fn](features.shape[1])
    # apply feature transform and window fn
    X = [fn(feature * win)[start_range:end_range:decimate] for feature in features]
    X = np.array(X)
    return X, labels


def load_data(kwargs):
    X = []
    y = []
    for i in range(5):
        fname = f"data/challenge_train_{i}.wav"
        wave_data = load_wav(fname)
        features, labels = load_features_window(
            data=wave_data,
            size=kwargs['size'],
            step=kwargs['step'],
            window_fn=kwargs['window_fn'],
            feature_fn=kwargs['feature_fn'],
            label=i,
            feature_range=kwargs["feature_range"],
            decimate=kwargs["decimate"]
        )
        X.append(features)
        y.append(labels)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print(f"Using {kwargs['feature_fn']} transform and a {kwargs['window_fn']} window.")    
    return X, y

import sklearn.neighbors


def knn_fit(X, y):
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7)
    knn.fit(X=X, y=y)
    return knn


def knn_classify(knn, wave_data, kwargs):
    
    features, _ = load_features_window(
        data=wave_data,
            size=kwargs['size'],
            step=kwargs['step'],
            window_fn=kwargs['window_fn'],
            feature_fn=kwargs['feature_fn'],            
            feature_range=kwargs["feature_range"],
            decimate=kwargs["decimate"],
            label=-1)
        
    
    print("Predicting...")
    labels = knn.predict(features)
    return labels

def load_test_wave_labels(basename):
    # load the data from wavfile
    wave = load_wav(basename + ".wav")
    labels = np.loadtxt(basename + ".labels")
    return wave, labels


def plot_test(knn, parameters, fname):
    print("="*80)
    print(f"Testing with {fname}")
    wave, labels_true = load_test_wave_labels(fname)
    labels_pred = knn_classify(knn, wave, parameters)
    plot_test_classification(wave, labels_true, labels_pred)

def run_secret_test(knn, parameters):
    import secret_test
    classify = lambda wave: knn_classify(knn, wave, parameters)
    secret_test.challenge_evaluate_performance(classify)

def plot_test_classification(wave_data, labels_true, labels_predicted):
    ## plot the classification of wave_data (should be a 1D 8Khz audio wave)
    ## and two sets of labels: true and predicted. They do not need
    ## to be the same length, but they should represent equally-sampled
    ## sections of the wave file
    sr = 4096
    ts = np.arange(len(wave_data)) / float(sr)

    try:
        len(labels_true)
    except:
        labels_true = [labels_true]

    # make sure there are at least 2 predictions, so interpolation does not freak out
    if len(labels_predicted) == 1:
        labels_predicted = [labels_predicted[0], labels_predicted[0]]
    if len(labels_true) == 1:
        labels_true = [labels_true[0], labels_true[0]]

    # predict every 10ms
    frames = ts[::80]

    true_inter = scipy.interpolate.interp1d(
        np.linspace(0, np.max(ts), len(labels_true)), labels_true, kind="nearest"
    )
    predicted_inter = scipy.interpolate.interp1d(
        np.linspace(0, np.max(ts), len(labels_predicted)),
        labels_predicted,
        kind="nearest",
    )

    true_interpolated = true_inter(frames)[:, None]
    predicted_interpolated = predicted_inter(frames)[:, None]
    # show colorblocks for the labels
    plt.figure(figsize=(16, 4))
    plt.imshow(
        true_interpolated.T,
        extent=[0, np.max(ts), 0, 1],
        interpolation="nearest",
        cmap="tab10",
        vmin=0,
        vmax=10,
    )
    plt.imshow(
        predicted_interpolated.T,
        extent=[0, np.max(ts), 0, -1],
        interpolation="nearest",
        cmap="tab10",
        vmin=0,
        vmax=10,
    )

    # plot the wave
    plt.plot(ts, wave_data, c="w", alpha=1)
    plt.text(0.5, 0.5, "True", color="w")
    plt.text(0.5, -0.5, "Predicted", color="w")
    plt.grid("off")
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")    

    print(f"Prediction accuracy {sklearn.metrics.accuracy_score(true_interpolated, predicted_interpolated):.3f}")
    print("Confusion matrix")
    print(sklearn.metrics.confusion_matrix(true_interpolated, predicted_interpolated))
    print()