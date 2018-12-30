import librosa
import numpy as np
import matplotlib.pyplot as plt


def make_pattern(N1, N2, f0, fs):
    # function[gn] = rosenberg(N1, N2, f0, fs)
    t = 1 / f0  # period in seconds
    pulse_length = int(t * fs)  # length of one period of pulse

    # select N1 and N2 for duty cycle
    N2 = int(pulse_length * N2)
    N1 = int(N1 * N2)
    output = np.zeros((N2))

    # calculate pulse samples for n=1:N1 - 1
    output[:N1] = [0.5 * (1 - np.cos(np.pi * (x - 1) / N1)) for x in range(N1)]
    output[N1:N2] = [np.cos(np.pi * (x - N1) / (N2 - N1) / 2) for x in range(N1, N2)]
    output = np.concatenate((output, np.zeros(pulse_length - N2)), axis=0)

    return output


def repeat_to_n_second(pattern: np.ndarray, t, fs):
    size = pattern.shape[0]
    repeat_times = fs * t // size + 1
    # print(repeat_times)
    return np.tile(pattern, repeat_times)[:t * fs]


if __name__ == '__main__':
    pitch = 120
    glottal_tense = 0.85
    fs = 44100

    p = make_pattern(glottal_tense, .65, pitch, fs)

    aud = repeat_to_n_second(p, 10, fs)
    # plt.plot(p)
    # plt.show()

    librosa.output.write_wav("test.wav", aud, fs)

    # t = np.arange(0, 1, 1 / fs)
    # d = np.arange(0, 1, 1 / 1 / pitch)
    # y = pulstran(t, d, p, fs) * 2.0 - 1.0
