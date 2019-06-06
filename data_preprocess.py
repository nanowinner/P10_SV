import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from configuration import get_config

config = get_config()   # get arguments from parser

# downloaded dataset path
audio_path = r'C:\VoxCeleb\vox1_dev_wav\wav' # VoxCeleb1 dataset


def save_spectrogram_tisv():
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is split by voice activity detection (VAD) using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VCTK)
    """
    print("start text independent utterance feature extraction")
    os.makedirs("C:\\Users\\erh\\PyCharmProjects\\Speaker_Verification\\train_tisv_NEW", exist_ok=True)   # make folder to save train file
    os.makedirs("C:\\Users\\erh\\PyCharmProjects\\Speaker_Verification\\test_tisv_NEW", exist_ok=True)    # make folder to save test file

    utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr    # lower bound of utterance length
    total_speaker_num = len(os.listdir(audio_path))
    train_speaker_num = (total_speaker_num//10)*9            # split total data 90% train and 10% test
    print("total speaker number : %d"%total_speaker_num)
    print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))

    for i, folder in enumerate(os.listdir(audio_path)):
        speaker_path = os.path.join(audio_path, folder)     # path of each speaker
        print("%dth speaker processing..."%i)
        utterances_spec = []
        k=0

        for video_ID in os.listdir(speaker_path):
            video_path = os.path.join(speaker_path, video_ID)               # path of each video belonging to same speakerID
            for utter_name in os.listdir(video_path):
                utter_path = os.path.join(video_path, utter_name)           # path of each utterance
                utter, sr = librosa.core.load(utter_path, config.sr)        # load utterance audio
                intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection
                for interval in intervals:
                    if (interval[1]-interval[0]) >= utter_min_len:           # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                              win_length=int(config.window * sr), hop_length=int(config.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances

                        if (interval[1] - interval[0]) > utter_min_len:
                            utterances_spec.append(S[:, :config.tisv_frame])  # first 180 frames of partial utterance
                            utterances_spec.append(S[:, -config.tisv_frame:])  # last 180 frames of partial utterance
                        else:
                            utterances_spec.append(S[:, :config.tisv_frame])  # first 180 frames of partial utterance

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        if i<train_speaker_num:      # save spectrogram as numpy file
            np.save(os.path.join("C:\\Users\\erh\\PyCharmProjects\\Speaker_Verification\\train_tisv_NEW", "speaker%d.npy"%i), utterances_spec)
        else:
            np.save(os.path.join("C:\\Users\\erh\\PyCharmProjects\\Speaker_Verification\\test_tisv_NEW", "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)


if __name__ == "__main__":
    # extract_noise()
    save_spectrogram_tisv()
