import tensorflow as tf
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import random
from configuration import get_config

config = get_config()


def generate_valid_batches(speaker_num=config.N, utter_num=config.M):
    """ EXPERIMENTAL IMPROVEMENT ON BATCH CREATION (def random_batch()) FOR VALIDATION
        Generate 1 batch, where frame length of 160 is applied to each batch.
        speaker_num : number of speaker of each batch
        utter_num : number of utterance per speaker of each batch
    :return: 1 random numpy batch (frames x batch(NM) x n_mels)
    """
    path = config.test_path

    # Get folder and total number of speakers
    np_file_list = os.listdir(path)
    total_speaker = len(np_file_list)

    # Select random N number of speakers
    selected_files = random.sample(np_file_list, speaker_num)

    enroll_utter_batch = []
    valid_utter_batch = []
    for file in selected_files:
        utters = np.load(os.path.join(path, file))  # load utterance spectrogram of selected speaker

        enroll_utter_index = np.random.randint(0, utters.shape[0], utter_num)  # select random M utterances per speaker
        enroll_utter_batch.append(utters[enroll_utter_index])  # each speakers utterance [M, n_mels, frames] is appended

        # Ensure valid_utter_batch has no utterances like enroll_utter_batch
        while True:
            valid_utter_index = np.random.randint(0, utters.shape[0],
                                                  utter_num)  # select random M utterances per speaker
            unique_list = True
            for each_valid_utter in valid_utter_index:
                if each_valid_utter in enroll_utter_index:
                    unique_list = False
                    break  # repeat to make a new batch of random M utterances

            if unique_list:
                break

        valid_utter_batch.append(utters[valid_utter_index])  # each speakers utterance [M, n_mels, frames] is appended

    enroll_utter_batch = np.concatenate(enroll_utter_batch, axis=0)  # utterance batch [batch(NM), n_mels, frames]
    enroll_utter_batch = enroll_utter_batch[:, :, :160]  # for test session, fixed length slicing of input batch
    enroll_utter_batch = np.transpose(enroll_utter_batch, axes=(2, 0, 1))  # transpose [frames, batch, n_mels]

    valid_utter_batch = np.concatenate(valid_utter_batch, axis=0)  # utterance batch [batch(NM), n_mels, frames]
    valid_utter_batch = valid_utter_batch[:, :, :160]  # for test session, fixed length slicing of input batch
    valid_utter_batch = np.transpose(valid_utter_batch, axes=(2, 0, 1))  # transpose [frames, batch, n_mels]

    return enroll_utter_batch, valid_utter_batch


def random_batch(speaker_num=config.N, utter_num=config.M, shuffle=True, utter_start=0, forceValidation=False):
    """ Generate 1 batch.
        For TI-SV, random frame length is applied to each batch of utterances (140-180 frames)
        speaker_num : number of speaker of each batch
        utter_num : number of utterance per speaker of each batch
        shuffle : random sampling or not
        noise_filenum : specify noise file or not (TD-SV)
        utter_start : start point of slicing (TI-SV)
    :return: 1 random numpy batch (frames x batch(NM) x n_mels)
    """

    if forceValidation == True:
        config.train = False

    # data path
    if config.train:
        path = config.train_path
    else:
        path = config.test_path

    # TI-SV
    np_file_list = os.listdir(path)
    total_speaker = len(np_file_list)

    if shuffle:
        selected_files = random.sample(np_file_list, speaker_num)  # select random N speakers
    else:  # possible also for validation
        selected_files = np_file_list[:speaker_num]                # select first N speakers

    utter_batch = []
    for file in selected_files:
        utters = np.load(os.path.join(path, file))        # load utterance spectrogram of selected speaker
        if shuffle:
            utter_index = np.random.randint(0, utters.shape[0], utter_num)   # select M utterances per speaker
            utter_batch.append(utters[utter_index])       # each speakers utterance [M, n_mels, frames] is appended
        else:
            utter_batch.append(utters[utter_start: utter_start+utter_num])

    utter_batch = np.concatenate(utter_batch, axis=0)     # utterance batch [batch(NM), n_mels, frames]

    if config.train:
        frame_slice = np.random.randint(140,181)          # for train session, random slicing of input batch
        utter_batch = utter_batch[:,:,:frame_slice]       # from [batch(NM), n_mels, frames], leave the first two intact, but from frames, start from 0 and end at frame_slice
    else:
        utter_batch = utter_batch[:,:,:160]               # for test session, fixed length slicing of input batch

    utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, batch, n_mels]

    return utter_batch


def batch_entire_valid_set(start_speaker, end_speaker):
    """ Generates an enrollment and validation batch. To be used within a loop that iterates the entire valid dataset.
        start_speaker : generate batch by starting from this speaker
        end_speaker : end batch generation at this speaker (EXCLUSIVE)
    :return: Ordered enrollment and validation batches (frames x batch(NM) x n_mels)
    """
    path = config.test_path
    np_file_list = os.listdir(path)
    selected_files = np_file_list[start_speaker:end_speaker]  # select N speakers itteratively to iterate thru valid_set

    enroll_utter_batch = []
    valid_utter_batch = []
    for file in selected_files:
        utters = np.load(os.path.join(path, file))        # load utterance spectrogram of selected speaker

        enroll_utter_batch.append(utters[0: config.M])    # enroll this user with first M amount of utterances
        valid_utter_batch.append(utters[config.M: (config.M + config.M)])  # verify this user with NEXT M amount

    enroll_utter_batch = np.concatenate(enroll_utter_batch, axis=0)     # utterance batch [batch(NM), n_mels, frames]
    enroll_utter_batch = enroll_utter_batch[:,:,:160]               # for test session, fixed length slicing of input batch
    enroll_utter_batch = np.transpose(enroll_utter_batch, axes=(2,0,1))     # transpose [frames, batch, n_mels]

    valid_utter_batch = np.concatenate(valid_utter_batch, axis=0)  # utterance batch [batch(NM), n_mels, frames]
    valid_utter_batch = valid_utter_batch[:, :, :160]  # for test session, fixed length slicing of input batch
    valid_utter_batch = np.transpose(valid_utter_batch, axes=(2, 0, 1))  # transpose [frames, batch, n_mels]

    return enroll_utter_batch, valid_utter_batch

def normalize(x):
    """ normalize the last dimension vector of the input matrix
    :return: normalized input
    """
    return x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keep_dims=True)+1e-6)


def cossim(x,y, normalized=True):
    """ calculate similarity between tensors
    :return: cos similarity tf op node
    """
    if normalized:
        return tf.reduce_sum(x*y)
    else:
        x_norm = tf.sqrt(tf.reduce_sum(x**2)+1e-6)
        y_norm = tf.sqrt(tf.reduce_sum(y**2)+1e-6)
        return tf.reduce_sum(x*y)/x_norm/y_norm


def similarity(embedded, w, b, N=config.N, M=config.M, P=config.proj, center=None):
    """ Calculate similarity matrix from embedded utterance batch (NM x embed_dim) eq. (9)
        Input center to test enrollment. (embedded for verification)
    :return: tf similarity matrix (NM x N)
    """
    embedded_split = tf.reshape(embedded, shape=[N, M, P])

    if center is None:
        center = normalize(tf.reduce_mean(embedded_split, axis=1))              # [N,P] normalized center vectors eq.(1)
        center_except = normalize(tf.reshape(tf.reduce_sum(embedded_split, axis=1, keep_dims=True)
                                             - embedded_split, shape=[N*M,P]))  # [NM,P] center vectors eq.(8)
        # make similarity matrix eq.(9)
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center_except[i*M:(i+1)*M,:]*embedded_split[j,:,:], axis=1, keep_dims=True) if i==j
                        else tf.reduce_sum(center[i:(i+1),:]*embedded_split[j,:,:], axis=1, keep_dims=True) for i in range(N)],
                       axis=1) for j in range(N)], axis=0)
    else :
        # If center(enrollment) exist, use it.
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keep_dims=True) for i
                        in range(N)],
                       axis=1) for j in range(N)], axis=0)

    S = tf.abs(w)*S+b   # rescaling

    return S


def loss_cal(S, type="softmax", N=config.N, M=config.M):
    """ calculate loss with similarity matrix(S) eq.(6) (7) 
    :type: "softmax" or "contrast"
    :return: loss
    """
    S_correct = tf.concat([S[i*M:(i+1)*M, i:(i+1)] for i in range(N)], axis=0)  # colored entries in Fig.1

    if type == "softmax":
        total = -tf.reduce_sum(S_correct-tf.log(tf.reduce_sum(tf.exp(S), axis=1, keep_dims=True) + 1e-6))
    elif type == "contrast":
        S_sig = tf.sigmoid(S)
        S_sig = tf.concat([tf.concat([0*S_sig[i*M:(i+1)*M, j:(j+1)] if i==j
                              else S_sig[i*M:(i+1)*M, j:(j+1)] for j in range(N)], axis=1)
                             for i in range(N)], axis=0)
        total = tf.reduce_sum(1-tf.sigmoid(S_correct)+tf.reduce_max(S_sig, axis=1, keep_dims=True))
    else:
        raise AssertionError("loss type should be softmax or contrast !")

    return total


def optim(lr):
    """ return optimizer determined by configuration
    :return: tf optimizer
    """
    if config.optim == "sgd":
        return tf.train.GradientDescentOptimizer(lr)
    elif config.optim == "rmsprop":
        return tf.train.RMSPropOptimizer(lr)
    elif config.optim == "adam":
        return tf.train.AdamOptimizer(lr, beta1=config.beta1, beta2=config.beta2)
    else:
        raise AssertionError("Wrong optimizer type!")


# for check
if __name__ == "__main__":
    random_batch()
    w= tf.constant([1], dtype= tf.float32)
    b= tf.constant([0], dtype= tf.float32)
    embedded = tf.constant([[0,1,0], [0,0,1], [0,1,0], [0,1,0], [1,0,0], [1,0,0]], dtype= tf.float32)
    sim_matrix = similarity(embedded,w,b,3,2,3)
    loss1 = loss_cal(sim_matrix, type="softmax",N=3,M=2)
    loss2 = loss_cal(sim_matrix, type="contrast",N=3,M=2)
    with tf.Session() as sess:
        print(sess.run(sim_matrix))
        print(sess.run(loss1))
        print(sess.run(loss2))
