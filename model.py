import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from utils import random_batch, normalize, similarity, loss_cal, optim, generate_valid_batches, batch_entire_valid_set
from configuration import get_config

config = get_config()
# Uncomment to run on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def train(path):
    tf.reset_default_graph()    # reset graph

    # Draw train graph
    train_batch = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32)  # input batch (time x batch x n_mel)
    lr = tf.placeholder(dtype=tf.float32)  # learning rate

    global_step = tf.Variable(0, name='global_step', trainable=False)
    w = tf.get_variable("w", initializer=np.array([10], dtype=np.float32))
    b = tf.get_variable("b", initializer=np.array([-5], dtype=np.float32))

    # Embedding LSTM (3-layer default)
    with tf.variable_scope("lstm", reuse=None):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        print(config.num_layer)
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # define lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=train_batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize
    print("embedded size: ", embedded.shape)

    # Define loss
    sim_matrix = similarity(embedded, w, b)
    print("similarity matrix size: ", sim_matrix.shape)
    loss = loss_cal(sim_matrix, type=config.loss)

    # Optimizer operation
    trainable_vars = tf.trainable_variables()                # get variable list
    optimizer = optim(lr)                                    # get optimizer (type is determined by configuration)
    grads, vars = zip(*optimizer.compute_gradients(loss))    # compute gradients of variables with respect to loss
    grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)       # l2 norm clipping by 3
    grads_rescale = [0.01*grad for grad in grads_clip[:2]] + grads_clip[2:]   # smaller gradient scale for w, b
    train_op = optimizer.apply_gradients(zip(grads_rescale, vars), global_step=global_step)  # gradient update operation

    # Check variables memory
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)

    # TensorBoard vars declaration
    lr_summ = tf.summary.scalar(name='My_LR', tensor=lr)
    loss_summary = tf.summary.scalar("loss_ORIG", loss)
    w_summary = tf.summary.histogram('My_Weights', w)
    b_summary = tf.summary.histogram('My_Bias', b)
    merged = tf.summary.merge_all()                 # merge all TB vars into one
    saver = tf.train.Saver(max_to_keep=40)          # create a saver, max_to_keep=40 w/ every 2500 steps = around 100000

    # Training session
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        os.makedirs(os.path.join(path, "Check_Point"), exist_ok=True)   # make folder to save model
        os.makedirs(os.path.join(path, "logs"), exist_ok=True)          # make folder to save TensorBoard logs
        os.makedirs("./Plots/", exist_ok=True)                          # make folder to save all plots and .txt logs
        os.makedirs("./Plots/" + path[11:], exist_ok=True)              # makes the subdirs for individual plots
        log_path = "./Plots/" + path[11:] + "/" + path[11:] + ".txt"    # declares .txt log files naming convention
        ckpt_perf_log_path = "./Plots/" + path[11:] + "/" + path[11:] + "_ckpts.txt"

        # Block of code to make folders of runs for TensorBoard visualization
        logspath = os.path.join(path, "logs")
        num_previous_runs = os.listdir('./tisv_model/logs')
        if len(num_previous_runs) == 0:
            run_number = 1
        else:
            run_number = max([int(s.split('run_')[1]) for s in num_previous_runs]) + 1
        curr_logdir = 'run_%02d' % run_number
        writer = tf.summary.FileWriter(os.path.join(logspath, curr_logdir), sess.graph)  # Define writer for TensorBoard
        # END of Block

        # epoch = 0      # not used
        lr_factor = 1    # LR decay factor (1/2 per 10000 iteration)
        loss_acc = 0     # accumulated loss (for calculating average of loss)

        # declares lists for figure creation
        EER_list = []         # collects the EER results every 100 steps for plotting
        train_loss_list = []  # collects the training loss results every 100 steps for plotting
        # LR_decay_list = []  # not used

        ckpt_at_iter = 2500

        for iter in range(config.iteration):
            # run forward and backward propagation and update parameters
            _, loss_cur, summary = sess.run([train_op, loss, merged],
                                  feed_dict={train_batch: random_batch(), lr: config.lr*lr_factor})

            loss_acc += loss_cur    # accumulated loss for each 100 iteration

            # write train_loss to TensorBoard
            if iter % 10 == 0:
                writer.add_summary(summary, iter)
            # perform validation
            if (iter+1) % 100 == 0:
                # print("(iter : %d) loss: %.4f" % ((iter+1),loss_acc/100))
                # print("==============VALIDATION START!============")

                # Draw validation graph, where enrollment AND validation batch batch (time x batch x n_mel)
                enroll = tf.placeholder(shape=[None, config.N * config.M, 40], dtype=tf.float32)
                valid = tf.placeholder(shape=[None, config.N * config.M, 40], dtype=tf.float32)
                val_batch = tf.concat([enroll, valid], axis=1)

                # Embedding LSTM (3-layer default)
                with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
                    lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in
                                  range(config.num_layer)]
                    lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)  # make lstm op and variables
                    outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=val_batch, dtype=tf.float32, time_major=True)
                    embedded = outputs[-1]          # the last output is the embedded d-vector
                    embedded = normalize(embedded)  # normalize
                # print("embedded size: ", embedded.shape)

                # enrollment embedded vectors (speaker model)
                enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N * config.M, :],
                                                                   shape=[config.N, config.M, -1]), axis=1))
                # validation embedded vectors
                valid_embed = embedded[config.N * config.M:, :]

                similarity_matrix = similarity(embedded=valid_embed, w=1., b=0., center=enroll_embed)

                # print("test file path : ", config.test_path)

                all_EER = []
                all_thres = []
                all_FAR = []
                all_FRR = []

                # Determine amount of batches able to run with current N
                total_speakers = len(os.listdir(config.test_path))
                total_possible_batches = total_speakers // config.N

                # Track time of total EER process per validation START
                time1 = time.time()

                # Calc EER for max amount of possible batches
                for i in range(total_possible_batches):

                    # Generate enrollment and validation batches
                    # and return similarity matrix after performing evaluation
                    enroll_batch, valid_batch = batch_entire_valid_set(start_speaker=i * config.N,
                                                                       end_speaker=(i * config.N) + config.N)
                    S = sess.run(similarity_matrix, feed_dict={enroll: enroll_batch, valid: valid_batch})
                    S = S.reshape([config.N, config.M, -1])

                    np.set_printoptions(precision=4)
                    # print("inference time for %d utterances : %0.2fs" % (2*config.M*config.N, time2-time1))
                    # print(S)    # print similarity matrix

                    # Declare vars to calculate EER
                    diff = 1
                    EER = 0
                    EER_thres = 0
                    EER_FAR = 0
                    EER_FRR = 0

                    # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
                    for thres in [0.01 * i + 0.5 for i in range(50)]:
                        S_thres = S > thres

                        # False acceptance ratio = false acceptance / mismatched population (enroll speaker != test speaker)
                        FAR = sum([np.sum(S_thres[i]) - np.sum(S_thres[i, :, i]) for i in range(config.N)]) / (
                                    config.N - 1) / config.M / config.N

                        # False reject ratio = false reject / matched population (enroll speaker = test speaker)
                        FRR = sum([config.M - np.sum(S_thres[i][:, i]) for i in range(config.N)]) / config.M / config.N

                        # Save threshold when FAR = FRR (=EER)
                        if diff > abs(FAR - FRR):
                            diff = abs(FAR - FRR)
                            EER = (FAR + FRR) / 2
                            EER_thres = thres
                            EER_FAR = FAR
                            EER_FRR = FRR

                    all_EER.append(EER)
                    all_thres.append(EER_thres)
                    all_FAR.append(EER_FAR)
                    all_FRR.append(EER_FRR)

                    # Print out individual validation batch EERs. Uncomment to work.
                    # print("Sub-EER num. %i : %0.4f (thres:%0.4f, FAR:%0.4f, FRR:%0.4f)" %
                    #       ((i + 1), EER, EER_thres, EER_FAR, EER_FRR))

                # Track time of total EER process per validation STOP
                time2 = time.time()

                # Average EER, Threshold, FAR and FRR for printing
                average_scores = [sum(all_EER) / len(all_EER),
                                  sum(all_thres) / len(all_thres),
                                  sum(all_FAR) / len(all_FAR),
                                  sum(all_FRR) / len(all_FRR)]

                print("(iter : %d) loss: %.4f || Final EER: %0.4f (thres:%0.4f, FAR:%0.4f, FRR:%0.4f) ||"
                      " inference time for %d utterances: %0.2fs" %
                      ((iter + 1),         # Current hundredth iteration
                       loss_acc / 100,     # Current loss
                       average_scores[0],  # EER
                       average_scores[1],  # Threshold
                       average_scores[2],  # FAR
                       average_scores[3],  # FRR
                       2 * config.M * config.N,  # Number of utterance
                       time2 - time1))  # Time it took to make in

                EER_list.append(average_scores[0])  # Append Final (averaged) EER value to list for plotting
                # print("==============VALIDATION END!==============")
                train_loss_list.append(loss_acc/100)

                # save figures
                if (iter+1) % 500 == 0:
                    plt.ioff()
                    fig_EER = plt.figure()
                    iter_list = [(i + 1) * 100 for i in range(len(EER_list))]
                    plt.plot(iter_list, EER_list, label="EER")
                    plt.xlabel("Steps")
                    plt.ylabel("EER")
                    plt.title("Equal error rate progress")
                    plt.grid(True)
                    plot_path = "./Plots/" + path[11:] + "/" + path[11:] + ".png"
                    print("Saving plot as: %s" % plot_path)
                    plt.savefig(plot_path)
                    plt.close(fig_EER)

                    plt.ioff()
                    fig_LOSS = plt.figure()
                    iter_list = [(i + 1) * 100 for i in range(len(EER_list))]
                    plt.plot(iter_list, train_loss_list, color="orange", label="train_loss")
                    plt.xlabel("Steps")
                    plt.ylabel("Training loss")
                    plt.title("Training progress")
                    plt.grid(True)
                    plot_path = "./Plots/" + path[11:] + "/" + path[11:] + "_LOSS.png"
                    print("Saving plot as: %s" % plot_path)
                    plt.savefig(plot_path)
                    plt.close(fig_LOSS)

                # Every 100 iterations, save a log of training progress
                with open(log_path, "a") as file:
                    file.write(str(iter+1) + "," + str(loss_acc/100) + "," +
                               str(average_scores[0]) + "," +
                               str(average_scores[1]) + "," +
                               str(average_scores[2]) + "," +
                               str(average_scores[3]) + "\n")

                loss_acc = 0                        # reset accumulated loss

            # decay learning rate
            if (iter+1) % ckpt_at_iter == 0:
                lr_factor /= 2                      # lr decay
                print("Learning Rate (LR) decayed! Current LR: ", config.lr*lr_factor)

            # save model checkpoint
            if (iter+1) % ckpt_at_iter == 0:
                saver.save(sess, os.path.join(path, "./Check_Point/model.ckpt"), global_step=iter//ckpt_at_iter)  # naming val
                with open(ckpt_perf_log_path, "a") as file:
                    file.write("Model %d, (iter : %d) || Ckpt EER: %0.4f (thres:%0.4f, FAR:%0.4f, FRR:%0.4f) ||"
                               " inference time for %d utterances: %0.2fs" %
                               (iter//ckpt_at_iter,
                                (iter + 1),         # Current hundredth iteration
                                # loss_acc / 100,   # Current loss, currently commented out because its reset above
                                average_scores[0],  # EER
                                average_scores[1],  # Threshold
                                average_scores[2],  # FAR
                                average_scores[3],  # FRR
                                2 * config.M * config.N,  # Number of utterance
                                time2 - time1) + "\n")  # Time it took to make it, plus end line for next ckpt

                print("Model checkpoint saved!")


# Test Session
def test(path):
    tf.reset_default_graph()

    # draw graph
    enroll = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    test = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32)  # test batch (time x batch x n_mel)
    batch = tf.concat([enroll, test], axis=1)

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    print("embedded size: ", embedded.shape)

    # check variables memory
    trainable_vars = tf.trainable_variables()
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)

    # enrollment embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))
    # test embedded vectors
    test_embed = embedded[config.N*config.M:, :]

    similarity_matrix = similarity(embedded=test_embed, w=1., b=0., center=enroll_embed)

    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # load model
        print("model path :", path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model.split('-')[-1]):  # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        print("test file path : ", config.test_path)

        # return similarity matrix after enrollment and test set
        time1 = time.time()  # for check inference time
        S = sess.run(similarity_matrix, feed_dict={enroll: random_batch(shuffle=False),
                                                   test: random_batch(shuffle=False, utter_start=config.M)})
        S = S.reshape([config.N, config.M, -1])
        time2 = time.time()

        np.set_printoptions(precision=4)
        print("inference time for %d utterances : %0.2fs" % (2*config.M*config.N, time2-time1))
        print(S)    # print similarity matrix

        # calculating EER
        diff = 1
        EER = 0
        EER_thres = 0
        EER_FAR = 0
        EER_FRR = 0

        # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
        for thres in [0.01*i+0.5 for i in range(50)]:
            S_thres = S>thres

            # False acceptance ratio = false acceptance / mismatched population (enroll speaker != test speaker)
            FAR = sum([np.sum(S_thres[i])-np.sum(S_thres[i,:,i]) for i in range(config.N)])/(config.N-1)/config.M/config.N

            # False reject ratio = false reject / matched population (enroll speaker = test speaker)
            FRR = sum([config.M-np.sum(S_thres[i][:,i]) for i in range(config.N)])/config.M/config.N

            # Save threshold when FAR = FRR (=EER)
            if diff > abs(FAR-FRR):
                diff = abs(FAR-FRR)
                EER = (FAR+FRR)/2
                EER_thres = thres
                EER_FAR = FAR
                EER_FRR = FRR

        print("\nEER : %0.4f (thres:%0.4f, FAR:%0.4f, FRR:%0.4f)" % (EER, EER_thres, EER_FAR, EER_FRR))


# Averaged test of 100 EERs
def averaged_test(path):
    tf.reset_default_graph()

    # draw graph
    enroll = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    test = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32)  # test batch (time x batch x n_mel)
    batch = tf.concat([enroll, test], axis=1)

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    print("embedded size: ", embedded.shape)

    # check variables memory
    trainable_vars = tf.trainable_variables()
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)

    # enrollment embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))
    # test embedded vectors
    test_embed = embedded[config.N*config.M:, :]

    similarity_matrix = similarity(embedded=test_embed, w=1., b=0., center=enroll_embed)

    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # load model
        print("model path :", path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model.split('-')[-1]):  # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        # print("test file path : ", config.test_path)

        # return similarity matrix after enrollment and test set
        time1 = time.time() # for check inference time

        all_EER = []
        all_thres = []
        all_FAR = []
        all_FRR = []

        for i in range(100):

            # S = sess.run(similarity_matrix, feed_dict={enroll: random_batch(shuffle=False),
            #                                            test: random_batch(shuffle=False, utter_start=config.M)})
            enroll_batch, valid_batch = generate_valid_batches(config.N, config.M)
            S = sess.run(similarity_matrix, feed_dict={enroll: enroll_batch, test: valid_batch})

            S = S.reshape([config.N, config.M, -1])
            time2 = time.time()

            np.set_printoptions(precision=4)
            # print("inference time for %d utterances : %0.2fs" % (2*config.M*config.N, time2-time1))
            # print(S)    # print similarity matrix

            # calculating EER
            diff = 1
            EER = 0
            EER_thres = 0
            EER_FAR = 0
            EER_FRR = 0

            # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
            for thres in [0.01*i+0.5 for i in range(50)]:
                S_thres = S>thres

                # False acceptance ratio = false acceptance / mismatched population (enroll speaker != test speaker)
                FAR = sum([np.sum(S_thres[i])-np.sum(S_thres[i,:,i]) for i in range(config.N)])/(config.N-1)/config.M/config.N

                # False reject ratio = false reject / matched population (enroll speaker = test speaker)
                FRR = sum([config.M-np.sum(S_thres[i][:,i]) for i in range(config.N)])/config.M/config.N

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thres = thres
                    EER_FAR = FAR
                    EER_FRR = FRR

            all_EER.append(EER)
            all_thres.append(EER_thres)
            all_FAR.append(EER_FAR)
            all_FRR.append(EER_FRR)
            print("\nEER num. %i : %0.4f (thres:%0.4f, FAR:%0.4f, FRR:%0.4f)" % ((i+1),EER,EER_thres,EER_FAR,EER_FRR))

        average_scores = [sum(all_EER) / len(all_EER),
                          sum(all_thres) / len(all_thres),
                          sum(all_FAR) / len(all_FAR),
                          sum(all_FRR) / len(all_FRR)]
        print("Final EER: %0.4f (thres:%0.4f, FAR:%0.4f, FRR:%0.4f)" % (average_scores[0], average_scores[1], average_scores[2], average_scores[3]) )


# Verify over the entire test(valid) set
def test_entire_valid_set(path):
    tf.reset_default_graph()

    # Draw graph
    enroll = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    valid = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32)  # test batch (time x batch x n_mel)
    val_batch = tf.concat([enroll, valid], axis=1)

    # Embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=val_batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    print("embedded size: ", embedded.shape)

    # check variables memory
    trainable_vars = tf.trainable_variables()
    variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
    print("total variables :", variable_count)

    # enrollment embedded vectors (speaker model)
    enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))
    # validation embedded vectors
    valid_embed = embedded[config.N*config.M:, :]

    similarity_matrix = similarity(embedded=valid_embed, w=1., b=0., center=enroll_embed)

    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # load model
        print("model path :", path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model.split('-')[-1]):  # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break

        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

        # print("test file path : ", config.test_path)

        all_EER = []
        all_thres = []
        all_FAR = []
        all_FRR = []

        # Determine amount of batches able to run with current N
        total_speakers = len(os.listdir(config.test_path))
        total_possible_batches = total_speakers // config.N

        # Track time of total EER process per validation START
        time1 = time.time()

        # Calc EER for max amount of possible batches
        for i in range(total_possible_batches):

            # Generate enrollment and validation batches
            # and return similarity matrix after performing evaluation
            enroll_batch, valid_batch = batch_entire_valid_set(start_speaker=i*config.N, end_speaker=(i*config.N)+config.N)
            S = sess.run(similarity_matrix, feed_dict={enroll: enroll_batch, valid: valid_batch})
            S = S.reshape([config.N, config.M, -1])

            np.set_printoptions(precision=4)
            # print("inference time for %d utterances : %0.2fs" % (2*config.M*config.N, time2-time1))
            # print(S)    # print similarity matrix

            # Declare vars to calculate EER
            diff = 1
            EER = 0
            EER_thres = 0
            EER_FAR = 0
            EER_FRR = 0

            # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
            for thres in [0.01*i+0.5 for i in range(50)]:
                S_thres = S>thres

                # False acceptance ratio = false acceptance / mismatched population (enroll speaker != test speaker)
                FAR = sum([np.sum(S_thres[i])-np.sum(S_thres[i,:,i]) for i in range(config.N)])/(config.N-1)/config.M/config.N

                # False reject ratio = false reject / matched population (enroll speaker = test speaker)
                FRR = sum([config.M-np.sum(S_thres[i][:,i]) for i in range(config.N)])/config.M/config.N

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thres = thres
                    EER_FAR = FAR
                    EER_FRR = FRR

            all_EER.append(EER)
            all_thres.append(EER_thres)
            all_FAR.append(EER_FAR)
            all_FRR.append(EER_FRR)
            print("\nEER num. %i : %0.4f (thres:%0.4f, FAR:%0.4f, FRR:%0.4f)" % ((i+1),EER,EER_thres,EER_FAR,EER_FRR))

        # Track time of total EER process per validation STOP
        time2 = time.time()

        # Average EER, Threshold, FAR and FRR for printing
        average_scores = [sum(all_EER) / len(all_EER),
                          sum(all_thres) / len(all_thres),
                          sum(all_FAR) / len(all_FAR),
                          sum(all_FRR) / len(all_FRR)]

        print("Final EER: %0.4f (thres:%0.4f, FAR:%0.4f, FRR:%0.4f) || inference time for %d utterances: %0.2fs" %
              (average_scores[0],        # EER
               average_scores[1],        # Threshold
               average_scores[2],        # FAR
               average_scores[3],        # FRR
               2 * config.M * config.N,  # Number of utterance
               time2 - time1))           # Time it took to make in
