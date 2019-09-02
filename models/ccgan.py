import tensorflow as tf
from helpers.conversion import AZ_MOVE_COUNT
from helpers.io_helpers import ensure_dir_exists
import numpy as np


def create_ccgan():
    NOISE_DIM = (1,)
    IN_NODES = (8, 8, 12)
    GENERATED_NODES = AZ_MOVE_COUNT
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    generator_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    # -- GENERATOR
    # -----------------------------------------------------------------------------------------------------
    generator_board_input = tf.keras.layers.Input(shape=IN_NODES, name="generator_board_input")
    generator_noise_input = tf.keras.layers.Input(shape=NOISE_DIM, name="generator_noise")
    generator_convolution = tf.keras.layers.Conv2D(2, 3, activation='relu')(generator_board_input)
    generator_flatten = tf.keras.layers.Flatten()(generator_convolution)
    generator_concat = tf.keras.layers.concatenate([generator_flatten, generator_noise_input])
    generator_dense = tf.keras.layers.Dense(256, activation='relu')(generator_concat)
    generator_output = tf.keras.layers.Dense(GENERATED_NODES, activation='softmax', name="generator_output")(generator_dense)

    generator = tf.keras.Model([generator_board_input, generator_noise_input], generator_output, name="generator")

    # -- DISCRIMINATOR
    # -------------------------------------------------------------------------------------------------
    discriminator_board_input = tf.keras.layers.Input(shape=(8, 8, 12), name="discriminator_board_input")
    discriminator_move_input = tf.keras.layers.Input(shape=(GENERATED_NODES,), name="discriminator_move_input")

    discriminator_board_convolve = tf.keras.layers.Conv2D(2, 3, activation=tf.nn.leaky_relu)(discriminator_board_input)
    discriminator_board_flatten = tf.keras.layers.Flatten()(discriminator_board_convolve)
    discriminator_concat = tf.keras.layers.concatenate([discriminator_move_input, discriminator_board_flatten])

    discriminator_dense = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)(discriminator_concat)

    discriminator_output = tf.keras.layers.Dense(1, activation='sigmoid', name="discriminator_output")(discriminator_dense)

    discriminator = tf.keras.Model([discriminator_board_input, discriminator_move_input], discriminator_output,
                                   name="discriminator")
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False

    # -- GAN
    # -----------------------------------------------------------------------------------------------------------
    gan_input_boards = tf.keras.layers.Input(shape=(8, 8, 12), name="gan_board_input")
    gan_input_noise = tf.keras.layers.Input(shape=NOISE_DIM, name="gan_noise_input")

    generated = generator([gan_input_boards, gan_input_noise])  # rename
    valid = discriminator([gan_input_boards, generated])  # rename

    gan = tf.keras.Model([gan_input_boards, gan_input_noise], valid)
    gan.compile(optimizer=generator_optimizer, loss='binary_crossentropy')

    return gan, generator, discriminator


def train_ccgan(dataset, generator, discriminator, gan, epochs, steps_per_epoch, nth_epoch=0, save_models_fn=None):
    # initialise
    next_pair_op = dataset.make_one_shot_iterator().get_next()
    history = []

    with tf.Session() as sess:
        for epoch in range(epochs):
            print("Epoch %d" % (epoch + 1))
            for step in range(steps_per_epoch):

                boards, moves = sess.run(next_pair_op)  # Fetch data, otherwise each training call increments the iterator

                generator_boards, real_boards = boards[::2], boards[1::2]
                _, real_moves = moves[::2], moves[1::2]

                # Need both to support odd-length batches
                half_shape_1st = tf.shape(moves[::2, 0])
                half_shape_2nd = tf.shape(real_moves[..., 0])
                full_shape = tf.shape(moves[..., 0])

                # GENERATE, uses half batch-size noise
                noise = tf.random.normal(half_shape_1st)

                fake = generator.predict([generator_boards, noise], steps=1)

                # TRAIN DISCRIMINATOR
                zeros = tf.zeros(half_shape_1st)
                ones = tf.ones(tf.shape(half_shape_2nd))

                discriminator_loss_fake = discriminator.train_on_batch([generator_boards, fake], zeros)  # half batch-size false

                discriminator_loss_real = discriminator.train_on_batch([real_boards, real_moves], ones)  # half batch-size true

                # average loss from generated and real moves
                discriminator_loss = 0.5 * np.add(discriminator_loss_fake, discriminator_loss_real)

                # TRAIN GENERATOR
                generator_loss = gan.train_on_batch([boards, tf.random.normal(full_shape)], tf.ones(full_shape))  # full batch-size noise, batch-size true

                # Include generator output in history every nth epoch
                history.append((discriminator_loss[0], discriminator_loss[1], generator_loss, fake if nth_epoch != 0 and (epoch % nth_epoch == 0 or epoch == epochs - 1) else None))


            current_epoch_history = history[epoch * steps_per_epoch:]
            d_loss_avg = sum(map(lambda x: x[0], current_epoch_history)) / steps_per_epoch
            d_acc_avg = sum(map(lambda x: x[1], current_epoch_history)) / steps_per_epoch
            g_loss_avg = sum(map(lambda x: x[2], current_epoch_history)) / steps_per_epoch
            print("\t[mean D loss: %f, mean acc.: %.2f%%] [mean G loss:% f]" % (d_loss_avg, 100 * d_acc_avg, g_loss_avg))

            # Make a checkpoint every nth epoch
            if save_models_fn is not None and ((epoch + 1) % nth_epoch == 0 or epoch == epochs - 1 or epoch == 0):
                save_models_fn(generator, discriminator, gan, epoch + 1)

    print("End of final epoch")
    return history


def get_saver(out_dir):  # Create a closure over out_dir so that train_ccgan need not handle the
                         # output path
    out_dir = out_dir if out_dir[-1] is '/' else out_dir + '/'  # Sanitise output directory
    ensure_dir_exists(out_dir)
    out_path_like = out_dir + "__%s/"

    def save_models(generator, discriminator, gan, epoch):
        nonlocal out_path_like  # Capture closed variable
        epoch_path = out_path_like % epoch
        ensure_dir_exists(epoch_path)
        epoch_path += "%s.h5"
        tf.keras.models.save_model(generator, epoch_path % "GENERATOR")
        tf.keras.models.save_model(discriminator, epoch_path % "DISCRIMINATOR")
        tf.keras.models.save_model(gan, epoch_path % "GAN")

    return save_models


def load_models(in_dir):  # Loads models saved by a function return from get_saver
    file_like = in_dir + "%s.h5"
    generator = tf.keras.models.load_model(file_like % "GENERATOR")
    discriminator = tf.keras.models.load_model(file_like % "DISCRIMINATOR")
    gan = tf.keras.models.load_model(file_like % "GAN")
    return generator, discriminator, gan
