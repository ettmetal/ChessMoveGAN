from dataset import paired_messagepack_file_dataset_generator
from models.ccgan_profiling import *
import msgpack
import numpy as np
import itertools

tf.logging.set_verbosity(tf.logging.ERROR) # ignore some irritating, but benign, tensorflow warnings

# File name prefix for logging / saving
run_name = ""

# Get networks
gan, generator, discriminator = create_ccgan()
print(gan.summary())
print(generator.summary())
print(discriminator.summary())

# Training parameters
epochs = 50
batch_size = 50
train_samples = 750
shuffle_buffer = 150

save_generated = True # Should the nth epoch's final batch's generated moves be saved?

# Data file paths
boards_path = "preprocessed_games/10__boards__standard_2018_human_morethan10.mpk"
moves_path = "preprocessed_games/10__moves__standard_2018_human_morethan10.mpk"
# Get training dataset generator
ccgan_dataset = tf.data.Dataset.from_generator(paired_messagepack_file_dataset_generator(boards_path, moves_path),
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=((8, 8, 12), (AZ_MOVE_COUNT)))
training_data = ccgan_dataset.take(train_samples).shuffle(shuffle_buffer).repeat(epochs).batch(batch_size)

# Train the GAN
print("Beginning training")

history = train_ccgan(training_data, generator, discriminator, gan, epochs, train_samples // batch_size, nth_epoch=5, run_name=run_name,
                        save_models_fn=get_saver("saved_models__%s/" % run_name) if save_generated else None)

print ("Training ended")

# Write captured generated samples
if save_generated:
    generated_path = "generated__%s.msgpack" % run_name
    with open(generated_path, 'ab+') as output:
        for epoch in range(epochs):
            d_loss, d_acc, g_loss, generated = history[epoch]
            print("\nEpoch %d:\n\tDiscriminator Loss: %f (Accuracy: %.2f%%)\n\tGenerator Loss:     %f" % (epoch + 1, d_loss, d_acc * 100, g_loss))
            if generated is not None:
                msgpack.pack((epoch, epochs, [np.argmax(move).item() for move in generated]), output)

# Log run metrics
with open("RunHistoryData__%s.csv" % run_name, 'a+') as history_csv:
    history_csv.write("Discriminator Loss,Discriminator Accuracy,Generator Loss\n")
    for d_loss, d_acc, g_loss, _ in itertools.chain(history):
        history_csv.write("%s,%s,%s\n" % (d_loss, d_acc, g_loss))
