import os
import time

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf

from models.pointer_model.model import PointerModel
from models import Generator, Discriminator, Reconstructor
from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab

from models.WGAN import WGAN

cp = ModelCheckpoint("./weights/model.{epoch:02d}-{val_loss:.2f}.hdf5",
                     monitor='val_loss',
                     verbose=0,
                     save_best_only=True,
                     save_weights_only=True,
                     mode='auto')

# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.train_batcher = Batcher(config.train_data_path,
                                     self.vocab,
                                     hps=config.hps,
                                     single_pass=False)
        self.val_batcher = Batcher(config.eval_data_path,
                                   self.vocab,
                                   hps=config.hps,
                                   single_pass=False)

    def setup_train_generator(self, model_file_path=None):
        generator = Generator(num_embeddings=config.vocab_size,  # 4999
                              embedding_dim=config.emb_dim,  # 128
                              n_labels=config.vocab_size,  # 4999
                              pad_length=config.padding,  # 20
                              encoder_units=config.hidden_dim,  # 256
                              decoder_units=config.hidden_dim,  # 256
                              )
        model = generator.model()
        model.summary()
        model.compile(optimizer='adagrad',
                      lr=config.lr,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print('Generator Compiled.')

        try:
            model.fit_generator(generator=self.train_batcher.next_batch(),
                                samples_per_epoch=5,
                                validation_data=self.val_batcher.next_batch(),
                                callbacks=[cp],
                                verbose=1,
                                nb_val_samples=1,
                                nb_epoch=config.max_iterations)

        except KeyboardInterrupt as e:
            print('Generator training stopped early.')
        print('Generator training complete.')

    def setup_train_discriminator(self):
        model = Discriminator().model()

        model.summary()
        model.compile(optimizer=Adam(lr=0.0001, beta_1=0.5, beta_2=0.9),
                      lr=config.lr,
                      loss='binary_crossentropy',
                      )
        print('Discriminator Compiled.')

        try:
            model.fit_generator(generator=self.train_batcher.next_batch_discriminator(),
                                samples_per_epoch=5,
                                validation_data=self.val_batcher.next_batch_discriminator(),
                                callbacks=[cp],
                                verbose=1,
                                nb_val_samples=1,
                                nb_epoch=config.max_iterations)

        except KeyboardInterrupt as e:
            print('Discriminator training stopped early.')
        print('Discriminator training complete.')

    def setup_train_wgan_model(self):
        generator = Generator(num_embeddings=config.vocab_size,  # 4999
                              embedding_dim=config.emb_dim,  # 128
                              n_labels=config.vocab_size,  # 4999
                              pad_length=config.padding,  # 20
                              encoder_units=config.hidden_dim,  # 256
                              decoder_units=config.hidden_dim,  # 256
                              ).model()
        reconstructor = Reconstructor(num_embeddings=config.vocab_size,  # 4999
                                      embedding_dim=config.emb_dim,  # 128
                                      n_labels=config.vocab_size,  # 4999
                                      pad_length=config.padding,  # 20
                                      encoder_units=config.hidden_dim,  # 256
                                      decoder_units=config.hidden_dim,  # 256
                                      ).model()
        discriminator = Discriminator().model()
        wgan = WGAN(generator=generator,
                    reconstructor=reconstructor,
                    discriminator=discriminator,
                    )
        try:
            wgan.train(self.train_batcher.next_batch())
        except KeyboardInterrupt as e:
            print('WGAN training stopped early.')
        print('WGAN training complete.')


if __name__ == '__main__':
    train_model = Train()
    train_model.setup_train_wgan_model()