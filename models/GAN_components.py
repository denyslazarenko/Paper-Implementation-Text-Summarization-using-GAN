import os
from keras import backend as K
from keras.layers import Layer
from keras.layers import Dense, Embedding, Activation, Permute, merge
from keras.layers import Input, Flatten, Dropout, MaxPooling2D, Reshape
from keras.layers import Convolution2D
from keras.models import Model
from .pointer_model import PointerModel
import models.config as config
import datetime
from keras.models import model_from_json


# ToDo after generator outputs destribution there should be argmax layer which predicts the most probable word
# after these steps are done I should have a symmarized text(encoded like input to Generator) which should be an immput to Discriminator
class Generator(object):
    def __init__(self, num_embeddings, embedding_dim, n_labels, pad_length, encoder_units, decoder_units):
        self.pointer_model = PointerModel(num_embeddings=num_embeddings,   #4999
                                          embedding_dim=embedding_dim,     #128
                                          n_labels=n_labels,               #4999
                                          pad_length=pad_length,           #20
                                          encoder_units=encoder_units,     #256
                                          decoder_units=decoder_units,     #256
                                          trainable=True,
                                          return_probabilities=False)

    def model(self):
        return self.pointer_model


class Reconstructor(object):
    def __init__(self, num_embeddings, embedding_dim, n_labels, pad_length, encoder_units, decoder_units):
        self.pointer_model = PointerModel(num_embeddings=num_embeddings,  # 4999
                                          embedding_dim=embedding_dim,    # 128
                                          n_labels=n_labels,              # 4999
                                          pad_length=pad_length,          # 20
                                          encoder_units=encoder_units,    # 256
                                          decoder_units=decoder_units,    # 256
                                          trainable=True,
                                          return_probabilities=False)

    def model(self):
        return self.pointer_model


class Discriminator(object):
    def __init__(self):
        self.vocabulary_size = config.VOCABULARY_SIZE
        self.sequence_length = config.MAX_SEQUENCE_LENGTH
        self.embedding_dim = config.EMBEDDING_DIM
        self.filter_sizes = config.filter_sizes
        self.dropout_rate = config.dropout_rate
        self.num_filters = config.num_filters
        self.output_dim = config.output_dim
        self.model_dir = config.models_dir
        self.model_path = config.MODEL
        self.weights_dir = config.weights_dir
        self.weights_path = config.WEIGHTS
        self.loss = config.loss
        self.optimizer = config.optimizer
        self.nb_epoch = config.nb_epoch
        self.batch_size = config.batch_size

    def model(self):
        inputs = Input(shape=(self.sequence_length, self.vocabulary_size))
        reshape_1 = Reshape((self.sequence_length, self.vocabulary_size, 1))(inputs)
        conv_1_0 = Convolution2D(self.num_filters, self.filter_sizes[0], self.embedding_dim, border_mode='valid', init='normal',
                                 activation='relu', dim_ordering='tf')(reshape_1)
        maxpool_1_0 = MaxPooling2D(pool_size=(self.sequence_length - self.filter_sizes[0] + 1, 1), strides=(1, 1),
                                   border_mode='valid', dim_ordering='tf')(conv_1_0)
        conv_1_1 = Convolution2D(self.num_filters, self.filter_sizes[1], self.embedding_dim, border_mode='valid', init='normal',
                                 activation='relu', dim_ordering='tf')(reshape_1)
        maxpool_1_1 = MaxPooling2D(pool_size=(self.sequence_length - self.filter_sizes[1] + 1, 1), strides=(1, 1),
                                   border_mode='valid', dim_ordering='tf')(conv_1_1)
        conv_1_2 = Convolution2D(self.num_filters, self.filter_sizes[2], self.embedding_dim, border_mode='valid', init='normal',
                                 activation='relu', dim_ordering='tf')(reshape_1)
        maxpool_1_2 = MaxPooling2D(pool_size=(self.sequence_length - self.filter_sizes[2] + 1, 1), strides=(1, 1),
                                   border_mode='valid', dim_ordering='tf')(conv_1_2)

        merged_tensor_1 = merge([maxpool_1_0, maxpool_1_1, maxpool_1_2], mode='concat', concat_axis=1)
        flatten_1 = Flatten()(merged_tensor_1)
        dropout_1 = Dropout(self.dropout_rate)(flatten_1)

        #ToDo should output one or two numbers fake/real
        output_1 = Dense(output_dim=self.output_dim, activation='linear')(dropout_1)
        model_1 = Model(input=[inputs], output=output_1)
        return model_1

    # ToDo this step is used when the model should be first pretrained
    def train(self):
        last_epoch, model_checkpoint_path = self.find_last_checkpoint(self.model_dir)
        initial_epoch = 0
        if model_checkpoint_path is not None:
            print('Loading epoch {0:d} from {1:s}'.format(last_epoch, model_checkpoint_path))
            model = self.load_model(model_path=model_checkpoint_path, weights_path=self.weights_path)
            initial_epoch = last_epoch + 1
        else:
            print('Building new model')
            model = self.model()
            model.compile(loss=self.loss,
                          optimizer=self.optimizer,
                          )

        print(model.summary())

        if initial_epoch < self.nb_epoch:
            training_start_time = datetime.datetime.now()
            print('{0}: Starting training'.format(training_start_time))
            X_train, X_test, y_train, y_test = self.load_training_samples()
            model.fit(X_train, y_train,
                      batch_size=self.batch_size,
                      nb_epoch=self.nb_epoch,
                      verbose=1,
                      validation_data=(X_test, y_test))
        self.save_model(model_path=self.model_path,
                        weights_path=self.weights_path)
        print('{0}: Finished'.format(datetime.datetime.now()))

    def load_training_samples(self):
        return (None, None, None, None)

    def find_last_checkpoint(self, checkpoint_dir, ptrn="*_[0-9]*.hdf5"):
        """
        Restore the most recent checkpoint in checkpoint_dir, if available.
        If no checkpoint available, does nothing.
        """

        import glob
        full_glob = os.path.join(checkpoint_dir, ptrn)
        all_files = glob.glob(full_glob)
        model_checkpoint_path = None
        epoch = 0

        for cur_fi in all_files:
            bname = os.path.basename(cur_fi)
            cur_epoch = bname.split('_')[-1].split('.')[0]
            cur_epoch = int(cur_epoch)
            if cur_epoch > epoch:
                epoch = cur_epoch
                model_checkpoint_path = cur_fi

        return epoch, model_checkpoint_path

    def save_model(self, model_path, weights_path):
        model_json = self.model().to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        weights = self.model().get_weights()
        # ToDo implement
        # helpers.save_file(weights, weights_path)

    def load_model(self, model_path, weights_path):
        _model = None
        weights = None
        print("load model")
        if _model is None:
            try:
                json_file = open(model_path, 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                _model = model_from_json(loaded_model_json)
                # ToDo implement
                # weights = helpers.get_file(weights_path)
                _model.set_weights(weights)
                _model.compile(optimizer="adam",
                               loss='categorical_crossentropy',
                               metrics=['top_k_categorical_accuracy'])
                print("Loaded model from disk")
            except IOError:
                _model = None
        return _model

