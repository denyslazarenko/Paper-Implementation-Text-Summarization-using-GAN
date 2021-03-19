from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from data_util import config
from .custom_recurrents import AttentionDecoder


def PointerModel(num_embeddings=config.vocab_size -1, #5000
                 embedding_dim=config.emb_dim,        #128
                 n_labels=config.vocab_size -1,       #4999
                 pad_length=config.padding,           #20
                 encoder_units=config.hidden_dim,     #256
                 decoder_units=config.hidden_dim,     #256
                 trainable=True,
                 return_probabilities=False):

    input_ = Input(shape=(pad_length,), dtype='float32')
    input_embed = Embedding(num_embeddings, embedding_dim,
                            input_length=pad_length,
                            trainable=trainable,
                            # weights=[np.eye(num_embeddings)],
                            name='OneHot'
                            )(input_)

    encoder = Bidirectional(LSTM(output_dim=encoder_units, return_sequences=True),
                            name='encoder',
                            merge_mode='concat',
                            trainable=trainable)(input_embed)

    decoder = AttentionDecoder(decoder_units,
                               name='attention_decoder_1',
                               output_dim=n_labels,
                               return_probabilities=return_probabilities,
                               trainable=trainable)(encoder)
    output_2 = Dense(output_dim=n_labels, activation='softmax')(decoder)
    model = Model(input=input_, output=output_2)

    return model
