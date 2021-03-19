from data_util import config

VOCABULARY_SIZE = config.vocab_size
EMBEDDING_DIM = config.vocab_size
MAX_SEQUENCE_LENGTH = config.max_enc_steps

RANDOM_SEED = 42

filter_sizes = [3, 4, 5]
num_filters = 32
dropout_rate = 0.5
nb_epoch = 5
batch_size = 4
output_dim = 1

loss = 'categorical_crossentropy'
optimizer = 'RMSprop'

models_dir = './saved_models'
weights_dir = '../weights'
MODEL = models_dir + '/Discriminator/model.json'
WEIGHTS = weights_dir + '/Discriminator/weights_final_v2'


