# Modifications Copyright https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
#
# The code is a modification from the blog by Jason Brownlee on July 17, 2019 in Generative Adversarial Networks.
# ==============================================================================

# example of a wgan for generating handwritten digits
from numpy import expand_dims
from numpy import mean
from numpy import ones
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.constraints import Constraint
from matplotlib import pyplot
#seminar imports
from models.GAN_components import Generator, Discriminator
from models.GAN import GAN
from data_util import config


# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


class WGAN(GAN):
    def __init__(self, **kwargs):
        super(WGAN, self).__init__(**kwargs)
        self.critic = self.define_critic()
        self.gan = self.define_gan()

    # calculate wasserstein loss
    def wasserstein_loss(srlf, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    # define the standalone critic model
    def define_critic(self):
        model = self.discriminator
        opt = RMSprop(lr=0.00005)
        model.compile(loss=self.wasserstein_loss, optimizer=opt)
        return model

    # define the combined generator and critic model, for updating the generator
    def define_gan(self):
        # make weights in the critic not trainable
        self.critic.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(self.generator)
        # add the critic
        model.add(self.critic)
        # compile model
        opt = RMSprop(lr=0.00005)
        model.compile(loss=self.wasserstein_loss, optimizer=opt)
        return model

    # select real samples
    def generate_real_samples(self, bath_generator):
        # choose random instances
        _, X = next(bath_generator)
        n_samples = X.shape[0]
        # generate class labels, -1 for 'real'
        y = -ones((n_samples, 1))
        return X, y

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, batch_generator):
        # generate points in latent space
        X, _ = next(batch_generator)
        # predict outputs
        X = self.generate(X)
        n_samples = X.shape[0]
        # create class labels with 1.0 for 'fake'
        y = ones((n_samples, 1))
        return X, y

    # train the generator and critic
    def train(self, batch_generator, n_steps=200, n_batch=4, n_critic=5, save_iter=20):
        c1_hist, c2_hist, g_hist = list(), list(), list()
        # manually enumerate epochs
        for i in range(n_steps):
            # update the critic more than the generator
            c1_tmp, c2_tmp = list(), list()
            for _ in range(n_critic):
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(batch_generator)
                # update critic model weights
                c_loss1 = self.critic.train_on_batch(X_real, y_real)
                c1_tmp.append(c_loss1)
                # generate 'fake' examples
                X_fake, y_fake = self.generate_fake_samples(batch_generator)
                # update critic model weights
                c_loss2 = self.critic.train_on_batch(X_fake, y_fake)
                c2_tmp.append(c_loss2)
            # store critic loshalf_batchs
            c1_hist.append(mean(c1_tmp))
            c2_hist.append(mean(c2_tmp))
            # prepare points in latent space as input for the generator
            X_gan, _ = next(batch_generator)
            y_gan = -ones((n_batch, 1))
            # update the generator via the critic's error
            g_loss = self.gan.train_on_batch(X_gan, y_gan)
            g_hist.append(g_loss)
            # summarize loss on this batch
            print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i + 1, c1_hist[-1], c2_hist[-1], g_loss))
            if i%save_iter == 0:
                print(f"Real input:{X_gan}")
                samples = self.generate(X_gan)
                print(f"GAN results:{samples} after iteration:{i}", )
        self.plot_history(c1_hist, c2_hist, g_hist)

    def plot_history(self, d1_hist, d2_hist, g_hist):
        # plot history
        pyplot.plot(d1_hist, label='crit_real')
        pyplot.plot(d2_hist, label='crit_fake')
        pyplot.plot(g_hist, label='gen')
        pyplot.legend()
        pyplot.savefig('plot_line_plot_loss.png')
        pyplot.close()

