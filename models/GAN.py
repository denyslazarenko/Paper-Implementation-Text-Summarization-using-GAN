from keras.models import Sequential


class GAN(object):
    def __init__(self, generator, discriminator, **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        self.gan = Sequential([generator, discriminator])

        self.coding = self.generator.input_shape[1]

        if 'init' in kwargs:
            init = kwargs['init']
            init(self.generator)
            init(self.discriminator)

        generator.summary()
        discriminator.summary()
        self.gan.summary()

    def generate(self, inputs):
        return self.generator.predict(inputs)
