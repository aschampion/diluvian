from keras.layers import AveragePooling3D, Convolution3D, Input, merge
from keras.layers.core import Activation, Lambda, Merge
from keras.models import Model, Sequential


INPUT_SHAPE = (33, 33, 17, 1)
NUM_MODULES = 8


def addConvolutionModule(model, first):
    if False and first:
        model = Convolution3D(32, 3, 3, 3,
                                activation='relu',
                                border_mode='same')(model)
        model = Convolution3D(32, 3, 3, 3,
                                border_mode='same')(model)
    else:
        model2 = Convolution3D(32, 3, 3, 3,
                                activation='relu',
                                border_mode='same')(model)
        model2 = Convolution3D(32, 3, 3, 3,
                                border_mode='same')(model2)
        model = merge([model, model2], mode='sum')
    # Note that the activation here differs from He et al. 2016, as that
    # activation is not on the skip connection path.
    model = Activation('relu')(model)

    return model


image_input = Input(shape=INPUT_SHAPE, dtype='float32', name='image_input')
mask_input = Input(shape=INPUT_SHAPE, dtype='float32', name='mask_input')
# model.add(Lambda(lambda x: x + 0, input_shape=INPUT_SHAPE))
# ffn = Sequential()
# model.add(Input(shape=INPUT_SHAPE, dtype='float32', name='image_input'))
# ffn.add(Activation('linear', input_shape=INPUT_SHAPE))
ffn = merge([image_input, mask_input], mode='concat')

# Convolve and activate before beginning the skip connection modules,
# as discussed in the Appendix of He et al 2016.
ffn = Convolution3D(32, 3, 3, 3,
                    activation='relu',
                    border_mode='same')(ffn)

for module_index in range(0, NUM_MODULES):
    ffn = addConvolutionModule(ffn, module_index == 0)

# mask_output = Activation('linear', name='mask_output')(ffn)
mask_output = Convolution3D(1, 1, 1, 1, name='mask_output')(ffn)
# mask_output = AveragePooling3D(pool_size=(1,1,1), name='mask_output')(ffn)
# mask_output = ffn
ffn = Model(input=[image_input, mask_input], output=[mask_output])
ffn.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['categorical_crossentropy'])
