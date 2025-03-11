from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell

convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [3, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(256, 256), input_channels=16, filter_size=5, num_features=64),  # Change shape to (256, 256)
        CLSTM_cell(shape=(128, 128), input_channels=64, filter_size=5, num_features=96),   # Change shape to (128, 128)
        CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=96)     # Change shape to (64, 64)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 3, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=96),    # Change shape to (64, 64)
        CLSTM_cell(shape=(128, 128), input_channels=96, filter_size=5, num_features=96),   # Change shape to (128, 128)
        CLSTM_cell(shape=(256, 256), input_channels=96, filter_size=5, num_features=64)    # Change shape to (256, 256)
    ]
]

convgru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [3, 16, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 64, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(256, 256), input_channels=16, filter_size=5, num_features=64),    # Change shape to (256, 256)
        CGRU_cell(shape=(128, 128), input_channels=64, filter_size=5, num_features=96),    # Change shape to (128, 128)
        CGRU_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=96)       # Change shape to (64, 64)
    ]
]

convgru_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [64, 16, 3, 1, 1],
            'conv4_leaky_1': [16, 3, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(64, 64), input_channels=96, filter_size=5, num_features=96),     # Change shape to (64, 64)
        CGRU_cell(shape=(128, 128), input_channels=96, filter_size=5, num_features=96),    # Change shape to (128, 128)
        CGRU_cell(shape=(256, 256), input_channels=96, filter_size=5, num_features=64)     # Change shape to (256, 256)
    ]
]