from keras.layers import Input, Conv1D, MaxPool1D, LSTM, Dense, BatchNormalization, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy


# Default values come from paper
def create_model(inputs: int, classify: bool = True, **options):
    input_layers = []
    concat_layers = []

    lookback = options.get('lookback', 30)
    filters = options.get('filters', 16)
    kernel_size = options.get('kernel_size', 2)
    input_neurons = options.get('input_neurons', 50)

    for _ in range(inputs):
        input_layer = Input((lookback, 1))
        conv_layer = Conv1D(filters=filters, kernel_size=kernel_size)(input_layer)
        pooling = MaxPool1D()(conv_layer)
        lstm = LSTM(input_neurons)(pooling)
        concat_layers.append(lstm)
        input_layers.append(input_layer)

    concat_layer = concatenate(concat_layers)
    dense1 = Dense(options.get('dense1', 256))(concat_layer)
    batch_nl1 = BatchNormalization()(dense1)
    dropout1 = Dropout(options.get('dropout1', 0.5))(batch_nl1)

    dense2 = Dense(options.get('dense2', 64))(dropout1)
    batch_nl2 = BatchNormalization()(dense2)
    dropout2 = Dropout(options.get('dropout2', 0.3))(batch_nl2)

    output_layer = Dense(2, activation='softmax', name='output') if classify else Dense(1, name='output')
    output_layer = output_layer(dropout2)
    loss = {
        'output': categorical_crossentropy if classify else options.get('loss', 'mae')
    }

    model = Model(inputs=input_layers, outputs=[output_layer])
    if classify:
        model.compile(
            loss=loss, 
            optimizer=Adam(learning_rate=options.get('learning_rate', 0.001)),
            metrics='acc'
        )
    else:
        model.compile(
            loss=loss, 
            optimizer=Adam(learning_rate=options.get('learning_rate', 0.001)),
        )


    return model