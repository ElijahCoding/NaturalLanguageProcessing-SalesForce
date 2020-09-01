import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 64

padded_shapes = ([None], ())

train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=padded_shapes)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    metrics=['accuracy']
)

history = model.fit(train_dataset, epochs=20, validation_data=test_dataset, validation_steps=30)


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sentence, pad, model):
    encoded_sample_pad_text = encoder.encode(sentence)
    if pad:
        encoded_sample_pad_text = pad_to_size(encoded_sample_pad_text, 64)
    encoded_sample_pad_text = tf.cast(encoded_sample_pad_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pad_text, 0))

    return predictions

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True, model=model) * 100
print('Probability this is a positive review %.2f' % predictions)

sample_text = ('Movie is shit')
predictions = sample_predict(sample_text, pad=True, model=model) * 100
print('Probability this is a positive review %.2f' % predictions)

second_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

second_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

history = second_model.fit(train_dataset, epochs=5, validation_data=test_dataset, validation_steps=30)

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True, model=second_model) * 100
print('Probability this is a positive review %.2f' % predictions)

sample_text = ('Movie is shit')
predictions = sample_predict(sample_text, pad=True, model=second_model) * 100
print('Probability this is a positive review %.2f' % predictions)