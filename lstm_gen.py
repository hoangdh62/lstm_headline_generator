from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, concatenate, Input
from tensorflow.keras.models import Model
import tensorflow.keras.utils as ku
import numpy as np
from imutils import paths
from tensorflow.keras.callbacks import ModelCheckpoint
import regex as re
from data_load import load_myVocab
from underthesea import pos_tag
from tqdm import tqdm

word2idx, idx2word = load_myVocab('Viet74k.txt')
unk_idx = word2idx.get("unk")

total_words = len(word2idx)
max_content_len = 512
max_title_len = 64
batch_size = 32

# D:/AI_RESEARCH/NLP_Research/headline-generator/data_processed
filePaths = list(paths.list_files('Data'))

def lower_case(text):
    result = []
    tags = pos_tag(text)
    for tag in tags:
        if tag[1] == 'Np':
            result.append(tag[0])
        else:
            result.append(tag[0].lower())
    return result

def _format_line(line):
    # line = re.sub(url_marker.WEB_URL_REGEX, "<link>", line)
    line = re.sub("[0-9]+", "num", line)
    line = re.sub("[.]+", ".", line)
    line = re.sub("[\?\!]", " eos", line)
    line = re.sub("[\'\"\(\)]", "", line)
    return lower_case(line)

def get_sequence_of_tokens(contents, corpus):
    content_sequences = []
    title_sequences = []
    for line_idx in range(len(corpus)):
        for i in range(1, len(corpus[line_idx])):
            content_sequences.append(contents[line_idx])
            n_gram_sequence = corpus[line_idx][:i + 1]
            title_sequences.append(n_gram_sequence)
    cont_sequences = np.array(pad_sequences(content_sequences, maxlen=max_content_len))
    return cont_sequences, title_sequences

def generate_padded_sequences(title_sequences):
    # max_sequence_len = max([len(x) for x in title_sequences])
    # input_sequences = np.array(pad_sequences(title_sequences, maxlen=max_title_len, padding='pre'))
    predictors, label = title_sequences[:, :-1], title_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label

title_sequences_total = np.concatenate((np.load("input_sequences1.npy"), np.load("input_sequences2.npy")),axis=0)
cont_sequences_total = np.concatenate((np.load("cont_sequences1.npy"), np.load("cont_sequences2.npy")),axis=0)
def generator(title_sequences_total, cont_sequences_total):
    while True:
        article_idxs = np.random.choice(range(len(title_sequences)), size=batch_size, replace=False)
        # corpus = []
        # contents = []
        # for article_path in (article_paths):
        #     article = open(article_path, 'r', encoding='utf-8').read().split('\n')
        #     title = 'sos ' + article[0] + ' eos'
        #     content = ""
        #     for k in range(1, len(article)):
        #         content += article[k] + " "
        #     corpus.append([word2idx.get(w, unk_idx) for w in _format_line(title)])
        #     contents.append([word2idx.get(ww, unk_idx) for ww in _format_line(content)])
        #
        # cont_sequences, title_sequences = get_sequence_of_tokens(contents, corpus)
        cont_sequences = cont_sequences_total[article_idxs]
        title_sequences = title_sequences_total[article_idxs]
        predictors, labels = generate_padded_sequences(title_sequences)
        yield [cont_sequences, predictors], labels

def create_model(iput_len1, input_len2, total_words):

    input1 = Input(shape=(iput_len1,))
    embed1 = Embedding(total_words, 128, input_length=iput_len1)(input1)
    lstm1 = LSTM(256)(embed1)
    drop1 = Dropout(0.25)(lstm1)

    input2 = Input(shape=(input_len2,))
    embed2 = Embedding(total_words, 128, input_length=32)(input2)
    lstm2 = LSTM(256)(embed2)
    drop2 = Dropout(0.25)(lstm2)

    final_layer = concatenate([drop1, drop2])
    final_layer = Dense(512, activation='relu')(final_layer)
    den = Dense(total_words, activation='softmax')(final_layer)

    # Add Output Layer
    model = Model(inputs=[input1, input2], outputs=den)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

checkpoint = ModelCheckpoint('checkpoint/' + 'ep{epoch:03d}-loss{loss:.03f}.h5',
        monitor='val_loss', save_weights_only=False, save_best_only=False, period=10)

lstm_model = create_model(max_content_len, max_title_len-1, total_words)
lstm_model.summary()
lstm_model.fit(generator(filePaths), steps_per_epoch=(len(filePaths)*32)//batch_size, epochs=100, callbacks=[checkpoint])
lstm_model.save('models/vocab74k_model3.h5')