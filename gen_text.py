from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import  numpy as np
import regex as re
from data_load import load_myVocab
from underthesea import pos_tag

# file = open('myVocab_test.txt' , 'r', encoding='utf-8')
# lines = file.readlines()
# vocab = ""
# for line in lines:
#     temp = ""
#     arr = line.split()
#     for i in range(1, len(arr)-1):
#         temp += " " + arr[i]
#     vocab += arr[0] + temp + '\n'
# file2 = open('myVocab_test2.txt' , 'w', encoding='utf-8')
# file2.write(vocab)

MAX_CONTENT_LEN = 512
MAX_TITLE_LEN = 27
next_words = 32

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

word2idx, idx2word = load_myVocab('myVocab.txt')
# word2idx, idx2word = load_vocab('vocab.txt')
unk_idx = word2idx.get("unk")

lstm_model = load_model("models/ep100-loss0.078.h5")

def generate_text(seed_text, next_words, model, input_len1, input_len2):
    result = ""
    for _ in range(next_words):
        token_list = [word2idx.get(w, unk_idx) for w in _format_line(seed_text)]
        token_list = pad_sequences([token_list], maxlen=input_len1, padding='pre')

        token_list2 = [word2idx.get(w, unk_idx) for w in _format_line("sos " +result)]
        token_list2 = pad_sequences([token_list2], maxlen=input_len2 - 1, padding='pre')

        predicted = np.argmax(model.predict([token_list, token_list2]))
        output_word = idx2word[predicted]
        if output_word == 'eos':
            return result
        result += " " + output_word
    return result

text = open('test.txt', 'r', encoding='utf-8').read()
print (generate_text(text, next_words, lstm_model, MAX_CONTENT_LEN, MAX_TITLE_LEN))