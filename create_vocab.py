from imutils import paths
from underthesea import word_tokenize, pos_tag
import regex as re
from tqdm import tqdm

def _format_line(line):
    # line = re.sub(url_marker.WEB_URL_REGEX, "<link>", line)
    line = re.sub("[0-9]+", "num", line)
    line = re.sub("[.]+", ".", line)
    line = re.sub("[\?\!]", " eos", line)
    line = re.sub("[\'\"\(\)\“\”]", "", line)
    return line

filePaths = list(paths.list_files('D:/AI_RESEARCH/NLP_Research/headline-generator/data_processed'))
vocab = {}
for filePath in tqdm(filePaths):
    word_total = ""
    article = open(filePath, 'r', encoding='utf-8').read()
    article = re.sub("[\n]+", " ", article)
    word_total =  _format_line(article)
    word_tokens = pos_tag(word_total)
    for word_token in word_tokens:
        if word_token[1] == 'Np':
            if word_token[0] in vocab:
                vocab[word_token[0]] = vocab[word_token[0]] + 1
            else: vocab[word_token[0]] = 0
        else:
            low_word = word_token[0].lower()
            if low_word in vocab:
                vocab[low_word] = vocab[low_word] + 1
            else:
                vocab[low_word] = 0
sort_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
print(len(sort_vocab))
file = open('myVocab.txt', 'w', encoding='utf-8')
vocabTxt = ""
n=0
for voc in sort_vocab:
    vocabTxt += voc[0] + "\n"
    n+=1
file.write(vocabTxt)



