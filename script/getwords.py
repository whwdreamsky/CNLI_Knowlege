import sys
import codecs
worddict = {}
default_key = ['_PAD','_UNK','_GO','_EOS']
for key in default_key:
    worddict[key] = len(worddict)

filename = sys.argv[1]
with codecs.open(filename,'r','utf-8') as f:
    for line in f:
        line = line.lower()
        tmp = line.strip().split('\t')
        if len(tmp)!=3:
            continue
        words = tmp[0].split(' ')
        words.extend(tmp[1].split(' '))
        for w in words:
            if w not in worddict and w!="":
                worddict[w] = len(worddict)
for key,value in worddict.items():
    result = key + '\t' + str(value)
    print(result.encode('utf-8'))

