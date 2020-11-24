import pandas as pd
import numpy as np

#docs = json.load(open("/home/samuel/Data/ace_2005_td_v7/proc/english.json"))

f = open("/home/samuel/Data/ace_2005_td_v7/proc/train.oneie.json")

df = pd.read_json(f, lines=True)

titles = df['doc_id']

for i in range(len(df)):
    print(i)
    doc = df.iloc[i]
    print()

#############

f = open("/home/samuel/Data/ace_2005_td_v7/proc/english.json")

df = pd.read_json(f, lines=True)

titles = df['doc_id']

lens = []

for i in range(len(df)):
    print(i)
    doc = df.iloc[i]
    title = doc['doc_id']
    sents = doc['sentences']
    print(title)
    fullText = []
    for j in sents:
        fullText += j['tokens']
        fullText.append("\n")
        #print(j['text'])
        #print(j['entities'])
        #print(j['relations'])
        #print(j['events'])
        #input()
    #print(fullText)
    #print(" ".join(fullText))
    print(len(fullText))
    lens.append(len(fullText))
    if len(fullText) < 200:
        print(" ".join(fullText))
        #input()
    #input()

lens = np.array(lens)
print(lens)
print(len(lens))
print(min(lens), max(lens))
for i in range(0, 2000, 128):
    print("above", i, ":", sum(lens > i))