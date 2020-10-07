import spacy
import pickle
import random
import sys,fitz

nlp_model=spacy.load("modeldonenlp")
train_data=pickle.load(open("train_data.pkl",'rb'))
print(train_data[0][0])

doc=nlp_model(train_data[0][0])
for ent in doc.ents:
    print(f'{ent.label_.upper():{30}}-{ent.text}')
    #print(f'{ent.label.upper}')
fname="Alice Clark CV.pdf"
doc=fitz.open(fname)
text=""
for page in doc:
    text=text+str(page.getText())
tx=" ".join(text.split('\n'))
print(tx)
print("******************")
doc1=nlp_model(tx)
for ent in doc1.ents:
    print(f'{ent.label_.upper():{30}}-{ent.text}')
    #print(f'{ent.label.upper}')