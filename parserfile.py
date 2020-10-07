import spacy
import pickle
import random

#text_file = open("train_data.txt", "r")
#print(text_file)
#lines = text_file.readlines()
#print(lines)
tas= open('train_data.txt', 'r', encoding="utf8").read()
#print(type(text_as_string))
train_data=pickle.load(open("train_data.pkl",'rb'))


#print(train_data[0])

nlp=spacy.blank('en')
def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
        ner=nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    for _,annotation  in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe !='ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer=nlp.begin_training()
        for itn in range(10):
            print("Start Iteration", str(itn))
            random.shuffle(train_data)
            losses = {}
            index=0
            for text, annotations in train_data:
                try:
                    nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    losses=losses)
                except Exception as e:
                    pass
            print("Losses", losses)

train_model(train_data)
nlp.to_disk("modeldonenlp")