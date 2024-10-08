# %%
"""

"""

# %%
#Install python3.9 first

# %%
!nvidia-smi

# %%
!wget https://svn.spraakdata.gu.se/sb-arkiv/pub/dalaj/datasetDaLAJsplit.csv


# %%
%pip install pandas
%pip install scikit-learn
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



# %%
dalaj_df = pd.read_csv("./content/datasetDaLAJsplit.csv")

# %%
#@title Fix arrangement Train

train_df = dalaj_df[dalaj_df['split'] == "train" ]

sorted_items = []
for idx, item in train_df.iterrows():
  correct_dict = {"Text": item["corrected sentence"],"Label":1}
  sorted_items.append(correct_dict)

  incorrect_dict = {"Text": item["original sentence"],"Label":0}
  sorted_items.append(incorrect_dict)

final_train_df = pd.DataFrame(sorted_items)
from sklearn.utils import shuffle
final_train_df = shuffle(final_train_df)

# %%
final_train_df

# %%
#@title Fix arrangement for test

test_df = dalaj_df[dalaj_df['split'] == "test" ]

sorted_items = []
for idx, item in test_df.iterrows():
  correct_dict = {"Text": item["corrected sentence"],"Label":1}
  sorted_items.append(correct_dict)

  incorrect_dict = {"Text": item["original sentence"],"Label":0}
  sorted_items.append(incorrect_dict)

final_test_df = pd.DataFrame(sorted_items)
from sklearn.utils import shuffle
final_test_df = shuffle(final_test_df)


# %%
final_test_df

# %%
#@title sentence model
#!pip install -U sentence-transformers
#from sentence_transformers import SentenceTransformer, models

#model_dir = 'Peltarion/xlm-roberta-longformer-base-4096'
#mod = "KB/bert-base-swedish-cased"
#sen_xlmr = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

#word_embedding_model = models.Transformer(mod, tokenizer_name_or_path= mod , max_seq_length=512)
#pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

#model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# %%
"""
#EDA
"""

# %%
import os

# Construct the absolute path using the current working directory
current_dir = os.getcwd()

#@title EDA
!git clone https://github.com/mosh98/swe_aug.git
%pip install -r requirements.txt
""" if cant install because of access denied: https://stackoverflow.com/questions/64278198/error-can-not-perform-a-user-install-user-site-packages-are-not-visible-in """
!wget https://www.ida.liu.se/divisions/hcs/nlplab/swectors/swectors-300dim.txt.bz2
!bzip2 -dk /content/swectors-300dim.txt.bz2


word_vec_path = os.path.join(current_dir, "content", "swectors-300dim.txt")

""" word_vec_path = "./content/swectors-300dim.txt" """

from swe_aug import EDA
print(type(EDA.Enkel_Data_Augmentation)) #for debugging purposes
aug = EDA.Enkel_Data_Augmentation(word_vec_path)

# %%
"""
# Fraction of Dataset

"""

# %%
!pip install transformers
!pip install MultiEncoder==0.0.6
!pip install ipywidgets
!pip install torch
!pip install ultralytics
#https://stackoverflow.com/questions/78114412/import-torch-how-to-fix-oserror-winerror-126-error-loading-fbgemm-dll-or-depen (if theres an error)

# %%
mod = "KB/bert-base-swedish-cased"
from mle import multi_layer_encoder
le = multi_layer_encoder.multi_layer_encoder(mod)

# %%
#@title

from sklearn.model_selection import train_test_split


def frac(dataframe, fraction,label):
    """Returns fraction of data"""
    if fraction == 1.0:
      return dataframe
      
    y = dataframe[label]
    train, test = train_test_split(dataframe,train_size=fraction,stratify = y)
    
    return train

def encode_df(dataframe, mod=None,col = "text"):
  encoded = []
  for idx, item in dataframe.iterrows():
        list_of_encoded_inputs, dect = le.multi_encode(item.Text)
        encoded.append(list_of_encoded_inputs[1])
        #encoded.append(model.encode(item[col])) 

  return encoded


reports = [] #clf_report (Before Augment, After Augment)


# %%
#@title
test_embed = []

y_test = final_test_df.Label

for idx, item in final_test_df.iterrows():
      list_of_encoded_inputs, dect = le.multi_encode(item.Text)
      test_embed.append(list_of_encoded_inputs[1])
#test_embed.append(model.encode(item.Text))
  

# %%
#@title
#from swe_aug.Other_Techniques import Text_Cropping
#frag = Text_Cropping.cropper(percent = 0.25)
!pip install SpaceAugmentation
from aug import Augmentation

ag = Augmentation.Augmentation()

# %%
#@title
from swe_aug.Other_Techniques import Type_SR
aug = Type_SR.type_DA(word_vec_path)

# %%
#@title Training Loop
from sklearn.linear_model import SGDClassifier

split_percentage = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

label = "Label"

for percentage in split_percentage:
    print("______________________________________________")
    
    print("Percentage",percentage)
    
    temp_train_df = frac(final_train_df, percentage/100,label)
    y_train = temp_train_df.Label
    
    print("Before Augmentation size: ",temp_train_df.shape)
    
    train_embed = encode_df( temp_train_df,  col="Text")

    logreg = SGDClassifier(max_iter=3000,n_jobs=-1)
    logreg.fit(train_embed, y_train)
    
    y_pred = logreg.predict(test_embed)
    
    r_1 = classification_report(y_test, y_pred)
    print("Before Augmentation")
    print(r_1)
    print(" ")

    # picking out bad samples
    incorrect_df = temp_train_df[temp_train_df['Label'] == 0]

    #Augmentation

    aug_samples = []
    for idx, item in incorrect_df.iterrows():
      txt = item.Text
      lab = item.Label
      list_of_augs = aug.type_synonym_sr(txt, token_type = "NOUN", n = 2)
      
      for element in list_of_augs:
          aug_samples.append({"Text":' '.join(element),"Label":lab})

    
    new_aug_samples = pd.DataFrame(aug_samples)
    

    
    
    
    new_df = pd.concat([temp_train_df,new_aug_samples],ignore_index=True)
    
    print("After Augmentation size: ",new_df.shape[0])


    #augmented train test

    #encode
    train_embed = encode_df( new_df,  col="Text")
    y_train = new_df.Label
    
    logreg_ = SGDClassifier(max_iter=3000, n_jobs=-1)
    logreg_.fit(train_embed, y_train)

    y_pred = logreg_.predict(test_embed)


    r_2 = classification_report(y_test, y_pred)
    print("After Augmentation")
    print(r_2)

    #save the reports
    reports.append((percentage, r_1,r_2))



    print("______________________________________________")








# %%

!pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer, models

#model_dir = 'Peltarion/xlm-roberta-longformer-base-4096'
#mod = "KB/bert-base-swedish-cased"
sen_xlmr = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

word_embedding_model = models.Transformer(sen_xlmr, tokenizer_name_or_path= sen_xlmr , max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# %%
    # picking out bad samples
incorrect_df = final_train_df[final_train_df['Label'] == 0]

    #Augmentation

aug_samples = []
original_augmented = [] #every element : orginal_text, (Tuple of augmented sentences)
for idx, item in incorrect_df.iterrows():
      txt = item.Text
      lab = item.Label
      list_of_augs = aug.type_synonym_sr(txt, token_type = "NOUN", n = 2)
      
      temp_list = []
      for element in list_of_augs:
            aug_samples.append({"Text":' '.join(element),"Label":lab})
            temp_list.append(' '.join(element))
      
      original_augmented.append((txt,temp_list))

    

new_aug_samples = pd.DataFrame(aug_samples)

# %%
from sentence_transformers import SentenceTransformer, util


def checkSimilarity(reference_string, list_of_strings, model):
    """
    This function takes a reference string and a list of strings and returns a list of strings that are similar to the reference string.

    :param reference_string:
    :param list_of_strings:
    :param model: sentence transformer model
    :return:
    """
    #1. Encode using Sentence Transformer
    #2. Calculate cosine similarity
    #3. Append the similirty value to the list
    # caclulcate percentage of elements below 0.9
    # return the list, and the percentage of elements below 0.9

    reference_encoded = model.encode(reference_string,convert_to_numpy=True) #encoded reference string
    list_of_cosine_similarity = []
    for string in list_of_strings:
        #print(string)
        encoded_string = model.encode(string,convert_to_numpy=True)

        #find coside similairity of enocded String and reference string
        similarity = util.cos_sim(reference_encoded, encoded_string)
        list_of_cosine_similarity.append(similarity.item())
    #print(list_of_cosine_similarity)
    #sum(i > 5 for i in j)
    percentage_of_elements_below_0_9 = sum(i < 0.9 for i in list_of_cosine_similarity) / len(list_of_cosine_similarity)

    return sum(i < 0.95 for i in list_of_cosine_similarity),percentage_of_elements_below_0_9

# %%
low_semantic_sentences = []
for item in original_augmented:
  num, percentage = checkSimilarity(item[0],item[1],model)
  low_semantic_sentences.append(num)

# %%
print("Original Sentences:",len(original_augmented))
print("Num of Augmented Sentences:",len(original_augmented)*2)


# %%
print("Num of Bad Augmented Sentences:",sum(low_semantic_sentences))


# %%
num_of_aug_samples = len(original_augmented)*2
print("Percentage of Bad Augmented Sentences:",(sum(low_semantic_sentences)/num_of_aug_samples)*100)