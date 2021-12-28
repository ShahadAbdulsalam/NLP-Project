#!/usr/bin/env python
# coding: utf-8

# # Object:
# 
# ### Analysing comments on COVID-⁠19 Vaccines plan in Canada.

# # Description:
# 
# ### We will analyse the text starting with turn these comments into a meaningful format, then cleaning data by:
# - Remove capital letters and replace them by lower case letters.
# - Remove punctiuation.
# - Remove stop words and numbers.
# ### AS a final steps we will use two of the topic modeling techniques, then converting comments into supervised data that we can explorate data using AMOD and counter.

# # Tools:
# - Numpy
# - Pandas
# - Sklearn
# - NLTK
# - RE
# - Spacy
# - Gensim

# In[1]:


import pandas as pd
import numpy as np
import spacy
import re, nltk, spacy, gensim
nlp = spacy.load('en_core_web_sm')
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()
from pprint import pprint


# In[2]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[3]:


from spacy.lang.en import English
get_ipython().system('pip install spacy && python -m spacy download en')


# In[4]:



data= pd.read_csv(r'C:\Users\sshah\OneDrive\المستندات\comment_on_plan.csv',encoding='latin-1')


# In[5]:


data


# In[6]:


data1= list(data.Comment)
data1


# ## Preprocessing:

# In[7]:


documents=[]
for i in data1:
    documents.append(nlp(i))
documents


# In[8]:


def preprocessing(docs):
    processed_data=[]
    for e in docs:
        tokens = []
        for token in nlp(e):
            if not token.is_stop:
                tt = gensim.utils.simple_preprocess(str(token.lemma_), deacc=True)
                for i in tt: 
                    tokens.append(i)
        processed_data.append(tokens)
    return processed_data


# In[9]:


preprocessed_data=preprocessing(documents)
preprocessed_data[1]


# ## LDA Model:

# In[10]:


dictionary= corpora.Dictionary(preprocessed_data)
dt_matrix= [dictionary.doc2bow(rev) for rev in preprocessed_data]


# In[11]:


lda = gensim.models.ldamodel.LdaModel(corpus=dt_matrix, num_topics=8, id2word=dictionary, passes=5)


# In[12]:


lda.print_topics()


# ## Evaluate LDA.

# In[13]:


cohrence_lda_model= CoherenceModel(model=lda, texts= preprocessed_data, dictionary= dictionary, coherence='c_v')
cohrence_lda= cohrence_lda_model.get_coherence()
print(f'\n Cohrence score: {cohrence_lda}')


# # determining best number of topics for LDA Model.

# In[14]:


def compute_cohrence_values(dictionary, corpus, texts, limit, start=2, step=1):
    coherence_values=[]
    model_list=[]
    for num_topics in range(start, limit, step):
        model= gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics= num_topics, id2word= dictionary)
        model_list.append(model)
        coherence_model= CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    return model_list, coherence_values


# In[15]:


model_list, coherence_values= compute_cohrence_values(dictionary= dictionary, corpus= dt_matrix, texts= preprocessed_data, limit=9, start= 2, step=1)


# In[16]:


start=2
limit=9
step=1

x= range(start,limit,step)


# In[17]:


for topic, cv in zip(x, coherence_values):
    print('Number of topics:', topic, 'has coherence score:', round(cv,4))


# ## From the previous, we found out that the best model will have 7 topics

# In[18]:


optimal_model= model_list[5]
model_topics= optimal_model.show_topics(formatted=False)
optimal_model.print_topics(num_words=10)


# ## Visualaization:

# In[19]:


visualaization= gensimvis.prepare(optimal_model,dt_matrix,dictionary)
visualaization


# ### now it's clear that we didn't achieve the high objective results with LDA model, so we'll try with another model.

# # CorEx Model:

# In[20]:


get_ipython().system('pip install corextopic')
get_ipython().system('pip install networkx')
from corextopic import corextopic as ct
from corextopic import vis_topic as vt


# In[21]:


vectorizer2 = CountVectorizer(max_features=20000,
                             stop_words='english', token_pattern="\\b[a-z][a-z]+\\b",
                             binary=True)

doc_word = vectorizer2.fit_transform(data1)
words = list(np.asarray(vectorizer2.get_feature_names()))


# In[22]:


topic_model = ct.Corex(n_hidden=4, words=words, seed=1)
topic_model.fit(doc_word, words=words, docs=data1)


# In[23]:


topics = topic_model.get_topics()
for n,topic in enumerate(topics):
    topic_words,_,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))
    
    categories = ['Vaccine.plan', 'Healthcare.in.Canada', 
              'Canadian.goverment', 'Trudeau.and.the.liberal.party.of.Canada']


# In[24]:


predictions = pd.DataFrame(topic_model.predict(doc_word), columns=['topic'+str(i) for i in range(4)])
predictions


# In[25]:


topic_model.fit(doc_word, words=words, docs=data1, 
                anchors=[['plan', 'decide'], ['healthcare','health','care','children'],['government','country','citizen','decision'],['canada','US','liberal','pay']], anchor_strength=10)

topics = topic_model.get_topics()
for n,topic in enumerate(topics):
    topic_words,_,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))


# In[26]:


topics


# In[27]:


predictions = pd.DataFrame(topic_model.predict(doc_word), columns=['topic'+str(i) for i in range(4)])
predictions


# ## Converting data into supervised data:

# In[28]:


data['spacy_doc'] = list(nlp.pipe(data.Comment))


# In[29]:


data['index']= range(0,1362)


# In[30]:


predictions['index']= range(0,1362)


# In[31]:


spacy_data=pd.merge(data,predictions, on= ['index','index'])


# In[32]:


spacy_data.rename(columns = {'topic0':'covid_plan', 'topic1':'healthcare', 'topic2':'canadian_government','topic3':'liberal_party'}, inplace = True)


# In[34]:


spacy_data


# In[35]:


type(spacy_data.spacy_doc[0])


# In[36]:


covid_plan_reviews = spacy_data[spacy_data.covid_plan==True]
healthcare_reviews = spacy_data[spacy_data.healthcare==True]
canadian_government_reviews = spacy_data[spacy_data.canadian_government==True]
liberal_party_reviews = spacy_data[spacy_data.liberal_party==True]


# ## EDA with Spacy(Amods & Counter):

# In[37]:


from spacy.symbols import amod


# In[38]:


def get_amods(noun, ser):
    amod_list = []
    for doc in ser:
        for token in doc:
            if (token.text) == noun:
                for child in token.children:
                    if child.dep == amod:
                        amod_list.append(child.text.lower())
    return sorted(amod_list)

def amods_by_sentiment(noun):
    print(f"Adjectives describing {str.upper(noun)}:\n")
    
    print("\nCovid plan topic:")
    pprint(get_amods(noun, covid_plan_reviews.spacy_doc))
    
    print("\nHealthcare topic:")
    pprint(get_amods(noun, healthcare_reviews.spacy_doc))
    print("\n Canadian government topic:")
    pprint(get_amods(noun, canadian_government_reviews.spacy_doc))
    print("\nLiberal party topic:")
    pprint(get_amods(noun, liberal_party_reviews.spacy_doc))
   


# In[45]:


amods_by_sentiment('Canada')


# In[40]:


covidplan_adj = [token.text.lower() for doc in covid_plan_reviews.spacy_doc for token in doc if token.pos_=='ADJ']
healthcare_adj = [token.text.lower() for doc in healthcare_reviews.spacy_doc for token in doc if token.pos_=='ADJ']
canadiangovernment_adj = [token.text.lower() for doc in canadian_government_reviews.spacy_doc for token in doc if token.pos_=='ADJ']
liberal_party_adj = [token.text.lower() for doc in liberal_party_reviews.spacy_doc for token in doc if token.pos_=='ADJ']


covidplan_noun = [token.text.lower() for doc in covid_plan_reviews.spacy_doc for token in doc if token.pos_=='NOUN']
healthcare_noun = [token.text.lower() for doc in healthcare_reviews.spacy_doc for token in doc if token.pos_=='NOUN']
canadiangovernment_noun = [token.text.lower() for doc in canadian_government_reviews.spacy_doc for token in doc if token.pos_=='NOUN']
liberal_party_noun = [token.text.lower() for doc in liberal_party_reviews.spacy_doc for token in doc if token.pos_=='NOUN']


# In[41]:


from collections import Counter


# In[42]:


Counter(canadiangovernment_adj).most_common(10)


# In[43]:


Counter(liberal_party_noun).most_common(10)


# In[44]:


Counter(healthcare_adj).most_common(10)


# In[164]:


Counter(canadiangovernment_noun).most_common(10)


# In[ ]:




