from __future__ import print_function
from time import time
import xml.etree.ElementTree as ET
import os,sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import json

n_topics = 10 #number of distributions
n_top_words = 20 #number of terms extracted in each distribution 

#helper function
def print_feature_names(feature_names):
    for f in feature_names:
        print(f)

#helper function to find term index in given vocabulary        
def find_term_idx(term,feature_names):
    for idx, f in enumerate(feature_names):
        if(f==term):
            return idx
    
#helper function to print most commonly co-occurred terms (with score) in each distribution    
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([str(topic[i])
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    

#helper function to find the distribution in which the target term is mostly "related" 
#and get the top terms in that distribution
def find_most_relevant(model, target_term_idx, ignore_idx):
    max_score = 0.0
    max_idx = -1
    
    for topic_idx, topic in enumerate(model.components_):
        if(topic[target_term_idx]>max_score and topic_idx != ignore_idx):
            max_score = topic[target_term_idx]
            max_idx = topic_idx
    
    return max_idx


def get_top_terms(model, feature_names, n_top_words, topic_idx):
    max_topic = model.components_[topic_idx]
    #relevant_terms = " ".join([feature_names[i] for i in max_topic.argsort()[:-n_top_words - 1:-1]])
    #print("Most Relevant Topic #%d:%s" % (max_idx,relevant_terms))
    relevant_terms = set()
    for i in max_topic.argsort()[:-n_top_words - 1:-1]:
        relevant_terms.add(feature_names[i])
    return relevant_terms
    
#helper function to convert a set object to list that can be directly json encoded   
def set_convert_to_json_list(terms_set):
    json_term_list = list()
    for term in terms_set:
        json_term_list.append({"name":term})
    return json_term_list

#helper function to convert a dict object to list that can be directly json encoded
def dict_convert_to_json_list(terms_dict):
    json_term_list = list()
    for key in terms_dict:
        json_term_list.append({"name":key, "children":set_convert_to_json_list(terms_dict[key])})
    return json_term_list
    

#exit if no user term is provided    
if(len(sys.argv)!=2):
    print ("Invalid input...exit")
    exit()
    
user_term = sys.argv[1]

#load the PMC dataset
print("Loading dataset...")
t0 = time()

data_samples = list()
for folder, dirs, files in os.walk('../data/'):
    for f in files:
        if f.endswith('.nxml'):
            tree = ET.parse(os.path.join(folder,f))
            root = tree.getroot()
            data_samples.append(' '.join(root.itertext())) #each document is represented by text content from that xml file
            
print("done in %0.3fs." % (time() - t0))
print("%d lines read" % len(data_samples))

#use tf (raw term count) features for LDA.
print("Extracting tfidf features for LDA...")
#max_df=0.95: when building the vocabulary ignore terms that have a document frequency strictly > 95% of documents
#min_df=2: when building the vocabulary ignore terms that have a document frequency strictly < 2 
#stop_words='english': a built-in stop word list for English is used
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))


print("Fitting LDA models with tfidf features")
#learning_method='online': method used to update _component
#in general, if the data size is large, the online update will be much faster than the batch update
#'online': Online variational Bayes method.

#learning_offset=50.:learning rate, a (positive) parameter that downweights early iterations in online learning
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tfidf)
print("done in %0.3fs." % (time() - t0))


tf_feature_names = tfidf_vectorizer.get_feature_names()
#for debugging purposes
#print("\nFeature Names in LDA model:")
#print_feature_names(tf_feature_names)
print("\nTopics in LDA model:")
print_top_words(lda, tf_feature_names, n_top_words)


#to extract terms(topics) that most commonly co-occurred with the user provided term (categorized in 2 levels)

#set containing terms already shown in previous levels
shown_terms = set()
shown_terms.add(user_term)
user_term_idx = find_term_idx(user_term,tf_feature_names)
#l1_relevant_terms meaning the level 1 topics that are mostly commonly co-occurred
l1_relevant_distribution_idx = find_most_relevant(lda, user_term_idx, -1)
l1_relevant_terms = get_top_terms(lda, tf_feature_names, n_top_words, l1_relevant_distribution_idx)
l1_relevant_terms = l1_relevant_terms.difference(shown_terms)
#update shown_terms set for l2_relevant_terms
shown_terms = shown_terms.union(l1_relevant_terms)
l2_relevant_terms = dict()
#level 2 topics are grouped by level 1 topics as keys
for term in l1_relevant_terms:
    term_idx = find_term_idx(term,tf_feature_names)
    tmp_distribution_idx = find_most_relevant(lda, term_idx, l1_relevant_distribution_idx)
    l2_relevant_terms[term] = get_top_terms(lda, tf_feature_names, n_top_words, tmp_distribution_idx)
    l2_relevant_terms[term] = l2_relevant_terms[term].difference(shown_terms)
    
#convert obj to json file
model = dict()
model["name"] = user_term
model["children"] = dict_convert_to_json_list(l2_relevant_terms)


output = open('../output/flare_v3_10.json','w')
output.write(json.dumps(model,indent=1)) # python will convert \n to os.linesep
output.close()




