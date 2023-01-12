#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:09:56 2022

@author: fpannach
"""
import rdflib
from rdflib import Graph, URIRef
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from rdflib.namespace import RDF, RDFS, OWL
import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Namespace
import os 
import numpy as np
import pandas as pd 
import seaborn as sns
import io
import pydotplus
from IPython.display import display, Image
from rdflib.tools.rdf2dot import rdf2dot
#%%
# gcdh = Namespace("http://teaching.gcdh.de/")

# graph_a = Graph()
# graph_a.bind("gcdh", "http://teaching.gcdh.de/")
# ### change this path
# graph_a.parse('/home/fpannach/PhD/Code//Hyleme_Onto/Mythos_Ontology/hylva/Domain_Ontologies/Dumuzi/Das_Versprechen_der_Fliege.ttl', format='ttl')

# graph_b = Graph()
# graph_b.bind("gcdh", "http://teaching.gcdh.de/")

# ### change this path
# graph_b.parse('/home/fpannach/PhD/Code/Hyleme_Onto/Mythos_Ontology/hylva/Domain_Ontologies/Orpheus/Orpheus_AB_3_2.ttl', format='ttl')

# graph_merged = Graph()
# graph_merged.parse('/home/fpannach/PhD/Code/Hyleme_Onto/Mythos_Ontology/hylva/Domain_Ontologies/Orpheus/Orpheus_MA_1.ttl', format='ttl')
# graph_merged.parse('/home/fpannach/PhD/Code//Hyleme_Onto/Mythos_Ontology/hylva/Domain_Ontologies/Orpheus/Orpheus_AB_32.ttl', format='ttl')

#%%
def remove_diagonal(df) :
    for column in df.columns :
        for index in df.index :
            if index == column : 
                df[column][index] = 0
    
    return df

def find_max_list(list):
    list_len = [len(i) for i in list]

    return max(list_len)

def depth(graph):
    indivs = get_classes(graph1, OWL.NamedIndividual)
    depth = 0 
    for indiv in indivs : 
        longest_indiv = find_max_list(find_indiv_classes(graph, indiv))
        if longest_indiv > depth : 
            depth = longest_indiv

    return depth 

#%%
def merge_graphs(input1, input2): 
    graph_merged = Graph()
    graph_merged.parse(input1)
    graph_merged.parse(input2)

def get_identifiers(graph, indivs) :
    inarray = np.array(indivs)

    for (s, p, value) in graph:
       #gcdh.WikidataID works initially but reverts to rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type') after code is run
        if s in indivs :
            if p == rdflib.term.URIRef("http://teaching.gcdh.de/WikidataID") :
                inarray =  np.where(inarray == s, value, inarray)
            elif p == rdflib.term.URIRef("http://teaching.gcdh.de/PleiadesID") :
                inarray =  np.where(inarray == s, value, inarray)

    return list(inarray)

def get_identifier(graph, indiv) :

    value = graph.value(subject=rdflib.term.URIRef(indiv), predicate=rdflib.term.URIRef("http://teaching.gcdh.de/WikidataID"), object=None, default=None, any=True)
    if not value :
        value = graph.value(subject=rdflib.term.URIRef(indiv), predicate=rdflib.term.URIRef("http://teaching.gcdh.de/PleiadesID"), object=None, default=None, any=True)

    return str(value)


def get_classes(graph, POS) :
    classes = []
    for subject in set(list(graph.subjects())):
        if (subject, RDF.type, POS) in graph:
            classes.append(subject)

    return classes

#%%
def calculate_overlap_score(list1, list2):
    shared = [x for x in list1 if x in list2]

    return len(list(shared))/len(list(set(list1+list2)))

def get_overlap(graph1, graph2):
    return get_class_overlap(graph1, graph2), get_indiv_overlap(graph1, graph2)

def get_class_overlap(graph1, graph2):
    cl_1 = list(get_classes(graph1, OWL.Class))
    cl_2 = list(get_classes(graph2, OWL.Class))

    return calculate_overlap_score(cl_1, cl_2)

def get_matches(graph1, graph2):
    cl_1 = get_classes(graph1, OWL.NamedIndividual)
    cl_2 = get_classes(graph2, OWL.NamedIndividual)
    matches, relation_matches = find_similar_entities(graph1, graph2, cl_1, cl_2)
    #print("Most similar individuals:")
    #print("Individual:", x, "ist ein/e ", klasse, "und tut:", Property, "mit" xy)
    return matches, relation_matches

def get_indiv_overlap(graph1, graph2):
    cl_1 = get_classes(graph1, OWL.NamedIndividual)
    cl_2 = get_classes(graph2, OWL.NamedIndividual)
    wd_id_1 = get_identifiers(graph1, cl_1)
    wd_id_2 = get_identifiers(graph2, cl_2)

    return calculate_overlap_score(wd_id_1, wd_id_2)

#%%
#lowest common class
#beispiel volksmenge/Zuhörer und Hörer_von_Orpheus_Musik

def find_ancestors(graph, classes): 
    ancestors = []
    for cl in classes : 
        for(s, p, o) in graph : 
            if s == rdflib.term.URIRef(cl) and p == RDFS.subClassOf :  #<-- RDFS.subClassOf 
                ancestors.append(o)

    return ancestors

def find_ancestor_relations(graph, classes): 
    ancestors = []
    for cl in classes :
        for(s, p, o) in graph : 
            if s == rdflib.term.URIRef(cl) and p == RDFS.subPropertyOf :  #<-- RDFS.subPropertyOf 
                ancestors.append(o)

    return ancestors


def find_indiv_classes(graph, indiv) :
    classes = []
    for (s, p, o) in graph:
        if s == rdflib.term.URIRef(indiv) and p == RDF.type :
            subclasses = []
            subclasses.append(o)
            children = [o]
            while find_ancestors(graph, children) : 
                children = find_ancestors(graph, children)
                for child in children : 
                    subclasses.append(child)
            classes.append(subclasses)

    return classes

def find_indiv_relations(graph, indiv) :
    relations = []
    for (s, p, o) in graph.triples((rdflib.term.URIRef(indiv), None, None)):
        if not p == RDF.type :
            subrelations = []
            subrelations.append(p)
            children = [p]
            while find_ancestor_relations(graph, children) : 
                children = find_ancestor_relations(graph, children)
                for child in children : 
                    subrelations.append(child)
            relations.append(subrelations)

    return relations


def find_common_relations(graph1, graph2, indiv1, indiv2): 
    cl_1 = find_indiv_relations(graph1, indiv1)
    cl_2 = find_indiv_relations(graph2, indiv2)
    plen,intersect = lcsubpath_length(cl_1, cl_2)
    # plen/depth(graph1)+depth(graph2) how to normalize this? 
    return plen

def find_common_indiv_classes(graph1, graph2, indiv1, indiv2): 
    cl_1 = find_indiv_classes(graph1, indiv1)
    cl_2 = find_indiv_classes(graph2, indiv2)
    plen,intersect = lcsubpath_length(cl_1, cl_2)
    # plen/depth(graph1)+depth(graph2) how to normalize this? 

    return plen

def match_indiv(graph1, graph2, indivs1, indivs2) : 
     matches = {}
     matches_rl = {}
     if len(indivs1) >= len(indivs2) : 
         src = indivs1
         src_graph = graph1
         trg = indivs2
         trg_graph = graph2
     else :
         src = indivs2
         src_graph = graph2
         trg = indivs1
         trg_graph = graph1

     for src_ind in src: 
         matches[src_ind] = {}
         matches_rl[src_ind] = {}

         for trg_ind in trg :
             no_common_classes = find_common_indiv_classes(src_graph, trg_graph, src_ind, trg_ind)
             no_common_relations = find_common_relations(src_graph, trg_graph, src_ind, trg_ind)
             if no_common_classes > 1 :
                 matches[src_ind][trg_ind] = no_common_classes
             if no_common_relations > 2 :  #Damit sowas wie 
                 matches_rl[src_ind][trg_ind] = no_common_relations

     return matches, matches_rl

#find the longest matching path
# longest = most specialised e.g. husband vs marriage partner = more similar 

def filter_identifiers(graph1, graph2, indivs1, indivs2):
    indivs1_fltrd = indivs1.copy()
    indivs2_fltrd = indivs2.copy()

    for indiv1 in indivs1 :
        ident1 = get_identifier(graph1, indiv1)
        for indiv2 in indivs2 :
            ident2 = get_identifier(graph2, indiv2)
            if ident1 == ident2:
                indivs1_fltrd.remove(indiv1)
                indivs2_fltrd.remove(indiv2)

    return indivs1_fltrd, indivs2_fltrd

def filter_aliasses(graph1, graph2, indivs1, indivs2):
    #deep copy of indivs
    indivs1_fltrd = indivs1.copy()
    indivs2_fltrd = indivs2.copy()

    for indiv1 in indivs1 :
        alias1 = graph1.value(subject=rdflib.term.URIRef(indiv1), predicate=rdflib.term.URIRef("http://teaching.gcdh.de/Alias"), object=None, default=None, any=True)
        pref_label1 = graph1.value(subject=rdflib.term.URIRef(indiv1), predicate=RDFS.label, object=None, default=None, any=True)
        for indiv2 in indivs2 : 
            alias2 = graph2.value(subject=rdflib.term.URIRef(indiv2), predicate=rdflib.term.URIRef("http://teaching.gcdh.de/Alias"), object=None, default=None, any=True)
            pref_label2 = graph2.value(subject=rdflib.term.URIRef(indiv2), predicate=RDFS.label, object=None, default=None, any=True)
#            print(pref_label1, alias2)
            if alias1 or alias2 :
                if str(alias1) == str(alias2) :
                    indivs1_fltrd.remove(rdflib.term.URIRef(indiv1))
                    indivs2_fltrd.remove(rdflib.term.URIRef(indiv2))
                    #doesnt work 
                    #indivs1_fltrd.remove(indiv1)
                    #indivs2_fltrd.remove(indiv2)
                    
                elif str(alias1) == str(pref_label2) : 
#                    print(alias1, pref_label2)

                    indivs1_fltrd.remove(rdflib.term.URIRef(indiv1))
                    indivs2_fltrd.remove(rdflib.term.URIRef(indiv2))
                    #indivs1_fltrd.remove(indiv1)
                    #indivs2_fltrd.remove(indiv2)
                elif str(alias2) == str(pref_label1) : 
                    indivs1_fltrd.remove(rdflib.term.URIRef(indiv1))
                    indivs2_fltrd.remove(rdflib.term.URIRef(indiv2))
                    #indivs1_fltrd.remove(indiv1)
                    #indivs2_fltrd.remove(indiv2)

    return indivs1_fltrd, indivs2_fltrd


def filter_lists(indivs1, indivs2) :
      tmp = indivs1
      indivs1 = [indiv for indiv in indivs1 if not indiv in indivs2]
      indivs2 = [indiv for indiv in indivs2 if not indiv in tmp]

      return indivs1, indivs2

def indiv_matching(graph1, graph2, indivs1, indivs2) :
      #select indiv that do not match by ID or Name/Alias
      indivs1, indivs2 = filter_lists(indivs1, indivs2)
      indivs1, indivs2 = filter_aliasses(graph1,graph2, indivs1, indivs2)
      indivs1, indivs2 = filter_identifiers(graph1, graph2, indivs1, indivs2)

      #take the wikidata ones out also
      return match_indiv(graph1, graph2, indivs1, indivs2)

def find_similar_entities(graph1, graph2, cl_1, cl_2): 
    return indiv_matching(graph1, graph2, cl_1, cl_2)

 #%%
def lcsubpath_length(a, b):
    l = 0
    for lst1 in a : 
        for lst2 in b : 
            intersect = [value for value in lst1 if value in lst2]
            length = len(intersect)
            if length > l : 
                l = length
    return l, intersect

#remove indivs matched by ID

#%%
dir = os.path.dirname(__file__) 
rel_path = "Orpheus/"
file_path = os.path.join(dir, rel_path)

columns = [filename.replace("Orpheus_", "").replace(".ttl","") for filename in os.listdir(file_path)]
df_cl = pd.DataFrame(columns=columns, index=columns)
df_ind = pd.DataFrame(columns=columns, index=columns)

#%% 
def plot_heatmap(df, mode, title="", ) :
    sns.set_theme(style="whitegrid")
    
    m = np.tril(df)
    f, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(title)
    x_axis_labels = df.index
    y_axis_labels = df.columns # labels for y-axis
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    mask = np.triu(np.ones_like(m, dtype=bool))
    if mode == 'diagonal' : 
        sns.heatmap(m, xticklabels=x_axis_labels, yticklabels=y_axis_labels, mask=mask, cmap=cmap, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    else :
        sns.heatmap(df, xticklabels=x_axis_labels, yticklabels=y_axis_labels, vmax=1.0, cmap=cmap, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
#%%
for filename1 in os.listdir(file_path):
    column = filename1.replace("Orpheus_", "").replace(".ttl","")
    clss = []
    inds = []
    matches = []
    f = os.path.join(file_path, filename1)
    print("Comparing", filename1)
    with open(f) as file :
        graph1 = Graph()
        graph1.parse(file, format='ttl')
        for filename2 in os.listdir(file_path):
            print("to", filename2)
            f = os.path.join(file_path, filename2)
            with open(f) as file :
                graph2 = Graph()
                graph2.parse(file, format='ttl')
                class_ovl, ind_ovl = get_overlap(graph1, graph2)
                #clss.append(class_ovl)
                #inds.append(ind_ovl)
                #mtc = get_matches(graph1, graph2)
                #matches.append(mtc)
                clss.append(get_overlap(graph1, graph2)[0])
                inds.append(get_overlap(graph1, graph2)[1])

    df_cl[column], df_ind[column] = clss, inds
                #IND.append(get_overlap(graph1, graph2)[1])
   # CL_all.append(CL)
   # IND_all.append(IND)
#%%
plot_heatmap(df_cl, 'full', 'Orpheus Class Overlap')
df_cl_max = remove_diagonal(df_cl)
for column in df_cl.columns: 
    max_cl = df_cl_max[column].max(axis=0)
    df_cl_max[column].values[df_cl_max[column].values < max_cl] = 0
plot_heatmap(df_cl_max, 'full', 'Orpheus  Max Class Overlap')

#%%
plot_heatmap(df_ind, 'full', 'Orpheus Individuals Overlap')
df_ind_max = remove_diagonal(df_ind)
for column in df_ind.columns: 
    max_ind = df_ind_max[column].max(axis=0)
    df_ind_max[column].values[df_ind_max[column].values < max_ind] = 0
plot_heatmap(df_ind_max, 'full', 'Orpheus Max Individuals Overlap')

#%% DUMUZI
rel_path = "Dumuzi/"
file_path = os.path.join(dir, rel_path)

columns = os.listdir(file_path)
df_cl_dm = pd.DataFrame(columns=columns, index=columns)
df_ind_dm = pd.DataFrame(columns=columns, index=columns)

for filename1 in os.listdir(file_path):
    #column = filename1.replace("Orpheus_", "").replace(".ttl","")
    column = filename1
    clss = []
    inds = []
    matches = []
    f = os.path.join(file_path, filename1)
    print("Comparing", filename1)
    with open(f) as file :
        graph1 = Graph()
        graph1.parse(file, format='ttl')
        for filename2 in os.listdir(file_path):
            print("to", filename2)
            f = os.path.join(file_path, filename2)
            with open(f) as file :
                graph2 = Graph()
                graph2.parse(file, format='ttl')
                class_ovl, ind_ovl = get_overlap(graph1, graph2)
                #clss.append(class_ovl)
                #inds.append(ind_ovl)
                #mtc = get_matches(graph1, graph2)
                #matches.append(mtc)
                clss.append(get_overlap(graph1, graph2)[0])
                inds.append(get_overlap(graph1, graph2)[1])

    df_cl_dm[column], df_ind_dm[column] = clss, inds
plot_heatmap(df_cl_dm, 'full', 'Dumuzi Class Overlap')

#%%
df_cl_dm_max = remove_diagonal(df_cl_dm)
#%%
for column in df_cl_dm.columns: 
    max_cl = df_cl_dm_max[column].max(axis=0)
    df_cl_dm_max[column].values[df_cl_dm_max[column].values < max_cl] = 0
plot_heatmap(df_cl_dm_max, 'full', 'Dumuzi  Max Class Overlap')

#%%
plot_heatmap(df_ind_dm, 'full', 'Dumuzi Individuals Overlap')
df_ind_dm_max = remove_diagonal(df_ind_dm)
for column in df_ind_dm.columns:
    max_ind = df_ind_dm_max[column].max(axis=0)
    df_ind_dm_max[column].values[df_ind_dm_max[column].values < max_ind] = 0
plot_heatmap(df_ind_dm_max, 'full', 'Dumuzi Max Individuals Overlap')

#%%
names = ["AB_3_2", "BCP_312", "FM_3", "HC_311", "HF_7", "KD_45", "MA_1", "MA_5", "MV_i", "OM_10", "P_9", "PS_179"]
dates = [150, 523, 550, -23, 300, 0, 25, 25, 975, 8, 150, -400]
df_time = pd.DataFrame(columns=names, index=names)

for date, title in zip(dates, names) : 
    column = [abs(date-x) for x in dates]
    df_time[title] = column

#plot_heatmap(df_time, 'full', 'Orpheus Time Distance')

# %%
group1 = ["HF_7", "PS_179"] # before 0 
group2 = ["HC_311", "KD_45", "MA_1", "MA_5", "OM_10"] # around 0 
group3 = ["BCP_312", "FM_3", "MV_i","AB_3_2", "P_9"]
#group3 = ["AB_3_2", "P_9"] # 1-2 jhd
#group4 = ["BCP_312", "FM_3", "MV_i"]#later



def traverse(lst, df):
    score = 0
    counter = 0
    for i in range(len(lst)):
        for j in range(len(lst[i+1:])):
            print(lst[i], lst[j+i+1])
            score += df[lst[i]][lst[j+i+1]] #TODO: replace with scoring funciton
            counter += 1
    return score/counter

group1_sim = traverse(group1, df_cl)
group2_sim = traverse(group2, df_cl)
group3_sim = traverse(group3, df_cl)
#group4_sim = traverse(group4, df_cl)
#%%
import random 

def calculate_random_sample(olist):
    rem_list = olist
    sampled_list1 = random.sample(olist, 5)
    rem_list = [x for x in olist if x not in sampled_list1]
    sampled_list2 = random.sample(rem_list, 5)
    rem_list = [x for x in rem_list if x not in sampled_list2]
    
    return sampled_list1, sampled_list2, rem_list

rand_list1,rand_list2, rem = calculate_random_sample(names)
random_group1_sim = traverse(rand_list1, df_cl)
random_group2_sim = traverse(rand_list2, df_cl)
random_group3_sim = traverse(rem, df_cl)
#%%
import numpy as np
from scipy.stats import ttest_ind, describe, kstest, normaltest, ttest_1samp


v1 = [group1_sim, group2_sim, group3_sim]
v2 = [random_group1_sim, random_group2_sim, random_group3_sim]

mean_pop = 0.7

t_stat, p_value = ttest_ind(v1, v2)

print(t_stat, p_value)   #idk why this is a list
#%%
print(describe(v1))  #skewed right, heavy tailed to normal distribution (kurtosis)
print(describe(v2))  #skewed right, heavy tailed to normal distribution (kurtosis)


#%%
# import matplotlib.pyplot as plt

# data =

# x_val = [x[0] for x in data]
# y_val = [x[1] for x in data]

# print x_val
# plt.plot(x_val,y_val)
# plt.plot(x_val,y_val,'or')
# plt.show()