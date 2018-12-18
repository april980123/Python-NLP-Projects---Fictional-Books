#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 01:49:37 2018

@author: reneehsu
"""
"""
Import the book and a text file named “firstname.txt” that contains all the common first names in English language (downloaded online).
Then, in the “firstname txt”, we used regular expression to find all the first names.
"""
import nltk
from nameparser import HumanName
from nltk import word_tokenize, pos_tag, ne_chunk
import numpy as np

with open('firstname.txt', 'r') as myfile:
  data = myfile.read().replace('\n', '')
# Type the filename, our three files are: 'The Prisoner Of Azkaban.txt', 'Pride-and-Prejudice.txt', 
# or 'Harry Potter 2 - Chamber of Secrets.txt'
filename = 'Pride-and-Prejudice.txt'
with open(filename, 'r') as myfile:
  book = myfile.read().replace('\n', '')
import re
firstname = re.findall(r'[A-Z]+',data)
firstname.append("PROFESSOR") # to catch professor Snape for Harry Potter

"""
The function “get_human_name” has functions as such:
1.	Tokenize each word in the book, add pos_tag on them, and loop through every word with “PERSON” tag.
2.	If the word is a full name (which means it has length longer than 1), it will be included in the character name list.
3.	If the input word is just a person’s first name, it will be checked against the name file, and then it will be included in the character list if it’s a legit first name.
"""
def get_human_names(text):
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    person_list = []
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):       
        for leaf in subtree.leaves():            
            person.append(leaf[0])
        if len(person) > 1: #avoid grabbing lone surnames
            include = False
            for part in person:               
                name += part + ' '
                if part.upper() in firstname:
                    include = True
            if name[:-1] not in person_list and include == True:                
                person_list.append(name[:-1])
                
        elif len(person) == 1:
            person = ''.join(person)
            if person.upper() in firstname:
                person_list.append(person)
        name = ''
        person = []
    
    return set(person_list)
"""
Get the character names in the book using get_human_name function.
"""
names = get_human_names(book)

"""
We manually removed some of the names in the character list to make it more accurate. 
"""
if filename == 'The Prisoner Of Azkaban.txt':
    names.remove("Harry UP")
    names.remove("Bye Ron Harry")
    names.remove("Famous Harry Potter")
    names.remove("Professor")
    names.remove("Ron")
    names.remove("Parlor Ron")
    names.remove("Sirius Harry")
    names.remove("Psst Harry")
    names.remove("Dear Harry")
    names.remove("Harry")
    names.remove("Harry James Potter")
elif filename == 'Pride-and-Prejudice.txt':
    names.remove("Elizabeth Do")
    names.remove("Miss Eliza Bennet")
    names.remove("Elizabeth")
    names.remove("Miss Elizabeth")
    names.remove("Miss")
    names.remove("Had Elizabeth")
    names.remove("Between Elizabeth")
    names.remove("Eliza")
    names.remove("Miss Eliza")
    names.remove("Did Mr. Darcy")
    names.remove("Miss Elizabeth Bennet")
elif filename == 'Harry Potter 2 - Chamber of Secrets.txt':
    names.remove('Had Harry')
    names.remove('Famous Harry Potter')
    names.remove('Happy Valentine')
    names.remove('Merry Christmas')
    names.remove('So')
    names.remove('Harry')
    names.remove('Professor')

"""
The names will be stored in an ordered dictionary, with each key being the full name and the value being its corresponding first name, last name and full name. This is done to make sure that the computer will still recognize that person’s name even when only his/her first name or last name is mentioned in the book.
"""
from collections import OrderedDict
character = OrderedDict()
for name in names:
    temp = [HumanName(name).last ,HumanName(name).first]
    character[name] = temp
    character[name].append(name)
 
"""
We created a n*n matrix of 0’s with n being the number of characters in the list.
"""
n=len(character)
adjMatrix = [[0]*n for _ in range(n)]

words = nltk.tokenize.word_tokenize(book)

lastcharacter = None
lastpos = None

"""
We created a nested for loop to put the values into the matrix. The algorithm is as such:
1.	Iterate through all the words in the book. If that word is in the values of the character dictionary (meaning if it’s a person’s name), update the position in the book that we’ve already searched, and the position of the “last character” found in the dictionary.
2.	for the next person’s name found, check if that person’s name is different as the last character’s name and if they have a distance smaller than 50 words in the book. If these two conditions are satisfied, update the value in the matrix by adding 1 to those two people’s connections. This is done to make sure that the distance between any two characters is close enough to be considered as “connections”. The positions values are updated as well.
3.	in the case where the two character have the same name, meaning the same person is mentioned twice in the range of 50 words, update the positions value but the values in the matrix are not changed.
"""
for p in range(len(words)):
    for i, (key, value) in enumerate(character.iteritems()):        
        if words[p] in value: 
            if lastcharacter == None:
                lastcharacter = i
                lastpos = p
                break
            elif key != character.keys()[lastcharacter] and p - lastpos < 50:
                adjMatrix[lastcharacter][i]+=1
                adjMatrix[i][lastcharacter]+=1
                lastcharacter = i
                lastpos = p
                break
            elif key == character.keys()[lastcharacter]:
                lastcharacter = i
                lastpos = p

#plotting
"""
Use networkx DiGraph to plot the graph. To make the graph more clean and easy-to-read, only those connections that have value larger than 5 are shown. Then sort the page rank value for each node and only picked out the nodes with top 40 page ranks (meaning if more than one node have the same Pagerank value, there could be more than 40 nodes presented) to be shown in the graph. Top 10 characters are shown in red and the rest characters are shown in yellow. 
"""
import matplotlib.pyplot as plt 
import networkx as nx
import numpy as np
adjMatrix = np.array(adjMatrix)
G = nx.DiGraph()
G.add_nodes_from(character.keys())
for i in range(adjMatrix.shape[0]):
    for j in range(adjMatrix.shape[1]): 
        if adjMatrix[i][j] >= 5:  
            G.add_edge(character.keys()[i],character.keys()[j],weight=adjMatrix[i][j])

pr = nx.pagerank(G, alpha=0.85) # the default damping parameter alpha = 0.85

prv = pr.values()
prv.sort()
filter_pagerank_value = prv[-40]
topten_pagerank_value = prv[-10]

color_map = []
    
for char in character.keys():
    if pr[char] < filter_pagerank_value:
        G.remove_node(char)
for char in G.nodes():
    if  pr[char] >= topten_pagerank_value:
        color_map.append('red')
    else:
        color_map.append('yellow')

plt.figure(3,figsize=(12,12))
pos = nx.spring_layout(G,k = 1.5, iterations = 20)
nx.draw(G,with_labels=True, font_size = 17, node_color = color_map, pos = pos)

plt.show()    