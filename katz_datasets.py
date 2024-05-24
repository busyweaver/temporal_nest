#!/usr/bin/env python
# coding: utf-8

# In[1]:

#need to install
# https://github.com/Feelx234/nestmodel/tree/develop and
# https://github.com/Feelx234/temp-nestmodel


import graph_characteristics_lib as gc
import os
import sys
import math
import pickle
import numpy as np
from tnestmodel.temp_fast_graph import SparseTempFastGraph
from tnestmodel.temp_centralities import calc_temp_katz_iter


# In[6]:


# edges = np.array([[0, 1, 1], [1, 2, 2]])

# G = SparseTempFastGraph.from_temporal_edges(edges, is_directed=False)

# katz = calc_temp_katz_iter(G, alpha=1/2, kind="broadcast")
# katz


# In[8]:


names = [ ["dnc","dnc","crimson","d"] , ["fb-forum", "fb", "green","d"], ["talk_eo", "eo" ,"olive","d"],
          ["email-eu3", "eu3", "yellow","d"], ["email-eu2", "eu2", "magenta","d"], 
          ["racoon", "rac", "brown","u"], ["primate","pri",  "blue","u"],  ["workplace_2013", "wp", "black","u"],
           ["ht09", "ht","red","u"], ["weaver", "wea",  "pink","u"] ]


# In[16]:


def read_dic(s):
    with open(s+".pkl", 'rb') as f:
        d = pickle.load(f)
    return d

def find_sep(x):
    for s in x:
        if s not in ['0','1','2','3','4','5','6','7','8','9']:
            return s
    return None

def read_graph(path,s, dire):
    g = set()
    f = open(path+s, "r")
    sep = find_sep(f.readline())
    f.close()
    f = open(path+s, "r")
    for x in f:
        x = x[:-1]
        r = x.split(sep)
        u,v,t = r[0],r[1],r[2]
        if v != u:
            t = float(t)
            g.add( (u,v,int(t)) )
            if dire == "u":
                g.add( (v,u,int(t)) )
    return g

def SAE(s,s2,dv_rev,dv2):
    # print(s, s[0])
    # l = list(map(lambda x : x[0], s.tolist()))
    # l2 = list(map(lambda x : x[0], s2.tolist()))
    val = 0
    print("sae len(s)",len(s), "len(s2)", len(s2))
    for i in range(len(s)):
        v = dv_rev[i]
        if v in dv2.keys():
            k = dv2[v]
            val += abs( s[i] - s2[k] )
    print("val",val)
    return val/len(s)


# In[15]:


fold_d = "datasets/networks/"
folder = "values_graphs/"
dep = 5
d = dict()
for i in range(1):
    print(names[i])
#     cur = fold_d + names[i][0]+".csv"
    d[names[i][1]] = dict()
    g = read_graph(fold_d,names[i][0]+".csv", names[i][-1])
    V = list(gc.nodes(g))
    dv = dict()
    dv_rev = dict()
    for j in range(len(V)):
        dv[V[j]] = j
        dv_rev[j] = V[j]
    new_g = [  [dv[e[0]],dv[e[1]],e[2]]    for e in g ]
    print("new_g", new_g[:10])
    edges = np.array(new_g)
    if names[i][-1] == "d":
        G = SparseTempFastGraph.from_temporal_edges(edges, is_directed=True)
    else:
        G = SparseTempFastGraph.from_temporal_edges(edges, is_directed=False)
    s = calc_temp_katz_iter(G, alpha=0.01, kind="broadcast")
    print("s",s)
    z = 0
    col = None
    while z < dep+1:
        nam = folder+names[i][1]+"_1_1.0_keep_"+ str(z)
        print(nam)
        if os.path.isfile(nam+".pkl"):
            col = read_dic(nam)
            
            
        if z == 0:
            g2 = gc.randomized_edge_same_time_gen(g,names[i][-1],len(g)*math.ceil(math.log(len(g))))
        else:
            if names[i][-1] == "d":
                g2 = gc.felix_flips_imp(g,len(g)*math.ceil(math.log(len(g))),col)
            else:
                g2 = gc.rewire_any_imp(g,col,names[i][-1],len(g)*math.ceil(math.log(len(g))))

        V2 = list(gc.nodes(g2))
        dv2 = dict()
        for j in range(len(V2)):
            dv2[V2[j]] = j

        new_g = [  [dv2[e[0]],dv2[e[1]],e[2]]    for e in g2 ]
        edges = np.array(new_g)

        if names[i][-1] == "d":
            G = SparseTempFastGraph.from_temporal_edges(edges, is_directed=True)
        else:
            G = SparseTempFastGraph.from_temporal_edges(edges, is_directed=False)

        s2 = calc_temp_katz_iter(G, alpha=0.01, kind="broadcast")
        print("s2",s2)
        d[names[i][1]][z] = SAE(s,s2,dv_rev, dv2)
        
        z += 1
    


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

print("res",d)

# evenly sampled time at 200ms intervals
x = [  de for de in range(0,dep+1) ]
x.sort()
y = dict()

for n in d.keys():
    for de in d[n].keys():
        if n not in y:
            y[n] = []
        y[n].append(d[n][de])
        
# red dashes, blue squares and green triangles
for n in y.keys():
    plt.plot(x,y[n],label=n)
plt.yscale("log")
plt.xlabel("depth")
plt.ylabel("SAE")
plt.title('Katz Centrality')
plt.legend()
plt.savefig('katz.png', dpi=300)
plt.show()

