from temporal_nest_lib import *
import math
import straph as sg
import straph.betweenness as bt
import operator
import networkx as nx
import straph.paths.meta_walks as mw
import straph.betweenness.optimal_paths as opt

import numpy

#max of a set with positive values
def max_not_inf(s):
    m = -1
    for e in s:
        if e > m and e != numpy.Infinity:
            m = e
    return m

def global_efficiency(g,p,eps = 10^(-6)):
    diameter = -1
    su = 0
    to_straph_file(g,"tmp")    
    b = operator.lt
    walk_type = "passive"
    if p == "d":
        fun =  opt.co_dur_im
    else:
        fun =  opt.co_sh_im
    S = sg.read_stream_graph(path_nodes="tmp_nodes.sg",
                          path_links="tmp_links.sg")
    event, event_reverse = bt.events_dic(S)
    link_ind = bt.link_index(S)
    neighbors, neighbors_inv = bt.neighbors_direct(S)
    for v in S.nodes:
        _, cur_best, _ = bt.dijkstra_directed_dis_gen(S, v, event, event_reverse, neighbors, neighbors_inv, link_ind, b, fun, walk_type)
#         _, cur_besttmp, _ = bt.dijkstra_directed_dis_gen(S, v, event, event_reverse, neighbors, neighbors_inv, link_ind, b, fun2, walk_type)
        for w in S.nodes:
            if v!=w:
                dvw = min(cur_best[w].values())
                
                if dvw != 0:
                    if p == "d":
                        su += (1/(dvw+1))
                    else:
                        su += (1/dvw)
                
                if dvw != numpy.Infinity:
                    if dvw > diameter:
                        diameter = dvw
    N = len(S.nodes)
    #if want to renormalize
    return  (1/(N*(N-1)))*su, diameter
    
    #return su, diameter

    

def topological_overlap(g,i,t):
    se = seq_graphs(g)
    ev = list(events(g))
    ev.sort()
    m = ev.index(t)
    V = nodes(g)
    gm = adj(se[t])
    gmp = adj(se[ev[m+1]])
    su = [0,0,0]
    for j in V:
        if (i in gm and j in gm[i]):
            su[1] += 1
        if (i in gmp and j in gmp[i]):
            su[2] += 1
        if (i in gm and j in gm[i]) and (i in gmp and j in gmp[i]):
            su[0] += 1
    if su[1] ==0 or su[2]==0:
        return 0
    else:
        return su[0]/(math.sqrt( su[1]*su[2]))
        
def average_topological_overlap(g,i):
    ev = events(g)
    m = max(ev)
    su = 0
    for t in ev:
        if t != m:
            su += topological_overlap(g,i,t)
    return (1/(m-1))*su
def average_clustering_network(g):
    V = nodes(g)
    return (1/len(V))*sum(  average_topological_overlap(g,i) for i in V  )

import itertools
import random
from scipy.stats import bernoulli
import numpy as np

# if horizon = -1 means infinite, if horizon not specified will be equal to spread rate
# either iterations is fixed or threshold if both are set whatever finishes first ends the function
def SI(g, rate,  thresh  = 0.03, horizon = "sr", rate_rec = 0, iterations = math.inf):
    #selcting random v,t
    ev = events(g)
    m = max(ev)
    V = nodes(g)
    num_thresh = int(thresh * (len(V)*len(ev)) )
    if horizon == "sr":
        lk = rate
    else:
        lk = horizon
        
    nei = all_graph_neighbour_imp(g, look_ahead = lk)
    source = random.choice(list(V))
    t = min(ev)
    while len(nei[(source,t)]) < 2:
        source = random.choice(list(V))
        t = min(ev)
        
#     print(source)
    healthy = set( [ element for element in itertools.product(list(V), list(ev))])
    healthy.remove((source,t))
    infected = {(source,t)}
#     tot = len(V)*len(ev)
    itera = 0
    capacity = 1
#     print("len inf", len(infected), "thresh", thresh)
    while len(infected) < num_thresh and  itera < iterations and capacity > 0:
#         print("len inf", len(infected), "thresh", thresh)
        current_infected = set(list(infected)[:])
#         print("current_infected", current_infected)
        capacity = diffuse_pandemic(g, nei, healthy, current_infected, infected, {}, rate, rate_rec)
        itera += 1
    return itera, len(infected)/(len(V)*len(ev)), capacity

def diffuse_pandemic(g, nei, healthy, current_infected, infected, recovered, rate_inf, rate_rec):
#     print("current_infected", current_infected)
    capacity = 0
    for v in current_infected:
#         print("nei[v]", nei[v])
        for w in nei[v]:
            if w in healthy:
                capacity += 1
                val = bernoulli.rvs(size=1,p=rate_inf)[0]
                if val == 1:
                    healthy.remove(w)
                    infected.add(w)
    for v in current_infected:
        val = bernoulli.rvs(size=1,p=rate_rec)[0]
        if val == 1:
            recovered.add(v)
            infected.remove(v) 
    return capacity
