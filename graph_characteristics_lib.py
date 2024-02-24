from temporal_nest_lib import *
import math
import straph as sg
import straph.betweenness as bt
import operator
import networkx as nx
import straph.paths.meta_walks as mw
import straph.betweenness.optimal_paths as opt
import random
import numpy

#max of a set with positive values
def max_not_inf(s):
    m = -1
    for e in s:
        if e > m and e != numpy.Infinity:
            m = e
    return m

def global_efficiency(g,p,approx):
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
    lS = list(S.nodes)
    event, event_reverse = bt.events_dic(S)
    link_ind = bt.link_index(S)
    neighbors, neighbors_inv = bt.neighbors_direct(S)
    sam = random.sample(lS, k = int(approx*len(lS)))
    for v in sam:
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
    N = len(sam)
    #print("N", N)
    #if want to renormalize
    if N > 1:
        return  (1/(N*(N-1)))*su, diameter
    else:
        return  su, diameter
    #return su, diameter


def topological_overlap(g,i,t,ev,se,V):
    m = ev.index(t)
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
def average_topological_overlap(g,i,ev,se,V):
    #max value
    m = ev[-1]
    su = 0
    for t in ev:
        if t != m:
            su += topological_overlap(g,i,t,ev,se,V)
    return (1/(m-1))*su
def average_clustering_network(g, approx):
    V = list(nodes(g))
    ev = list(events(g))
    ev.sort()
    se = seq_graphs(g)
    sam = random.sample(V, k = int(approx*len(V)))
    if len(sam) == 0:
        return 0
    return (1/len(sam))*sum(  average_topological_overlap(g,i,ev,se,V) for i in sam )

import itertools
import random
from scipy.stats import bernoulli
import numpy as np

# if horizon = -1 means infinite, if horizon not specified will be equal to spread rate
# either iterations is fixed or threshold if both are set whatever finishes first ends the function
def SI(g, rate,  thresh  = 0.03, horizon = "sr", rate_rec = 0, iterations = math.inf):
    #selcting random v,t
    ev = events(g)
    l_ev = list(ev)
    m = l_ev[-1]
    t = l_ev[0]
    V = nodes(g)
    lV = list(V)
    num_thresh = int(thresh * (len(V)*len(ev)) )
    if horizon == "sr":
        lk = rate
    else:
        lk = horizon
    look_ahead_num = int( lk * m )
    nei = all_graph_neighbour_node(g, V, l_ev, look_ahead = lk)
    #nei = all_graph_neighbour_imp(g, look_ahead = lk)
    possible_start = []
    for v in lV:
        if len(neighb(nei[v],t, look_ahead_num)) >= 1:
            possible_start.append(v)

    if len(possible_start) == 0:
        print("no possible pandemy")
        return 0,0,0

    source = random.choice(possible_start)
#     print(source)
    healthy = set( [ element for element in itertools.product(lV, l_ev)])
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
        capacity = diffuse_pandemic(g, nei, healthy, current_infected, infected, {}, rate, rate_rec, look_ahead_num)
        itera += 1
    return itera, len(infected)/(len(lV)*len(l_ev)), capacity

def diffuse_pandemic(g, nei, healthy, current_infected, infected, recovered, rate_inf, rate_rec, lkn):
#     print("current_infected", current_infected)
    capacity = 0
    for v in current_infected:
#         print("nei[v]", nei[v])
        for w in neighb(nei[v[0]],v[1], lkn): 
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
