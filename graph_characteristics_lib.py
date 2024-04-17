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
import fibheap as fib
import subprocess

#max of a set with positive values
def max_not_inf(s):
    m = -1
    for e in s:
        if e > m and e != numpy.Infinity:
            m = e
    return m


def global_efficiency(g,p,approx = 1):
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
    #print("ids", S.node_to_label)
    #print("node", S.nodes, "edges", S.links, "presence", S.link_presence)
    event, event_reverse = bt.events_dic(S)
    link_ind = bt.link_index(S)
    neighbors, neighbors_inv = bt.neighbors_direct(S)
    #print("neigbour", neighbors)
    sam = random.sample(lS, k = int(approx*len(lS)))
    for v in sam:
        _, cur_best, _ = bt.dijkstra_directed_dis_gen(S, v, event, event_reverse, neighbors, neighbors_inv, link_ind, b, fun, walk_type)
#         _, cur_besttmp, _ = bt.dijkstra_directed_dis_gen(S, v, event, event_reverse, neighbors, neighbors_inv, link_ind, b, fun2, walk_type)
        for w in S.nodes:
            if v!=w:
                #print("cur_best[w]",w,cur_best[w])
                dvw = min(cur_best[w].values())
                #print("v", v, "w", w, "dvw", dvw)
                if dvw != 0:
                    if p == "d":
                        su += (1/(dvw+1))
                    else:
                        su += (1/dvw)
                if dvw != numpy.Infinity:
                    if dvw > diameter:
                        diameter = dvw
    return  su, diameter

def read_file_cpp(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=0):
            try:
                value = int(line.strip())
                if value != 2147483647:
                    result_dict[line_number] = value
                else:
                    result_dict[line_number] = numpy.Infinity
            except ValueError:
                print(f"Skipping line {line_number}: '{line.strip()}' is not a valid integer.")
    return result_dict

def global_efficiency_cpp(g,p):
    V = nodes(g)
    ev = list(events(g))
    ev.sort()
    se = seq_graphs(g)
    diameter = -1
    su = 0
    d = to_himmel(g,"tmp")
    d_rev = { d[v]:v   for v in d.keys()   }
    for v in V:
        if p == "d":
            subprocess.run(["./implementation/tempath", "-e", "tmp.csv", "-s", str(d[v]), "-a", "fastest"])
        else:
            subprocess.run(["./implementation/tempath", "-e", "tmp.csv", "-s", str(d[v]), "-a", "hopcount"])

        cur_tmp = read_file_cpp("pipe_cpp_python")
        cur_best = { d_rev[i]:cur_tmp[i] for i in cur_tmp.keys()    }
        #print(cur_tmp)
        #print(cur_best)
        for w in V:
            if v!=w:
                dvw = cur_best[w]
                #print("v", v, "w", w, "dvw", dvw)
                if p == "d":
                    su += (1/(dvw+1))
                else:
                    su += (1/dvw)
                if dvw != numpy.Infinity:
                    if dvw > diameter:
                        diameter = dvw
    subprocess.run(["rm", "pipe_cpp_python"])
    subprocess.run(["rm", "tmp.csv"])
    return  su, diameter

def global_efficiency_imp(g,p):
    V = nodes(g)
    ev = list(events(g))
    ev.sort()
    se = seq_graphs(g)
    diameter = -1
    su = 0
    if p == "d":
        delta =  [1,0]
    else:
        delta =  [0,1]

    for v in V:
        cur_best = optimal(g,v,delta, V, ev, se)
        for w in V:
            if v!=w:
                dvw = cur_best[w]
                #print("v", v, "w", w, "dvw", dvw)
                if p == "d":
                    su += (1/(dvw+1))
                else:
                    su += (1/dvw)
                if dvw != numpy.Infinity:
                    if dvw > diameter:
                        diameter = dvw
    return  su, diameter

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
def average_clustering_network(g, approx = 1):
    V = list(nodes(g))
    ev = list(events(g))
    ev.sort()
    se = seq_graphs(g)
    sam = random.sample(V, k = int(approx*len(V)))
    if len(sam) == 0:
        return 0
    return (1/len(sam))*sum(  average_topological_overlap(g,i,ev,se,V) for i in sam )

def average_clustering_network_imp(g):
    V = list(nodes(g))
    ev = list(events(g))
    ev.sort()
    ma = ev[-1]
    se = seq_graphs(g)
    su = 0
    for t in range(len(ev)):
        if t != len(ev) - 1:
            nb = { ii : 0 for ii in V }
            adj1 = adj(se[ev[t]])
            adj2 = adj(se[ev[t+1]])
            for (i,j) in se[ev[t]]:
                if i in adj2 and j in adj2[i]:
                    nb[i] += 1
            for ii in nb.keys():
                if nb[ii] != 0:
                    nb[ii] = nb[ii]/( math.sqrt(len(adj1[ii])*len(adj2[ii])))
            su += sum(nb.values())
    return su
    #return su/( len(V)*(len(ev)-1))


# The graph is already instanteneous
#delta has 2 values corresponding to delta_3 and delta_6 in the paper

import bisect
#bisect.insort(numbers, element)

def deleteRedundant(l):
    i = 0
    tmp = len(l)
    while i < tmp - 1:
        ti = l[i]
        tj = l[i+1]
        ai, opti = ti
        aj, optj = tj
        if aj <= ai:
            l.pop(i)
            i -= 1
        i += 1
        tmp = len(l)
    return l


def optimal(g,s,delta, V, ev, se):
    T = ev[-1]
    opt = {  v:numpy.Infinity for v in V }
    L = {  v : [] for v in V }
    for t in ev:
        #print("time", t)
        nod, gt, er, dt, dr = generateGraph(se[t], s, t, T, delta, L)
        Vp, optt = modDijkstra(s, nod, gt, er, dt, dr)
        for v in Vp:
            opt[v] = min( opt[v] , delta[0]*(t - T) + optt[v])
            bisect.insort(L[v], (t, optt[v]))
            #print("beofre delete",v, L[v])
            deleteRedundant(L[v])
            #print("after delete", v,L[v])
    return opt

def generateGraph(g, s, t, T, delta, L):
    #print("generateGraph(g, s, t, T, delta, L)", g, s, t, T, delta, L)
    Er = set()
    nod = node_static(g)
    nod.update([s])
    # dr = {  (v,w) : numpy.Infinity   for v in nod for w in nod}
    # dt = {  (v,w) : numpy.Infinity   for v in nod for w in nod}
    dr = dict()
    dt = dict()
    for (v,w) in g:
        if v==s:
            dt[(v,w)] = delta[0]*(T-t) + delta[1]
        else:
            dt[(v,w)] = delta[1]
    for v in nod:
        if v != s:
            # following not necessary since no maximal waiting time is fixed
            # if len(L[v]) != 0:
            #     print("v", v,  "L[v]", L[v], "t", t)
            #     i = 0
            #     (a,opta) = L[v][i]
            #     while a < t:
            #         L[v].pop(i)
            #         if len(L[v]) == 0:
            #             break
            #         (a,opta) = L[v][i]
            if len(L[v]) != 0:
                Er.add((s,v))
                #print("opt", L[v])
                opt = min( map(lambda x : x[1], L[v]  )  )
                #print("opt = ", opt)
                dr[(s,v)] = opt
    # i dont think we have to return nod as well
    return nod, g, Er, dt, dr


def modDijkstra(s, vt, gt, er, dt, dr):
    # print("modDijkstra(s, vt, gt, er, dt, dr)", s, vt, gt, er, dt, dr)
    optt = { v: numpy.Infinity for v in vt   }
    r = { v: numpy.Infinity for v in vt }
    r[s] = 0
    Vp = set()
    Q = fib.FibonacciHeap()
    nod = dict()
    for v in vt:
        nod[v] = Q.insert( (r[v], v)  )
    while Q.total_nodes != 0:
        (x,v) = Q.extract_min().data
        del nod[v]
        for (v,w) in gt.union(er):
            x = numpy.Infinity
            if (v,w) in dr:
                x = dr[(v,w)]
            y = numpy.Infinity
            if (v,w) in dt:
                y = dt[(v,w)]
            r[w] = min( r[w], r[v] + min( x, y  ) )
            if (v,w) in gt:
                optt[w] = min( optt[w], r[v] + y  )
                Vp.add(w)
    return Vp, optt


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
    look_ahead_num = int( lk * (m -t) )
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
    return itera, len(infected), capacity

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
