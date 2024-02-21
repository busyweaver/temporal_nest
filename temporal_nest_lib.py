import math
import random
import networkx as nx
from collections import Counter, defaultdict
from hashlib import blake2b
from functools import reduce

import pickle

def save_dic(s,d):
    with open(s+".pkl","wb") as f:
        pickle.dump(d,f)

def read_dic(s):
    with open(s+".pkl", 'rb') as f:
        d = pickle.load(f)
    return d

def graphs_equality(g,r):
    for e in g:
        if e not in r:
            return False
    for e in r:
        if e not in g:
            return False
    return True

def events(g):
    ev = set()
    for (a,b,t) in g:
        ev.add(t)
    return ev
def nodes(g):
    no = set()
    for (a,b,t) in g:
        no.add(a)
        no.add(b)
    return no


def is_undirected(g):
    sg = set(g)
    for (u,v,t) in sg:
        if (v,u,t) not in sg:
            return False
    return True

def to_undirected(g):
    gs = set(g)
    gn = set()
    for (a,b,t) in gs:
        gn.add((a,b,t))
        gn.add((b,a,t))
    if type(g) == list:
        return list(gn)
    else:
        return gn
# def neigbour_inst(v,t,g):
#     res = set()
#     for (a,b,tp) in g:
#         if t == tp and a == v:
#             res.add(b)
#         if t == tp and b == v:
#             res.add(a)
#     return res

def neighbour_temp(v,t,g):
    res = set()
    for (a,b,tpp) in g:
        l = [(a,b,tpp)]
        for (a,b,tp) in l:
            if a == v and tp >= t:
                res.add((b,tp))
    return res

def reverse_neighbour_temp(v,t,g):
    res = set()
    for (a,b,tpp) in g:
        l = [(a,b,tpp)]
        for (a,b,tp) in l:
            if a == v and tp <= t:
                res.add((b,tp))
    return res



def all_graph_neighbour_node(g, nod, l_ev, reverse = False, look_ahead = 1):
    res = dict()
    m = l_ev[-1]
    look_ahead_num = int( look_ahead * m )
    for (a,b,tpp) in g:
        if reverse:
            if a not in res:
                res[a] = [ (tpp,b) ]
            else:
                res[a].append( (tpp,b) )

        else:
            if a not in res:
                res[a] = [ (tpp,b) ]
            else:
                res[a].append( (tpp,b) )
    #     print("nodes", nodes(g))
    for v in nod:
        if v not in res:
            res[v] = []
    return res


def all_graph_neighbour_imp(g, nod, ev, reverse = False, look_ahead = 1):
    res = dict()
    l_ev = list(ev)
    m = max(ev)
    look_ahead_num = int( look_ahead * m )
    for (a,b,tpp) in g:
        if reverse:
            l_ev.sort(reverse = True)
            for i in range(len(l_ev)):
                if l_ev[i] >= tpp and (l_ev[i] - tpp) <= look_ahead_num:
#                     print(a, b, "l_ev[i]", l_ev[i], "tpp",tpp)
                    current_nodes = [a,b]
                    for j in range(1):
                        if (current_nodes[j],l_ev[i]) not in res:
                            res[(current_nodes[j],l_ev[i])] = { (current_nodes[1-j],tpp) }
                        else:
                            res[(current_nodes[j],l_ev[i])].add((current_nodes[1-j],tpp))
                else:
                    break

        else:
            l_ev.sort()
            for i in range(len(l_ev)):
                if l_ev[i] <= tpp and (tpp - l_ev[i]) <= look_ahead_num:
                    current_nodes = [a,b]
                    for j in range(1):
                        if (current_nodes[j],l_ev[i]) not in res:
                            res[(current_nodes[j],l_ev[i])] = { (current_nodes[1-j],tpp) }
                        else:
                            res[(current_nodes[j],l_ev[i])].add((current_nodes[1-j],tpp))
                else:
                    break
#     print("nodes", nodes(g))
    for v in nod:
        for t in ev:
            if (v,t) not in res:
                res[(v,t)] = {}
    return res


def all_graph_neighbour(g, reverse):
    res = dict()
    ev = events(g)
    for v in nodes(g):
        for t in range(1,max(ev)+1):
            if reverse:
                res[(v,t)] = reverse_neighbour_temp(v,t,g)
            else:
                res[(v,t)] = neighbour_temp(v,t,g)
    return res

# def all_graph_neighbour_inst(g):
#     res = dict()
#     ev = events(g)
#     for v in nodes(g):
#         for t in range(min(ev),max(ev)+1):
#             res[(v,t)] = neighbour_inst(v,t,g)
#     return res

def neighb(nei, t, lo_num):
    res = []
    for (tp,v) in nei:
        if tp >= t and (tp - t) <= lo_num:
            res.append( (v,tp) )
        elif tp >= t and (tp - t) > lo_num:
            return res
    return res


# l_ev[-1] is largest event since list is ordered
def degrees(g, nod, l_ev, reverse, look_ahead):
    res = dict()
    print("begin all neigbour")
    nei = all_graph_neighbour_node(g, nod, l_ev, reverse, look_ahead)
    print("end all neigbour")
    look_ahead_num = int( look_ahead * l_ev[-1] )
    for v in nod:
        for t in l_ev:
            res[(v,t)] = len(neighb(nei[v],t, look_ahead_num))
    return res, nei

def graph_time(g,t):
    res = set()
    for (a,b,tp) in g:
        if tp == t:
            res.add((a,b,tp))
    return res

def graph_cut(g, dire, tau):
    lg = list(g)
    lgo = [ [e[2],e] for e in lg ]
    lgo.sort()
    lgo = [ e[1] for e in lgo ]
    val = int(len(lgo)*tau)
    g_new = lgo[:val]
    if dire == "u":
        g_new = to_undirected(g_new)
    return g_new


def seq_graphs(g):
    d = dict()
    for e in g:
        u,v,t = e
        if t not in d:
            d[t] = {(u,v)}
        else:
            d[t].add((u,v))
    return d


def _hash_label(label, digest_size):
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()
def _init_node_labels(G, nod, l_ev, reverse, look_ahead):
    d,nei = degrees(G, nod, l_ev, reverse, look_ahead)
    return {u: str(d[u]) for u in d.keys()}, nei

def _neighborhood_aggregate(v, t, node_labels, nei, m, look_ahead):
    """
    Compute new labels for given node by aggregating
    the labels of each node's neighbors.
    """
    #print("nei agg", v, t, nei)
    look_ahead_num = int( look_ahead * m )
    label_list = []
#     print(node_labels)
    for nbr in neighb(nei, t, look_ahead_num):
        prefix = ""
        label_list.append(prefix + node_labels[nbr])
    return node_labels[(v,t)] + "".join(sorted(label_list))

import itertools

def dist_colours(col):
    res = dict()
    for e in col.keys():
        if col[e] not in res:
            res[col[e]] = {e}
        else:
            res[col[e]].add(e)
    return res

def check_convergence_node_labels(new,old):
    x, y =dist_colours(new), dist_colours(old)
    for e in x.keys():
        set_new = x[e]
        elem = list(set_new)[0]
        col_old = old[elem]
        set_old = y[col_old]
#         print(set_new,set_old)
        if not set_new == set_old:
            return False
    return True

def weisfeiler_lehman_graph_hash(
        G, iterations = -1, digest_size=16, keep_iterations=False, reverse = False, look_ahead = 1, save_each_step = False, name_save = ""):
    #print("salut")
    def weisfeiler_lehman_step(G, nei, labels, nod, l_ev, reverse, look_ahead):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        """
        new_labels = {}
        #nei = all_graph_neighbour_imp(G, nod, ev, reverse, look_ahead)
        for node in nod:
            for t in l_ev:
                label = _neighborhood_aggregate(node, t, labels, nei[node], l_ev[-1], look_ahead)
                new_labels[(node,t)] = _hash_label(label, digest_size)
        return new_labels
    nod = nodes(G)
    ev = events(G)
    l_ev = list(ev)
    l_ev.sort()

    if iterations == -1:
        iterations = len(nod)*len(ev)
    keep = dict()
#     keep = { 0:{e:'1'   for e in itertools.product(nodes(G),[i for i in ev])} }
    # set initial node labels
    node_labels, nei = _init_node_labels(G, nod, l_ev, reverse, look_ahead)
    #print("end node label")
    if save_each_step:
        save_dic(name_save+"_"+str(1) , node_labels)
    #print("end firt save")
    subgraph_hash_counts = []
    i = -1
    for i in range(1,iterations +1):
        print("iteration", i)
        node_labels_new = weisfeiler_lehman_step(G, nei, node_labels, nod, l_ev, reverse, look_ahead)
        counter = Counter(node_labels_new.values())
        # sort the counter, extend total counts
        subgraph_hash_counts.extend(sorted(counter.items(), key=lambda x: x[0]))
        if check_convergence_node_labels(node_labels_new, node_labels):
#             print("conv = ",i+1," ",end="")
            break
        else:
            if keep_iterations:
                keep[i+1] = node_labels
            if save_each_step:
                save_dic(name_save+"_"+str(i+1) , node_labels)
            node_labels = node_labels_new

    # hash the final counter
#     if keep_iterations:
#         keep.append(node_labels_new)
    return _hash_label(str(tuple(subgraph_hash_counts)), digest_size), node_labels, keep, i+1

# works only for directed

def possible_flips_at(node_set, col,se,t):
    res = set()
    for x in se[t]:
        for v in node_set:
            if col[(x[1],t)] == col[(v,t)] and x[1] != v and x[0] != v:
                if t in res:
                    res.add(( (x[0],x[1],t), (x[0],v,t) ))
                else:
                    res = {( (x[0],x[1],t), (x[0],v,t) )}
                    
    return res
    
def possible_felix_flips(node_set,ev,col,se):
    res = dict()
    for t in ev:
        res[t] = possible_flips_at(node_set,col,se,t)
    return res

import random

def felix_flip(g,se,col,possible_flips):
    l = list(reduce(lambda x,y : x.union(y) , possible_flips.values(), set() ))
#     print("l",l)
    if len(l) == 0:
        print("no possible flips")
        return -1
    x = random.choice(  l )
#     print("chosen", x)
    g.remove(x[0])
    g.add(x[1])
    t = x[0][2]
    se[t].remove((x[0][0],x[0][1]))
    se[t].add((x[1][0],x[1][1]))
    return g,se,x

def felix_flips(gg,n,col):
    g = set(gg.copy())
    se = seq_graphs(g)
    node_set = nodes(g)
    ev = events(g)
    flips = possible_felix_flips(node_set,ev,col,se)
#     if len(flips) == 0:
#         print('no possible flips')
#         return -1
    for _ in range(n):
        g,se,x = felix_flip(g,se,col,flips)
        t = x[0][2]
        flips[t] = possible_flips_at(node_set, col,se,t)
#         flips.remove(x)
#         flips.add((x[1],x[0]))
    return list(g)   

                

def list_rewirings(g,col):
    res = []
    non = []
    edges = seq_graphs(g)
    ev = events(g)
    for t in ev:
        if len(edges[t]) > 1:
            for (a,b) in edges[t]:
                for (c,d) in edges[t]:
                    if (a,b) != (c,d):
                        if col[(a,t)] == col[(c,t)] and col[(b,t)] == col[(d,t)]:
                            if (a,d,t) not in g and (c,b,t) not in g and a!=d and b!=c and ((c,d,t),(a,b,t)) not in res:
                                res.append(((a,b,t),(c,d,t)))
    return res

def list_rewirings_inter(g,col):
    res = []
    non = []
    edges = seq_graphs(g)
    ev = events(g)
    for t in ev:
        for tp in ev:
            if len(edges[t]) >= 1 and len(edges[tp]) >= 1:
                for (a,b) in edges[t]:
                    for (c,d) in edges[tp]:
                        if (a,b) != (c,d):
                            if col[(a,t)] == col[(c,tp)] and col[(b,t)] == col[(b,tp)] and col[(b,t)] == col[(d,tp)] and col[(d,t)] == col[(d,tp)]:
                                if (a,d,t) not in g and (c,b,tp) not in g and a!=d and b!=c and ((c,d,tp),(a,b,t)) not in res:
                                    res.append(((a,b,t),(c,d,tp)))                            
    return res

def rewirings(g,col, dire):
    l = []
    se = seq_graphs(g)
    node_set = nodes(g)
    ev = events(g)
    if dire == "u":
#     print("rewirirings *")
#     print("max ev", max(ev))
        for t in ev:
            l = l + list(rewirings_at_time(se,col,t))
    else:
        for t in ev:
            l = l + list(possible_flips_at(node_set, col,se,t))       
    return l

def rewirings_at_time(seq_g,col,t):
    res = set()
    edges = seq_g[t]
    ev = seq_g.keys()
#     for tp in ev:
#         edges2 = seq_g[tp]
    if len(edges) >= 2:
        for (a,b) in edges:
            for (c,d) in edges:
                if (a,b) != (c,d):
#                         print("color", (a,t), (c,t), (b,t), (d,t))
                    # maybe add condition if a==c , not rewire because it does not change anything : update done
                    if col[(a,t)] == col[(c,t)] and col[(b,t)] == col[(b,t)]:
                        if (a,d) not in edges and (c,b) not in edges and a!= c and a!=d and b!=c and ((c,d,t),(a,b,t)) not in res:
                            res.add(((a,b,t),(c,d,t)))
    return res

def rewirings_at_time_felix(seq_g,col,t):
    res = set()
    edges = seq_g[t]
    ev = seq_g.keys()
#     print(edges,t)
    for tp in ev:
        edges2 = seq_g[tp]
        if tp>=t and len(edges) >= 1 and len(edges2) >= 1:
            for (a,b) in edges:
                for (c,d) in edges2:
                    if (a,b) != (c,d):
#                         print("color", (a,t), (c,t), (b,t), (d,t))
                        # maybe add condition if a==c , not rewire because it does not change anything : update done
                        if col[(d,t)] == col[(b,t)] and col[(d,tp)] == col[(c,tp)]:
                            if (a,d) not in edges and (c,b) not in edges2 and a!= c and a!=d and b!=c and ((c,d,tp),(a,b,t)) not in res:
                                res.add(((a,b,t),(c,d,tp)))
    return res

def check_seq_g(g,se):
    print("checking")
    for t in se.keys():
        for x in se[t]:
            if (x[0],x[1],t) not in g:
                print("check : PROBLEM in se not g")
                return False
    for e in g:
        if e[2] not in se.keys() or (e[0],e[1]) not in se[e[2]]:
            print("check : PROBLEM in g not se")
            return False
    return True
            
def check_rewire_all(g,rewire):
    for t in rewire.keys():
        for e in rewire[t]:
            if e[0] not in g or e[1] not in g:
                if e[0] not in g:
                    print("check : PROBLEM rewire 1", e)
                else:
                    print("check : PROBLEM rewire 2", e)
                return False
    return True

def rewiring_one(g_new, rewire, se, dire):
    #print("rewiring_one")
#     check_seq_g(g_new,se)
#     check_rewire_all(g_new,rewire)
    su = 0
    for t in rewire.keys():
        su += len(rewire[t])
#     print("is possible?", possible_rewire)
#     m = max(events(g))
    r = -1
    if su != 0:
        i = random.randint(0,su-1)
        lk = list(rewire.keys())
        for j in range(len(lk)):
            i = i - len(rewire[lk[j]])
            if i < 0:
                break
        (a,b,t),(c,d,tp) = rewire[lk[j]][i]

        g_new.remove((a,b,t))
        if dire=="u":
            g_new.remove((b,a,t))
        se[t].remove((a,b))
        if dire=="u":
            se[t].remove((b,a))
            
        g_new.remove((c,d,tp))
        if dire=="u":
            g_new.remove((d,c,tp))
            
        se[tp].remove((c,d))
        if dire=="u":
            se[tp].remove((d,c))
                
        g_new.add((a,d,t))
        if dire=="u":
            g_new.add((d,a,t))
        se[t].add((a,d))
        if dire=="u":
            se[t].add((d,a))

        g_new.add((c,b,tp))
        if dire=="u":
            g_new.add((b,c,tp))
        
        se[tp].add((c,b))
        if dire=="u":
            se[tp].add((b,c))
            
        r = t
    return g_new,se,r,tp

def rewire_any(gg,n,col,dire):
    g = set(gg.copy())
    #print(g, is_undirected(g))
    rewire = dict()
    se = seq_graphs(g)
    ev = events(g)
    index = []
    l = []
    for t in ev:
        index.append(t)
        rewire[t] = list(rewirings_at_time(se,col,t))
        l = l+rewire[t]
        
    for i in range(n):
#         print("i",i)
        g,se,t,tp = rewiring_one(g,rewire,se,dire)
        if t != -1:
            rewire[t] = list(rewirings_at_time(se,col,t))
            if t != tp:
                rewire[tp] = list(rewirings_at_time(se,col,tp))

    return list(g)

def nb_randomized_edge(g):
    edges = list(g)
    d = seq_graphs(g)
    nb = 0
    for (i,j,t) in edges:
        for (ip,jp,tp) in edges:
            r = edges.index((i,j,t))
            s = edges.index((ip,jp,tp))
            if (r != s and i!=jp and j!=ip and (i,jp) not in d[t] and (j,ip) not in d[tp]) or (r != s and i!=ip and j!=jp and (i,ip) not in d[t] and (j,jp) not in d[tp]):
                nb += 1
    return nb
    
    
def randomized_edge(g, dire, tout = -1):
    edges = list(g)
    d = seq_graphs(g)
    random.shuffle(edges)
    if tout == -1:
        fin = len(edges)
    else:
        fin = tout
    nb_rewired = 0
    rewired = False
    while nb_rewired < fin:
#         print("r", r,end =" ")
#         nb = nb_randomized_edge(g)
#         print("nb possible", nb, "over ", len(g)*len(g), "len edges", len(edges))
#         print("edges", edges)
        r = random.randint(0,len(edges)-1)
        i,j,t = edges[r]
#         print("ijt",i,j,t)
        s = random.randint(0,len(edges)-1)
        ip, jp, tp = edges[s]
#         print("ipjptp",ip,jp,tp)
        b = random.randint(0,1)
        if b and r != s and i!=jp and j!=ip and (i,jp) not in d[t] and (j,ip) not in d[tp]:
            edges.pop(r)
            d[t].remove((i,j))
#             print("edges in", edges)
            if dire == "u":
                x = edges.index((j,i,t))
                edges.pop(x)
                d[t].remove((j,i))
                
            x = edges.index((ip,jp,tp))
            edges.pop(x)
            d[tp].remove((ip,jp))
            if dire == "u":
                x = edges.index((jp,ip,tp))
                edges.pop(x)
                d[tp].remove((jp,ip))
                
            edges.append( (i,jp,t) )
            d[t].add((i,jp))
            if dire == "u":
                edges.append( (jp,i,t) )
                d[t].add((jp,i))
                
            edges.append( (j,ip,tp) )
            d[tp].add((j,ip))
            if dire == "u":
                edges.append( (ip,j,tp) )
                d[tp].add((ip,j))
                        
            rewired = True
        elif not b and r != s and i!=ip and j!=jp and (i,ip) not in d[t] and (j,jp) not in d[tp]:
            edges.pop(r)
            d[t].remove((i,j))
            
            if dire == "u":
                x = edges.index((j,i,t))
                edges.pop(x)
                d[t].remove((j,i))
            #for s
            x = edges.index((ip,jp,tp))
            edges.pop(x)
            d[tp].remove((ip,jp))
            if dire == "u":
                x = edges.index((jp,ip,tp))
                edges.pop(x)
                d[tp].remove((jp,ip))
                
            edges.append( (i,ip,t) )
            d[t].add((i,ip))
            if dire == "u":
                edges.append( (ip,i,t) )
                d[t].add((ip,i))
                
            edges.append( (j,jp,tp) )
            d[tp].add((j,jp))
            if dire == "u":
                edges.append( (jp,j,tp) )
                d[tp].add((jp,j))
                
            rewired = True
        if rewired:
            nb_rewired += 1
            rewired = False
    if nb_rewired < fin:
        print("randomized edge : probleme no possible rewirings to finish job ", r, "rewirings done out of ", fin)
    return set(edges)


def check_rewire(d,i,j):
    su = 0
    for ip,jp in d:
        if i!=jp and j!=ip and (i,jp) not in d and (j,ip) not in d:
            su += 1
        if i!=ip and j!=jp and (i,ip) not in d and (j,jp) not in d:
            su += 1
    if su > 0:
        return True, su
    else:
        return False, su
    
def number_rewirings_randomized_same_time(g):
    d = seq_graphs(g)
    nb = 0
    for t in events(g):
        for (i,j) in d[t]:
            x = check_rewire(d[t],i,j) 
            if x[0]:
                nb += x[1]
    return nb
    
        
    
def randomized_edge_same_time(g, dire, tout = -1):
    edges = list(g)
    d = seq_graphs(g)
    random.shuffle(edges)
    if tout == -1:
        fin = len(edges)
    else:
        fin = tout
    r = 0
    rr = 0            
    
    rewired = False
#     print("fin", fin)
    while r < fin:
        
        if rr == len(edges):
            print("randomized_edge_same_time : probleme no possible rewirings to finish job ", r, "rewirings done out of ", fin)
            return set(edges)
        i,j,t = edges[rr]
        
        while rr<len(edges)-1 and not check_rewire(d[t],i,j)[0]:
            rr += 1
            i,j,t = edges[rr]
        if not check_rewire(d[t],i,j)[0]:
            print("randomized_edge_same_time : probleme no possible rewirings to finish job ", r, "rewirings done out of ", fin)
            return set(edges)
        s = random.randint(0,len(d[t])-1)
        ip, jp = list(d[t])[s]
        b = random.randint(0,1)
#         print("selected", (i,j), (ip,jp), d[t])
        if b == 0 and i!=jp and j!=ip and (i,jp) not in d[t] and (j,ip) not in d[t]:
            x = edges.index( (i,j,t) )
            edges.pop(x)
            if dire == "u":
                x = edges.index( (j,i,t) )
                edges.pop(x)
            
#             print("1", (ip,jp,t) in edges or (jp,ip,t) in edges)
            x = edges.index( (ip,jp,t) )
            edges.pop(x)
            if dire == "u":
                x = edges.index( (jp,ip,t) )
                edges.pop(x)
                
            
            d[t].remove((i,j))
            if dire == "u":
                d[t].remove((j,i))
            
            d[t].remove((ip,jp))
            if dire == "u":
                d[t].remove((jp,ip))
            
            d[t].add((i,jp))
            if dire == "u":
                d[t].add((jp,i))
                
            d[t].add((j,ip))
            if dire == "u":
                d[t].add((ip,j))
                
            edges.append( (i,jp,t) )
            if dire == "u":
                edges.append( (jp,i,t) )
            edges.append( (j,ip,t) )
            if dire == "u":
                edges.append( (ip,j,t) )
                
            rewired = True
        elif b == 1 and i!=ip and j!=jp and (i,ip) not in d[t] and (j,jp) not in d[t]:
            x = edges.index( (i,j,t) )
            edges.pop(x)
            if dire == "u":
                x = edges.index( (j,i,t) )
                edges.pop(x)
#             print("2", (ip,jp,t) in edges or (jp,ip,t) in edges)
            x = edges.index( (ip,jp,t) )
            edges.pop(x)
            if dire == "u":
                x = edges.index( (jp,ip,t) )
                edges.pop(x)
                
            d[t].remove((i,j))
            if dire == "u":
                d[t].remove((j,i))
            d[t].remove((ip,jp))
            if dire == "u":
                d[t].remove((jp,ip))
            
            d[t].add((i,ip))
            if dire == "u":
                d[t].add((ip,i))
                
            d[t].add((j,jp))
            if dire == "u":
                d[t].add((jp,j))
            edges.append( (i,ip,t) )
            if dire == "u":
                edges.append( (ip,i,t) )
                
            edges.append( (j,jp,t) )
            if dire == "u":
                edges.append( (jp,j,t) )
            rewired = True
        if rewired:
            r += 1
            rr += 1
            rewired = False
    return set(edges)


#variant of the preceding a bit more efficient not rewiring if index already rewired
# i did not add the case for directed temporal networks yet
def randomized_edge_var(g, tout = -1):
    edges = list(g)
    d = seq_graphs(g)
    random.shuffle(edges)
    if tout == -1:
        fin = len(edges)
    else:
        fin = tout
    selected = set()
    r = 0
    while r < fin:
        i,j,t = edges[r]
        s = random.randint(0,len(edges))
        if s not in selected:
            ip, jp, tp = edges[k]
            b = random.randint(0,1)
            if b and i!=jp and j!=ip and (i,jp) not in d[t] and (j,ip) not in d[tp]:
                mi,ma = min(r,s), max(r,s)
                edges.pop(mi)
                edges.pop(ma-1)
                edges.insert(mi, (i,jp,t) )
                edges.insert(ma, (j,ip,tp) )
                rewired = True
            elif not b and i!=ip and j!=jp and (i,ip) not in d[t] and (j,jp) not in d[tp]:
                mi,ma = min(r,s), max(r,s)
                edges.pop(mi)
                edges.pop(ma-1)
                edges.insert(mi, (i,ip,t) )
                edges.insert(ma, (j,jp,tp) )
                rewired = True
            if rewired:
                selected.add(s)
                rewired = False
                r += 1
    return set(edges)
    
def permuted_times(g):
    edges = list(g)
    time = []
    for i in range(len(edges)):
        u,v,t = edges[i]
        time.append(t)
    random.shuffle(time)
    for i in range(len(edges)):
        u,v,t = edges[i]
        edges[i] = (u,v,time[i])
    return set(edges)


def random_times(g):
    res = set()
    d = seq_graphs(g)
    for t in d.keys():
        node = list(nodes(d[t]))
        gp = set()
        while i < len(d[t]):
            x = random.sample(node,2)
            if (x[0],x[1]) not in gp and (x[1],x[0]) not in gp:
                gp.add((x[0],x[1]))
                i += 1
        d[t] = gp
    for t in d.keys():
        for e in d[t]:
            res.add((e[0],e[1],t))
    return res
        
    #generate gnm according to edges
    
def aggregated_graph(g):
    d = {}
    for e in g:
        u,v,t = e
        if (u,v) in d:
            d[(u,v)].append(t)
        else:
            d[(u,v)] = [t]
    return d
def random_contacts(g):
    res = set()
    ev = events(g)
    d = aggregated_graph(g)
    cmp = 0
    lk = list(d.keys())
    for e in lk:
        cmp += (len(d[e])-1)
        d[e] = [random.choice(ev)]
    for i in range(cmp):
        edge = random.randint(0,len(lk)-1)
        t = random.choice(ev)
        d[edge].append(t)
    for e in d.keys():
        for t in d[e]:
            res.add((e[0],e[1],t))
    return res

def to_straph_file(g,s):
    tmp = aggregated_graph(g)
    ev = events(g)
    no = nodes(g)
    ma,mi = max(ev), min(ev)
    with open(s+"_nodes.sg","w") as f:
        for v in no:
            f.write(v+" "+str(mi)+" "+str(ma)+"\n")
    with open(s+"_links.sg","w") as f:
        for u,v in tmp.keys():
            for (x,y) in [ (u,v),(v,u) ]: 
                for t in tmp[(u,v)]:
                    f.write(x+" "+y+" "+str(t)+" "+str(t)+"\n")
#for static graphs
def adj(g):
    tmp = dict()
    for u,v in g:
        for x,y in [(u,v),(v,u)]:
            if x in tmp:
                tmp[x].add(y)
            else:
                tmp[x] = {y}
    return tmp

def randomize(g,n,col,dire):
#     print("start ****")
    g_tmp = g[:]
    if dire == "u":
        g1 = rewire_any(g,n,col,dire)
    else:
        g1 = felix_flips(g,n,col)
    print("fin g1")
    g2 = randomized_edge_same_time(g,dire, n)
    print("fin g2")
    g3 = randomized_edge(g,dire,n)
    print("fin g3")
    return g1, g2, g3

