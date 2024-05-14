#!/usr/bin/env python
# coding: utf-8

# In[6]:


from graph_characteristics_lib import *
import os
import sys
import math


list_rewires = dict()

path = "datasets/networks/"
names = [ ["bison_dire","bis","crimson","d"] , ["cattle_dire", "cat", "green","d"],
          ["email-eu2", "eu2", "magenta","d"], ["sheep_dire", "she" ,"olive","d"],
          ["email-eu3", "eu3", "yellow","d"], ["primate","pri",  "blue","u"],
          ["racoon", "rac", "brown","u"],  ["weaver", "wea",  "pink","u"],
          ["workplace_2013", "wp", "black","u"], ["ht09", "ht","red","u"]]

#names = [ ["opsahl", "opsahl", "black","d"], ["email-eu2", "eu2", "magenta","d"], ["dnc", "dnc", "brown","d"], ["email-eu3", "eu3", "yellow","d"], ["highschool_2011", "hs11", "purple","u"], ["hospital_ward", "hw", "blue","u"], ["ht09", "ht","red","u"], ["workplace_2013", "wp", "green","u"] ]
look_aheads = [0.0,0.2,0.4,0.6,0.8,1]
iterations = 1
cut = float(sys.argv[1])
print("cut", cut)
if len(sys.argv) <= 2:
    nb_graphs = 1
else:
    nb_graphs = int(sys.argv[2])

print("average nb graphs", nb_graphs)

def find_sep(x):
    for s in x:
        if s not in ['0','1','2','3','4','5','6','7','8','9']:
            return s
    return None


def read_graph(path,s, dire):
    g = set()
    f = open(path+s+".csv", "r")
    sep = find_sep(f.readline())
    f.close()
    f = open(path+s+".csv", "r")
    for x in f:
        x = x[:-1]
        r = x.split(sep)
    #     r = x.split("\t")
        u,v,t = r[0],r[1],r[2]
        if v != u:
            t = float(t)
            g.add( (u,v,int(t)) )
            if dire == "u":
                g.add( (v,u,int(t)) )
    return g
folder = "values_graphs/"
str_dict = ["keep", "rewire", "time"]
folder_res = "results_"+str(cut)+"/"
if not os.path.exists(folder_res): 
    os.makedirs(folder_res)


# In[119]:


from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,int(len(names)/2))
for k in range(len(names)):
    e = names[k]
    print(e)
    g = read_graph(path,e[0],e[-1])
    g = graph_cut(g,names[k][-1],cut)
    i = k//2
    j = k%2
    d = seq_graphs(g)
    l_ev = list(events(g))
    l_ev.sort()
    mi, ma = l_ev[0], l_ev[-1]
    print("mi",mi,"ma",ma)
    l_ev_norm = [ (e-mi)/(ma-mi) for e in l_ev ]
#     print(l_ev_norm)
#     print("//////////////////////////////////////////",mi,ma)
    l = [  len(d[e]) for e in l_ev ]
#     for it in range(iterations):
    #     print(l)
    axs[j, i].set_title(names[k][1])
    axs[j, i].scatter( l_ev_norm ,l, alpha=1, color = names[k][2])
#     axs[j, i].legend()
    axs[j,i].set(xlabel = "% times" , ylabel='# nb edges ', )
#     axs[j,i].yaxis.set_label_coords(-0.3,0.5)
    axs[j,i].xaxis.set_label_coords(0.5,-0.07)
    #     axs[j,i].set_xticks([0,20,40,60,80,100], labels=[0,0.2,0.4,0.6,0.8,1])
    #     axs[j,i].set_yticks([0,0.2,0.4,0.6,0.8,1],labels=[0,0.2,0.4,0.6,0.8,1])
fig.set_size_inches(15, 10)
fig.savefig(folder_res+"edge_dist.svg", format="svg", transparent = True, dpi=150)
fig.show()

import re

def find_max_wl(folder, name):
    pattern = re.compile("^("+name+"_"+"[0-9]+"+".pkl)")
    files = [f for f in os.listdir(folder) if os.path.isfile(folder+f) and pattern.match(folder+f)]
    m = 1
    for e in files:
        i = -5
        tmp = ""
        current = e[i]
        while current != "_":
            tmp += current
            i -= 1
            current = e[i]
        val = int(tmp)
        if val > m:
            m = val
    return m




# In[3]:




#do not change the look aheads sequence
import time
def stats_numberrewirings_conv(path, names, look_aheads, iterations, cut, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    files = [f for f in os.listdir(folder) if os.path.isfile(folder+f)]
    d_exec = dict()
    d = dict()
    d_con = dict()
    d_time = dict()
    for e in names:
        d = dict()
        d_time = dict()
        d_con = dict()
        d_exec = dict()
        g = read_graph(path,e[0],e[-1])
        g_new = graph_cut(g,e[-1],cut)
#         for i in range(iterations+1):
#             print(d,e[0],i)
        if not e[1]+"_"+str(look_aheads[0])+"_"+str(cut)+"_rewire.pkl" in files:
            print("compute wl for", e[0])
            for lo in look_aheads:
                start = time.time()
                _, col, _, con = weisfeiler_lehman_graph_hash(g_new,iterations = -1, reverse = False, look_ahead = lo, save_each_step = True, name_save = folder+e[1]+"_"+str(lo)+"_"+str(cut)+"_keep")
                print("end computer color", lo)
    #             print(col,keep)
                end = time.time()
                d[lo] = dict()
                d_exec[lo] = end-start
                d_time[lo] = dict()
                d_con[lo] = con
                max_it = find_max_wl(folder, folder+e[1]+"_"+str(lo)+"_"+str(cut)+"_keep")
                for i in range(1,max_it+1):
    #                 print("keep",len(keep))
    #                 print("i",i, "j",j, "keep",len(keep))
    #                 possible_rewire,_ = list_rewirings(g_new,keep[j])
                    keep = read_dic(folder+e[1]+"_"+str(lo)+"_"+str(cut)+"_keep_"+str(i))
                    tmp = dict()
                    if e[-1] == "u":
                        possible_rewire = rewirings(g_new,keep, e[-1])
                        list_rewires[e[0]] = dict()
                        if i == 1 or i == max_it:
                            #for now not used, maybe get rid of it
                            list_rewires[e[0]][i] = possible_rewire
                        print("possible", len(possible_rewire))
                        for (a,b) in possible_rewire:
                            if a[2] not in tmp:
                                tmp[a[2]] = 1
                            else:
                                tmp[a[2]] += 1
                        nb_possible = len(possible_rewire)
                    else:
                        node_set = nodes(g_new)
                        se = seq_graphs(g_new)
                        ev = events(g_new)
                        possible_at, nb_possible = nb_felix_flips_improved(ev, se, node_set, keep)
                        tmp = possible_at
                        
                    d_time[lo][i] = tmp
                    d[lo][i] = nb_possible
                    possible_rewire = []
#                 del keep[0]
                #save_dic(folder+e[1]+"_"+str(lo)+"_"+str(cut)+"_keep" , keep)
                #save_dic( folder+e[1]+"_"+str(lo)+"_"+str(cut)+"_exec_time" , d_exec[e[0]])
                save_dic( folder+e[1]+"_"+str(lo)+"_"+str(cut)+"_rewire" , d[lo])
                save_dic( folder+e[1]+"_"+str(lo)+"_"+str(cut)+"_time" , d_time[lo])

            save_dic( folder+e[1]+"_"+str(lo)+"_"+str(cut)+"_exec_time" , d_exec)
        else:
            print(e, "already present")
#     return d, d_con, d_time, keep


# In[4]:


stats_numberrewirings_conv(path, names, look_aheads, iterations, cut, folder)


# In[5]:


# write properties
def write_characteristics_table(folder, folder_values, names, cut):
    files = [f for f in os.listdir(folder) if os.path.isfile(folder+f)]
    s = "table_gen.tex"
    if s not in files:
        fg = open(folder+"table_gen.tex", "w")
        for k in range(len(names)):
            e = names[k]
            g = read_graph(path,e[0],e[-1])
            g = graph_cut(g,e[-1],cut)
            ev = events(g)
            nb_temp_edges = len(g)
            nod = nodes(g)
            nb_nodes = len(nod)
            rew_dic = read_dic(folder_values+e[1]+"_"+str(1)+"_"+str(cut)+"_rewire")
            exec_dic = read_dic(folder_values+e[1]+"_"+str(1)+"_"+str(cut)+"_exec_time")
            name_keep = folder_values+names[k][1]+"_"+str(1)+"_"+str(cut)+"_keep"
            m = find_max_wl(folder_values, name_keep)
            #m = max(rew_dic.keys())
            fg.write(e[1]+" & "+ e[3]+ " & " +str(nb_nodes)+" & "+str(len(ev))+
                     " & "+str(nb_temp_edges)+" & "+str(m)+ " & " +str(exec_dic[1])[:5] + " & "+ str(rew_dic[m])+ "\\\\ \n")
        fg.close()
    else:
        print("table gen already present")
write_characteristics_table(folder_res, folder, names, cut)


# In[13]:


from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,int(len(names)/2))
for k in range(len(names)):
#     print(names[k][0])
    dd = dict()
    l = []
    for e in look_aheads:
        name_keep = folder+names[k][1]+"_"+str(e)+"_"+str(cut)+"_keep"
        max_wl = find_max_wl(folder, name_keep)
        l.append(max_wl)
#         f = read_dic(name_keep+"_"+str(max_wl))
#         f2 = read_dic(name_keep+"_"+str(1))
# #         print("look for", f)
#         dd[e] = {1: f, max_wl : f2 }
# #         print(dd[e])
# #         print(max(dd[e].keys()))
    i = k//2
    j = k%2
    #l = [  max(dd[e].keys()) for e in look_aheads ]
#     for it in range(iterations):
    #     print(l)
    axs[j, i].set_title(names[k][1])
    axs[j, i].scatter( look_aheads ,l, alpha=1, color = names[k][2])
#     axs[j, i].legend()
    axs[j,i].set(xlabel = "% lookahead" , ylabel='# convergence', )
#     axs[j,i].yaxis.set_label_coords(-0.3,0.5)
    axs[j,i].xaxis.set_label_coords(0.5,-0.07)
    #     axs[j,i].set_xticks([0,20,40,60,80,100], labels=[0,0.2,0.4,0.6,0.8,1])
    #     axs[j,i].set_yticks([0,0.2,0.4,0.6,0.8,1],labels=[0,0.2,0.4,0.6,0.8,1])
fig.set_size_inches(15, 10)
fig.savefig(folder_res+"conv_steps.svg", format="svg", transparent = True, dpi=150)
fig.show()


# In[14]:


from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,int(len(names)/2))
for k in range(len(names)):
    dd = dict()
    for e in look_aheads:
        f = folder+names[k][1]+"_"+str(e)+"_"+str(cut)+"_"+str_dict[1]
#         print("look for", f)
        dd[e] = read_dic(f)
    i = k//2
    j = k%2
    axs[j, i].set_title(names[k][1])
    m = -1
    for e in look_aheads:
        x = max(dd[e].keys())
        if x > m:
            m = x
    for it in range(0,m):
        l = [ dd[e][max(dd[e].keys())] if it not in dd[e] else dd[e][it] for e in look_aheads ]

        if i <= 1 and j <= 1:
            axs[j, i].semilogy( look_aheads ,l, alpha=(it+1)/(m+1), color = names[k][2], label= "iter "+str(it+1))
        else:
            axs[j, i].plot( look_aheads ,l, alpha=(it+1)/(m+1), color = names[k][2], label= "iter "+str(it+1))
        axs[j, i].legend()

    axs[j,i].set(xlabel = "% lookahead" , ylabel='# rewirings', )
#     axs[j,i].yaxis.set_label_coords(-0.3,0.5)
    axs[j,i].xaxis.set_label_coords(0.5,-0.07)
#     axs[j,i].set_xticks([0,20,40,60,80,100], labels=[0,0.2,0.4,0.6,0.8,1])
#     axs[j,i].set_yticks([0,0.2,0.4,0.6,0.8,1],labels=[0,0.2,0.4,0.6,0.8,1])
fig.set_size_inches(20, 12)
fig.savefig(folder_res+"number_rewire.svg", format="svg", transparent = True, dpi=150)
fig.show()


# In[16]:


from matplotlib.pyplot import figure
from matplotlib.pyplot import cm

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,int(len(names)/2))
for k in range(len(names)):
    dd = dict()
    for e in look_aheads:
        f = folder+names[k][1]+"_"+str(e)+"_"+str(cut)+"_"+str_dict[2]
#         print("look for", f)
        dd[e] = read_dic(f)
    i = k//2
    j = k%2
    axs[j, i].set_title(names[k][1])
    
    color = iter(cm.rainbow(np.linspace(0, 1, len(look_aheads))))
    for a in look_aheads:
        c = next(color)
        m_it = max(dd[a].keys())
        d1 =  dd[a][1] 
        d2 =  dd[a][m_it]
#         print(dd[a][0])
        if len(dd[a][1].keys()) != 0:
            m_ev1 = max(dd[a][1].keys())
            l_ev1 = list(dd[a][1].keys())
            l_ev1.sort()
            l_val1 = [ dd[a][1][t] for t in l_ev1  ]
            y1 = [dd[a][1][t] for t in l_ev1  ]
            #if cummulative uncomment following
            y1 = [  sum(l_val1[:ii+1]) for ii in range(len(l_ev1))  ]
            l_ev1 = [  e/m_ev1 for e in l_ev1 ]
            if i <= 1 and j <= 1:
                axs[j, i].semilogy( l_ev1 ,y1, alpha=1/2, color = c, label= "iter 0 "+str(a)[:4])
            else:
                axs[j, i].plot( l_ev1 ,y1, alpha=1/2, color = c, label= "iter 0 "+str(a)[:4])

        if len(dd[a][m_it].keys()) != 0:
            m_ev2 = max(dd[a][m_it].keys())
            l_ev2 = list(dd[a][m_it].keys())
            l_ev2.sort()
            l_val2 = [ dd[a][m_it][t] for t in l_ev2  ]
            y2 = [dd[a][m_it][t] for t in l_ev2  ]
            y2 = [  sum(l_val2[:ii+1]) for ii in range(len(l_ev2))  ]
            l_ev2 = [  e/m_ev2 for e in l_ev2 ]
            if i <= 1 and j <= 1:
                axs[j, i].semilogy( l_ev2 ,y2, alpha=1, color = c, label= "iter inf"+str(a)[:4])
            else:
                axs[j, i].plot( l_ev2 ,y2, alpha=1, color = c, label= "iter inf"+str(a)[:4])
        axs[j, i].legend()

    axs[j,i].set(xlabel = "% time" , ylabel='# rewirings', )

    axs[j,i].xaxis.set_label_coords(0.5,-0.07)

fig.set_size_inches(20, 12)
fig.savefig(folder_res+"dist_rewire.svg", format="svg", transparent = True, dpi=150)
fig.show()


# In[17]:


from matplotlib.pyplot import figure
from matplotlib.pyplot import cm

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,int(len(names)/2))
for k in range(len(names)):
    print(names[k][0])
    dd = dict()
    for e in look_aheads:
        f = folder+names[k][1]+"_"+str(e)+"_"+str(cut)+"_"+str_dict[2]
#         print("look for", f)
        dd[e] = read_dic(f)
    i = k//2
    j = k%2
    axs[j, i].set_title(names[k][1])
    color = iter(cm.rainbow(np.linspace(0, 1, len(look_aheads))))
    l1 = []
    l2 = []
    for a in look_aheads:
        m_it = max(dd[a].keys())
        if len(dd[a][1].keys())!=0 :
            m_ev1 = max(dd[a][1].keys())
            s1 = float(sum( (e/m_ev1)*dd[a][1][e]  for e in dd[a][1].keys() )/sum(dd[a][1].values()))
            l1.append(s1)
        else:
            l1.append(-1)
        if len(dd[a][m_it].keys())!=0 :
            m_ev2 = max(dd[a][m_it].keys())
            s2 = float(sum( (e/m_ev2)*dd[a][m_it][e]  for e in dd[a][m_it].keys() )/sum(dd[a][m_it].values()))
            l2.append(s2)
        else:
            l2.append(-1)
    c = next(color)
    axs[j, i].scatter( look_aheads ,l1, alpha=1/2, color = c, label= "iter 0 ")
    c = next(color)
    axs[j, i].scatter( look_aheads ,l2, alpha=1, color = c, label= "iter inf")
    axs[j, i].legend()

    axs[j,i].set(xlabel = "% lookahead" , ylabel='% average rewiring time', )

    axs[j,i].xaxis.set_label_coords(0.5,-0.07)

fig.set_size_inches(20, 12)
fig.savefig(folder_res+"expected_rewire.svg", format="svg", transparent = True, dpi=150)
fig.show()


# In[2]:


# def statistics_rewirings(path,names,cut,nb_rewire, iter_pandemy, folder, folder_res, approx, wl_it = -1,look_ahead = 1):
#     print("stats rewirings")
#     files = [f for f in os.listdir(folder_res) if os.path.isfile(folder_res+f)]
#     s = "table_charac.tex"
#     s2 = "table_diff_"+str(iter_pandemy)+".tex"
#     if s not in files:
#         fg = open(folder_res+s, "w")
#         fg2 = open(folder_res+s2, "w")
#         for k in range(0,len(names)):
#             print("statistics_rewirings",  names[k][0])
#             g = read_graph(path,names[k][0], names[k][-1])
#             g_new = graph_cut(g,names[k][-1], cut)
#             m = max(events(g_new))
#             names_keep = folder+names[k][1]+"_"+str(1)+"_"+str(cut)+"_keep"
#             max_wl = find_max_wl(folder, names_keep)
#             # if want max iteration do this
#             # col = read_dic(names_keep+"_"+str(max_wl))
#             col = read_dic(names_keep+"_"+str(1))
#             nb = 1
#             if nb != 0:
#                 x,y,z = randomize(g_new,nb_rewire,col,names[k][-1])
#                 fg.write(names[k][1])
#                 for e in [g_new,x,y,z]:
#                     fg.write(" & $ "+str(float(average_clustering_network2(e,approx)))[:5]+" $,")
#                     print("clust ok")
#                 for e in [g_new,x,y,z]:
#                     fg.write(" & $ "+str(float(global_efficiency(e,"s",approx)[0]))[:5]+" $ ,")
#                     print("effi shor ok")
#                 for e in [g_new,x,y,z]:
#                     fg.write(" & $ "+str(float(global_efficiency(e,"d",approx)[0]))[:5]+" $ ,")
#                     print("effi dur ok")
#                 fg.write("\\\\ \n")

#                 # DIFFUSIONS
#                 l = [g_new,x,y,z]
#                 rates = [0.1,0.2,0.3]
#                 average = dict( { e:dict(  {ee : 0 for ee in range(len(l)) } )  for e in rates }  )
#                 pan = 0
#                 while pan < iter_pandemy:
#                     print("pan", pan)
#                     for rate in rates:
#                         for i in range(len(l)):
#                             r = SI(l[i], rate, iterations = 100)
#                             average[rate][i] += (r[1]/iter_pandemy)
#                     pan += 1
#                 fg2.write(names[k][1]+" ")
#                 for rate in rates:
#                     for i in range(len(l)):
#                         fg2.write("$ "+str(float(average[rate][i]))[:5]+" $ ")
#                 fg2.write("\\\\ \n")

#         fg.close()
#         fg2.close()
#     else:
#         print("file statistics_rewirings already present")


# In[3]:


def statistics_rewirings_clus(path,names,cut,nb_rewire,folder, folder_res, iter_pandemy, wl_it = -1,look_ahead = 1):
    print("stats rewirings")
    files = [f for f in os.listdir(folder_res) if os.path.isfile(folder_res+f)]
    for zz in ["1","conv"]:
        s = "table_charac_"+zz+".tex"
        # s2 = "table_diff_"+str(iter_pandemy)+".tex"
        if s not in files:
            fg = open(folder_res+s, "w")
            # fg2 = open(folder_res+s2, "w")
            for k in range(0,len(names)):
                print("statistics_rewirings",  names[k][0])
                g = read_graph(path,names[k][0], names[k][-1])
                g_new = graph_cut(g,names[k][-1], cut)
                m = max(events(g_new))
                names_keep = folder+names[k][1]+"_"+str(1)+"_"+str(cut)+"_keep"
                max_wl = find_max_wl(folder, names_keep)
                # if want max iteration do this
                if zz == "1":
                    col = read_dic(names_keep+"_"+str(max_wl))
                else:
                    col = read_dic(names_keep+"_"+str(1))
                nb = 1
                if nb != 0:
                    m = len(g_new)
                    clus = [[],[],[]]
                    eff_s = [[],[],[]]
                    eff_d = [[],[],[]]
                    for z in range(nb_graphs):
                        print("randomize iteration number ", z+1)
                        ll = randomize(g_new,int(nb_rewire*m*math.log(m)),col,names[k][-1])
                        for ii in range(3):
                            clus[ii].append(float(average_clustering_network_imp(ll[ii])))
                            print("clust ok")
                            eff_s[ii].append(float(global_efficiency_cpp(ll[ii],"s")[0]))
                            print("effi shor ok")
                            eff_d[ii].append(float(global_efficiency_cpp(ll[ii],"d")[0]))
                            print("effi dur ok")

                    av_clus = [ sum(clus[ii])/nb_graphs  for ii in range(3)]
                    av_eff_s = [ sum(eff_s[ii])/nb_graphs  for ii in range(3)]
                    av_eff_d = [ sum(eff_d[ii])/nb_graphs  for ii in range(3)]

                    if nb_graphs > 1:
                        dev_clus = [ math.sqrt(sum( (clus[ii][jj]-av_clus[ii])**2 for jj in range(len(clus[ii]))  )/(nb_graphs - 1))  for ii in range(3)]
                        dev_clus = [0] + dev_clus
                        dev_eff_s = [ math.sqrt(sum( (eff_s[ii][jj]-av_eff_s[ii])**2 for jj in range(len(eff_s[ii]))  )/(nb_graphs - 1))  for ii in range(3)]
                        dev_eff_s = [0] + dev_eff_s
                        dev_eff_d = [ math.sqrt(sum( (eff_d[ii][jj]-av_eff_d[ii])**2 for jj in range(len(eff_d[ii]))  )/(nb_graphs - 1))  for ii in range(3)]
                        dev_eff_d = [0] + dev_eff_d
                    else:
                        dev_clus = [ 0  for ii in range(3)]
                        dev_clus = [0] + dev_clus
                        dev_eff_s = [ 0 for ii in range(3)]
                        dev_eff_s = [0] + dev_eff_s
                        dev_eff_d = [ 0  for ii in range(3)]
                        dev_eff_d = [0] + dev_eff_d


                    av_clus = [float(average_clustering_network_imp(g_new))] + av_clus
                    print("clus orig ok")
                    av_eff_s = [float(global_efficiency_cpp(g_new,"s")[0])] + av_eff_s
                    print("eff shor orig ok")
                    av_eff_d = [float(global_efficiency_cpp(g_new,"d")[0])] + av_eff_d
                    print("eff dur orig ok")

                    fg.write(names[k][1])
                    for e in range(4):
                        fg.write(" & $ "+str(av_clus[e])[:5]+" $,")
                    for e in range(4):
                        fg.write(" & $ "+str(av_eff_s[e])[:5]+" $ ,")
                    for e in range(4):
                        fg.write(" & $ "+str(av_eff_d[e])[:5]+" $ ,")
                    fg.write("\\\\ \n")

                    for e in range(4):
                        fg.write(" & $ "+str(dev_clus[e])[:5]+" $,")
                        print("clust ok")
                    for e in range(4):
                        fg.write(" & $ "+str(dev_eff_s[e])[:5]+" $ ,")
                        print("effi shor ok")
                    for e in range(4):
                        fg.write(" & $ "+str(dev_eff_d[e])[:5]+" $ ,")
                        print("effi dur ok")
                    fg.write("\\\\ \n")

                    # DIFFUSIONS
                    # l = [g_new,x,y,z]
                    # rates = [0.1,0.2,0.3]
                    # average = dict( { e:dict(  {ee : 0 for ee in range(len(l)) } )  for e in rates }  )
                    # pan = 0
                    # while pan < iter_pandemy:
                    #     print("pan", pan)
                    #     for rate in rates:
                    #         for i in range(len(l)):
                    #             r = SI(l[i], rate, iterations = 10)
                    #             average[rate][i] += (r[1]/iter_pandemy)
                    #     pan += 1
                    # fg2.write(names[k][1]+" ")
                    # for rate in rates:
                    #     for i in range(len(l)):
                    #         fg2.write("$ "+str(float(average[rate][i]))[:5]+" $ ")
                    # fg2.write("\\\\ \n")


            fg.close()
            # fg2.close()
        else:
            print("file statistics_rewirings already present")



# In[4]:


nb_rewire = 2
nb_pandemy = 1
statistics_rewirings_clus(path,names,cut,nb_rewire, folder, folder_res, nb_pandemy)
