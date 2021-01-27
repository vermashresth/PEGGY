



import numpy as np
arr = np.load('sp_utter.npy', allow_pickle=True)
arr
# size = (s_pop, islands, conc)
vocab = 5
def freq(lst):
    d = {}
    for i in lst:
        if d.get(i):
            d[i] += 1
        else:
            d[i] = 1
    return d
def distr(lst):
    ar = [0 for _ in range(vocab)]
    d = freq(lst)
    for k in d:
        ar[k] = d[k]
    return ar
def kl_divergence(p1, q1, e=0.00001):
    p = p1+np.full(p1.shape, e)
    q = q1+np.full(q1.shape, e)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


sp_isle_dict = {}
for speaker_id, i in enumerate(arr):
    for island_id, j in enumerate(i):
        bidim_distr = []
        for concept_id, k in enumerate(j):
            my_distr = distr(k)
            if sum(k)!=0:
              bidim_distr.append(my_distr)
        if len(bidim_distr)>0:
            print(speaker_id,island_id)
            sp_isle_dict[(speaker_id,island_id)] = np.array(bidim_distr)/np.sum(np.array(bidim_distr))











islands = 3
import itertools
speakers = 6
g=list(itertools.combinations(list(range(speakers)),2))
# g.extend(itertools.combinations(g2,2))
travellers = [6,7]

within = []
for j in range(islands):
    for sp1, sp2 in g:
        if (sp1, j) in sp_isle_dict.keys() and (sp2, j) in sp_isle_dict.keys():
            p = sp_isle_dict[(sp1, j)]
            q = sp_isle_dict[(sp2, j)]
            within.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))

print('local lanuage diff within regions', np.mean(within))

g=list(itertools.combinations(list(range(speakers)),2))
ig = list(itertools.combinations(list(range(islands)),2))
# g.extend(itertools.combinations(g2,2))
across = []
for i1, i2 in ig:
    for sp1, sp2 in g:
        if (sp1, i1) in sp_isle_dict.keys() and (sp2, i2) in sp_isle_dict.keys():
            p = sp_isle_dict[(sp1, i1)]
            q = sp_isle_dict[(sp2, i2)]
            across.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
        if (sp1, i2) in sp_isle_dict.keys() and (sp2, i1) in sp_isle_dict.keys():
            p = sp_isle_dict[(sp1, i2)]
            q = sp_isle_dict[(sp2, i1)]
            across.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))

print('local lanuage diff across regions', np.mean(across))

base=[]
for sp1, sp2 in g:
    for i in range(islands):
        for j in range(islands):
                if (sp1, i) in sp_isle_dict.keys() and (sp2, j) in sp_isle_dict.keys():
                    p = sp_isle_dict[(sp1, i)]
                    q = sp_isle_dict[(sp2, j)]
                    base.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
print("Baseline population difference in language", np.mean(base))

diff = []
for sp in travellers:
    for i1, i2 in ig:
        if (sp, i1) in sp_isle_dict.keys() and (sp, i2) in sp_isle_dict.keys():
            p = sp_isle_dict[(sp, i1)]
            q = sp_isle_dict[(sp, i2)]
            diff.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
print('traveller difference in language in diff regions ', np.mean(diff))





# arr
# travellers = [9,10]
# def freq(lst):
#     d = {}
#     for i in lst:
#         if d.get(i):
#             d[i] += 1
#         else:
#             d[i] = 1
#     return d
# def distr(lst):
#     ar = [0 for _ in range(vocab)]
#     d = freq(lst)
#     for k,v in d:
#         ar[k] = v
#     return ar
# def kl_divergence(p1, q1, e=0.00001):
#     p = p1+np.full(p1.shape, e)
#     q = q1+np.full(q1.shape, e)
#     return np.sum(np.where(p != 0, p * np.log(p / q), 0))
# size = (s_pop, islands, conc)
# vocab
#
# sp_isle_dict = {}
# for speaker_id, i in enumerate(arr):
#     for island_id, j in enumerate(i):
#         bidim_distr = []
#         for concept_id, k in enumerate(j):
#             my_distr = distr(k)
#             bidim_distr.append(my_distr)
#         if len(bidim_distr)>0:
#             sp_isle_dict[(i,j)] = np.array(bidim_distr)
#
#
# g=list(itertools.combinations(list(range(speakers)),2))
# # g.extend(itertools.combinations(g2,2))
# within = []
# for j in range(islands):
#     for sp1, sp2 in g:
#         if (sp1, j) in sp_isle_dict.keys() and (sp2, j) in sp_isle_dict.keys():
#             p = sp_isle_dict[(sp1, j)]
#             q = sp_isle_dict[(sp2, j)]
#             within.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
#
# print('local lanuage diff within regions', np.mean(within))
#
# g=list(itertools.combinations(list(range(speakers)),2))
# ig = list(itertools.combinations(list(range(islands)),2))
# # g.extend(itertools.combinations(g2,2))
# across = []
# for i1, i2 in ig:
#     for sp1, sp2 in g:
#         if (sp1, i1) in sp_isle_dict.keys() and (sp2, i2) in sp_isle_dict.keys():
#             p = sp_isle_dict[(sp1, i1)]
#             q = sp_isle_dict[(sp2, i2)]
#             across.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
#         if (sp1, i2) in sp_isle_dict.keys() and (sp2, i1) in sp_isle_dict.keys():
#             p = sp_isle_dict[(sp1, i2)]
#             q = sp_isle_dict[(sp2, i1)]
#             across.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
#
# print('local lanuage diff across regions', np.mean(across))
#
#
# diff = []
# for sp in travellers:
#     for i1, i2 in ig:
#         if (sp, i1) in sp_isle_dict.keys() and (sp, i2) in sp_isle_dict.keys():
#             p = sp_isle_dict[(sp, i1)]
#             q = sp_isle_dict[(sp, i2)]
#             diff.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
# print('traveller difference in language in diff regions ', np.mean(diff))
#
#
#
# base=[]
# for sp1, sp2 in g:
#     for i in range(islands):
#         for j in range(islands):
#                 if (sp1, i) in sp_isle_dict.keys() and (sp2, j) in sp_isle_dict.keys():
#                     p = sp_isle_dict[(sp1, i)]
#                     q = sp_isle_dict[(sp2, j)]
#                     base.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
# print("Baseline population difference in language beween any two speaker")
#
# same = []
# for sp in travellers:
#     for j in range(islands):
#         if (sp, i1) in sp_isle_dict.keys() and (sp, i2) in sp_isle_dict.keys():
#             p = sp_isle_dict[(sp, i1)]
#             q = sp_isle_dict[(sp, i2)]
#             diff.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
#
# print('traveller difference in language with locals', np.mean(across))
