# s_allow={0:[1],
#          1:[0],
#          2:[3],
#          3:[2],
#          40:[0,1],
#          41:[2,3]}
def kl_divergence(p1, q1, e=0.00001):
    p = p1+np.full(p1.shape, e)
    q = q1+np.full(q1.shape, e)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

s_allow={0:[1],
         1:[0],
         2:[3],
         3:[2],
         40:[0,1],
         41:[2,3]}
g1 = [0,1]
g2 = [2,3]

t1 = [40]
t2 = [41]

new_dist = {}
for i in lang_dist:
    s_id = int(i[0])
    if s_id!=n_population-1:
        allowed = s_allow[s_id]
        new_dist[s_id] = np.array(lang_dist[str(s_id)+'-'+str(s_id)].copy())
        for j in allowed:
            new_dist[s_id]+=np.array(lang_dist[str(s_id)+'-'+str(j)])
        new_dist[s_id]/=np.sum(np.array(new_dist[s_id]).flatten())
    else:
        multi = [s_id*10+j for j in range(n_villages)]
        for k in multi:
            allowed = s_allow[k]
            new_dist[k] = np.zeros((5,5))
            for j in allowed:
                new_dist[k]+=np.array(lang_dist[str(s_id)+'-'+str(j)])
            new_dist[k]/=np.sum(np.array(new_dist[k]).flatten())



import itertools

g=list(itertools.product(g1, g2))
# g.extend(list(itertools.product(g2, g3)))
# g.extend(list(itertools.product(g1, g1)))
across = []
for i,j in g:
    p, q = np.array(new_dist[i]), np.array(new_dist[j])
    across.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
print('local lanuage diff across regions', np.mean(across))

g=list(itertools.combinations(g1,2))
g.extend(itertools.combinations(g2,2))
# g.extend(itertools.combinations(g3,2))
within = []
for i,j in g:
    p, q = np.array(new_dist[i]), np.array(new_dist[j])
    within.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
print('local language diff within region ', np.mean(within))



t=list(itertools.product(t1, t2))
# t.extend(list(itertools.product(t3, t2)))
# t.extend(list(itertools.product(t1, t3)))
across = []
for i,j in t:
    p, q = np.array(new_dist[i]), np.array(new_dist[j])
    across.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
print('traveller difference in language in diff regions ', np.mean(across))


t=list(itertools.combinations(t1,2))
t.extend(itertools.combinations(t2,2))
# t.extend(itertools.combinations(t3,2))





t=list(itertools.product(t1, g1))
t.extend(list(itertools.product(t2, g2)))
# t.extend(list(itertools.product(t3, g3)))
across = []
for i,j in t:
    p, q = np.array(new_dist[i]), np.array(new_dist[j])
    print(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
    across.append(kl_divergence(p,q, 1e-4)+kl_divergence(q,p, 1e-4))
print('traveller difference in language with locals', np.mean(across))







        
