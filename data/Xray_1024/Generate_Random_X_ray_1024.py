# import the necessary library
import numpy as np
from sklearn.cluster import AffinityPropagation

nbr_graphs = 100
nbr_node = 10# int(100*np.random.rand(nbr_graphs))
# create a dummy array
arr = np.random.rand(1024, nbr_graphs,nbr_node )
for k in range (nbr_graphs) :
    for i in range(nbr_node) :
        for j in range(nbr_node):
            arr[k,i,j]=int(100*arr[k,i,j])
# use the tofile() method
# and use ',' as a separator
# as we have to generate a csv file
arr.tofile('X_ray_1024_node.csv', sep=',')

# using loadtxt()
arr2 = np.reshape(np.loadtxt('X_ray_1024_node.csv',
                 delimiter=",", dtype=float), (1024,nbr_graphs,nbr_node))

e = np.random.rand(nbr_graphs, nbr_node, nbr_node)
tresh = 0.3
for k in range (nbr_graphs) :
    for i in range(nbr_node) :
        for j in range(nbr_node):
            if i == j :
                e[k,i,j]=1
            else :
                if e[k,i,j] <= tresh:
                    e[k,i, j] =0
                else :
                    e[k, i, j] = e[k, i, j]


e.tofile('X_ray_1024_edge.csv', sep=',')

# using loadtxt()
e2 = np.reshape(np.loadtxt('X_ray_1024_edge.csv',
                 delimiter=",", dtype=float), (nbr_graphs,nbr_node,nbr_node))



# create a dummy array
label = np.random.rand(1, nbr_graphs,nbr_node)
for i in range (nbr_graphs):
    for j in range (nbr_node):
        label[0,i,j] = int(3*label[0,i,j])



# use the tofile() method
# and use ',' as a separator
# as we have to generate a csv file
label.tofile('X_ray_1024_label.csv', sep=',')

# using loadtxt()
label2 = np.reshape(np.loadtxt('X_ray_1024_label.csv',
                 delimiter=",", dtype=float), (1,nbr_graphs,nbr_node))
a = 5



# credit to Stack Overflow user in the source link
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# some dummy data
word_vectors = ((1, 39),((1, 39)))

word_cosine = cosine_distances(word_vectors)
affprop = AffinityPropagation(affinity = 'precomputed', damping = 0.5)
af = affprop.fit(word_cosine)

b=2












