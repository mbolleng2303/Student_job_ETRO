# import the necessary library
import numpy as np
from sklearn.cluster import AffinityPropagation

nbr_graphs = 4
nbr_node = 20# int(100*np.random.rand(nbr_graphs))
# create a dummy array
arr = np.random.rand(1024, nbr_graphs,nbr_node )
for k in range (1024) :
    for i in range(nbr_graphs) :
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













