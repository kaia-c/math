"""
Assignment 1
Kaia Chapman
Purpose:
Create 3 directed graphs based on pictures in assignment. Write funtions to
make a complete graph (graph with max number of neighbors associated with each
key), compute_in_degrees of a given dimentional graph, and to compute
in_degree_distribution of a given dimentional graph.

"""
import random
#Define three constants whose values are dictionaries corresponding to the
#three directed graphs shown below.
EX_GRAPH0 = {0: set([1, 2]),
             1: set([]),
             2: set([])};
EX_GRAPH1 = {0: set([1, 4, 5]),
             1: set([2, 6]),
             2: set([3]),
             3: set([0]),
             4: set([1]),
             5: set([2]),
             6: set([])}
EX_GRAPH2 = {0: set([1,4,5]),
             1: set([2,6]),
             2: set([3,7]),
             3: set([7]),
             4: set([1]),
             5: set([2]),
             6: set([]),
             7: set([3]),
             8: set([1,2]),
             9: set([0,4,5,6,7,3])
             };
print(EX_GRAPH1.values())



def make_complete_graph(num_nodes):
    """
    Takes the number of nodes num_nodes and returns a dict with:
    keys=ints in range 0 -> num_nodes-1, values=max connections (num_nodes-1)
    """
    directed_graph_dict={}
    if num_nodes > 0:
        for i in range(0, num_nodes):            
            directed_graph_dict[i]={j for j in range(0, num_nodes) if (j!=i)}
    return directed_graph_dict

#testing
directed_graph_dict=make_complete_graph(8)
print (directed_graph_dict.items())


def compute_in_degrees(digraph):
    """
    Takes a directed graph digraph (represented as a dictionary) and return dict with:
    keys=same int ids in range 0 -> num_nodes-1, values=in-degrees
    Counts number of times 

    NOTE FOR WHOEVER GRADES THIS...As you may guess looking at the below code,
    I'm familiar with more c-like languages then python. I know there has to be
    some 1 or 2 line solution using dict comprehension, but I can't quite see it.
    If anyone with more experience has a 'pythonic' solution they would share,
    I'd really appreciate it!
    """
    degrees_dict=dict.fromkeys(digraph.keys(),[])
    for every_key in degrees_dict:
        count=0
        for key in digraph:
            for item in digraph[key]:
                if (item==every_key):
                    count+=1
        degrees_dict[every_key]=count
    return degrees_dict

#testing
#degrees_dict=compute_in_degrees(EX_GRAPH1)
#print(degrees_dict.items())



def in_degree_distribution(digraph):
    """
    Takes a directed graph digraph (represented as a dictionary) and computes the
    unnormalized distribution of the in-degrees of the graph. Returns a dict with:
    keys = each in-degree from digraph, values=count of nodes with in-degree
    """
    degrees_dict=compute_in_degrees(digraph)
    in_degree_dist_dict={}
    for key in degrees_dict:
        if degrees_dict[key] not in in_degree_dist_dict.keys():
            in_degree_dist_dict[degrees_dict[key]]=1
        else:
            in_degree_dist_dict[degrees_dict[key]]+=1
    return in_degree_dist_dict

#testing
in_degree_dist_dict=in_degree_distribution(EX_GRAPH2)
print(in_degree_dist_dict.items())

def testER(nodes, prob):
    v={x:0 for x in range(0, nodes-1)}
    e=0
    for i in v:
        if (i!=v[i]):
            a=random.random()
            if a<prob:
                v[i]={x for x in range(v[i], i)}
                if(e not in v[i]):
                    v[i].add(e)
    return v
c=testER(50,.5)
print(c.items())
       
