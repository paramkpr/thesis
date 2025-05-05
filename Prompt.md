Prompt

You are provided with a python script that is a simple, yet CORRECT version of a fair hierarchical clustering algorithm. 
You are also provided with two model classes: Node and Hierarchy. Your job is to write the sub routines in the given file such that: 

1. write a version of average linkage that outputs a `Hierarchy`
2. write a version of split root that outputs a `Hierarchy`
3. write a version of make fair that outputs a `Hierarchy`

you need to make sure that all the tree operations work as in the given code and you must rewrite them as part of providing me the algorithms
call out if the Node or Hierarchy classes do not provide a method that is required to implement one of the algorithms
we will go step by step, first we will implement average linkage


    n = 32
    c = 1
    eps = 1 / (c * math.log2(n)) # 1/16
    h = 4
    k = 2