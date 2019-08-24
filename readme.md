# biBIT\*
a bidirectional version of BIT\*

## Modifiaction from BIT\*

two trees
* **GTree** 
    * root: start
* **HTree**
    * root: goal
* alg aims at connect two tree

### Data Structure

#### Vertex Information(added):

* isGTree
* isConnectedWithAnotherTree

#### Graph Information

* treeConnPair - connection edge between two trees
* treeConnPairs
* bestConnPairs - represent the current solution

Expand Edge:
* To FreeSample
* To SameTree -  rewire
* To AnotherTree - new solution

### Heuristic / SearchQueue

costToGo/costToCome

* Vertex Queue Value: `g_T(v) + c_hat(v,w) + h_T(w)`, `w` is the nearest vertex in Htree.
    * `c_hat(v,w)`
* Vertex Enqueue Value: `g_hat(v) + h_hat(v) < bestCost`
* Edge Queue Value: `g_T(v) + c_hat(v,x) + c_hat(x,w) + h_T(w)`
    * `c_hat(v,x) + c_hat(x,w)`, `c_hat(v,x)`
* Edege Enqueue Value: `g_hat(v) + c_hat(v,x) + h_hat(x)`

### Sample Bias

TODO

to connect two trees quickly

