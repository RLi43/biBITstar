#!/usr/bin/env python3
#coding:utf-8

### ---------------
# BIBITSTAR - biBiT* - bidirectional Batch Informed Tree 
# Author: RLi
### ---------------

import copy
import math
import platform
import random
import time
import numpy as np
import queue

import matplotlib.pyplot as plt

show_animation = True

### helpful functions
def _dis(x1,x2):
    return np.linalg.norm(np.array(x1)-np.array(x2))    

### map
# including problem defination and enviroment
# frankly, it has every thing other than a planner
class Map:
    def __init__(self,dim=2,obs_num=20,obs_size_max=2.5,xinit=[0,0],xgoal=[23,23],randMax=[30,30],randMin=[-5,-5]):
        self.dimension = dim
        self.xinit = xinit
        self.xgoal = xgoal
        self.randMax = randMax
        self.randMin = randMin
        self.obstacles = []
        self.DISCRETE = 0.05

        self.obstacles = [[3,3,3],[10,10,5],[4,15,3],[20,5,4],[7,6,2],[15,25,3],[20,13,2],[0,20,3],[17,17,2]]
        # # random obstacles
        # for i in range(obs_num):
        #     #TODO
        #     ob = []
        #     for j in range(dim):
        #         ob.append(random.random()*20+1.5)
        #     ob.append(random.random()*obs_size_max+0.2)
        #     self.obstacles.append(ob)

        ## informed data    
        self.cMin = _dis(self.xinit,self.xgoal)
        self.xCenter = (np.array(xinit)+np.array(xgoal))/2
        a1 = np.transpose([(np.array(xgoal)-np.array(xinit))/self.cMin])        
        # first column of idenity matrix transposed
        id1_t = np.array([1.0]+[0.0,]*(self.dimension-1)).reshape(1,self.dimension)
        M = a1 @ id1_t
        U, S, Vh = np.linalg.svd(M, 1, 1)
        self.C = np.dot(np.dot(U, 
            np.diag([1.0,]*(self.dimension-1)+[np.linalg.det(U) * np.linalg.det(np.transpose(Vh))]))
            , Vh)

    ## collision
    def collision(self,x):
        for ob in self.obstacles:
            if _dis(x,ob[:-1])<=ob[-1]:
                return True
        return False
    def collisionLine(self,x1,x2):
        dis = _dis(x1,x2)
        if dis<self.DISCRETE:
            return False
        nums = int(dis/self.DISCRETE)
        direction = (np.array(x2)-np.array(x1))/_dis(x1,x2)
        for i in range(nums+1):
            x = np.add(x1 , i*self.DISCRETE*direction)
            if self.collision(x): return True
        if self.collision(x2): return True
        return False

    ## sample
    def randomSample(self):
        x = []
        for j in range(self.dimension):
            x.append(random.random()*(self.randMax[j]-self.randMin[j])+self.randMin[j])
        return x
    def freeSample(self):
        x = self.randomSample()
        while self.collision(x):
            x = self.randomSample()
        return x
    def inLimit(self,x):
        for i in range(self.dimension):
            if x[i]>self.randMax[i] or x[i] < self.randMin[i]:
                return False
        return True
    def informedSample(self,cMax):
        L = np.diag([cMax/2]+[math.sqrt(cMax**2-self.cMin**2)/2,]*(self.dimension-1))
        cl = np.dot(self.C,L)
        while True:
            x = np.dot(cl,self.ballSample())+self.xCenter
            if not self.collision(x) and self.inLimit(x): break
        return list(x)
    def ballSample(self):
        ret = []
        for i in range(self.dimension):
            ret.append(random.random()*2-1)
        ret = np.array(ret)
        return ret/np.linalg.norm(ret)*random.random()

    # animation 
    def drawMap(self):
        if self.dimension==2:
            plt.clf()
            sysstr = platform.system()
            if(sysstr =="Windows"):
                scale = 16
            elif(sysstr == "Linux"):
                scale = 20
            else: scale = 20
            for (ox, oy, size) in self.obstacles:
                plt.plot(ox, oy, "ok", ms=scale * size)
            
            plt.plot(self.xinit[0],self.xinit[1], "xr")
            plt.plot(self.xgoal[0],self.xgoal[1], "xr")
            plt.axis([self.randMin[0]-2,self.randMax[0]+2,self.randMin[1]-2,self.randMax[1]+2])
            plt.grid(True)

### planner 
class BiBITstar(object):
    def __init__(self,_map,maxIter =300, bn=10):
        # parameters
        self.map = _map
        self.batchSize = bn
        self.maxIter = maxIter
        # solution
        self.bestCost = float('inf')
        self.hasExactSolution = False
        self.bestConn = None

        self.x = [_map.xinit,_map.xgoal] # store all the point(samples, vertices)
        self.r = float('inf')
        self.qe = queue.PriorityQueue() # ecost,[vtind, xind]
        self.qv = queue.PriorityQueue() # the index in Tree
        self.vold = []                  # the index in Tree
        self.Tree = [0,1]
        self.X_sample = []
        # because python alway copy a vertex for me ... :(
        self.isGtree = [True,False] # accroding to the order in Tree
        self.cost = [0,0]           # accroding to the order in Tree
        self.parent = [None,None]   # accroding to the order in tree
        self.children = [[],[]]     # accroding to the order in Tree
        self.depth = [0,0]          # accroding to the order in Tree
        self.conn = {}

        self.rewire = False
        self.pruned = []

        self.pruneNum = 0
        self.gVertexNum = 1
        self.hVertexNum = 1
        # added other method
        self.beacons = None            # the index in Tree

        # initialization
        self.qv.put((self.distance(0,1),0))
        self.qv.put((self.distance(0,1),1))

        # if show_animation:
        #     self.map.drawMap()

    # helpful functions    
    def qeAdd(self,vt,x):
        self.qe.put((self.edgeQueueValue(vt,x),[vt,x]))

    def bestInQv(self):
        vc,vmt = self.qv.get()
        self.qv.put((vc,vmt))
        return (vc,vmt)
    def bestInQe(self):
        ec,[wmt,xm] = self.qe.get()
        self.qe.put((ec,[wmt,xm]))
        return (ec,[wmt,xm])
    def getCost(self,ind):
        try:
            tind = self.Tree.index(ind)
            cost = self.cost[tind]
            return cost
        except:
            return float('inf')

    # -- Main Part --
    # decide the order
    # ---------------
    def vertexQueueValue(self,vt):
        if not self.hasExactSolution:
            return self.cost[vt] + self.costTjHeuristicVertex(vt)
        v = self.Tree[vt]
        vn,dis = self.nearest(v,not self.isGtree[vt])
        return dis
        
    def edgeQueueValue(self,wt,x):
        if self.hasExactSolution:
            return self.cost[wt] + self.distance(self.Tree[wt],x) + self.costTgHeuristic(x,self.isGtree[wt])
        vn,dis = self.nearest(x,self.isGtree[wt]) # !!problem: can't update during a batch
        return self.distance(self.Tree[wt],x) + dis
        
    # -- costHelper
    def lowerBoundHeuristicEdge(self,vt,x):
        return self.costFgHeuristic(self.Tree[vt],not self.isGtree[vt]) + \
                    self.costFgHeuristic(x, self.isGtree[vt]) + \
                        self.distance(self.Tree[vt],x)
    def lowerBoundHeuristicVertex(self,x):
        x = self.Tree[x]
        return self.costFgHeuristic(x,True) + self.costFgHeuristic(x,False) 
    def lowerBoundHeuristic(self,x):
        return self.costFgHeuristic(x,True) + self.costFgHeuristic(x,False) 
    def costFgHeuristic(self,x,h=False):
        if h: target = 1
        else: target = 0
        return self.distance(target,x)

    def costTgHeuristic(self,ind,h=False):
        Vnearest,nearDis = self.nearest(ind,not h)
        return self.cost[Vnearest] + nearDis

    def costTjHeuristicVertex(self,vt,i=False):
        if i:
            return self.cost[vt]
        else:
            return self.costTgHeuristic(self.Tree[vt],self.isGtree[vt])
    def costTjVertex(self,vertex,i=False):
        if i:
            return self.getCost(vertex)
        else:
            try:
                vcon = self.conn[vertex]
                return self.cost[vcon] + self.distance(vertex,vcon)
            except:
                return float('inf')
    # --

    
    def nearest(self,indx, inGtree):
        nearestDis = float('inf')
        vn = None
        for vt in range(len(self.Tree)):
            if(self.isGtree[vt] == inGtree):
                dis = self.distance(self.Tree[vt],indx)
                if dis < nearestDis:
                    vn = vt
                    nearestDis = dis
        return vn,nearestDis

    def distanceTree(self,tind1,tind2):
        return np.linalg.norm(np.array(self.x[self.Tree[tind1]]) - np.array(self.x[self.Tree[tind2]]))
    def distance(self,ind1,ind2):
        return np.linalg.norm(np.array(self.x[ind1]) - np.array(self.x[ind2]))

    # -- main --
    def solve(self):
        for iterateNum in range(self.maxIter):
            print("iter: ",iterateNum)
            if show_animation:
                self.drawGraph()
            if self.isEmpty():
                print("newBatch")
                self.newBatch()
            else:
                ec,[wmt,xm] = self.qe.get()
                wm = self.Tree[wmt]
                if self.lowerBoundHeuristicEdge(wmt,xm) > self.bestCost:
                    # end it.
                    while not self.qe.empty():
                        self.qe.get()
                    while not self.qv.empty():
                        vc,vt = self.qv.get()
                        self.vold.append(vt)
                    continue
                
                ## ExpandEdge
                if self.collisionEdge(wm,xm):
                    continue
                # it's a simple demo, we don't care too much about time-cost
                # if there's no collision, we add this edge.
                trueEdgeCost = self.distance(wm,xm)
                try:
                    xmt = self.Tree.index(xm)
                    isG = self.isGtree[xmt]                    
                    # same tree
                    ## delay rewire?
                    if isG == self.isGtree[wmt]:
                        if self.cost[wmt] + trueEdgeCost >= self.cost[xmt]:
                            continue
                        oldparent = self.parent[xmt]
                        self.children[oldparent].remove(xmt)
                        self.parent[xmt] = wmt
                        self.children[wmt].append(xmt)
                        self.cost[xmt] = self.cost[wmt] + trueEdgeCost # has not update the children TODO
                        self.depth[xmt] = self.depth[wmt] + 1
                        self.updateCost(xmt)

                    # another tree
                    else:
                        try:
                            wcont = self.conn[wmt]
                            if self.cost[wcont] + self.distance(wm,self.Tree[wcont]) <= \
                                self.cost[xmt] + self.distance(wm,xm):
                                continue
                            self.conn.pop(wcont)
                        except:
                            pass
                        try:
                            xcont = self.conn[xmt]                        
                            if self.cost[xcont] + self.distance(xm,self.Tree[xcont]) <= \
                                self.cost[wmt] + self.distance(xm,wm):
                                continue
                            self.conn.pop(xcont)
                        except:
                            pass
                        # update or create one                        
                        self.conn[wmt] = xmt
                        self.conn[xmt] = wmt
                        newCost = self.cost[wmt] + self.cost[xmt] + self.distance(wm,xm)
                        if newCost < self.bestCost:
                            if not self.hasExactSolution:
                                self.batchSize -= 3 #test
                                self.hasExactSolution = True
                            # new better solution
                            self.bestCost = newCost
                            self.bestConn = [wmt,xmt]
                            self.smartPath()

                            # update beacons:
                            self.beacons = []
                            curv = wmt
                            while curv != None:
                                self.beacons.append(curv)
                                curv = self.parent[curv]                                
                            curv = xmt
                            while curv != None:
                                self.beacons.append(curv)
                                curv = self.parent[curv]

                            # report?
                            if self.bestCost == self.map.cMin:
                                break
                # v->sample
                except:
                    xmt = len(self.Tree)
                    self.Tree.append(xm)
                    self.isGtree.append(self.isGtree[wmt])
                    self.parent.append(wmt)
                    self.children[wmt].append(xmt)
                    self.children.append([])
                    self.cost.append(self.cost[wmt] + trueEdgeCost)
                    self.depth.append(self.depth[wmt]+1)
                    self.X_sample.remove(xm)
                    self.qv.put((self.vertexQueueValue(xmt),xmt))
                    if self.isGtree[xmt]:
                        self.gVertexNum += 1
                    else:
                        self.hVertexNum += 1

        if not self.hasExactSolution:
            print("plan failed")
        else:
            # self.updateCost(0)
            # self.updateCost(1)
            if show_animation:
                path = self.getPath()       
                plt.plot([self.x[ind][0] for ind in path], [self.x[ind][1] for ind in path], '-o') 
                # plt.show() 
            print("plan finished with cost: ",self.bestCost)
        # print plan information
        print("Plan Info:")
        print("total samples:",len(self.x),"Gtree:",self.gVertexNum,"Htree:",self.hVertexNum)
        print("pruned: v: ",len(self.pruned)," sample:",len(self.x)-len(self.X_sample)-len(self.Tree),len(self.pruned)-(len(self.Tree)-self.gVertexNum-self.hVertexNum))
        ## TODO more informations?
    
    def smartPath(self,nearRewire=True):
        if nearRewire:
            for i in range(2):
                cvt = self.bestConn[i]
                while cvt!=None:
                    ## expand vertex
                    # # expand to free sample
                    v = self.Tree[cvt]
                    # xnearby = self.nearby(v,self.X_sample)
                    # for xind in xnearby:
                    #     if self.edgeInsertConditionSample(vt,xind):
                    #         self.qeAdd(vt,xind)

                    ## expand to tree
                    # expand to the same tree
                    inear = self.nearbyT(v,self.isGtree[cvt])
                    for ivt in inear:
                        xm = self.Tree[ivt]
                        if self.edgeInsertConditionSameTree(cvt,ivt) and not self.collisionEdge(v,xm):
                            # self.qeAdd(cvt,self.Tree[ivt])
                            trueEdgeCost = self.distance(v,xm)
                            if self.cost[cvt] + trueEdgeCost >= self.cost[ivt]:
                                continue
                            oldparent = self.parent[ivt]
                            self.children[oldparent].remove(ivt)
                            self.parent[ivt] = cvt
                            self.children[cvt].append(ivt)
                            self.cost[ivt] = self.cost[cvt] + trueEdgeCost # has not update the children TODO
                            self.depth[ivt] = self.depth[cvt] + 1
                            self.updateCost(ivt)

                    # # expand to another tree
                    # jnear = self.nearbyT(v,not self.isGtree[cvt])
                    # for jvt in jnear:
                    #     if self.edgeInsertConditionAnotherTree(cvt,jvt):
                    #         self.qe.put((0,[cvt,self.Tree[jvt]]))
                    #         # self.qeAdd(cvt,self.Tree[jvt])
                    cvt = self.parent[cvt]
        """ # RRT*-SMART: A Rapid Convergence Implementation of RRT*"""
        self.beacons = []
        for i in range(2):
            cvt = self.bestConn[i]
            self.beacons.append(cvt)
            while True:
                if self.beacons[-1] != cvt:
                    self.beacons.append(cvt)
                pvt = self.parent[cvt]
                if pvt == None:
                    break
                gpvt = self.parent[pvt]
                if gpvt == None:
                    break
                # because: self.cost[gpvt] + self.distanceTree(cvt,gpvt) < self.cost[cvt] : triangle
                if not self.collisionEdge(self.Tree[cvt],self.Tree[gpvt]):
                    self.parent[cvt] = gpvt
                    self.children[pvt].remove(cvt)
                    self.children[gpvt].append(cvt)
                    self.cost[cvt] = self.cost[gpvt] + self.distanceTree(cvt,gpvt)
                    # cvt = gpvt
                cvt = pvt
        self.updateCost(0)
        self.updateCost(1)
        

    ## ---
    # while BestQueueValue(Qv) <= BestQueueValue(Qe):
    #     ExpandVertex(BestValueIn(Qv))
    def isEmpty(self):
        while not self.qv.empty():
            if self.qe.empty():
                self.expandVertex()
            else:
                vcost,vmt = self.bestInQv()
                ecost,[wmt,xm] = self.bestInQe()
                if(ecost>=vcost):
                    self.expandVertex()
                else:
                    break
        while self.qe.empty() and not self.qv.empty():
            self.expandVertex()

        return self.qe.empty()

    # f_hat(v,x) < bestCost
    def edgeInsertConditionSample(self,vt,xind):
        return  self.lowerBoundHeuristicEdge(vt,xind) < self.bestCost
    # f_hat(v,x) < bestCost AND (better solution)
    # Ti_hat(v) + c(v,x) < Ti(x) (optimal tree)
    def edgeInsertConditionSameTree(self,vt,ivt):
        if self.parent[vt] == ivt:
            return False
        if self.parent[ivt] == vt:
            return False
        v = self.Tree[vt]
        iv = self.Tree[ivt]
        costTargetHeuristic = self.costFgHeuristic(v,not self.isGtree[vt]) + \
                                self.distance(v,iv)
        return costTargetHeuristic < self.cost[ivt] and \
                self.costFgHeuristic(iv, self.isGtree[vt]) + \
                    costTargetHeuristic < self.bestCost 
    def edgeInsertConditionAnotherTree(self,vt,jvt):
        v = self.Tree[vt]
        jv = self.Tree[jvt]
        cvx = self.distance(v,jv)
        if self.costFgHeuristic(v,not self.isGtree[vt]) + \
                self.costFgHeuristic(jv, self.isGtree[vt]) + \
                    cvx >= self.bestCost:
            return False
        # if it is better than current connEdge
        try:
            vcont = self.conn[vt]
            if vcont == jvt or self.cost[jvt] + cvx > self.cost[vcont] + self.distance(self.Tree[vcont],v):
                return False
        except:
            pass
        try:
            jcont = self.conn[jvt]
            if jcont == vt or self.cost[vt] + cvx > self.cost[jcont] + self.distance(self.Tree[jcont],jv):
                return False
        except:
            pass
        return True

    def expandVertex(self):
        (vcost,vt) = self.qv.get()
        self.vold.append(vt)
        if self.lowerBoundHeuristicVertex(vt) > self.bestCost:
            while not self.qv.empty():
                vc,vt = self.qv.get()
                self.vold.append(vt)
        else:
            ## expand vertex
            # expand to free sample
            v = self.Tree[vt]
            xnearby = self.nearbyS(v)
            for xind in xnearby:
                if self.edgeInsertConditionSample(vt,xind):
                    self.qeAdd(vt,xind)

            ## expand to tree
            # expand to the same tree
            # delay rewire?
            if self.rewire:
                inear = self.nearbyT(v,self.isGtree[vt])
                for ivt in inear:
                    if self.edgeInsertConditionSameTree(vt,ivt):
                        self.qeAdd(vt,self.Tree[ivt])
            # expand to another tree
            jnear = self.nearbyT(v,not self.isGtree[vt])
            for jvt in jnear:
                if self.edgeInsertConditionAnotherTree(vt,jvt):
                    # TODO if there's no solution, should we give some reward?
                    # if self.hasExactSolution:
                    #     self.qeAdd(vt,self.Tree[jvt])
                    # else:
                    self.qe.put((0,[vt,self.Tree[jvt]]))
            # # trying hard to connect...
            # if len(jnear) == 0:
            #     jnt,dis = self.nearest(v,not self.isGtree[vt])
            #     self.qe.put((0,[vt,self.Tree[jnt]]))


    """
    return nearby(self.r) x in thelist
    """
    def nearbyS(self,vind):
        near = []
        for ind in self.X_sample: # 太暴力…… 下次试试r近邻……
            if self.distance(ind,vind) < self.r:
                near.append(ind)
        return near
    def nearbyT(self,vind,isG):
        near = []
        for ti in range(len(self.Tree)):
            if self.isGtree[ti] == isG:
                if self.distance(vind, self.Tree[ti]) < self.r:
                    near.append(ti)
        return near

    import math
    # to avoid too much sample near the tree, speed up the connection of trees
    # 新的机遇！
    def betweenTreesSample(self):
        while True:
            x = self.map.freeSample()
            if self.r == float('inf'):
                break
            # this rejection should be impled before collision test
            disg = float('inf')
            for v in self.Tree:
                if v == None:
                    continue
                dis = np.linalg.norm(np.array(self.x[v]) - np.array(x))
                if dis < disg:
                    disg = dis
            # penalize the dis < self.r
            # tan(45) = 1
            if disg > self.r: break
            prob = math.tan( math.radians(disg/self.r * 45) )
            if random.random() <= prob:
                break
        return x
    def betweenTreesSampleInformed(self):
        avoidList = list(range(len(self.x)))
        for px in self.pruned:
            avoidList.remove(px)
        while True:
            x = self.map.informedSample(self.bestCost)
            if self.r == float('inf'):
                break
            # this rejection should be impled before collision test
            disg = float('inf')
            for v in avoidList:
                dis = np.linalg.norm(np.array(self.x[v]) - np.array(x))
                if dis < disg:
                    disg = dis
            # penalize the dis < self.r
            # tan(45) = 1
            if disg > self.r: break
            prob = math.tan( math.radians(disg/self.r * 45) )
            if random.random() <= prob:
                break
        return x
    
    # 描边大师
    def intelligentSample(self):
        #TODO
        if self.hasExactSolution:
            while True:
                randt = random.randint(0,len(self.beacons)-1)
                v = self.Tree[self.beacons[randt]]
                x = self.x[v] + self.map.ballSample() * self.r 
                if not self.map.collision(x):
                    break
            return x


    def sample(self):
        if not self.hasExactSolution:
            for i in range(self.batchSize):
                self.X_sample.append(len(self.x))
                #self.x.append(self.map.freeSample())
                self.x.append(self.betweenTreesSample())
        else:
            for i in range(self.batchSize - 3):
                self.X_sample.append(len(self.x))
                self.x.append(self.betweenTreesSampleInformed())
                # self.x.append(self.map.informedSample(c))
            for i in range(3):
                self.X_sample.append(len(self.x))
                self.x.append(self.intelligentSample())

    def collisionEdge(self,vind,xind):
        return self.map.collisionLine(self.x[vind],self.x[xind])

    def newBatch(self):
        # --debug
        while not self.qv.empty():
            vc,vt = self.qv.get()
            self.vold.append(vt)
        while not self.qe.empty():
            self.qe.get()
        # --
        # self.updateCost()
        for item in self.conn:
            self.smartPath([item,self.conn[item]])
        self.prune()
        if self.hasExactSolution:
            self.rewire = True
        self.sample()
        self.updateRadius()
        while len(self.vold) > 0:
            vt = self.vold.pop()
            self.qv.put((self.vertexQueueValue(vt),vt))
    
    def updateRadius(self):
        q = len(self.x) - len(self.pruned)
        self.r = self.radius(q)

    def radius(self,q):
        return 30.0 * math.sqrt((math.log(q) / q))

    # update the cost of vertex (might be out-of-date because of rewire)
    # there shoule be some more efficient way (but it's just a simple demo ...
    def updateCost(self,vt):
        waitingToUpdate = queue.Queue()
        for cd in self.children[vt]:
            waitingToUpdate.put(cd)
                
        while not waitingToUpdate.empty():
            curV = waitingToUpdate.get()
            self.cost[curV] = self.cost[self.parent[curV]] + self.distance(self.Tree[curV],self.Tree[self.parent[curV]])
            for cd in self.children[curV]:
                waitingToUpdate.put(cd)
        if self.hasExactSolution:
            self.bestCost = self.cost[self.bestConn[0]] + self.cost[self.bestConn[1]]\
                + self.distance(self.Tree[self.bestConn[0]],self.Tree[self.bestConn[1]])
        #     for conk in self.conn.keys():
        #         coni = self.conn[conk]
        #         thisCost = self.cost[conk] + self.cost[coni]\
        #             + self.distance(self.Tree[conk],self.Tree[coni])
        #         if thisCost < self.bestCost:
        #             self.bestCost = thisCost
        #             self.bestConn = [conk,coni]
        #             self.smartPath(self.bestConn)


    def prune(self):
        # if prune ...
        if self.hasExactSolution:
            # self.updateCost(prune=True)
            for x in self.X_sample:
                if self.lowerBoundHeuristic(x) > self.bestCost:
                    self.X_sample.remove(x)
                    self.pruned.append(x)
            pruneVertices = []
            for vt in range(len(self.Tree)):
                if self.Tree[vt] == None:
                    continue
                if self.lowerBoundHeuristicVertex(vt) > self.bestCost:   
                    self.deleteVertex(vt,pruneVertices) 
            pruneVertices.sort(reverse=True)
            for vtp in pruneVertices:
                try:
                    vtcon = self.conn[vtp]
                    self.conn.pop(vtcon)
                    self.conn.pop(vtp)
                except:
                    pass
                self.vold.remove(vtp) 
                # there's lots of work if we delete it...
                # so we just mark it as pruned...
                self.children[vtp] = None # if children have children?
                self.Tree[vtp] = None 
                if self.isGtree[vtp]:
                    self.gVertexNum -= 1
                else:
                    self.hVertexNum -= 1
                self.isGtree[vtp] = None
                self.cost[vtp] = None
                self.parent[vtp] = None
                self.depth[vtp] = None
                    
    def deleteVertex(self,vt,pruneVertices):
        while len(self.children[vt]):
            print("waring/debug: prune a vertex which has children")
            cdt = self.children[vt][-1]  
            self.deleteVertex(cdt,pruneVertices)
        # prune       
        if self.Tree[vt] != None:
            if self.lowerBoundHeuristicVertex(vt) < self.bestCost: 
                self.X_sample.append(self.Tree[vt])
            else:
                self.pruned.append(self.Tree[vt])
            pruneVertices.append(vt)
            self.Tree[vt] = None
            pt = self.parent[vt]
            self.children[pt].remove(vt)
        

    def getPath(self):
        reversePath = []
        if self.isGtree[self.bestConn[0]]: # hope that this value is not "None"
            vg = self.bestConn[0]
            vh = self.bestConn[1]
        else:
            vg = self.bestConn[1]
            vh = self.bestConn[0]
        curV = vg
        if vg != 0:
            while self.parent[curV] != 0:
                reversePath.append(self.Tree[curV])
                curV = self.parent[curV]
            reversePath.append(self.Tree[curV])
        
        # reverse
        path = [0]
        while len(reversePath)>0:
            path.append(reversePath.pop())

        curV = vh
        if vh != 1:
            while self.parent[curV] != 1:
                path.append(self.Tree[curV])
                curV = self.parent[curV]
            path.append(self.Tree[curV])
        path.append(1)

        return path

    def drawGraph(self):
        plt.clf()
        if self.map.dimension == 2:
            self.map.drawMap()
            for xind in self.X_sample:
                plt.plot(self.x[xind][0],self.x[xind][1],'ob')            

            for vt in range(len(self.Tree)):
                v = self.Tree[vt]
                if v == None:
                    continue
                cl = 'r'
                if self.isGtree[vt]:
                    cl = 'g'
                plt.plot(self.x[v][0],self.x[v][1],'o'+cl)   
                if self.parent[vt]!=None:          
                    plt.plot([self.x[v][0], self.x[self.Tree[self.parent[vt]]][0]], 
                        [self.x[v][1], self.x[self.Tree[self.parent[vt]]][1]], '-'+cl)
            
            for vconnt in self.conn.keys():
                vconn = self.Tree[vconnt]
                vcon2 = self.Tree[self.conn[vconnt]]
                plt.plot([self.x[vconn][0], self.x[vcon2][0]], 
                    [self.x[vconn][1], self.x[vcon2][1]], '-y')
                
        plt.pause(0.01)
        #plt.show()


if __name__ == '__main__':
    map2Drand = Map()
    bit = BiBITstar(map2Drand)
    # show map
    if show_animation:
        bit.map.drawMap()
        # plt.pause(10)
    start_time = time.time()
    bit.solve()
    print("time_use: ",time.time()-start_time)
    plt.show()

