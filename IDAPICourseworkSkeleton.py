#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
import pygraphviz

#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
    dataSize = len(theData)
    for i in theData:
      prior[i[root]] += 1/float(dataSize)
    return prior

# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
    cAcc = zeros((noStates[varC]), float )
    pAcc = zeros((noStates[varP]), float )
    cAndPAcc = zeros((noStates[varC], noStates[varP]), float )
    
    for i in theData:
      cAndPAcc[i[varC]][i[varP]] += 1
      pAcc[i[varP]] += 1
    
    for i in range(0, len(cPT)):
      for j in range(0, len(cPT[i])):
        cPT[i][j] = cAndPAcc[i][j] / pAcc[j]
        
    return cPT
    
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
    
    for i in theData:
      jPT[i[varRow]][i[varCol]] += 1/float(len(theData))
    
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    aJPT = aJPT.transpose()
    for i in aJPT:
      i /= sum(i)
    return aJPT.transpose()

# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
    
    prior = naiveBayes[0]
    childNodes = naiveBayes[1:]
    for i in range(0, len(rootPdf)):
      rootPdf[i] = prior[i]
      for j in range(0, len(theQuery)):
        childNode = childNodes[j]
        rootPdf[i] = rootPdf[i] * childNode[theQuery[j],i]
        
    total = sum(rootPdf)
    for i in range(0, len(rootPdf)):
      rootPdf[i] = rootPdf[i]/total
    return rootPdf

def createNaiveBayes(theData, noStates, prior):
  childCPT = createChildCPT(theData, noStates)
  return [prior] + childCPT
  
def createChildCPT(theData, noStates):
  cpt = []
  for i in range(1, 6):
    cpt.append(CPT(theData, i, 0, noStates))
  return cpt

# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
    for i in range(0, len(jP)):
      for j in range(0, len(jP[i])):
        if jP[i][j] == 0.0:
          continue
        Pd = jP[i][j]
        temp = log2(Pd/(sum(jP[i])*sum(jP.transpose()[j])))
        mi += Pd*temp
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
    for i in range(0, len(MIMatrix)):
      for j in range(0, len(MIMatrix[i])):
        jpt = JPT(theData, i, j, noStates)
        MIMatrix[i][j] = MutualInformation(jpt)
# Coursework 2 task 2 should be inserted here



# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    for i in range(0, len(depMatrix)):
      for j in range(i+1, len(depMatrix[i])):
        depList.append((depMatrix[i][j], i, j))
# end of coursework 2 task 3
    depList = sorted(depList, reverse=True)
    return array(depList)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4
  
def generateGraph(spanningTree, noVariables):
  graph = {}
  for (x, i, j) in spanningTree:
    if not i in  graph:
      graph[i] = []
      graph[i].append(j)
    else:
      graph[i].append(j)
    if not j in graph:
      graph[j] = []
      graph[j].append(i)
    else:
      graph[j].append(i)
  for i in range(0, noVariables):
    if not i in graph:
      graph[i] = []
    
  return graph

def bfs(x, graph):
  visited = {}
  xSet = []
  q = []
  
  q.append(x)
  visited[x] = True
  
  while q:
    y = q.pop()
    for i in graph[y]:
      if not i in visited:
        xSet.append(i)
        q.append(i)
        visited[i] = True
  
  return set(xSet)

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    for (x, i, j) in depList:
      g = generateGraph(spanningTree, noVariables)
      setI = bfs(i, g)
      setJ = bfs(j, g)
      
      if not setJ.intersection(setI):
        spanningTree.append((x, i, j))
    
    return array(spanningTree)
    
def createGraph(spanningTree, noVariables):
  G = pygraphviz.AGraph(strict=False)
  for i in range(0, noVariables):
    G.add_node(str(noVariables))
  for (x, i, j) in spanningTree:
    G.add_edge(str(i), str(j), str(x))
  
  return G
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
   

# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here


# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here


# Coursework 3 task 5 ends here 
    return mdlAccuracy
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#

def coursework1():
  noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
  theData = array(datain)
  AppendString("results.txt","Coursework One Results: Jamal Khan - jzk09")
  AppendString("results.txt","") #blank line


  AppendString("results.txt","The prior probability distribution of node 0:")
  prior = Prior(theData, 0, noStates)
  AppendList("results.txt", prior)

  AppendString("results.txt","The conditional probability matrix P (2|0) calculated from the data:")
  cpt = CPT(theData, 2,0, noStates)
  AppendArray("results.txt", cpt)

  AppendString("results.txt","The joint probability matrix P (2&0) calculated from the data:")
  jpt = JPT(theData, 2, 0, noStates)
  AppendArray("results.txt", jpt)

  AppendString("results.txt","The conditional probability matrix P (2|0) calculated from the joint probability matrix P (2&0):")
  jpt2cpt = JPT2CPT(jpt)
  AppendArray("results.txt", jpt2cpt)

  AppendString("results.txt","The results of queries [4,0,0,0,5] and [6, 5, 2, 5, 5] respectively on the naive network:")

  naiveBayes = createNaiveBayes(theData, noStates, prior)
  queryA = Query([4,0,0,0,5], naiveBayes)
  AppendList("results.txt", queryA)

  queryB = Query([6,5,2,5,5], naiveBayes)
  AppendList("results.txt", queryB)


def coursework2():
  noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
  theData = array(datain)
  jpt = JPT(theData, 2, 0, noStates)
  MutualInformation(jpt)
  dm = DependencyMatrix(theData, noVariables, noStates)
  dl = DependencyList(dm)
  #print dl
  st = SpanningTreeAlgorithm(dl, noVariables)
  print st
  g = createGraph(st)
  g.draw('test.png')
  
  #print dl
  #AppendArray("lol.txt", dl)
  

if __name__ == "__main__":
  coursework2()
  
  
