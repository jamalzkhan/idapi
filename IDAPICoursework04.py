#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
import pydot

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
    
def makeName(number):
  return str(int(number))
  
def createGraph(spanningTree, noVariables):
  g = pydot.Dot(graph_type='graph')
  for i in range(0, noVariables):
    g.add_node(pydot.Node(makeName(i)))
  for (x, i, j) in spanningTree:
    g.add_edge(pydot.Edge(makeName(i), makeName(j)))#, label=str(x)))
  return g
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
    for i in theData:
      cPT[i[child]][i[parent1]][i[parent2]] += 1
    
    for i in range(0, noStates[parent1]):
      for j in range(0, noStates[parent2]):
        sum = 0.0
        for k in range(0, len(cPT)):
          sum += cPT[k][i][j]
        for k in range(0, len(cPT)):
          if not sum == 0:
            cPT[k][i][j] = cPT[k][i][j] / sum
    #print cPT
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

def HepatitisBayesianNetwork(theData, noStates):
  
  # dm = DependencyMatrix(theData, noVariables, noStates)
  # dl = DependencyList(dm)
  # st = SpanningTreeAlgorithm(dl, noVariables)
  
  cpt0 = Prior(theData, 0, noStates)
  cpt1 = Prior(theData, 1, noStates)
  cpt2 = CPT(theData, 2, 0, noStates)
  cpt3 = CPT(theData, 3, 4, noStates)
  cpt4 = CPT(theData, 4, 1, noStates)
  cpt5 = CPT(theData, 5, 4, noStates)
  cpt6 = CPT(theData, 6, 1, noStates)
  cpt7 = CPT_2(theData, 7, 0, 1, noStates)
  cpt8 = CPT(theData, 8, 7, noStates)
  
  arcList = [[0], [1], [2,0], [3,4], [4,1], [5,4], [6, 1], [7, 0, 1], [8, 7]]
  cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, cpt7, cpt8]  
  
  #g = createGraph(st, noVariables)
  #g.write_png('spanning_tree_coursework_3.png')
    
  return arcList, cptList

# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here
    for i in range(0, len(noStates)):
      tempSize = noStates[i] - 1
      for p in arcList[i][1:]:
        tempSize *= noStates[p]
      mdlSize += tempSize
    mdlSize = (mdlSize) * log2(noDataPoints)/2
# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
    for i in range(0, len(dataPoint)):
      table = cptList[i]
      arc = arcList[i]
      val = 1.0
      if len(arc) == 1:
        val = table[dataPoint[i]]
      elif len(arc) == 2:
        val = table[dataPoint[i]][dataPoint[arc[1]]]
      else:
        val = table[dataPoint[i]][dataPoint[arc[1]]][dataPoint[arc[2]]]
        
      if val > 0.0:
        jP *= val
# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
    for i in theData:
      j = JointProbability(i, arcList, cptList)
      mdlAccuracy += log2(j)

# Coursework 3 task 5 ends here 
    return mdlAccuracy

def MDLScore(theData, noDataPoints, noStates, arcList, cptList):
  modelSize = MDLSize(arcList, cptList, noDataPoints, noStates)
  modelAccuracy = MDLAccuracy(theData, arcList, cptList)
  return modelSize - modelAccuracy
  

def removeElem(i, list):
  new_list = list[:]
  del new_list[i]
  return new_list
  

def minMDL(theData, noDataPoints, noStates, arcList, cptList):
  mini = float('inf'), -1, -1
  for arc in arcList:
      copyCPT = cptList[:]
      for a in arc[1:]:
          arc.remove(a)
          index = arcList.index(arc)
          copyCPT.pop(arc[0])
          tempCPT = None
          if len(arc) == 1:
              tempCPT = Prior(theData, arc[0], noStates)
          elif len(arc) == 2:
              tempCPT = CPT(theData, arc[0], arc[1], noStates)
          copyCPT.insert(arc[0], tempCPT)
          score = MDLScore(theData, noDataPoints, noStates, arcList, copyCPT)
          if mini[0] > score:
              mini = score, arc[0], a
          arc.append(a)
  return mini
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
    for i in range(0, noVariables):
      mean.append(0.0)
    for i in realData:
      for j in range (0, noVariables):
        mean[j] += i[j] / len(realData)
      
    # Coursework 4 task 1 ends here
    return array(mean)

def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    
    # Coursework 4 task 2 begins here
    U = realData - Mean(theData)
    U_T = transpose(U)
    covar = dot(U_T, U) / (len(realData) - 1)
    # Coursework 4 task 2 ends here
    return covar

def CreateEigenfaceFiles(theBasis):
    # Coursework 4 task 3 begins here
    for i in range(0,len(theBasis)):
      name = "PrincipalComponent" + str(i) + ".jpg"
      SaveEigenface(theBasis[i], name)

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here
    theFaceImageData = ReadOneImage(theFaceImage)
    tr = transpose(theBasis)
    magnitudes = dot((theFaceImageData - theMean), tr)
    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, magnitudes):
    # Coursework 4 task 5 begins here
    SaveEigenface(aMean, "Reconstructed_0" + ".jpg")
    for i in range(0, len(magnitudes)):
      reconstruction = add(dot(transpose(aBasis[0:i]), magnitudes[0:i]), aMean)
      SaveEigenface(reconstruction, "Reconstructed_"+str(i+1)+".jpg")      
    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    mean = Mean(theData)
    
    U = theData - mean
    U_T= transpose(U)
    U_U_T = dot(U, U_T)
    
    # these need to be normalised
    eigenValues , eigenVectors = linalg.eig(U_U_T)
    tempMatrix = dot(U_T, eigenVectors)
    tempMatrix = tempMatrix.transpose()
    
    for i in range(0, len(tempMatrix)):
      magnitude = sqrt(dot(tempMatrix[i], transpose(tempMatrix[i])))
      tempMatrix[i] /= magnitude
    
    data = zip(eigenValues, tempMatrix)
    list.sort(data, reverse = True)
    eV, result = zip(*data)
    
    orthoPhi = result
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
  
  AppendString("results.txt","Coursework Two Results: Jamal Khan - jzk09")
  AppendString("results.txt","") #blank line
  
  AppendString("results.txt","The dependency matrix for HepatitisC data set:")
  dm = DependencyMatrix(theData, noVariables, noStates)
  AppendArray("results.txt", dm)
  
  AppendString("results.txt","The dependency list for HepatitisC data set:")
  dl = DependencyList(dm)
  AppendArray("results.txt", dl)
  
  AppendString("results.txt","The nodes for the spanning tree are: ")
  st = SpanningTreeAlgorithm(dl, noVariables)
  AppendArray("results.txt", st)
  
  g = createGraph(st, noVariables)
  g.write_png('spanning_tree.png')

def coursework3():
  noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
  theData = array(datain)
  cpt_2 = CPT_2(theData, 7, 1, 2, noStates)
  arcList, cptList = HepatitisBayesianNetwork(theData, noStates)
  mdlSize = MDLSize(arcList, cptList, noDataPoints, noStates)
  
  AppendString("results.txt","Coursework Three Results: Jamal Khan - jzk09")
  AppendString("results.txt","") #blank line

  AppendString("results.txt","The MDLSize of the network for Hepatitis C data set is:")
  AppendString("results.txt",mdlSize)
  AppendString("results.txt","") #blank line
  
  dataPoint = [0,8,0,1.8,6,0,5,0]
  jp = JointProbability(dataPoint, arcList, cptList)
  
  mdlAccuracy = MDLAccuracy(theData, arcList, cptList)
  AppendString("results.txt","The MDLAccuracy of the network for Hepatitis C data set is:")
  AppendString("results.txt",mdlAccuracy)
  AppendString("results.txt","") #blank line
  
  mdlScore = MDLScore(theData, noDataPoints, noStates, arcList, cptList)
  AppendString("results.txt","The MDLScore of the network for Hepatitis C data set is:")
  AppendString("results.txt",mdlScore)
  AppendString("results.txt","") #blank line
  
  bestScore, nodeX, nodeY = minMDL(theData, noDataPoints, noStates, arcList, cptList)
  
  AppendString("results.txt","The MDLScore of the best network with one arc removed from node " + str(nodeX) + " to node " + str(nodeY) + " is:")
  AppendString("results.txt",bestScore)
  

def coursework4():
  noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
  #theData = array(datain)
  #cov = Covariance(theData)
  
  AppendString("results.txt","Coursework Three Results: Jamal Khan - jzk09")
  AppendString("results.txt","") #blank line
  
  hepatitisMean = Mean(array(datain))
  
  AppendString("results.txt","The mean vector for the Hepatitis C data set:")
  AppendList("results.txt", hepatitisMean)
  
  
  hepatitisCovariance = Covariance(array(datain))
  AppendString("results.txt","The covariance matrix for the Hepatitis C data set:")
  AppendArray("results.txt", hepatitisCovariance)
  
  
  #theBasis = ReadEigenfaceBasis()
  #aMean = array(ReadOneImage("MeanImage.jpg"))
  #CreateEigenfaceFiles(theBasis)
  theFaceImage = "c.pgm"
  
  #magnitudes = ProjectFace(theBasis, aMean, theFaceImage)
  #CreatePartialReconstructions(theBasis, aMean, magnitudes)
  
  imageData = array(ReadImages())
  newMean = Mean(imageData)
  
  
  theNewBasis = PrincipalComponents(imageData)
  CreateEigenfaceFiles(theNewBasis)
  newMagnitudes = ProjectFace(theNewBasis, newMean, theFaceImage)
  
  AppendString("results.txt","The component magnitudes for image 'c.pgm' :")
  AppendList("results.txt", newMagnitudes)
  
  CreatePartialReconstructions(theNewBasis, newMean, newMagnitudes)

if __name__ == "__main__":
  coursework4()