import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import random
sys.setrecursionlimit(20000)
rows = 15       #ROWS
cols = 15       #COLS
grid = []

grid = [[0 for x in range(rows)]for y in range(cols)]
goal = [rows-1,cols-1]

def heuristic(i,j):
    return(((int(goal[0])-i)**2+(int(goal[1])-j)**2)**2)

def calculate(i,j,g,pi,py):
    i = i
    j = j
    g = g+1
    h = heuristic(i,j)
    f = g + h
    openset.append([i,j,g,h,f,pi,py])

def addneigh(i,j):
    neigh = []
    if j < cols-1:
        neigh.append([i,j+1, i, j])
    if i < rows-1:
        neigh.append([i+1,j, i, j])
    if i < rows-1 and j < cols-1:
        neigh.append([i+1, j+1, i, j])
    if i < rows-1 and j > 0:
        neigh.append([i+1, j-1, i, j])
    if i > 0 and j < cols-1:
        neigh.append([i-1, j+1, i, j])
    if i > 0:
        neigh.append([i-1, j, i, j])
    if j > 0:
        neigh.append([i,j-1, i, j])
    if i > 0 and j > 0:
        neigh.append([i-1,j-1,i,j])
    
    

    print("Neighbours", neigh)
    return(neigh)
        
inith = heuristic(0,0)
openset = [[0,0,0,inith,inith,0,0]]
closeset = []
reachedspots = []
ax = plt.axes()
plt.xlim(-1, rows)
plt.ylim(-1,cols)
pathi = []
pathj = []
obstacle = []
for i in range(rows):
    for j in range(cols):
        if random.random() < 0.4 and [i,j] != goal:    ####Change the value for more obstacle
            obstacle.append([i,j])
            ax.plot(i,j, 'ko')
        else:
            ax.plot(i,j, 'bo')
            plt.draw()
print("The obstacles are: ", obstacle)
def plotpath(i,j):
    ax.plot(i,j,'ro')
    plt.draw()
    

def findpath(i,j):
    for k in range(len(closeset)):
        if (closeset[k][0],closeset[k][1]) == (i,j):
            if (closeset[k][0],closeset[k][1]) == (0,0):
                pathi.append(i)
                pathj.append(j)
                print("The path is found")
                plotpath(pathi,pathj)
                plt.show()
                quit()
            else:
                pathi.append(i)
                pathj.append(j)
                findpath(closeset[k][5],closeset[k][6])
                print("finding path")
        
   
while len(openset) != 0:
    lowestf = openset[0][4]
    for i in range(len(openset)):
        if lowestf >= openset[i][4]:
            current = openset[i]
            lowestf = openset[i][4]
            
    print("Current Point:", current)
    ax.plot(current[0],current[1],'ro')
    plt.draw()
    plt.pause(0.001)
    for j in range(len(openset)):       #find the current point in openset
        #print("Openset ", j,  openset[j])
        if current == openset[j]:
            print("Found in openset")
            openset.remove(openset[j])
            if (current[0],current[1]) == (goal[0],goal[1]):
                print("End point reached")
                pathi.append(current[0])
                pathj.append(current[1])
                findpath(current[5],current[6])
            break
    closeset.append(current)   
    neigh = addneigh(current[0],current[1])
    
    for i in range(len(neigh)):
        flag = 0
        print("Length of closed set:", len(closeset))
        print("Length of open set:", len(openset))
        for j in range(len(closeset)):
            if (neigh[i][0], neigh[i][1]) == (closeset[j][0], closeset[j][1]):
                flag = 1
        x = [neigh[i][0], neigh[i][1]]
        flag1 = 1
        if x in obstacle:
            flag1 = 0
        #If neighbour is not in the close set and not obstacle
        if flag == 0 and flag1 == 1:
            tempg = current[2] + 1
            flag1 = 0
            for j in range(len(openset)):
                if (neigh[i][0], neigh[i][1]) == (openset[j][0], openset[j][1]):
                    flag1 = 1
                    dummy = openset[j]
                    break
            #if neighbour is in open set
            if flag1 == 1:
                if tempg < openset[j][2]:
                    openset[j][2] = tempg
                    openset[j][4] = openset[j][2] + openset[j][3]
            else:
                calculate(neigh[i][0],neigh[i][1],tempg, neigh[i][2],neigh[i][3])
                y = len(openset)-1
                openset[y][2] = tempg
                openset[y][4] = openset[y][2] + openset[y][3]
                
            
    for i in range(len(openset)):
        ax.plot(openset[i][0], openset[i][1], 'go')
    for i in range(len(closeset)):
        ax.plot(closeset[i][0], closeset[i][1], 'go')
    plt.draw()
    plt.pause(0.0000001)
    

#result = addneigh(2,2)
#print(result)


#plt.show()
