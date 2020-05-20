import cv2 as cv
import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats
import os
import time
os.chdir(r'/home/samvandha/Desktop')
img = cv.imread('one array.jpeg', 1)
width = int(img.shape[1] * 2)
height = int(img.shape[0] * 2)
img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

landmarks = [[24,28],[24,348],[850,20],[854,342],[24,118],[24,263],[269,26],[578,22],[259,345],[599,344],[850,110],[853,258]]
#landmarks = [[24,28],[24,348],[850,20],[854,342]]
#landmarks = [[269, 360]]
NL = len(landmarks)
x_range = np.array([0,width])
y_range = np.array([0,height])
N = 500
trajectory = []

def create_particles(x_range,y_range,N):
    particles = np.empty((N,2))
    particles[:,0] = uniform(x_range[0], x_range[1], size=N)
    particles[:,1] = uniform(y_range[0], y_range[1], size=N)
    return(particles)
       
particles = create_particles(x_range,y_range,N)
weights = np.array([1.0]*N)

def robot_path(i,j, sj):
    if j < 6:
        if j % 2 == 0:
            if i < 845:
                i += 1
            else:
                sj += 1
                if sj > 53:
                    sj = 0
                    j += 1
                
        elif j % 2 == 1:
            if i > 35:
                i -= 1
            else:
                if j < 5:
                    sj += 1
                    if sj > 53:
                        sj = 0
                        j += 1
    return(i,j,sj)  

i = 40
j = 0
sj = 0
def predict(particles, heading, distance, dt=1):
    N = len(particles)
    dist = (distance* dt) + np.random.randn(N) * 2
    particles[:,0] += np.cos(heading) * dist
    particles[:,1] += np.sin(heading) * dist
    return particles

def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance = np.power((particles[:,0] - landmark[0])**2 + (particles[:,1] - landmark[1])**2,0.5)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])
    weights += 1.e-300
    weights /= sum(weights)

def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random())/ N

    indexes = np.zeros(N, 'i')
    cumsum = np.cumsum(weights)
    i, j = 0,0
    while i < N and j < N:
        if positions[i] < cumsum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def resample(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)

while(1):
    img = cv.imread('one array.jpeg', 1)
    width = int(img.shape[1] * 2)
    height = int(img.shape[0] * 2)
    img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)
    previous_x = i
    previous_y = 40 + 56*j+sj
    (i,j,sj) = robot_path(i,j, sj)
    x = i
    y = 40 + 56*j + sj
    trajectory.append([x,y])
    heading = np.arctan2((y-previous_y),(x-previous_x))
    distance = ((y-previous_y)**2 + (x-previous_x)**2)**0.5
    particles = predict(particles, heading, distance, dt=1)
    center = np.array([[x,y]])
    zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * 0.5))
    update(particles, weights, z = zs, R=50, landmarks = landmarks)
    indexes = systematic_resample(weights)
    resample(particles, weights, indexes)
    for k in range(len(trajectory)):
        cv.circle(img, (trajectory[k][0], trajectory[k][1]), 6, (200,0,0), -1)
    for particle in particles:
        cv.circle(img, (int(particle[0]),int(particle[1])), 2,(0,0,255), -1)
    #time.sleep(0.0001)
    for landmark in landmarks:
        cv.circle(img, tuple(landmark), 10, (55,200,100), -1)
    k = cv.waitKey(1) & 0xFF
    cv.putText(img, "Landmarks-", (30, 20), 2, 1.0, (0,255,255))
    cv.circle(img, (245,12), 10, (55,200,100), -1)
    cv.putText(img, "Particles", (300, 20), 2, 1.0, (0,255,255))
    cv.circle(img, (460,12), 8, (0,0,255), -1)
    cv.putText(img, "Robot trajectory(Ground Truth)", (300, 330), 2, 1.0, (0,255,255))
    cv.circle(img, (813,324), 8, (255,0,0), -1)
    
    cv.imshow("Image", img)
    if k == 27:
        break

cv.destroyAllWindows()
