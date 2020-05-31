####Samvandha Pathak May 2020####
####Anti balistic missile using particle filter to predict the missile####
import numpy as np
import cv2 as cv
import os
import random
from numpy.random import uniform
import scipy.stats
import time
os.chdir(r'/home/samvandha/Desktop')
img = cv.imread("circles1.png")
img_missile = cv.imread("missile.png")
(x1,y1) = 200,200
(x2,y2) = x1 + 70, y1+ 20
radars = np.array([[200,300],[300,200], [400,300], [300, 400]])
print("Radars:", radars[:,0])
old_x, old_y = 0,0
allslopes = []
N = 2000
x_range = np.array([0,600])
y_range = np.array([0,600])
def create_particles(x_range, y_range,N):
    particles = np.empty((N,2))
    particles[:,0] = uniform(x_range[0], x_range[1], size = N)
    particles[:,1] = uniform(y_range[0], y_range[1], size = N)
    return(particles)
    
def missile_proj(x,y, tag):
    slope = allslopes[tag]
    x = x+ 1
    y = y + slope
    return(x,y)

def generate_proj(x,y):
    global allslopes
    if x == 0:
        if y < 100 :
            slope = 2
        elif 100 < y < 200:
            slope = random.randint(1,1)
        elif 400<y<500:
            slope = -random.randint(1,1)
        elif 500<y<600:
            slope = -2
        else:
            slope = 0
        allslopes.append(slope)
    return()
tag = 0
def missilelaunch():
    global old_x,old_y
    global tag
    if (old_x, old_y) == (0,0):
        x = 0
        y = random.randint(0,600)
        x , y = 0, y
        old_x = x
        old_y = y
        tag = 0
        generate_proj(x,y)
        print("Projectile generated")
    else:
        old_x, old_y = missile_proj(old_x,old_y, tag)
    return(old_x,old_y)

def update(particles, weights, z, R, lands):
    weights.fill(1.)
    for i, land in enumerate(lands):
        distance = np.power((particles[:,0] - land[0])**2 + (particles[:,1] - land[1])**2,0.5)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])
    weights += 1.e-300
    weights /= sum(weights)
       
def estimate_distance(x,y,x1,y1):
    global radars
    distance = np.empty((len(radars), 1))
    angle = np.empty((len(radars),1))
    est_center = np.empty((len(radars),2))
    for i in range(len(radars)):
        distance[i] = ((radars[i][0]-(x+x1)/2)**2+(radars[i][1]-(y+y1)/2)**2)**0.5
        if random.random()<0.5:
            distance[i] = distance[i] + np.random.random()/10 ###This is the error we are assuming in the measurement
            angle[i] = np.arctan2((radars[i][0]-(x+x1)/2),(radars[i][1]-(y+y1)/2)) + np.random.random()/100 #error in angle
        else:
            distance[i] = distance[i] - np.random.random()/10 ###This is the error we are assuming in the measurement
            angle[i] = np.arctan2((radars[i][0]-(x+x1)/2),(radars[i][1]-(y+y1)/2)) - np.random.random()/100 #error in angle
        est_center[i][0] = radars[i][0] - np.sin(angle[i])*distance[i]
        est_center[i][1] = radars[i][1] - np.cos(angle[i])*distance[i]
    x = 0
    y = 0
    for i in range(len(est_center)):
        x += est_center[i][0]
        y += est_center[i][1]
    x = x/len(est_center)
    y = y /len(est_center)
    return(distance, x, y)

particles = create_particles(x_range, y_range, N)
oldxc1 = 0
oldyc1 = 0
def init_shoot(hitx,hity):
    angle = np.arctan2((hity-300),(hitx-300))
    init = 1
    target_dist = ((hitx-300)**2+(hity-300)**2)**0.5
    return(init, angle, target_dist)
    
def resample1(particles, weights, center):
    new_particles = np.empty((N,2))
    new_weights = np.array([0.000001]*N)
    avg_weight = np.sum(weights)/len(weights)
    for i in range(len(weights)):
        if avg_weight < weights[i]:
            new_particles[i] = particles[i]
            new_weights[i] = weights[i]
        else:
            h = random.random()
            if h < 0.25:
                new_particles[i] = [center[0][0] + 10*np.random.random(), center[0][1] + 10*np.random.random()] 
                new_weights[i] = avg_weight - 1*np.random.random()
            elif 0.25 < h < 0.5:
                new_particles[i] = [center[0][0] + 10*np.random.random(), center[0][1] - 10*np.random.random()]
                new_weights[i] = avg_weight - 1*np.random.random()
            elif 0.5 < h < 0.75:
                new_particles[i] = [center[0][0] - 10*np.random.random(), center[0][1] + 10*np.random.random()]
                new_weights[i] = avg_weight - 1*np.random.random()
            else:
                new_particles[i] = [center[0][0] - 10*np.random.random(), center[0][1] - 10*np.random.random()]
                new_weights[i] = avg_weight - 1*np.random.random()           
    weights = new_weights
    particles = new_particles
    return(particles,weights)
      
def shoot(sx,sy,angle, target_dist, frames, count):
    global frame_count
    k = target_dist/(1*frames)
    frame_count += 1
    sx += k*np.cos(angle)
    sy += k*np.sin(angle)
    return(sx, sy, frame_count)

def predict_tra_missile(useful_trajectory):
    theta = 0
    samples = 100
    for i in range(samples):
        j = random.randint(0, len(useful_trajectory)-1)
        theta += np.arctan2((useful_trajectory[j][0]-useful_trajectory[0][0]),(useful_trajectory[j][1]-useful_trajectory[0][1]))
    theta /= samples
    return(theta)

init = 0
sx = 300
sy = 300
trajectory= []
point_decided = 0
point_decided1 = 0
inside = 0
collision = 0
frame_count = 0
while(collision == 0):
    img = cv.imread("circles.png")
    img = cv.resize(img, (600,600), interpolation = cv.INTER_AREA)
    x, y = missilelaunch()
    (x2,y2) = x + 30, y+ 10
    cv.rectangle(img,(x,y),(x2,y2),(0,0,255), -1)
    distance, xc1, yc1 = estimate_distance(x,y,x2,y2)
    particles[:,0] -= oldxc1-xc1
    particles[:,1] -= oldyc1-yc1
    cv.circle(img, (int(xc1),int(yc1)), 5, (0,255,0), -1)
    weights = np.array([0.1]*N)
    center = np.array([[xc1,yc1]])
    NL = len(radars)
    zs = (np.linalg.norm(radars - center, axis=1) + (np.random.randn(NL) * 2))
    update(particles, weights, z = zs, R=50, lands = radars)
    particles,weights = resample1(particles, weights, center)
    x_particle = 0
    y_particle = 1
    for particle in particles:
        cv.circle(img, (int(particle[0]),int(particle[1])), 2,(0,255,255), -1)
        x_particle += particle[0]
        y_particle += particle[1]
    x_particle /= N
    y_particle /= N
    trajectory.append([x_particle,y_particle])
    trajectory = trajectory[:]
    useful_trajectory = trajectory[10:]
    kc1 = ((x_particle-300)**2+(y_particle-300)**2)**0.5
    if kc1 < 250 and  inside == 0:
        #print("Missile has entered dangerous position")
        theta = predict_tra_missile(useful_trajectory)
        for i in range(0, 100, 10):
            h1 = np.sin(theta) * i + x_particle
            h2 = np.cos(theta) * i + y_particle
            cv.circle(img, (int(h1),int(h2)), 3, (100,255,0), -1)
        if point_decided1 == 0:
            hitx = np.sin(theta) * 50 + x_particle
            hity = np.cos(theta) * 50 + y_particle
            point_decided1 = 1
            loop = 1
            new_loops = []
        if point_decided1 == 1:
            loop+= 1
        if point_decided1 == 1 and  -3 < x_particle-hitx < 3 and  -3 < y_particle-hity < 3:
            print("Loops for collision", loop)
            new_loops.append(loop)
    kc = ((x_particle-300)**2+(y_particle-300)**2)**0.5
    if kc < 180:
        inside = 1
        if len(new_loops) != 0 and inside == 1:
            required_frame = int(sum(new_loops)/len(new_loops))    
        else:
            required_frame = 25
        theta = predict_tra_missile(useful_trajectory)
        for i in range(0, 100, 10):
            h1 = np.sin(theta) * i + x_particle
            h2 = np.cos(theta) * i + y_particle
            cv.circle(img, (int(h1),int(h2)), 3, (100,255,0), -1)
        if point_decided == 0:
            hitx = np.sin(theta) * 50 + x_particle
            hity = np.cos(theta) * 50 + y_particle
            point_decided = 1
            init, angle, target_dist = init_shoot(hitx,hity)
        cv.circle(img, (int(hitx),int(hity)), 5, (100,255,200), -1) 
    for i in range(len(trajectory)):
        cv.circle(img, (int(trajectory[i][0]), int(trajectory[i][1])), 3, (150,150,100), -1)
    cv.circle(img, (int(x_particle),int(y_particle)), 3, (255,0,0), -1)
    cv.circle(img, (300,300), 180, (0,0,255), 1)
    cv.circle(img, (300,300), 250, (0,0,255), 1)
    cv.circle(img, (300,300), 5, (0,0,255), 1)
    if init == 1:
        sx,sy, frame_count = shoot(sx,sy, angle, target_dist, required_frame, frame_count)
        cv.circle(img, (int(sx),int(sy)), 3, (100,100,100), -1)
        if 0 <(sx-(x+15))<20 and 0<(sy-(y+5))< 10:
            print("Collision")
            collision = 1
            cv.circle(img, (int(sx), int(sy)), 50, (0, 100, 255), -1)    
    for i in range(len(radars)):
        cv.circle(img, tuple(radars[i]), 10,(0,100,200), -1)
        
        cv.circle(img, tuple(radars[i]), distance[i], (100,100,100), 1)
    cv.putText(img, "Robot trajectory", (350, 500), 2, 1.0, (150,150,100))
    cv.putText(img, "Particles", (350, 530), 2, 1.0, (0,255,255))
    cv.putText(img, "Predicted path", (350, 560), 2, 1.0, (100,255,0))
    cv.putText(img, "Radars", (350, 590), 2, 1.0, (0,100,255))
    cv.imshow("Circles", img)
    oldxc1 = xc1
    oldyc1 = yc1
    if cv.waitKey(1) & 0xFF == 27:
        cv.destroyAllWindows()
        break

print("Completed")
f = input("Press y to exit: ")
if f == 'y' or f =='Y':
    cv.destroyAllWindows()
