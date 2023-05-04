import pandas as pd

cities = pd.read_csv('city.csv')
cities.head(5)

"""# Sort and select 30 most populated cities"""

# top 30 cities by population
top30 = cities.sort_values(by=['population'])[::-1][:30]

"""### Import libraries"""

from random import sample, random
from math import radians, sin, cos, atan2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

"""## Helpful methods"""

# function to translate latitude distance into km
def lon_lat_to_km(point1, point2):
  lat1, lon1 = point1['geo_lat'], point1['geo_lon']
  lat2, lon2 = point2['geo_lat'], point2['geo_lon']
  radius = 6371  # km
  latitude_dist, longitude_dist = radians(lat2-lat1), radians(lon2-lon1) 
  a = sin(latitude_dist/2) * sin(latitude_dist/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(longitude_dist/2) * sin(longitude_dist/2)
  c = 2 * atan2(a**0.5, (1-a)**0.5)
  return radius*c

# function to calculate the total distance of the current path
def curr_distance(df, path):
  # add the distance between the first and the last city in the path 
  total_dist = lon_lat_to_km(df.iloc[path[0]], df.iloc[path[-1]])
  for i in range(len(path)-1):
    point1, point2 = df.iloc[path[i]], df.iloc[path[i+1]]
    total_dist += lon_lat_to_km(point1,point2)
  return total_dist


# simulated annealing

def sim_annealing(T, stop_T, cool_rate, old_path, df, time):

  paths = [old_path]           # for storing all generated paths
  old_total_dist = curr_distance(df, old_path)
  distances = [old_total_dist] # storing distances

  while T>stop_T:
    print(T, end='\r') 
    new_path = old_path.copy()
    i,j = sample([i for i in range(len(new_path))],2)# choose 2 random cities to swap
    new_path[i], new_path[j] = new_path[j], new_path[i] # swap them 

    new_total_dist = curr_distance(df, new_path)
    dist = new_total_dist - old_total_dist
    alpha = np.exp(-dist/T) 

    u = random()   # generating u ~ U(0;1)
    if u <= alpha: # condition to accept the new state
      old_path = new_path
      old_total_dist = new_total_dist 

    T *= cool_rate # reduce the temperature
    paths.append(old_path)
    distances.append(old_total_dist)  
    time += 1

  return paths, distances,time

"""## Generate paths for each cooling rate"""

x0 = [i for i in range(len(top30))] # initial order of the cities

t = 0       # set time step 
T = 10000   # set init temperature
stop_T = 50 # stopping temp
fast_cooling_rate = 0.8
med_cooling_rate = 0.9
slow_cooling_rate = 0.99

fast_paths, fast_dist, fast_t = sim_annealing(T, stop_T, fast_cooling_rate, x0, top30, t)
med_paths, med_dist, med_t = sim_annealing(T, stop_T, med_cooling_rate, x0, top30, t)
slow_paths, slow_dist, slow_t = sim_annealing(T, stop_T, slow_cooling_rate, x0, top30, t)

fast_t = np.linspace(0,fast_t,len(fast_dist))
med_t = np.linspace(0,med_t,len(med_dist))
slow_t = np.linspace(0,slow_t,len(slow_dist))


# plot distance optimization over time for each cooling rate
plt.figure(figsize=(14,8))
plt.plot(fast_t, fast_dist, color='#FE938C', label='Fast cooling rate')
plt.plot(med_t, med_dist, color='#9893DA', label='Medium cooling rate')
plt.plot(slow_t, slow_dist, color='#305252', label='Slow cooling rate')

plt.xlabel('Iteration')
plt.ylabel('Distance, km')
plt.title('The speed of convergence for three different values of the annealing rate')

plt.legend()
plt.grid()

"""### Conclusion

As seen on the figure, the fast annuealing rate allows the distance to converge quickly, at about $30^{th}$ iteration, however, at the cost of accuracy. The convergence is unstable.

The medium cooling rate causes the convergence to occur later, at about $50^{th}$ iteration.

The slow cooling rate coverges much later at $500^{th}$ iteration, but it displays at about half time it became more accurate and most future choices of the path decreased the total distance, rather than increasing it due to inaccuracies, much unlike the previous two cases.

# Animations for each annealing rate

### Make lists of city names in the same order as generated paths to use as node labels in animation
"""

cities_slow = [[top30.iloc[ind]["address"] for ind in path] for path in slow_paths]
cities_medium = [[top30.iloc[ind]["address"] for ind in path] for path in med_paths]
cities_fast = [[top30.iloc[ind]["address"] for ind in path] for path in fast_paths]

"""### Library for graph visualization"""
import networkx as nx

# make overlay
import json
 
# Opening JSON file
with open('russia.json') as json_file:
    source = json.load(json_file)[0]['geojson']['coordinates']

coord = []
for i in source:
    coord = coord + i[0]

# Since some part of Russia is located in west semisphere, for proper display we need to convert such coordinates
# add 360 to the degree value
for i in range(len(coord)):
    x = coord[i][0]
    if x < 0:
        coord[i][0] += 360
coord = np.array(coord)

"""### Frame update"""

def update(frame, params):
  city_names, path, ax = params
  # erase last frame
  ax.clear()
  # add the outline of the country
  ax.plot(coord[:,0], coord[:,1], 'bo')

  graph = nx.DiGraph()
  for i in range(len(path[frame])):
    # add named nodes for all cities in path
    lon = top30.iloc[path[frame][i]]['geo_lon']
    lat = top30.iloc[path[frame][i]]['geo_lat']
    graph.add_node(city_names[frame][i], pos=(lon,lat))
    
  # add paths as edges of graph
  for i in range(len(city_names[frame])-1):
    node1, node2 = city_names[frame][i], city_names[frame][i+1]
    graph.add_edge(node1,node2)
  # add edge between the 1st and last node
  graph.add_edge(city_names[frame][0], city_names[frame][-1])
  # draw the graph
  pos=nx.get_node_attributes(graph,'pos')
  nx.draw(graph, pos, with_labels=True, ax=ax)

"""### Function to save animation"""

fig,ax = plt.subplots(figsize=(18,10))
ani = FuncAnimation(fig, partial(update, params=(cities_fast, fast_paths,ax)), frames=len(cities_fast), interval=100)
ani.save('fast_rate.gif')

fig,ax = plt.subplots(figsize=(18,10))
ani = FuncAnimation(fig, partial(update, params=(cities_medium, med_paths,ax)), frames=len(cities_medium), interval=100)
ani.save('med_rate.gif')

fig,ax = plt.subplots(figsize=(18,10))
ani = FuncAnimation(fig, partial(update, params=(cities_slow, slow_paths,ax)), frames=len(cities_slow), interval=100)
ani.save('slow_rate.gif')
