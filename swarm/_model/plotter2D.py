import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

def plotSwarm2D( sim, t, followcenter):
	fig = plt.figure()
	ax = fig.gca()
	locations = []
	directions = []
	for fish in sim.fishes:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	cmap = cm.jet
	norm = Normalize(vmin=0, vmax=sim.N)
	ax.quiver(locations[:,0],locations[:,1],
		      directions[:,0], directions[:,1],
		      color=cmap(norm(np.arange(sim.N))))
	if (followcenter):
		center = sim.computeCenter()
		ax.set_xlim([center[0]-3,center[0]+3])
		ax.set_ylim([center[1]-3,center[1]+3])
	else:
		ax.set_xlim([-1,1])
		ax.set_ylim([-1,1])
	plt.savefig("_figures/swarm_t={:04d}.png".format(t))
	plt.close()

def plotSwarmSphere( sim, t, i ):
	fig = plt.figure()
	locations = []
	directions = []
	for fish in sim.swarm:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	ax.quiver(locations[:,0],locations[:,1],
		      directions[:,0], directions[:,1])
	# Create a sphere
	r = 1
	pi = np.pi
	cos = np.cos
	sin = np.sin
	phi = np.mgrid[0.0:pi:100j]
	x = r*cos(phi)
	y = r*sin(phi)
	ax.plot_surface(x, y,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
	ax.set_aspect('equal', 'box')
	#ax.set_xlim([-2,2])
	#ax.set_ylim([-2,2])
	#ax.set_zlim([-2,2])
	plt.savefig("_figures/swarm_t={}_sphere_i={}.png".format(t,i))
	plt.close()

def plotFishs( fishs, i, t, type ):
	if fishs.size == 0:
		print("no fish of type {}".format(type))
		return
	fig = plt.figure()
	locations = []
	directions = []
	for fish in fishs:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	ax.quiver(locations[:,0],locations[:,1],
		      directions[:,0], directions[:,1])
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	plt.savefig("_figures/{}_t={}_i={}.png".format(type, t, i))
	plt.close()

def plotFish( fish, i, t ):
	fig = plt.figure()
	loc = fish.location
	vec = fish.curDirection
	ax.quiver(loc[0], loc[1], vec[0], vec[1])
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	plt.savefig("_figures/fish_t={}_i={}.png".format(t, i))
	plt.close()

def plotRot( vec1, vec2, rotvec, angle ):
	fig = plt.figure()
	locations = [vec1,vec2,rotvec]
	vecs = np.array([vec1,vec2,rotvec])
	loc = np.zeros(2)
	ax.quiver(loc, loc, vecs[:,0], vecs[:,1], color=['green','red'])
	ax.set_title("rotation by {} degree".format(angle))
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	plt.show()
