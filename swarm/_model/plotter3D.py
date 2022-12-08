import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

def plotSwarm3D( sim, t, followcenter, step, numTimeSteps, dynamicscope=True):
	fig = plt.figure()
	if (step > numTimeSteps - 3):
		fig, (_, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(15, 15), dpi=300)
	else:
		fig, (_, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(15, 15))
	_.set_visible(False)
	ax = fig.add_subplot(211, projection='3d')
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
	ax.quiver(locations[:,0],locations[:,1],locations[:,2],
		      directions[:,0], directions[:,1], directions[:,2], 
		      color=cmap(norm(np.arange(sim.N))))
	displ = 5
	if (followcenter):
		center = sim.computeCenter()
		if (dynamicscope):
			avgdist = sim.computeAvgDistCenter(center)
			displx = avgdist/2.
			ax.set_xlim([center[0]-displx-displ,center[0]+displx+displ])
			ax.set_ylim([center[1]-displx-displ,center[1]+displx+displ])
			ax.set_zlim([center[2]-displx-displ,center[2]+displx+displ])
		else:
			ax.set_xlim([center[0]-displ,center[0]+displ])
			ax.set_ylim([center[1]-displ,center[1]+displ])
			ax.set_zlim([center[2]-displ,center[2]+displ])
	else:
		ax.set_xlim([-displ,displ])
		ax.set_ylim([-displ,displ])
		ax.set_zlim([-displ,displ])
	x  = np.arange(0, step+1)
	ax2.plot(x, np.array(sim.angularMoments), '-b', label='Angular Moment')
	ax2.plot(x, np.array(sim.polarizations), '-r', label='Polarization')
	ax2.set_xlim([0, numTimeSteps])
	ax2.set_ylim([0.,1.])
	#ax2.legend(frameon=False, loc='upper center', ncol=2)
	plt.savefig("_figures/swarm_t={:04d}_3D.png".format(t))
	plt.close('all')

def finalplotSwarm3D( sim, t, followcenter, step, numTimeSteps, dynamicscope=True):
	if (step == numTimeSteps - 1):
		x  = np.arange(0, step+1)
		plt.figure(figsize=(452.0 / 72.27, 452.0*(5**.5 - 1) / 2 / 72.27), dpi=300)
		plt.plot(x, np.array(sim.angularMoments), '-b', label='Angular Moment')
		plt.plot(x, np.array(sim.polarizations), '-r', label='Polarization')
		plt.xlim([0, numTimeSteps])
		plt.ylim([0.,1.])
		#ax2.legend(frameon=False, loc='upper center', ncol=2)
		plt.savefig("_figures/3Dswarm_t={:04d}_3D.png".format(t))
		plt.close('all')

def plotSwarmSphere( sim, t, i ):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	locations = []
	directions = []
	for fish in sim.swarm:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	ax.quiver(locations[:,0],locations[:,1],locations[:,2],
		      directions[:,0], directions[:,1], directions[:,2])
	# Create a sphere
	r = 1
	pi = np.pi
	cos = np.cos
	sin = np.sin
	phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
	x = r*sin(phi)*cos(theta)
	y = r*sin(phi)*sin(theta)
	z = r*cos(phi)
	ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
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
	ax = fig.gca(projection='3d')
	locations = []
	directions = []
	for fish in fishs:
		locations.append(fish.location)
		directions.append(fish.curDirection)
		#print(np.linalg.norm(fish.curDirection))
	locations = np.array(locations)
	directions = np.array(directions)
	ax.quiver(locations[:,0],locations[:,1],locations[:,2],
		      directions[:,0], directions[:,1], directions[:,2])
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	ax.set_zlim([-2,2])
	plt.savefig("_figures/{}_t={}_i={}.png".format(type, t, i))
	plt.close('all')

def plotFish( fish, i, t ):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	loc = fish.location
	vec = fish.curDirection
	ax.quiver(loc[0], loc[1], loc[2], vec[0], vec[1], vec[2])
	ax.set_xlim([-2,2])
	ax.set_ylim([-2,2])
	ax.set_zlim([-2,2])
	plt.savefig("_figures/fish_t={}_i={}.png".format(t, i))
	plt.close()

def plotRot( vec1, vec2, rotvec, angle ):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	locations = [vec1,vec2,rotvec]
	vecs = np.array([vec1,vec2,rotvec])
	loc = np.zeros(3)
	ax.quiver(loc, loc, loc, vecs[:,0], vecs[:,1], vecs[:,2], color=['green','red','black'])
	ax.set_title("rotation by {} degree".format(angle))
	ax.set_xlim([-1,1])
	ax.set_ylim([-1,1])
	ax.set_zlim([-1,1])
	plt.show()
