import imageio

frames = []

time = 1000
for t in range(1,time):
	fname = "_figures/swarm_t={:04d}_3D.png".format(t)
	image = imageio.imread(fname)
	frames.append(image)


imageio.mimsave('./swarm.gif', # output gif
                frames,          # array of input frames
                fps = 5)         # optional: frames per second
