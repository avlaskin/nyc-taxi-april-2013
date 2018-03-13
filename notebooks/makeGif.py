import imageio

period = 'week'
images = []
for i in range(168-1):
    filename = ('./images/%s_hour_%d.png' % (period, i))
    images.append(imageio.imread(filename))

imageio.mimsave('./images/%s_animation.gif' % period, images, fps=1.0)
