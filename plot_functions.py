from basic_libs import *

def plot_hist(data):
    plt.figure()
    plt.hist(data,color="black")

def plot_image(image):
    plt.imshow(image)
    plt.axis('off')
