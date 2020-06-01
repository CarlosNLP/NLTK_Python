import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

# Giving a style to the graph
style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

def animate(i):
    # Reading and splitting the log per lines
    pullData = open("twitter_log.txt", "r").read()
    lines = pullData.split('\n')

    # Creating the x and y array lists that will be used to make the graph dynamic
    xar = []
    yar = []

    # Starting positions for x and y
    x = 0
    y = 0

    # Each line has the classification of that tweet "pos" or "neg"
    for line in lines:
        x += 1 # x will be incrementing per number of tweets
        if "pos" in line:
            y += 1 # function increases per positive tweet
        elif "neg" in line:
            y -= 1 # function decreases per negative tweet

        # Populating the array lists for x and y
        xar.append(x)
        yar.append(y)
    
    ax1.clear()
    ax1.plot(xar, yar)

# Running the dynamic graph up to 1,000 tweets
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
