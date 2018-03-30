from rplidar import RPLidar
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


DMAX = 2000
IMIN = 0
IMAX = 500
flag = True
lidar = RPLidar('/dev/ttyUSB0')

def update_line(num, iterator, line):
    scan = next(iterator)
    offsets1 = np.array([(meas[2]*np.cos(np.radians(meas[1])), meas[2]*np.sin(np.radians(meas[1]))) for meas in scan])
    line.set_offsets(offsets1)
    intens = np.array([meas[0] for meas in scan])
    line.set_array(intens)
    print(line)
    return line


def main():
    info = lidar.get_info()
    print(info)

    health = lidar.get_health()
    print(health)

    fig = plt.figure()
    ax = plt.subplot(111)
    line = ax.scatter([0, 0], [0, 0], s=5, c=[IMIN, IMAX],
                      cmap=plt.cm.Greys_r, lw=0)
    plt.plot([0], [0], 'ro')

    ax.set_xlim(-DMAX, DMAX)
    ax.set_ylim(-DMAX, DMAX)
    ax.grid(True)
    iterator = lidar.iter_scans()
    ani = animation.FuncAnimation(fig, update_line,
                                  fargs=(iterator, line), interval=50)
    plt.show()

    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()





if __name__== '__main__':
    main()
