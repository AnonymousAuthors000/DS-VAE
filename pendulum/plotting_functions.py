import matplotlib.pyplot as plt
import numpy as np

def plot_thetas(t, theta1s, theta2s, i=0, title=None, ax=None, alpha=.2, do_y_label=True, linestyle='-'):
    if ax is None:
        show = True
        ax = plt.gca()
    else:
        show = False
    if title is not None:
        ax.set_title(title)
    ax.plot(t, theta1s, label=f'Angular Displacement (rad) {i}', color='b', alpha=alpha, linestyle=linestyle)
    ax.plot(t, theta2s, label=f'Angular Velocity (rad/s) {i}', color='r', alpha=alpha, linestyle=linestyle)
    ax.set_xlabel('Time (s)')
    if do_y_label:
        ax.set_ylabel('No Damping, Angular Disp. (rad) and Angular Vel. (rad/s)')

    if show:
        plt.show()
    return ax


def plot_theta(t, thetas, i=0, title=None, ax=None, alpha=.2, do_y_label=True, do_title=True, do_x_axis=True, linestyle='-', color="blue"):
    if ax is None:
        show = True
        ax = plt.gca()
    else:
        show = False
    if do_title:
        if title is not None:
            ax.set_title(title, size=15)
    ax.plot(t, thetas, label=f'Angular Displacement (rad) {i}', color=color, alpha=alpha, linestyle=linestyle)
    if do_y_label:
        ax.set_ylabel('Angular Displacement (rad)', size = 15)
        
    if  do_x_axis:
        ax.set_xlabel('Time (s)', size = 20)

    if show:
        plt.show()
    return ax
    
    
def plot_random_test_point(t, test_set, learned_simulator, seed=None):
    if seed is not None:
        test_idx = np.random.RandomState(seed=seed).randint(0, len(test_set) -1)
    else:
        test_idx = np.random.randint(0, len(test_set) -1)
    predicted_thetas = learned_simulator(test_set[test_idx][0].unsqueeze(0)).squeeze(0).detach().numpy()
    predicted_theta1s = predicted_thetas[:t.shape[0]]
    predicted_theta2s = predicted_thetas[t.shape[0]:]

    actual_thetas = test_set[test_idx][1].squeeze(0).numpy()
    actual_theta1s = actual_thetas[:t.shape[0]]
    actual_theta2s = actual_thetas[t.shape[0]:]
    
    plot_thetas(t, actual_theta1s, actual_theta2s)
    plot_thetas(t, predicted_theta1s, predicted_theta2s)
