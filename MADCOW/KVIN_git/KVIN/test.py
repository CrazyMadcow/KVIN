import sys
import argparse

import PIL.Image as pilimg
from scipy import io
import math
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch.autograd import Variable

from dataset.dataset import *
from model import *


def main(config,
         n_domains=1,
         n_traj=1,
         n_actions=8):
    # Correct vs total:
    correct, total = 0.0, 0.0
    # Automatic swith of GPU mode if available
    use_GPU = torch.cuda.is_available()
    # Instantiate a VIN model
    vin = VIN(config)
    # Load model parameters
    vin.load_state_dict(torch.load(config.weights))
    # Use GPU if available
    if use_GPU:
        vin = vin.cuda()

    # dom size
    dom_size = [16, 16]

    for dom in range(n_domains):
        # Randomly select goal position
        # find goal
        goal_path = 'data\\scenarios_grid\\Scenario_goal' + str(5) + '.png'

        goal_grid = pilimg.open(goal_path)
        goal_grid = goal_grid.convert('L')
        goal_pix = np.array(goal_grid)
        for i in range(dom_size[0]):
            for j in range(dom_size[1]):
                if goal_pix[i, j] == 200:
                    goal_index = np.array([i, j])
        goal = [40, 40]
        # Generate obstacle map
        grid_path = 'data\\scenarios_grid\\Scenario_' + str(5) + '_grid.png'

        grid = pilimg.open(grid_path)
        grid = grid.convert('L')

        # Display image
        # grid.show()

        # Fetch image pixel data to numpy array
        pix = np.array(grid)


        # get empty domain
        dom = np.zeros(dom_size)

        # obs grid domain generation
        im_try = np.copy(dom)

        for i in range(dom_size[0]):
            for j in range(dom_size[1]):
                if pix[i, j] == 0:
                    im_try[i, j] = 1

        dom = im_try

        im_obs = np.copy(dom)
        im_obs = np.max(im_obs) - im_obs
        im_obs = im_obs / np.max(im_obs)

        # Get value prior
        # reward grid domain generation
        #reward_prior = -1 * np.ones((dom_size[0], dom_size[1]))
        #reward_prior[goal_index[0], goal_index[1]] = 10
        #value_prior = reward_prior

        value_prior = np.zeros((dom_size[0], dom_size[1]))
        value_prior[goal_index[0], goal_index[1]] = 10


        # Sample random trajectories to our goal
        # Read simulation data

        mat_path = 'data\\scenario_states\\Gen' + str(5) + '.mat'
        Scenario = io.loadmat(mat_path)
        X = np.array(Scenario['X'])
        Y = np.array(Scenario['Y'])
        acc = np.array(Scenario['acc'])
        ns = X.shape[0] - 1

        for i in range(n_traj):
            if len(X) > 1:

                # Get number of steps to goal
                L = len(X) * 2
                # Allocate space for predicted steps
                pred_traj = np.zeros((L, 2))
                # Set starting position
                pred_traj[0, :] = [X[0], Y[0]]

                gamma = math.radians(45)


                for j in range(1, L):
                    print(j)
                    # Transform current state data
                    state_data = pred_traj[j - 1, :]
                    state_data = state_data.astype(np.float32)
                    # Transform domain to Networks expected input shape
                    im_data = im_obs.astype(np.int)
                    im_data = 1 - im_data
                    im_data = im_data.reshape(1, 1, config.imsize,
                                              config.imsize)
                    # Transfrom value prior to Networks expected input shape
                    value_data = value_prior.astype(np.int)
                    value_data = value_data.reshape(1, 1, config.imsize,
                                                    config.imsize)



                    # Get inputs as expected by network
                    X_in = torch.from_numpy(
                        np.append(im_data, value_data, axis=1)).float()
                    S1_in = torch.from_numpy(state_data[0].reshape(
                        [1, 1]))
                    S2_in = torch.from_numpy(state_data[1].reshape(
                        [1, 1]))
                    # Send Tensors to GPU if available
                    if use_GPU:
                        X_in = X_in.cuda()
                        S1_in = S1_in.cuda()
                        S2_in = S2_in.cuda()
                    # Wrap to autograd.Variable
                    X_in, S1_in, S2_in = Variable(X_in), Variable(
                        S1_in), Variable(S2_in)
                    #if j==110:
                    #   break
                    # Forward pass in our neural net
                    a = vin(X_in, S1_in, S2_in, config)
                    # Transform prediction to indices
                    s = np.array([pred_traj[j - 1, 0], pred_traj[j - 1, 1]])
                    print(s)
                    print("action :" + str(a))
                    print("gamma :" + str(gamma))
                    ns, gamma = get_next_states(s, gamma, a)
                    nx = ns[0]
                    ny = ns[1]
                    pred_traj[j, 0] = nx
                    pred_traj[j, 1] = ny
                    if abs(nx - goal[0]) < 0.5 and abs(ny - goal[1]) < 0.5:
                        # We hit goal so fill remaining steps
                        pred_traj[j + 1:, 0] = nx
                        pred_traj[j + 1:, 1] = ny
                        break

                # Plot optimal and predicted path (also start, end)
                if pred_traj[-1, 0] == goal[0] and pred_traj[-1, 1] == goal[1]:
                    correct += 1
                total += 1
                if config.plot == True:
                    visualize(im_obs, np.array([X[i], Y[i]]), pred_traj)
        #sys.stdout.write("\r" + str(int((float(dom) / n_domains) * 100.0)) + "%")
        #sys.stdout.flush()
    sys.stdout.write("\n")
    #print('Rollout Accuracy: {:.2f}%'.format(100 * (correct / total)))


def visualize(dom, states_xy, pred_traj):
    fig, ax = plt.subplots()
    implot = plt.imshow(dom, cmap="Greys_r")
    ax.plot(states_xy[:, 0], states_xy[:, 1], c='b', label='Optimal Path')
    ax.plot(
        pred_traj[:, 0], pred_traj[:, 1], '-X', c='r', label='Predicted Path')
    ax.plot(states_xy[0, 0], states_xy[0, 1], '-o', label='Start')
    ax.plot(states_xy[-1, 0], states_xy[-1, 1], '-s', label='Goal')
    legend = ax.legend(loc='upper right', shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-small')  # the legend text size
    for label in legend.get_lines():
        label.set_linewidth(0.5)  # the legend line width
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)


def get_next_states(s, gamma, a):
    dT = 0.1
    x = s[0]
    y = s[1]

    x = x + 0.25 * math.cos(gamma) * dT
    y = y + 0.25 * math.sin(gamma) * dT

    gamma = gamma + (a / 0.25) * dT
    ns = np.array([x, y])
    return ns, gamma


if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        default='trained/vin_16x16.pth',
        help='Path to trained weights')
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--imsize', type=int, default=16, help='Size of image')
    parser.add_argument(
        '--k', type=int, default=20, help='Number of Value Iterations')
    parser.add_argument(
        '--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument(
        '--l_h',
        type=int,
        default=150,
        help='Number of channels in first hidden layer')
    parser.add_argument(
        '--l_q',
        type=int,
        default=3,
        help='Number of channels in q layer (~actions) in VI-module')
    config = parser.parse_args()
    # Compute Paths generated by network and plot
    main(config)
