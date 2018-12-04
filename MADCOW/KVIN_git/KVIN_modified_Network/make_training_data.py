import sys
import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt
from scipy import io


def make_data(dom_size,n_domains):
    X_l=[]
    S1_l=[]
    S2_l=[]
    gamma_l=[]
    Labels_l=[]

    for k in n_domains:
        # Read image
        scenario_num=k
        grid_path = 'data\\scenarios_grid\\Scenario_' + str(scenario_num) + '_grid.png'


        grid = pilimg.open(grid_path)
        grid = grid.convert('L')

        # Display image
        # grid.show()

        # Fetch image pixel data to numpy array
        pix = np.array(grid)

        # Read simulation data
        mat_path = 'data\\scenario_states\\Gen' + str(scenario_num) + '.mat'
        Scenario = io.loadmat(mat_path)
        X = np.array(Scenario['X'])
        Y = np.array(Scenario['Y'])
        gamma= np.array(Scenario['gamma'])
        acc = np.array(Scenario['acc'])
        ns = X.shape[0] - 1



        # find goal
        goal_path = 'data\\scenarios_grid\\Scenario_goal' + str(scenario_num) + '.png'

        goal_grid = pilimg.open(goal_path)
        goal_grid = goal_grid.convert('L')
        goal_pix = np.array(goal_grid)
        for i in range(dom_size[0]):
            for j in range(dom_size[1]):
                if goal_pix[i, j] == 200:
                    goal = np.array([i, j])


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

        image = 1 - im_obs
        image_data = np.resize(image, (1, 1, dom_size[0], dom_size[1]))


        # reward grid domain generation
        value_prior = np.zeros((dom_size[0], dom_size[1]))
        value_prior[goal[0], goal[1]] = 10
        value_data = np.resize(value_prior, (1, 1, dom_size[0], dom_size[1]))

        # get image observation
        iv_mixed = np.concatenate((image_data, value_data), axis=1)




        X_current = np.tile(iv_mixed, (ns, 1, 1, 1))
        S1_current = np.expand_dims(X[0:ns, 0], axis=1)
        S2_current = np.expand_dims(Y[0:ns, 0], axis=1)
        gamma_current = np.expand_dims(gamma[0:ns, 0], axis=1)
        Labels_current = np.expand_dims(acc, axis=1)

        X_l.append(X_current)
        S1_l.append(S1_current)
        S2_l.append(S2_current)
        gamma_l.append(gamma_current)
        Labels_l.append(Labels_current)

    X_f = np.concatenate(X_l)
    S1_f = np.concatenate(S1_l)
    S2_f = np.concatenate(S2_l)
    gamma_f = np.concatenate(gamma_l)
    Labels_f = np.concatenate(Labels_l)

    return X_f, S1_f, S2_f, gamma_f, Labels_f


print("\nNow making  training data...")
dom_size = np.array([16, 16])
training_n_domains=[1,2,3,4,5,6,7,8,9,10]
X_out_tr, S1_out_tr, S2_out_tr, gamma_out_tr, Labels_out_tr = make_data(dom_size, training_n_domains)
sys.stdout.write("\nThe " + str(X_out_tr.shape[0]) + " of training data was generated\n\n")

print("\nNow making  testing data...")
test_n_domains=[8]
X_out_ts, S1_out_ts, S2_out_ts,gamma_out_ts,Labels_out_ts = make_data(dom_size, test_n_domains)
sys.stdout.write("\nThe " + str(X_out_ts.shape[0]) + " of training data was generated\n\n")

print("\nNow saving dataset...")
save_path = "dataset/gridworld_{0}x{1}".format(dom_size[0], dom_size[1])
np.savez_compressed(save_path, X_out_tr, S1_out_tr, S2_out_tr,gamma_out_tr,Labels_out_tr, X_out_ts, S1_out_ts, S2_out_ts,gamma_out_ts,Labels_out_ts)
print("\ndataset saved")

