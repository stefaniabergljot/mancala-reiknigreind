from array import array
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
import sys

sys.path.append("../../../")

from mancala.game import (
    play_turn,
    initial_board,
    still_going,
    is_finished,
    winner,
    copy_board,
    flip_board,
    legal_actions,
    copy_board,
    is_finished,
    Board,
    Player,
    board_repr,
    get_score,
    RANGE0,
    RANGE1
)

NAME = "EV-45"
nb = 20

def random_policy(board, legal_actions, player):
    return random.choice(legal_actions)

# assume that player 0 is allways the one to play!
def one_hot_encode(brd):
    nf = 12*nb + 1 # +1 just code my own Manacala pit
    x = np.zeros(nf)
    x[-1] = (brd[13]-brd[6]) / 24.0 # my own mancala
    for i in RANGE0:
        j = min(brd[i],nb-1)
        x[nb*i+j] += 1
    for i in RANGE1:
        j = min(brd[i],nb-1)
        x[nb*(i-1)+j] += 1  
    return x

def getfeatures(board, legal_actions):
    nf = 12*nb + 1 # +1 just code my own Manacala pit
    x = np.zeros((nf,len(legal_actions)))
    for i in range(len(legal_actions)):
        brd = list(board).copy()
        play_turn(brd, 0, legal_actions[i]) # always assumes player 0 is the one to play
        x[:,i] = one_hot_encode(brd)
    return(x)

def softmax_policy(xa,  model):
    (nx,na) = xa.shape 
    x = Variable(torch.tensor(xa, dtype = torch.float, device = device)) 
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(model[1],x) + model[0] @ torch.ones((1,na),device = device)  # matrix-multiply x with input weight w1 and add bias
    h_tanh = h.tanh() # squash this with a sigmoid function
    y = torch.mm(model[3],h_tanh) + model[2] # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach() # .cpu()
    # now for the actor:
    pi = torch.mm(model[4],h_tanh).softmax(1)
    m = torch.multinomial(pi, 1) # soft
    #m = torch.argmax(pi) # greedy
    value = va.data[0,m] # assuming the reward is zero this is the actual target value
    advantage = value - torch.sum(pi*va)
    xtheta_mean = torch.sum(torch.mm(h_tanh,torch.diagflat(pi)),1)
    h_tanh = torch.squeeze(h_tanh[:,m],1)
    grad_ln_pi = h_tanh.view(1,len(xtheta_mean)) - xtheta_mean.view(1,len(xtheta_mean))
    x_selected = Variable(torch.tensor(xa[:,m], dtype = torch.float, device = device)).view(nx,1)
    return va, m, x_selected, grad_ln_pi, value, advantage.item()

def softmax_policy_(xa,  model):
    (nx,na) = xa.shape 
    x = Variable(torch.tensor(xa, dtype = torch.float, device = device)) 
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(model[1],x) + model[0] @ torch.ones((1,na), device = device)  # matrix-multiply x with input weight w1 and add bias
    h_tanh = h.tanh() # squash this with a sigmoid function
    y = torch.mm(model[3],h_tanh) + model[2] # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach()
    # now for the actor:
    pi = torch.mm(model[4],h_tanh).softmax(1)
    m = torch.multinomial(pi, 1) # soft
    #####m = torch.argmax(pi) # greedy
    return va, m

def update_model(model, trace, alpha, phi, value, reward, I, gradlnpi, advantage, gamma = 1.0, lam = 0.7):
    # extract all the weights used by the neural network
    # b1 = model[0] w1 = model[1] b2 = model[2] w2 = model[3] theta = model[4]
  
    x = Variable(torch.tensor(phi.transpose(), dtype = torch.float, device = device)).view((len(phi),1))
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(model[1],x) + model[0] # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh() # squash this with a sigmoid function
    y = torch.mm(model[3],h_sigmoid) + model[2] # multiply with the output weights w2 and add bias
    y_sigmoid = y.sigmoid()
    # now compute all gradients
    y_sigmoid.backward()
    # update the eligibility traces using the gradients then zero the gradients
    for i in range(4):
        trace[i] = gamma * lam * trace[i] + model[i].grad.data
        model[i].grad.data.zero_()
    trace[4] = gamma * lam * trace[4] + I * gradlnpi 
    delta = reward + gamma * value - y_sigmoid.detach() # this is the usual TD error
    
    # perform now the update for the weights
    delta =  torch.tensor(delta, dtype = torch.float, device = device)
    for i in range(2):
        model[i].data = model[i].data + alpha[1] * delta * trace[i]
    for i in range(2,4):
        model[i].data = model[i].data + alpha[2] * delta * trace[i]   
    
    #model[4].data = model[4].data + alpha[0] * advantage * trace[4]
    model[4].data = model[4].data + alpha[0] * delta * trace[4]
    # Book-keeping stuff, I'm assuming gamma = 1 here
    I = gamma * I
    return model, I

def learnit(model, alpha = [0.01, 0.001, 0.001], epsilon = 0, debug = False):
    nx = 12*nb+1
    phi = np.zeros((nx,2))
    phiold = np.zeros((nx,2))
    # initialize all traces, to zero
    traces = traces = [len(model) * [None]] * 2
    for p in range(2):
        for m in range(len(model)):
            traces[p][m] = torch.zeros(model[m].size(), device = device, dtype = torch.float)
    I = [1.0,1.0]
    grad_ln_pi = [0.0, 0.0] # no update made for first moves!
    advantage = [0.0, 0.0] # no update made for the first moves!
    board = initial_board()
    # the player to start is randomly chosen
    player = random.randint(0,1)
    while still_going(board):
        if player == 1: # player 0 owns the neural network, player 1 borrows it!
            flip_brd = flip_board(board)
            possible_actions = legal_actions(flip_brd, 0) # I pretend to be player 0
            x = getfeatures(flip_brd, possible_actions)
        else:
            possible_actions = legal_actions(board, player)
            x = getfeatures(board, possible_actions)
        va, k, x_selected, grad_ln_pi[player], value, advantage[player] = softmax_policy(x, model)

        action = possible_actions[k] # get the actual corresponding for the stochastic policy used
        phi[:,player] = x[:,k] # lets keep a track of the current after-state
        if player == 1: 
            action = action + 7 # flip the action for player 1, since we flipped the board
        new_player = play_turn(board, player, action) # do the actual move
        # check if the game is over
        if False == still_going(board):
            wplayer = winner(board) # check who won
            if wplayer >= 0: # obviously not a draw, eithe 0 or 1 won
                # first we update the last know states for the two players and reward according to their win/loss
                model, I[wplayer] = update_model(model, traces[wplayer], alpha, phiold[:,wplayer], value=0.0, reward=1.0, I = I[wplayer], gradlnpi = grad_ln_pi[wplayer], advantage = advantage[wplayer])
                model, I[1-wplayer] = update_model(model, traces[1-wplayer], alpha, phiold[:,1-wplayer], value=0.0, reward=0.0, I = I[1-wplayer], gradlnpi = grad_ln_pi[1-wplayer], advantage = advantage[1-wplayer])
                if wplayer == player:
                    # if the last player to move won, then we also update this state
                    model, I[player] = update_model(model, traces[player], alpha, phi[:,player], value=0.0, reward=1.0, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
                else:
                    model, I[player] = update_model(model, traces[player], alpha, phi[:,player], value=0.0, reward=0.0, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
            else: # we have a draw so we update both players with 0.5 reward
                model, I[player] = update_model(model, traces[player], alpha, phiold[:,player], value=0.0, reward=0.5, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
                model, I[1-player] = update_model(model, traces[1-player], alpha, phiold[:,1-player], value=0.0, reward=0.5, I = I[1-player], gradlnpi = grad_ln_pi[1-player], advantage = advantage[1-player])            
                model, I[player] = update_model(model, traces[player], alpha, phi[:,player], value=0.0, reward=0.5, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
            break
        # if the game is not over, then we update the last known state for the player that just moved
        model, I[player] = update_model(model, traces[player], alpha, phiold[:,player], value=value, reward=0.0, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
        phiold[:,player] = phi[:,player]
        #grad_ln_pi[player] = gradlnpi
        player = new_player

    return model

def competition(model):
    n = 2 # two players!
    nx = 12*nb+1
    phi = np.zeros((nx,2))
    board = initial_board()
    rand_player = random.randint(0,1)
    player = random.randint(0,1)
    while still_going(board):
        if rand_player == player:
            possible_actions = legal_actions(board, player)
            action = random_policy(board, possible_actions, player)
        else:
            if player == 1:
                flip_brd = flip_board(board)
                possible_actions = legal_actions(flip_brd, 0)
                x = getfeatures(flip_brd, possible_actions)
            else:
                possible_actions = legal_actions(board, player)
                x = getfeatures(board, possible_actions)
            va, k = softmax_policy_(x, model)
            #k = torch.argmax(va)
            action = possible_actions[k] # get the actual corresponding epsilon greedy move
            if player == 1:
                action = action + 7
        player = play_turn(board, player, action)

    wplayer = winner(board)
    if wplayer < 0:
        return 0.5
    if wplayer == rand_player:
        return 0.0
    return 1.0

# here is the main program:

start = time.time()

# cuda will only create a significant speedup for large/deep networks and batched training
device = torch.device('cpu')

# parameters for the training algorithm
alpha = [0.01, 0.001, 0.001]  # step size for PG and then each layer of the neural network

lam = 0.0 # lambda parameter in TD(lam-bda)
# define the parameters for the single hidden layer feed forward neural network
# randomly initialized weights with zeros for the biases
nx = nb*12 + 1 # number of input neurons
nh = int(nx/2) # number of hidden neurons

# now perform the actual training and display the computation time
delta_train_steps = 1000 # how many training steps to perform before testing
train_steps = 3000 # how many training steps to perform in total (should be a multiple of delta_train_steps)

model = 5 * [None]  # initialize the model size

if False: # this is a comment for when you want to load a previously trained model, the set True to False
    loadtrainstep = 999 # choose the training step to load and continue training
    model[0] = torch.load('./ac/b1_trained_'+str(loadtrainstep)+'.pth')
    model[1] = torch.load('./ac/w1_trained_'+str(loadtrainstep)+'.pth')
    model[2] = torch.load('./ac/b2_trained_'+str(loadtrainstep)+'.pth')
    model[3] = torch.load('./ac/w2_trained_'+str(loadtrainstep)+'.pth')
    model[4] = torch.load('./ac/theta_'+str(loadtrainstep)+'.pth')
    wins_against_random = np.load('./ac/wins_against_random.npy')
    wins_against_random = np.concatenate((wins_against_random, np.zeros(train_steps-loadtrainstep-1)))
    wins_against_random[loadtrainstep:] = 0.0
    comp_time = np.load('./ac/comp_time.npy')
    comp_time = np.concatenate((comp_time, np.zeros(train_steps-loadtrainstep-1)))
else:
    loadtrainstep = 0
    print("nx = %d, nh = %d" % (nx,nh))
    model[0] = Variable(torch.zeros((nh,1), device = device, dtype=torch.float), requires_grad = True)
    model[1] = Variable(0.1*torch.randn(nh,nx, device = device, dtype=torch.float), requires_grad = True)
    model[2] = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)
    model[3] = Variable(0.1*torch.randn(1,nh, device = device, dtype=torch.float), requires_grad = True)
    model[4] = Variable(0.1*torch.randn(1,nh, device = device, dtype=torch.float), requires_grad = True)
    wins_against_random = np.zeros(train_steps)
    comp_time = np.zeros(train_steps)

for trainstep in range(loadtrainstep,train_steps):
    print("Train step ", trainstep, " / ", train_steps)
    start = time.time()
    for i in range(100):
        war = competition(model)
        wins_against_random[trainstep] += war
    print("wins against random = ", wins_against_random[trainstep]/100*100)
    for k in range(delta_train_steps):
        model = learnit(model, alpha)
    torch.save(model[0], './ac/b1_trained_'+str(trainstep)+'.pth')
    torch.save(model[1], './ac/w1_trained_'+str(trainstep)+'.pth')
    torch.save(model[2], './ac/b2_trained_'+str(trainstep)+'.pth')
    torch.save(model[3], './ac/w2_trained_'+str(trainstep)+'.pth')
    torch.save(model[4], './ac/theta_'+str(trainstep)+'.pth')
    np.save('./ac/wins_against_random.npy', wins_against_random)
    # estimate the computation time to complete
    end = time.time()
    comp_time[trainstep] = np.round((end - start)/60)
    print("estimated time remaining: ", comp_time[trainstep]*(train_steps-loadtrainstep-trainstep+1)/60," (hours)")
    np.save('./ac/comp_time.npy', comp_time)
