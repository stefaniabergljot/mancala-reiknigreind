from array import array
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
import sys
import time

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

from mancala.groups.minmax.action import action as minmax_action

NAME = "Dyna-Q"
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
        #if brd[i] > j:
        #    print("Warning: encountered state with more than " + str(nb) + " beans")
        x[nb*i+j] += 1
    for i in RANGE1:
        j = min(brd[i],nb-1)
        #if brd[i] > j:
        #    print("Warning: encountered state with more than " + str(nb) + " beans")
        x[nb*(i-1)+j] += 1
    return x

# helper function for debugging
def one_hot_decode(x):
    # phi has 241 items
    # last one is the score difference divided by 24
    brd = [0]*14
    for i in RANGE0:
        for j in range(nb):
            if x[nb*i + j] == 1:
                brd[i] = j
                break
    for i in RANGE1:
        for j in range(nb):
            if x[nb*(i-1) + j] == 1:
                brd[i] = j
                break
    #if x[-1] > 0.0:
    #    brd[13] = x[-1] * 24.0
    #else:
    #    brd[6] = x[-1] * 24.0
    brd[13] = x[-1] * 24.0
    total = sum(brd)
    if total < 6*4*2 and (6*4*2 - total) % 2 == 0 :
        missing_beans = 6*4*2 - total
        brd[6] += missing_beans / 2
        brd[13] += missing_beans / 2
    return brd

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

def softmax_policy_(xa,  model, hard=False):
    (nx,na) = xa.shape
    x = Variable(torch.tensor(xa, dtype = torch.float, device = device))
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(model[1],x) + model[0] @ torch.ones((1,na), device = device)  # matrix-multiply x with input weight w1 and add bias
    h_tanh = h.tanh() # squash this with a sigmoid function
    y = torch.mm(model[3],h_tanh) + model[2] # multiply with the output weights w2 and add bias
    va = y.sigmoid().detach()
    # now for the actor:
    pi = torch.mm(model[4],h_tanh).softmax(1)
    if not hard:
        m = torch.multinomial(pi, 1) # soft
    else:
        m = torch.argmax(pi) # greedy
    return va, m

def update_model(model, trace, alpha, phi, value, reward, I, gradlnpi, advantage, gamma = 1.0, lam = 0.95):
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
    #trace[4] = gamma * lam * trace[4] + I * gradlnpi
    #trace[4] = gamma * lam * trace[4] + I * gradlnpi
    trace[4] = gamma * lam * trace[4] + gradlnpi
    delta = reward + gamma * value - y_sigmoid.detach() # this is the usual TD error

    # perform now the update for the weights
    delta =  torch.tensor(delta, dtype = torch.float, device = device)
    for i in range(2):
        model[i].data = model[i].data + alpha[1] * delta * trace[i]
    for i in range(2,4):
        model[i].data = model[i].data + alpha[2] * delta * trace[i]

    model[4].data = model[4].data + alpha[0] * advantage * trace[4]
    # Book-keeping stuff, I'm assuming gamma = 1 here
    #I = gamma * I
    return model, I

def getTerminalStateValue(board, player):
    win = winner(board)
    if win == player:
        rew = 1.0
    elif win == 1-player:
        rew = 0.0
    else:
        rew = 0.5
    return rew

def evaluate(phi):
    #phi = one_hot_encode(board)
    # evaluate the position from the perspective of the current player
    x = Variable(torch.tensor(phi.transpose(), dtype=torch.float, device=device)).view((len(phi), 1))
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(model[1], x) + model[0]  # matrix-multiply x with input weight w1 and add bias
    h_sigmoid = h.tanh()  # squash this with a sigmoid function
    y = torch.mm(model[3], h_sigmoid) + model[2]  # multiply with the output weights w2 and add bias
    return y.sigmoid().detach()[0]

def initial_board2() -> array:
    return initial_board()
    import array
    #return array.array('i', [1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 0])
    #return array.array("i", 2 * (6 * [1] + [0]))
    #return initial_board()
    #return array.array("i", 2 * (4 * [0] + [2] + [1] + [0]))
    # array('i', [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 1, 0])
    import array
    #return array.array("i", [
    #    0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0]
    #                   )
    #return array.array("i", [
    #                         0, 0, 0, 3, 2, 1, 1,
    #                         0, 0, 0, 0, 2, 1, 0]
    #                   )

def learnit2(model, alpha = [0.01, 0.001, 0.001], mmdepth = 3):
    nx = 12 * nb + 1
    phi = np.zeros((nx, 2))
    phiold = np.zeros((nx, 2))
    # initialize all traces, to zero
    traces = len(model) * [None]
    for m in range(len(model)):
        traces[m] = torch.zeros(model[m].size(), device=device, dtype=torch.float)
    I = 1.0,
    grad_ln_pi = 0.0  # no update made for first moves!
    advantage = 0.0  # no update made for the first moves!
    board = initial_board2()
    # the player to start is randomly chosen
    #currentplayer = random.randint(0, 1)
    minmaxplayer = 1
    learningplayer = 0
    data = []
    priority_data = []

    if random.randint(0,1) == 1:
        # random chance that minmax starts
        while True:
            possible_actions = legal_actions(board, 0)
            action = minmax_action(board, possible_actions, 0, mmdepth)
            new_player = play_turn(board, 0, action)  # do the actual move
            if new_player == 1:
                board = flip_board(board)
                break
    # current player's turn
    is_terminal = False
    phiold = one_hot_encode(board)
    while still_going(board):
        # learning player's turn, board faces us
        possible_actions = legal_actions(board, 0)
        x = getfeatures(board, possible_actions)
        va, k, x_selected, grad_ln_pi, value, advantage = softmax_policy(x, model)
        action = possible_actions[k]  # get the actual corresponding for the stochastic policy used
        phi = x[:, k]  # lets keep a track of the current after-state

        new_player = play_turn(board, 0, action)  # do the actual move

        if not still_going(board):
            # learning player just performed terminal move (could be winning/losing)
            # TODO update and terminate training game
            is_terminal = True
            outcome = getTerminalStateValue(board, 0)
        elif new_player == 1:
            board = flip_board(board)
            while True:
                possible_actions = legal_actions(board, 0)
                action = minmax_action(board, possible_actions, 0, mmdepth)
                #action = random_policy(board, possible_actions, 0)
                new_player = play_turn(board, 0, action)  # do the actual move
                if not still_going(board):
                    # minmax player just performed terminal move (could be winning/losing)
                    # TODO update and terminate training game
                    is_terminal = True
                    outcome = getTerminalStateValue(board, 1)
                    break
                elif new_player == 1:
                    # it's learning player's turn again, break out
                    board = flip_board(board)
                    break
        if is_terminal:
            rew = outcome
            #rew = -1 * (phi[-1] - phiold[-1])  #
            data.append((phiold, phi, action, rew, 0))
            if rew > 0.0:
                priority_data.append((phiold, phi, action, rew, 0))
            #rew = -1 * (phi[-1] - phiold[-1])  #
            model, _ = update_model(model, traces, alpha, phiold, 0.0, rew, I, grad_ln_pi, advantage)
        else:
            phi = one_hot_encode(board) # is this right?
            rew = -1 * (phi[-1] - phiold[-1]) #
            data.append((phiold, phi, action, 0.0, 0))
            model, _ = update_model(model, traces, alpha, phiold, evaluate(phi), 0.0, I, grad_ln_pi, advantage)
        phiold = phi
    return model, data, priority_data, outcome


def learnit(model, alpha = [0.01, 0.001, 0.001], epsilon = 0, debug = False, mmdepth = 3):
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
    board = initial_board2()
    # the player to start is randomly chosen
    player = random.randint(0,1)
    minmaxplayer = random.randint(0,1)
    data = []
    while still_going(board):
        if player == 1: # player 0 owns the neural network, player 1 borrows it!
            flip_brd = flip_board(board)
            possible_actions = legal_actions(flip_brd, 0) # I pretend to be player 0
            x = getfeatures(flip_brd, possible_actions)
        else:
            possible_actions = legal_actions(board, player)
            x = getfeatures(board, possible_actions)
        #va, k, x_selected, grad_ln_pi[player], value, advantage[player] = softmax_policy(x, model)
        if minmaxplayer == player:
            if player == 1:
                action = minmax_action(flip_brd, possible_actions, 0, mmdepth)
            else:
                action = minmax_action(board, possible_actions, player, mmdepth)
        else:
            va, k, x_selected, grad_ln_pi[player], value, advantage[player] = softmax_policy(x, model)
            action = possible_actions[k] # get the actual corresponding for the stochastic policy used
            # TODO
            phi[:,player] = x[:,k] # lets keep a track of the current after-state
        if player == 1:
            action = action + 7 # flip the action for player 1, since we flipped the board
        new_player = play_turn(board, player, action) # do the actual move
        # check if the game is over
        if False == still_going(board):
            # only update things from the perspective of the winning player, right?
            wplayer = winner(board) # check who won

            if wplayer >= 0: # obviously not a draw, either 0 or 1 won
                # first we update the last know states for the two players and reward according to their win/loss
                if player != minmaxplayer:
                    if minmaxplayer != wplayer:
                        model, I[wplayer] = update_model(model, traces[wplayer], alpha, phiold[:,wplayer], value=0.0, reward=1.0, I = I[wplayer], gradlnpi = grad_ln_pi[wplayer], advantage = advantage[wplayer])
                    else:
                        model, I[1-wplayer] = update_model(model, traces[1-wplayer], alpha, phiold[:,1-wplayer], value=0.0, reward=0.0, I = I[1-wplayer], gradlnpi = grad_ln_pi[1-wplayer], advantage = advantage[1-wplayer])
                if wplayer == player:
                    # if the last player to move won, then we also update this state
                    if player != minmaxplayer:
                        model, I[player] = update_model(model, traces[player], alpha, phi[:,player], value=0.0, reward=1.0, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
                else:
                    if player != minmaxplayer:
                        model, I[player] = update_model(model, traces[player], alpha, phi[:,player], value=0.0, reward=0.0, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
                if player != minmaxplayer:
                    data.append((phiold[:,player],phi[:,player],action,wplayer==player,player))
            else: # we have a draw so we update both players with 0.5 reward
                if player != minmaxplayer:
                    model, I[player] = update_model(model, traces[player], alpha, phiold[:,player], value=0.0, reward=0.5, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
                    #model, I[1-player] = update_model(model, traces[1-player], alpha, phiold[:,1-player], value=0.0, reward=0.5, I = I[1-player], gradlnpi = grad_ln_pi[1-player], advantage = advantage[1-player])
                    model, I[player] = update_model(model, traces[player], alpha, phi[:,player], value=0.0, reward=0.5, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
                if player != minmaxplayer:
                    data.append((phiold[:,player],phi[:,player],action,0.5,player))
            break
        # if the game is not over, then we update the last known state for the player that just moved
        if player != minmaxplayer:
            model, I[player] = update_model(model, traces[player], alpha, phiold[:,player], value=value, reward=0.0, I = I[player], gradlnpi = grad_ln_pi[player], advantage = advantage[player])
        if player != minmaxplayer and sum(phiold[:,player] > 0.0):
            data.append((phiold[:,player],phi[:,player],action,0,player))
        phiold[:,player] = phi[:,player]
        #grad_ln_pi[player] = gradlnpi


        player = new_player

    return model, data

def competition(model):
    n = 2 # two players!
    nx = 12*nb+1
    phi = np.zeros((nx,2))
    board = initial_board2()
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
            va, k = softmax_policy_(x, model, True)
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

def dynaQ(model, iter = 1000):

    trace = [[]] * 4
    for _ in range(iter):
        (phiold,phi,action,reward,player) = DATA[np.random.randint(len(DATA))]
        x = Variable(torch.tensor(phi.transpose(), dtype = torch.float, device = device)).view((len(phi),1))
        # now do a forward pass to evaluate the board's after-state value
        h = torch.mm(model[1],x) + model[0] # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.tanh() # squash this with a sigmoid function
        y = torch.mm(model[3],h_sigmoid) + model[2] # multiply with the output weights w2 and add bias
        value = y.sigmoid()

        x = Variable(torch.tensor(phiold.transpose(), dtype = torch.float, device = device)).view((len(phi),1))
        # now do a forward pass to evaluate the board's after-state value
        h = torch.mm(model[1],x) + model[0] # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.tanh() # squash this with a sigmoid function
        y = torch.mm(model[3],h_sigmoid) + model[2] # multiply with the output weights w2 and add bias
        y_sigmoid = y.sigmoid()
        # now compute all gradients
        y_sigmoid.backward()
        # update the eligibility traces using the gradients then zero the gradients
        for i in range(4):
            trace[i] = model[i].grad.data.clone()
            model[i].grad.data.zero_()
        delta = reward +  value - y_sigmoid.detach() # this is the usual TD error

        # perform now the update for the weights
        delta =  torch.tensor(delta, dtype = torch.float, device = device)
        for i in range(2):
            model[i].data = model[i].data + alpha[1] * delta * trace[i]
        for i in range(2,4):
            model[i].data = model[i].data + alpha[2] * delta * trace[i]

    trace = [[]] * 4
    if len(PRIORITY_DATA) > 0:
        for _ in range(iter):
            (phiold, phi, action, reward, player) = PRIORITY_DATA[np.random.randint(len(PRIORITY_DATA))]
            x = Variable(torch.tensor(phi.transpose(), dtype=torch.float, device=device)).view((len(phi), 1))
            # now do a forward pass to evaluate the board's after-state value
            h = torch.mm(model[1], x) + model[0]  # matrix-multiply x with input weight w1 and add bias
            h_sigmoid = h.tanh()  # squash this with a sigmoid function
            y = torch.mm(model[3], h_sigmoid) + model[2]  # multiply with the output weights w2 and add bias
            value = y.sigmoid()

            x = Variable(torch.tensor(phiold.transpose(), dtype=torch.float, device=device)).view((len(phi), 1))
            # now do a forward pass to evaluate the board's after-state value
            h = torch.mm(model[1], x) + model[0]  # matrix-multiply x with input weight w1 and add bias
            h_sigmoid = h.tanh()  # squash this with a sigmoid function
            y = torch.mm(model[3], h_sigmoid) + model[2]  # multiply with the output weights w2 and add bias
            y_sigmoid = y.sigmoid()
            # now compute all gradients
            y_sigmoid.backward()
            # update the eligibility traces using the gradients then zero the gradients
            for i in range(4):
                trace[i] = model[i].grad.data.clone()
                model[i].grad.data.zero_()
            delta = reward + value - y_sigmoid.detach()  # this is the usual TD error

            # perform now the update for the weights
            delta = torch.tensor(delta, dtype=torch.float, device=device)
            for i in range(2):
                model[i].data = model[i].data + alpha[1] * delta * trace[i]
            for i in range(2, 4):
                model[i].data = model[i].data + alpha[2] * delta * trace[i]

    # Book-keeping stuff, I'm assuming gamma = 1 here
    return model

# here is the main program:

start = time.time()

# cuda will only create a significant speedup for large/deep networks and batched training
device = torch.device('cpu')

# parameters for the training algorithm
alpha = [0.01, 0.001, 0.001]  # step size for PG and then each layer of the neural network
#alpha = [0.01, 0.005, 0.005]

lam = 0.0 # lambda parameter in TD(lam-bda)
# define the parameters for the single hidden layer feed forward neural network
# randomly initialized weights with zeros for the biases
nx = nb*12 + 1 # number of input neurons
nh = int(nx/2) # number of hidden neurons

# now perform the actual training and display the computation time
delta_train_steps = 100 # how many training steps to perform before testing
train_steps = 7000 # how many training steps to perform in total (should be a multiple of delta_train_steps)

model = 5 * [None]  # initialize the model size

if True: # this is a comment for when you want to load a previously trained model, the set True to False
    loadtrainstep = 1806 # choose the training step to load and continue training
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

DATA = list()
PRIORITY_DATA = list()
training_games = 0
training_wins = 0
recent_training_wins = 0
dynaQ_iterations = 2000
for trainstep in range(loadtrainstep,train_steps):
    print("Train step ", trainstep, " / ", train_steps)
    start = time.time()
    for i in range(100):
        war = competition(model)
        wins_against_random[trainstep] += war
    print("wins against random = ", wins_against_random[trainstep]/100*100)
    recent_training_wins = 0
    for k in range(delta_train_steps):
        #start_time = time.time()
        #model, data, priority_data, rew = learnit2(model, alpha, 1)
        model, data, priority_data, rew = learnit2(model, [0.01, 0.001, 0.001], 4)
        #print("Iteration took " + str(time.time() - start_time))
        DATA.extend(data)
        PRIORITY_DATA.extend(priority_data)
        training_games += 1
        recent_training_wins += rew
    training_wins += recent_training_wins
    print("Won " + str(round(recent_training_wins/delta_train_steps, 4)) + " of recent training games")
    print("Won " + str(round(training_wins/training_games, 4)) + " of training games")

    model = dynaQ(model, dynaQ_iterations)
    if len(DATA) > 2000:
        DATA = DATA[-2001:-1]
    if len(PRIORITY_DATA) > 4000:
        PRIORITY_DATA = PRIORITY_DATA[-4001:-1]

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
