import autograd.numpy as np   
from autograd import grad     #Kali ini akan dipakai autograd yang tersedia di Numpy.

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z) )

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exps = np.exp(z - np.max(z, axis = 1, keepdims = True))
    return exps/np.sum(exps, axis = 1, keepdims = True)

def create_input(size):
    return [{'dim':size, 'act': identity}]

def add_forward(layers, size, activation):
    layers.append({'dim':size, 'act':activation})
    return layers

def identity(z):
    return z

def compute_activation(act, A, W, b):
    return act(np.dot(W,A) + b)

def init_weights(layers):
    weights = []
    for l in range(1,len(layers)):
        W = np.random.randn( layers[l]['dim'], layers[l-1]['dim'] )
        b = np.zeros( (layers[l]['dim'], 1) )
        weights.append( [W,b] )
    return weights

def init_vw(layers):
    vw = []
    for l in range(1,len(layers)):
        vw_init = np.zeros( [layers[l]['dim'], layers[l-1]['dim']] )
        vb_init = np.zeros( (layers[l]['dim'], 1) )
        vw.append( [vw_init,vb_init] )
    return vw

def init_adam(layers):
    Mw = []
    Vw = []
    for l in range(1, len(layers)):
        mw_init = np.zeros( (layers[l]['dim'], layers[l-1]['dim']) )
        mb_init = np.zeros( (layers[l]['dim'], 1) )
        Mw.append( [mw_init,mb_init] )
        Vw.append( [mw_init,mb_init] )
    return Mw, Vw

def forward_pass(X, layers, weights):
    A_prev = X
    for (l,[W,b]) in zip(layers[1:], weights):
        A = compute_activation(l['act'], A_prev, W, b)
        A_prev = A
    return A_prev

def cost(yhat,y ):  
    loss = (y - np.transpose(yhat).reshape((1,-1)))**2
    eps = 1e-18   #Digunakan di cross entropy loss function
    #loss = -(y*np.log(yhat.flatten() + eps) + (1-y)*np.log(1 - yhat.flatten() + eps))   #Cross entropy
    costs = np.squeeze(np.mean(loss, axis = 1))
    return costs

def grad_descent(xs, ys, cost, net, weights, learning_rate = 0.001):
    fw = lambda W: forward_pass(xs, net, W)
    costs = lambda W: cost(fw(W), ys) 
    
    dw = grad(costs)(weights)
    #print(dw)
    for i in range(0,len(weights)):
        weights[i][0] -= dw[i][0] * learning_rate
        weights[i][1] -= dw[i][1] * learning_rate
    return costs(weights), weights

def NAG(xs, ys, cost, net, weights, vw_1, vw_2, vw, learning_rate = 0.001, gamma=0.99):
    fw = lambda W: forward_pass(xs, net, W)
    costs = lambda W: cost(fw(W), ys) 
    
    dw = grad(costs)(weights)
    for i in range(0,len(weights)):
        vw_1 = vw[i][0] + learning_rate * dw[i][0]
        vw_2 = vw[i][1] + learning_rate * dw[i][1]
        weights[i][0] -= vw_1 
        weights[i][1] -= vw_2 
    return costs(weights), weights

def ADAM(xs, ys, cost, net, weights, Mw, Vw, learning_rate=2):
    beta1 = 0.99
    beta2 = 0.99
    
    fw = lambda W: forward_pass(xs, net, W)
    costs = lambda W: cost(fw(W), ys)
    
    dw = grad(costs)(weights)
    eps = 1e-10
    for i in range(0, len(weights)):
        Mw[i][0] = beta1*Mw[i][0] + (1-beta1)*dw[i][0]
        Mw[i][1] = beta1*Mw[i][1] + (1-beta1)*dw[i][1]
        
        Vw[i][0] = beta2*Vw[i][0] + (1-beta2)*np.power(dw[i][0], 2)
        Vw[i][1] = beta2*Vw[i][1] + (1-beta2)*np.power(dw[i][1], 2)
        
        weights[i][0] -= learning_rate*Mw[i][0] / (np.sqrt(Vw[i][0]) + eps)
        weights[i][1] -= learning_rate*Mw[i][1] / (np.sqrt(Vw[i][1]) + eps)
    
    return costs(weights), weights, Mw, Vw

def RMS(xs, ys, cost, net, weights, vw_1, vw_2, vw, learning_rate = 0.001, gamma=0.99, beta=0.9, eps=1e-8):
    fw = lambda W: forward_pass(xs, net, W)
    costs = lambda W: cost(fw(W), ys) 
    
    dw = grad(costs)(weights)
    for i in range(0,len(weights)):
        vw_1 = beta * vw[i][0] + (1 - beta) * (dw[i][0])**2 
        vw_2 = beta * vw[i][1] + (1 - beta) * (dw[i][1])**2  
        weights[i][0] -= (learning_rate/(np.sqrt(vw_1)+eps)) * dw[i][0] 
        weights[i][1] -= (learning_rate/(np.sqrt(vw_2)+eps)) * dw[i][1]
        
    return costs(weights), weights



