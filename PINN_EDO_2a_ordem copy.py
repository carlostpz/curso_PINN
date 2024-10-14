import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

print(tf.config.list_physical_devices("GPU"))


# collocation points
train_t = (np.random.rand(1000)*10).reshape(-1, 1) # definindo os pontos no domínio e que serão utilizados para treino
train_f = - 2 * np.exp(-2*train_t) + 0.5 * np.exp(-3*train_t) + 0.5 * np.exp(-train_t)

# true values of a, b, c
# a = 1
# b = 5
# c = 6

# log normal density
def log_dnorm(x, mu, sd):
    return -0.5* tf.math.log(2*np.pi) -tf.math.log(sd) - 0.5* tf.math.square(x - mu)/ tf.math.square(sd) 


def ode(t, NN, a, b, c, y, 
        a_mu, a_sd,
        b_mu, b_sd, 
        c_mu, c_sd):

    t = t.reshape(-1,1)
    t = tf.constant(t, dtype = tf.float32)
    t_0 = tf.zeros((1))

    # calculate NN'(t) and NN''(t)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            f = NN(t)
        f_t = tape.gradient(f, t)
    f_tt = tape2.gradient(f_t, t)
        
    # likelihood loss
    likelihood_loss = tf.reduce_sum( tf.square(y - f) )

    # ODE loss
    ode_loss = tf.reduce_sum( tf.square( a*f_tt + b*f_t + c*f - tf.math.exp(-1*t) ) )
    
    #with tf.GradientTape(persistent = True) as tape3:
    #    tape3.watch(a)
    #    tape3.watch(b)
    #    tape3.watch(c)    
    #    ode_loss = tf.reduce_sum( tf.square( a*f_tt + b*f_t + c*f - tf.math.exp(-1*t) ) )
    #dL_da = tape3.gradient(ode_loss, a)
    #dL_db = tape3.gradient(ode_loss, b)
    #dL_dc = tape3.gradient(ode_loss, c)
    
    # loss for initial condition
    IC_loss = tf.reduce_sum( tf.square( NN(t_0) + 1 ) ) # f(0) = -1

    # loss for the boundary condition
    with tf.GradientTape(persistent=True) as tape0:
        tape0.watch(t_0)
        f_t0 = NN(t_0)
    df_dt0 = tape0.gradient(f_t0, t_0)
    boundary_loss = tf.square( df_dt0 - 2) # f'(0) = 2

    # prior loss
    prior_loss = -log_dnorm(x = a, mu = tf.zeros(1) + a_mu, sd = tf.ones(1)*a_sd ) 
    prior_loss = prior_loss - log_dnorm(x = b, mu = tf.zeros(1) + b_mu, sd = tf.ones(1)*b_sd )
    prior_loss = prior_loss - log_dnorm(x = c, mu = tf.zeros(1) + c_mu, sd = tf.ones(1)*c_sd )

    # total loss
    total_loss = 100000*ode_loss + IC_loss + boundary_loss + likelihood_loss + prior_loss

    return total_loss
    
NN = tf.keras.models.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(units = 60, activation = 'tanh'),
        tf.keras.layers.Dense(units = 60, activation = 'tanh'),
        tf.keras.layers.Dense(units = 1, activation = 'linear')
])



# prior distributions for a, b, c
a_mu = 1.0; a_sd = 0.0001
b_mu = 5.0; b_sd = 0.0001
c_mu = 0.0; c_sd = 10.0
optm = tf.keras.optimizers.Adam(learning_rate = 0.01)

a = tf.Variable([1.5], trainable = True, name = 'a')
b = tf.Variable([7.0], trainable = True, name = 'b')
c = tf.Variable([4.0], trainable = True, name = 'c')

trainable = NN.trainable_variables
trainable.append(a)
trainable.append(b)
trainable.append(c)

n_epochs = 20000

train_loss_record = np.zeros([n_epochs])
a_record = np.zeros([n_epochs])
b_record = np.zeros([n_epochs])
c_record = np.zeros([n_epochs])

for epoch in range(n_epochs):
    with tf.GradientTape() as tape:
        
        tape.watch(trainable)
        
        train_loss = ode(t = train_t, 
                         NN = NN,
                         a = a,
                         b = b,
                         c = c,
                         a_mu = a_mu,
                         a_sd = a_sd,
                         b_mu = b_mu,
                         b_sd = b_sd,
                         c_mu = c_mu,
                         c_sd = c_sd,
                         y = train_f)
            
    grad_w = tape.gradient(train_loss, trainable)
    optm.apply_gradients(zip(grad_w, trainable ))
    
    # record training_loss and ode parameters a, b, c
    train_loss_record[epoch] = train_loss     
    a_record[epoch] = a     
    b_record[epoch] = b     
    c_record[epoch] = c     


    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Training loss: {train_loss.numpy()}, a: {a.numpy()}, b: {b.numpy()}, c: {c.numpy()}')

# imprime o gráfico da perda com relação as épocas
plt.figure(figsize = (7,4))
plt.plot(train_loss_record)
plt.show()

test_t = np.linspace(0, 10, 100)
true_f = - 2 * np.exp(-2*test_t) + 0.5 * np.exp(-3*test_t) + 0.5 * np.exp(-test_t)

pred_f = NN.predict(test_t).ravel()

plt.figure(figsize = (7,4))
plt.plot(train_t, train_f, 'ok', label = 'Collocation Points')
plt.plot(test_t, true_f, '-k',label = 'Analytic')
plt.plot(test_t, pred_f, '--r', label = 'PINN')
plt.legend(fontsize = 9)
plt.xlabel('t', fontsize = 10)
plt.ylabel('f(t)', fontsize = 10)
plt.show()

plt.figure(figsize = (7,4))
iterations = np.arange(n_epochs)
plt.plot(iterations, a_record, '-k', label = 'Value of a')
plt.plot(iterations, b_record, '-r', label = 'Value of b')
plt.plot(iterations, c_record, '-b', label = 'Value of c')
plt.legend(fontsize = 9)
plt.xlabel('Iteration', fontsize = 10)
plt.ylabel('a', fontsize = 10)
plt.show()

d = "blablabla"
