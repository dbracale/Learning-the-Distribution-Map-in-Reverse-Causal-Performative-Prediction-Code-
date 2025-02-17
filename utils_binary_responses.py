import numpy as np
import scipy.stats as stats
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
import random
from scipy.stats import beta as sp_beta
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def main_parameters():
    inf_X = -5
    sup_X = 5
    r_q   = 0.5
    r_u   = 0.5
    wage  = 1
    return inf_X, sup_X, r_q,r_u, wage

inf_X = -5 #replace for beta
#inf_X = 0 #replace for beta
sup_X = 5
r_q     = 0.5
r_u     = 0.5
wage      = 1

range_X = sup_X-inf_X

def gaussian_cdf(x, mu, sigma):
    cdf = stats.norm.cdf(x, loc=mu, scale=sigma)
    return cdf

def gaussian_pdf(x, mu, sigma):
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    return pdf
    
def score_cdf(x, alpha, beta):
    #beta_dist = sp_beta(alpha, beta) #replace for beta
    #cdf = beta_dist.cdf(x) #replace for beta
    cdf = gaussian_cdf(x, alpha, beta)
    return cdf

def score_pdf(x, alpha, beta):
    #beta_dist = sp_beta(alpha, beta) #replace for beta
    #pdf = beta_dist.pdf(x) #replace for beta
    pdf = gaussian_pdf(x, alpha, beta)
    return pdf

#def indicator(x, lower, upper):
#    return np.where((x < lower) | (x > upper), 0, 1)

def indicator(x, lower, upper):
    if isinstance(x, (int, float)):
        return 1 if lower <= x <= upper else 0
    elif isinstance(x, (list, np.ndarray)):
        x_numeric = [float(val) for val in x]  # Convert elements to numeric type
        return np.where((np.array(x_numeric) >= lower) & (np.array(x_numeric) <= upper), 1, 0)
    


# F_u: unqualified CDF, F_u_bar = 1-F_u
# F_q qualified CDF, F_q_bar = 1-F_q
# f_u: unqualified pdf
# f_q: qualified pdf
# Incentive = workers’ incentive to invest
# r_u = cost of hiring an unqualified worker.
# r_q = firm’s reward for hiring a qualified worker

def F_u_bar(x):
    #return 1-score_cdf(x/range_X, 2, 8) #replace for beta
    return 1-score_cdf(x, -1, 1) #replace for gaussian

def F_q_bar(x):
    #return 1-score_cdf(x/range_X, 8, 2) #replace for beta
    return 1-score_cdf(x, 1, 0.5) #replace for gaussian

def f_u(x):
    #return score_pdf(np.array(x)/range_X, 2, 8)/range_X #replace for beta
    return score_pdf(np.array(x), -1, 1) #replace for gaussian

def f_q(x):
    #return score_pdf(np.array(x)/range_X, 8, 2)/range_X #replace for beta
    return score_pdf(np.array(x), 1, 0.5) #replace for gaussian

def Incentive(x):
    return wage*(F_q_bar(x)-F_u_bar(x))

lambd = lambda x: f_q(x)/f_u(x)
t_star = lambda pi: r_u*(1-pi)/(r_q*pi)

# f_X: pdf of X with weights p
def f_X(x,p):
    return p*f_q(x)+(1-p)*f_u(x)

# Minimize Incentive
max_funct = minimize_scalar(lambda x: -Incentive(x))
argmax_I = max_funct.x
max_I = Incentive(argmax_I)

# Split Incentive into left and right part
def left_I(x):
    return Incentive(x)*int(x<=argmax_I)
    
def right_I(x):
    return Incentive(x)*int(x>=argmax_I)
    
# define the proportion of data in the left and right
prop_left = (argmax_I-inf_X)/(sup_X-inf_X)

def firm_utility(x,F_C):
    return (r_q*F_C(Incentive(x))*F_q_bar(x)-r_u*(1-F_C(Incentive(x)))*F_u_bar(x))/(F_C(Incentive(x))*F_q_bar(x)+(1-F_C(Incentive(x)))*F_u_bar(x))


# Generate n samples from the PDF of the score distribution using the accept-reject method
def sample_from_pdf(n, pdf_func, lower, upper):
    if n ==1:
        finished = False
        while finished == False:
            x = np.random.uniform(lower, upper)
            y = np.random.uniform(0, 1)  # Uniform random variable for acceptance
            if y < pdf_func(x):
                finished = True
                return x
    else:
        samples = []
        while len(samples) < n:
            x = np.random.uniform(lower, upper)
            y = np.random.uniform(0, 1)  # Uniform random variable for acceptance
            if y < pdf_func(x):
                samples.append(x)
        return np.array(samples)



# Define the log-likelihood function
def neg_log_likelihood(params, data):
    # Extract the transformed parameter p
    p = 1 / (1 + np.exp(-params[0]))  # Transform to (0, 1)
    likelihood = np.sum(np.log(p * f_q(data) + (1 - p) * f_u(data)))
    return -likelihood  # Negative because we want to maximize



## Below, write a function that takes $T$ uniform $(0,wage)$ and returns the corresponding thetas. Since Incentive is not invertible but is unimodal, we have split the part where it is invertible in the left and right part, where the argmax is the divider.
# Find thetas such that Incentive(theta) is uniform
def get_theta(T,design_pdf):

  #plot Incentive and it's maximum
  #x = np.linspace(inf_X, sup_X, 100)
  #y = Incentive(x)
  #plt.plot(x, y)
  #plt.scatter(argmax_I, 0, color='red', marker='o', label='Point at x=2')
  #plt.axvline(argmax_I, color='green', linestyle='--', label='Vertical Line at x=2')
  #plt.xlabel('x')
  #plt.ylabel('Incentive(x)')
  #plt.title('Plot Incentive')
  #plt.grid(True)
  #plt.show()

  theta = []
  I_val = []
  for i in range(T-1):
    # Given dataset of y values
    I_val_i = sample_from_pdf(1,design_pdf,0+10**(-6),max_I-10**(-6))
    I_val.append(I_val_i)
    # say 1 if left and 0 if right
    sample = random.choices([0, 1], weights=[1-prop_left, prop_left], k=1)[0]
    if sample == 1:
      result = root_scalar(lambda x: left_I(x) - I_val_i, bracket=[inf_X, argmax_I])  # Bracket contains a range where the root is expected
      theta.append(result.root)
    else:
      result = root_scalar(lambda x: right_I(x) - I_val_i, bracket=[argmax_I, sup_X])  # Bracket contains a range where the root is expected
      theta.append(result.root)

  theta.append(argmax_I)
  I_val.append(max_I)
  return theta, I_val

def get_pi_hat(theta,n,pi,plot_hist):
    pi_hat = []
    for index, theta_t in enumerate(theta):
      ############### SAMPLE from X at time t ###############
      # Sample n points from the custom PDF
      n_t = n[index]
      pi_t = pi[index]
      data = sample_from_pdf(n_t, lambda x: f_X(x,pi_t), inf_X, sup_X)
    
      ################ Fit classifier ##########################
      x_0 = sample_from_pdf(100, lambda x: f_u(x), inf_X, sup_X)
      x_1 = sample_from_pdf(100, lambda x: f_q(x), inf_X, sup_X)
      x = np.concatenate([x_0, x_1], axis = 0)
      x = x.reshape((-1, 1))
      y = np.array([0] * 100 + [1] * 100)
      classifier = LogisticRegression().fit(x, y)
    
      ############### Estimate pi with Lipton's method ###############
      mu_0 = np.mean(classifier.predict(x_0.reshape((-1, 1))))
      mu_1 = np.mean(classifier.predict(x_1.reshape((-1, 1))))
      estimated_p = (np.mean(classifier.predict(data.reshape((-1, 1)))) - mu_0) / (mu_1 - mu_0)
      estimated_p = np.clip(estimated_p, 0, 1)
      
      pi_hat.append(estimated_p)
    return pi_hat


def get_pi_hat_v2(theta,n,pi,plot_hist):
    pi_hat = []
    for index, theta_t in enumerate(theta):
      ############### SAMPLE from X at time t ###############
      # Sample n points from the custom PDF
      n_t = n[index]
      pi_t = pi[index]
      data = sample_from_pdf(n_t, lambda x: f_X(x,pi_t), inf_X, sup_X)
    
      # Plot the histogram of the generated samples
      if plot_hist:
          plt.hist(data, bins=30, density=True, alpha=0.5, label='Sampled Data')
          x = np.linspace(inf_X, sup_X, 1000)
          plt.plot(x, f_X(x,pi_t), 'r', linewidth=2, label='True PDF')
          plt.legend()
          plt.xlabel('x')
          plt.ylabel('PDF')
          plt.title('Sampling from a Bimodal Distribution')
          plt.show()
    
      ############### Estimate pi with MLE ###############
      initial_p_guess = 0.3 # Initial guess for the transformed parameter
      initial_param_guess = np.log(initial_p_guess / (1 - initial_p_guess))
      result = optimize.minimize(neg_log_likelihood, [initial_param_guess], args=(data,))# Perform unconstrained optimization to find the MLE of the transformed parameter
      estimated_param = result.x[0]
      estimated_p = 1 / (1 + np.exp(-estimated_param))# Transform the estimated parameter back to (0, 1)
      #print(f"The difference p-p_hat is: {pi_t-estimated_p}")
      pi_hat.append(estimated_p)
    return pi_hat
    
def get_cdf(name):
    def cdf1(x):
        return 1 / (1 + np.exp(-x/wage * 10 + 5))

    def cdf2(x):
        return gaussian_cdf(x/wage, 0.5, 0.15) 

    def cdf3(x):
        m = 0.5
        b = 0.2
        return 0.5 * (1-np.exp(-np.abs(x/wage - m) / b)) * np.sign(x/wage - m) +0.5

    def cdf4(x):
        alpha = 3
        beta_value = 8
        beta_dist = sp_beta(alpha, beta_value)
        cdf_value = beta_dist.cdf(x/wage)
        return cdf_value

    def cdf5(x):
        alpha = 3
        beta_value = 3
        beta_dist = sp_beta(alpha, beta_value)
        cdf_value = beta_dist.cdf(x/wage)
        return cdf_value

    def cdf6(x):
        alpha = 4
        beta_value = 3
        beta_dist = sp_beta(alpha, beta_value)
        cdf_value = beta_dist.cdf(x/wage)
        return cdf_value

    def cdf7(x):
        if x/wage <= 0:
            return 0
        elif 0 < x/wage <= 0.3:
            return 0.2
        elif 0.3 < x/wage <= 0.6:
            return 0.5
        elif 0.6 < x/wage <= 1:
            return 0.8
        elif x > 1:
            return 1

    def cdf8(x):
        return gaussian_cdf(x/wage, 0.5, 0.2)

    cdf_mapping = {
        'Logistic': cdf1,
        'Gaussian': cdf2,
        'Laplace': cdf3,
        'Beta(3,8)': cdf4,
        'Beta(3,3)': cdf5,
        'Beta(4,3)': cdf6,
        'Jump': cdf7,
        'Gaussian_cost': cdf8,
    }

    if name in cdf_mapping:
        return cdf_mapping[name]
    else:
        print(f"Invalid name: {name}. Please provide a valid name.")
        return None