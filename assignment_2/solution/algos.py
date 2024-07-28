from typing import Callable, Literal

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np
import sys

'Bisection Algo for Inexact Line Search'
def bisec(inital_point, f, d_f, d_k):
    # Initializations
    c1 = 0.001
    c2 = 0.1
    alpha_0 = 0
    t = 1.0
    beta_0 = 10**6
    k = 0 # iterations

    x_k = inital_point

    alpha, beta = alpha_0, beta_0
    while (k <= 1000): # stopping condition
        if (f(x_k + t * d_k) > (f(x_k) + c1 * t * np.dot(d_f(x_k).T, d_k))):
        # print(f"Intermediate Bisection: {f.__name__} | K={k} AND lp.norm(d_f(x_k))={lp.norm(d_f(x_k))}")
            beta, t = t, 1/2 * (alpha + beta) # reset        

        elif ((d_f(x_k + t * d_k).T @ d_k) < (c2 * np.dot(d_f(x_k).T, d_k))):
            alpha, t = t, 1/2 * (alpha + beta) # reset
            # print(f"Intermediate Bisection: {f.__name__} | while 1 | ELIF LHS={-(d_f(x_k - t * d_f(x_k)).T @ d_f(x_k))} | IF RHS: {(c2 * lp.norm(d_f(x_k)) ** 2)} | alpha={alpha} | beta={beta} | t={t}")
        else:
            break

        k = k + 1

    # print(f"Bisection: {f.__name__} | final x_k: {x_k}")
    # print("")
    return t

def fx_iteration_plot(x, y, plot_name, title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("iterations")
    plt.ylabel("f(x)")
    plt.savefig(plot_name)
    plt.close()

def f2x_iteration_plot(x, y, plot_name, title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("iterations")
    plt.ylabel("|d_f(x)|")
    plt.savefig(plot_name)
    plt.close()

def contour_plot(x1, x2, f, plot_name, title):
    fig, ax = plt.subplots()

    x = np.arange(-4, 4, 0.1)
    y = np.arange(-4, 4, 0.1)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    ax.contour(X, Y, Z, 100)
    ax.plot(x1, x2, 'o-')
    for i in range(1, len(x1)):
        ax.annotate('', xy=(x1[i], x2[i]), xytext=(x1[i-1], x2[i-1]),
                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                    va='center', ha='center')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    plt.grid(True)
    # plt.show()
    plt.savefig(plot_name)
    plt.close()

def conjugate_descent(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:
    
    k = 0
    counter = 0
    x_old = inital_point
    x1 = np.array([])
    x2 = np.array([])
    x1 = x_old[0].astype(np.float64)
    x2 = x_old[1].astype(np.float64)
    k_array = np.array([])
    k_array = np.append(k_array, counter)
    f_k_array = np.array([])
    f_k_array = np.append(f_k_array, f(x_old))
    d_f_k_array = np.array([])
    d_f_k_array = np.append(d_f_k_array, np.linalg.norm(d_f(x_old)))

    epsilon = 10**-6  # IS THIS AN ACCURATE ASSUMPTION?? In slide it says, epsilon>0
    d_k = - d_f(x_old) # Initialize the descent direction
    
    n = d_f(x_old).shape[0]

    while ((np.linalg.norm(d_f(x_old)) > epsilon)):
        alpha_k = bisec(x_old, f, d_f, d_k)
        x_new = x_old + alpha_k * d_k
        if (k < (n-1)):
            if approach == "Fletcher-Reeves":
                beta_k = np.linalg.norm(d_f(x_new))**2/np.linalg.norm(d_f(x_old))**2
                # beta_k = (d_f(x_new).T @ d_f(x_new))/(d_f(x_old).T @ d_f(x_old))
            elif approach == "Polak-Ribiere":
                beta_k = np.dot(d_f(x_new).T, (d_f(x_new)-d_f(x_old)))/np.linalg.norm(d_f(x_old))**2 
                # beta_k = max(0, beta_k)
                # beta_k = (d_f(x_new).T @ (d_f(x_new) - d_f(x_old)))/(d_f(x_old).T @ d_f(x_old))
            elif "Hestenes-Stiefel":
                beta_k = np.dot(d_f(x_new), d_f(x_new) - d_f(x_old))/np.dot(d_k, d_f(x_new) - d_f(x_old))
                # beta_k = (d_f(x_new).T @ (d_f(x_new) - d_f(x_old)))/(d_k.T @ ((d_f(x_new) - d_f(x_old))))

            d_k = - d_f(x_new) + beta_k * d_k # Newly updated direction
            # print("--------")
            # print(f"At k = {k}, For approach {approach}; function {f.__name__}, with initial point {inital_point} has -> d_k: {d_k}, beta_k: {beta_k}, function update: {f(x_new)-f(x_old)}")
            x_old = x_new
            x1 = np.append(x1, x_old[0])
            x2 = np.append(x2, x_old[1])
            k = k + 1
            counter= counter + 1

            k_array = np.append(k_array, counter)
            f_k_array = np.append(f_k_array, f(x_old))
            d_f_k_array = np.append(d_f_k_array, np.linalg.norm(d_f(x_old)))

        else:
            x_old = x_new
            d_k = - d_f(x_old)
            k = 0

    # Plot: f(x) vs iterations
    fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{approach}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{approach}")
    
    # Plot: |f'(x)| vs iterations
    f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{approach}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{approach}")
    
    n = d_f(x_old).shape[0]
    if (n<=2):
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{approach}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{approach}")
    
    return x_old

def sr1(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    k = 0
    x_old = inital_point
    x1 = np.array([])
    x2 = np.array([])
    x1 = x_old[0].astype(np.float64)
    x2 = x_old[1].astype(np.float64)
    k_array = np.array([])
    k_array = np.append(k_array, k)
    f_k_array = np.array([])
    f_k_array = np.append(f_k_array, f(x_old))
    d_f_k_array = np.array([])
    d_f_k_array = np.append(d_f_k_array, np.linalg.norm(d_f(x_old)))
    
    # I = np.identity(d_f(x_old).shape[0], dtype = float)
    # B_k = I # Choosing I as symmetric positive definite matrix (as in book)
    B_k = np.eye(d_f(x_old).shape[0]) # Choosing I as symmetric positive definite matrix (as in book)
    epsilon = 10**-6  # IS THIS AN ACCURATE ASSUMPTION?? In slide it says, epsilon>0
    
    while (np.linalg.norm(d_f(x_old)) > epsilon):
        d_k = - (np.dot(B_k, d_f(x_old)))  # Selection of the direction
        alpha_k = bisec(x_old, f, d_f, d_k)
        
        # step_size = line_search(f=f, myfprime=d_f, xk=x_old, pk=d_k, c1=c1, c2=c2)[0] # Selecting the step length
        # if step_size!=None:
        #     x_new = x_old + step_size * d_k

        # # find alpha_k / step_size
        # step_size = 1 # assignment 1 inspired
        # # Slides use Arminjo with Wolfe, but we calculate alpha_k using Backtracking with armijo condition
        # while ((f(x_old) - f(x_old + step_size * d_k)) < - (c1 * step_size * np.dot(d_f(x_old).T, d_k))): # while armijo condition not satisfied
        #     step_size = rho * step_size
        x_new = x_old + alpha_k * d_k

        # Find new B_k
        gamma = d_f(x_new) - d_f(x_old)
        delta = x_new - x_old
        w = delta - np.dot(B_k, gamma)
        wT = w.T
        sigma = 1/np.dot(wT, gamma)
        den = np.dot(wT, gamma)
        W = np.outer(w, w) # Outer product between w and the transpose of w
        rest = sigma*W

        # if abs(np.dot(wT, gamma)) >= 10**-8*np.linalg.norm(gamma)*np.linalg.norm(w): # update criterion # making sure the denominator is not approaching zero
        # if np.dot(wT, gamma) >= 10**-6: # update criterion # making sure the denominator is not close to zero and not negative
        B_k = B_k + rest # update
        # print(f"k = {k}")
        # print(f" for function {f.__name__}, and initial point {inital_point}, Updated B_k in SR1 is: {B_k}")
        # print(f"Is Updated B_k PD: {np.all(np.linalg.eigvals(B_k) >= 0)} and denominator value of update formula is {den}")
        
        x_old = x_new

        x1 = np.append(x1, x_old[0])
        x2 = np.append(x2, x_old[1])
        k = k + 1

        k_array = np.append(k_array, k)
        f_k_array = np.append(f_k_array, f(x_old))
        d_f_k_array = np.append(d_f_k_array, np.linalg.norm(d_f(x_old)))

        # Are we supposed to check in SR1 method that the B_k 
        # obtained satisfies Quasi-Newton conditon?

    # Plot: f(x) vs iterations
    fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{'SR1'}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{'SR1'}")
    
    # Plot: |f'(x)| vs iterations
    f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{'SR1'}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{'SR1'}")
    
    n = d_f(x_old).shape[0]
    if (n<=2):
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{'SR1'}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{'SR1'}")
    
    return x_old

def dfp(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    k = 0
    x_old = inital_point
    x1 = np.array([])
    x2 = np.array([])
    x1 = x_old[0].astype(np.float64)
    x2 = x_old[1].astype(np.float64)
    k_array = np.array([])
    k_array = np.append(k_array, k)
    f_k_array = np.array([])
    f_k_array = np.append(f_k_array, f(x_old))
    d_f_k_array = np.array([])
    d_f_k_array = np.append(d_f_k_array, np.linalg.norm(d_f(x_old)))
    # I = np.identity(d_f(x_old).shape[0], dtype = float)
    # B_k = I # Choosing I as symmetric positive definite matrix (as in book)
    B_k = np.eye(d_f(x_old).shape[0]) # Choosing I as symmetric positive definite matrix (as in book)
    epsilon = 10**-6  # IS THIS AN ACCURATE ASSUMPTION?? In slide it says, epsilon>0
    
    while (np.linalg.norm(d_f(x_old)) > epsilon):
        d_k = - (np.dot(B_k, d_f(x_old)))  # Selection of the direction

        # step_size = line_search(f=f, myfprime=d_f, xk=x_old, pk=d_k, c1=c1, c2=c2)[0] # Selecting the step length
        # if step_size!=None:
        #     x_new = x_old + step_size * d_k
        
        # # find alpha_k / step_size
        # step_size = 1 # assignment 1 inspired
        # # Slides use Arminjo with Wolfe, but we Calculate alpha_k using Backtracking with armijo condition
        # while ((f(x_old) - f(x_old + step_size * d_k)) < - (c1 * step_size * np.dot(d_f(x_old).T, d_k))): # while armijo condition not satisfied
        #     step_size = rho * step_size
        # x_new = x_old + step_size * d_k

        alpha_k = bisec(x_old, f, d_f, d_k)
        x_new = x_old + alpha_k * d_k

        # Find new B_k
        gamma = d_f(x_new) - d_f(x_old)
        delta = x_new - x_old
        w1 = delta
        w2 = np.dot(B_k, gamma)
        w1T = w1.T
        w2T = w2.T
        sigma1 = 1/(np.dot(w1T, gamma))
        sigma2 = -1/(np.dot(w2T, gamma))
        W1 = np.outer(w1, w1)
        W2 = np.outer(w2, w2)
        rest = sigma1*W1 + sigma2*W2
        B_k = B_k + rest

        x_old = x_new
        x1 = np.append(x1, x_old[0])
        x2 = np.append(x2, x_old[1])
        k = k + 1

        k_array = np.append(k_array, k)
        f_k_array = np.append(f_k_array, f(x_old))
        d_f_k_array = np.append(d_f_k_array, np.linalg.norm(d_f(x_old)))

        # Are we supposed to check in SR1 method that the B_k 
        # obtained satisfies Quasi-Newton conditon?

    # Plot: f(x) vs iterations
    fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{'DFP'}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{'DFP'}")
    
    # Plot: |f'(x)| vs iterations
    f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{'DFP'}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{'DFP'}")
    
    n = d_f(x_old).shape[0]
    if (n<=2):
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{'DFP'}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{'DFP'}")

    return x_old

def bfgs(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    k = 0
    x_old = inital_point
    x1 = np.array([])
    x2 = np.array([])
    x1 = x_old[0].astype(np.float64)
    x2 = x_old[1].astype(np.float64)
    k_array = np.array([])
    k_array = np.append(k_array, k)
    f_k_array = np.array([])
    f_k_array = np.append(f_k_array, f(x_old))
    d_f_k_array = np.array([])
    d_f_k_array = np.append(d_f_k_array, np.linalg.norm(d_f(x_old)))
    # I = np.identity(d_f(x_old).shape[0], dtype = float)
    B_k = np.eye(d_f(x_old).shape[0]) # Choosing I as symmetric positive definite matrix (as in book)
    epsilon = 10**-6  # IS THIS AN ACCURATE ASSUMPTION?? In slide it says, epsilon>0
    
    while (np.linalg.norm(d_f(x_old)) > epsilon):
        d_k = - (np.dot(B_k, d_f(x_old)))  # Selection of the direction

        # step_size = line_search(f=f, myfprime=d_f, xk=x_old, pk=d_k, c1=c1, c2=c2)[0] # Selecting the step length
        # if step_size!=None:
        #     x_new = x_old + step_size * d_k
        
        # # find alpha_k / step_size
        # step_size = 1 # assignment 1 inspired
        # # Slides use Arminjo with Wolfe, but we calculate alpha_k using Backtracking with armijo condition
        # while ((f(x_old) - f(x_old + step_size * d_k)) < - (c1 * step_size * np.dot(d_f(x_old).T, d_k))): # while armijo condition not satisfied
        #     step_size = rho * step_size
        # x_new = x_old + step_size * d_k

        alpha_k = bisec(x_old, f, d_f, d_k)
        x_new = x_old + alpha_k * d_k

        # Find new B_k
        gamma = d_f(x_new) - d_f(x_old)
        delta = x_new - x_old
        
        den = np.dot(delta, gamma)
        num = np.dot(B_k, gamma)

        L = 1 + np.dot(num, gamma)/den
        M = np.outer(delta, delta)/den
        N = np.outer(delta, num)/den
        O = np.outer(num, delta)/den

        rest = L*M - N - O
        B_k = B_k + rest
        
        x_old = x_new
        x1 = np.append(x1, x_old[0])
        x2 = np.append(x2, x_old[1])
        k = k + 1

        k_array = np.append(k_array, k)
        f_k_array = np.append(f_k_array, f(x_old))
        d_f_k_array = np.append(d_f_k_array, np.linalg.norm(d_f(x_old)))

        # Are we supposed to check in SR1 method that the B_k 
        # obtained satisfies Quasi-Newton conditon?

    # Plot: f(x) vs iterations
    fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{'BFGS'}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{'BFGS'}")
    
    # Plot: |f'(x)| vs iterations
    f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{'BFGS'}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{'BFGS'}")
    
    n = d_f(x_old).shape[0]
    if (n<=2):
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{'BFGS'}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}\n{'BFGS'}")
    return x_old