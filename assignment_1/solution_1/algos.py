from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy import linalg as lp 
import sys

def fx_iteration_plot(x, y, plot_name, title):
    plt.plot(x, y)
    plt.title("f(x) vs iterations | " + title)
    plt.xlabel("iterations")
    plt.ylabel("f(x)")
    plt.savefig(plot_name)
    plt.close()

def f2x_iteration_plot(x, y, plot_name, title):
    plt.plot(x, y)
    plt.title("d_f(x) vs iterations | " + title)
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


# Do not rename or delete this function
def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking", "Bisection"],
) -> npt.NDArray[np.float64]:
    # Complete this function

    if condition == "Backtracking":
        # Initializations
        x_k = inital_point
        x1 = np.array([])
        x2 = np.array([])
        x1 = x_k[0].astype(np.float64)
        x2 = x_k[1].astype(np.float64)
        x_k_array = np.array([])
        f_k_array = np.array([])
        d_f_k_array = np.array([])
        k_array = np.array([])
        step_size_0 = 10.0 # take a large step size
        rho = 0.75
        c1 = 0.001
        k = 0 # iterations
        epsilon = 10**-6

        x_k_array = np.append(x_k_array, x_k)
        f_k_array = np.append(f_k_array, f(x_k))
        d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
        k_array = np.append(k_array, k)
        
        counter = 0
        while ((k <= 10**4 and lp.norm(d_f(x_k)) > epsilon) and counter<=50): # stopping condition
            step_size = step_size_0
            while (f(x_k - step_size * d_f(x_k)) > (f(x_k) - c1 * step_size * ((lp.norm(d_f(x_k)))**2))): # while armijo condition not satisfied
                step_size = rho * step_size

            x_k = x_k - step_size * d_f(x_k)
            x1 = np.append(x1, x_k[0])
            x2 = np.append(x2, x_k[1])
            k = k + 1

            x_k_array = np.append(x_k_array, x_k)
            f_k_array = np.append(f_k_array, f(x_k))
            d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
            k_array = np.append(k_array, k)

            if (f_k_array[-1] - f_k_array[-2]) >= 0:
                print("Non-convergent, incrementing counter:", f_k_array[-1] - f_k_array[-2])
                counter += 1

        # Plot: f(x) vs iterations
        fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        # Plot: |f'(x)| vs iterations
        f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")

        # print("")
        # print(f"Backtracking: {f.__name__} | final x_k: {x_k}")
        # print("")
        return x_k

    elif condition == "Bisection":
    
        # Initializations
        x_k = inital_point
        x1 = np.array([])
        x2 = np.array([])
        x1 = x_k[0].astype(np.float64)
        x2 = x_k[1].astype(np.float64)
        x_k_array = np.array([])
        f_k_array = np.array([])
        d_f_k_array = np.array([])
        k_array = np.array([])
        c2 = 0.1
        alpha_0 = 0
        t = 1.0
        beta_0 = 10**6
        c1 = 0.001
        k = 0 # iterations
        epsilon = 10**-6
        f_k_array = np.array([])

        x_k_array = np.append(x_k_array, x_k)
        f_k_array = np.append(f_k_array, f(x_k))
        d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
        k_array = np.append(k_array, k)

        counter = 0
        while ((k <= 10**4 and lp.norm(d_f(x_k)) > epsilon) and counter<=50): # stopping condition
            # print(f"Intermediate Bisection: {f.__name__} | K={k} AND lp.norm(d_f(x_k))={lp.norm(d_f(x_k))}")
            alpha, beta = alpha_0, beta_0
            
            while 1:
                if (f(x_k - t * d_f(x_k)) > (f(x_k) - c1 * t * (lp.norm(d_f(x_k)) ** 2))):
                    # print(f"Intermediate Bisection: {f.__name__} | while 1 | IF LHS={f(x_k - t * d_f(x_k))} | IF RHS: {f(x_k) - c1 * t * (lp.norm(d_f(x_k)) ** 2)}")
                    beta, t = t, 1/2 * (alpha + beta) # reset
                elif (-(d_f(x_k - t * d_f(x_k)).T @ d_f(x_k)) < (c2 * lp.norm(d_f(x_k)) ** 2)):
                    alpha, t = t, 1/2 * (alpha + beta) # reset
                    # print(f"Intermediate Bisection: {f.__name__} | while 1 | ELIF LHS={-(d_f(x_k - t * d_f(x_k)).T @ d_f(x_k))} | IF RHS: {(c2 * lp.norm(d_f(x_k)) ** 2)} | alpha={alpha} | beta={beta} | t={t}")
                else:
                    break
        
            x_k = x_k - t * d_f(x_k)
            x1 = np.append(x1, x_k[0])
            x2 = np.append(x2, x_k[1])
            k = k + 1

            x_k_array = np.append(x_k_array, x_k)
            f_k_array = np.append(f_k_array, f(x_k))
            d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
            k_array = np.append(k_array, k)

            if (f_k_array[-1] - f_k_array[-2]) >= 0:
                print("Non-convergent, incrementing counter:", f_k_array[-1] - f_k_array[-2])
                counter += 1
        
        # Plot: f(x) vs iterations
        fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        # Plot: |f'(x)| vs iterations
        f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        # print(f"Bisection: {f.__name__} | final x_k: {x_k}")
        # print("")
        return x_k
    
    else:
        pass

# Do not rename or delete this function
def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:
    # Complete this function   

    if condition == "Pure":
        # Initializations
        k = 0
        epsilon = 10**-6
        x_k = inital_point
        x1 = np.array([])
        x2 = np.array([])
        x1 = x_k[0].astype(np.float64)
        x2 = x_k[1].astype(np.float64)
        x_k_array = np.array([])
        f_k_array = np.array([])
        d_f_k_array = np.array([])
        k_array = np.array([])

        x_k_array = np.append(x_k_array, x_k)
        f_k_array = np.append(f_k_array, f(x_k))
        d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
        k_array = np.append(k_array, k)
        
        counter=0
        while ((k <= 10**4 and lp.norm(d_f(x_k)) > epsilon) and counter<=50): # stopping condition
            det = np.linalg.det(d2_f(x_k)) # calculating the determinant of matrix
            if det:
                h_inverse = np.linalg.inv(d2_f(x_k))
            else:
                print(f"Pure Newton: Inverse of the function {d2_f.__name__} doesn't exist!")
                break

            if det <= 10**-6:
                print("Determinant is very small, hence exiting the code.")
                break

            d_k = - np.matmul(h_inverse, d_f(x_k)) # Newton direction
            x_k = x_k + d_k
            x1 = np.append(x1, x_k[0])
            x2 = np.append(x2, x_k[1])
            k = k + 1

            x_k_array = np.append(x_k_array, x_k)
            f_k_array = np.append(f_k_array, f(x_k))
            d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
            k_array = np.append(k_array, k)

            if (f_k_array[-1] - f_k_array[-2]) >= 0:
                print("Non-convergent, incrementing counter:", f_k_array[-1] - f_k_array[-2])
                counter += 1

        # Plot: f(x) vs iterations
        fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        # Plot: |f'(x)| vs iterations
        f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")

        # print(f"Pure Newton: {f.__name__}: final x_k: {x_k}")
        # print("")
        return x_k

    elif condition == "Damped":

        k = 0
        epsilon = 10**-6
        x_k = inital_point
        x1 = np.array([])
        x2 = np.array([])
        x1 = x_k[0].astype(np.float64)
        x2 = x_k[1].astype(np.float64)
        x_k_array = np.array([])
        f_k_array = np.array([])
        d_f_k_array = np.array([])
        k_array = np.array([])

        # params for backtracking procedure
        alpha = 0.001
        beta = 0.75 
        t_k = 1

        x_k_array = np.append(x_k_array, x_k)
        f_k_array = np.append(f_k_array, f(x_k))
        d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
        k_array = np.append(k_array, k)

        counter = 0
        while ((k <= 10**4 and lp.norm(d_f(x_k)) > epsilon) and counter<=50): # stopping condition
            det = np.linalg.det(d2_f(x_k)) # calculating the determinant of matrix
            if det:
                h_inverse = np.linalg.inv(d2_f(x_k))
            else:
                print(f"Damped Newton: Inverse of the function {d2_f.__name__} doesn't exist!")
                break

            if det <= 10**-6:
                print("Determinant is very small, hence exiting the code.")
                break
            
            d_k = - np.matmul(h_inverse, d_f(x_k))

            t_k = 1
            while ((f(x_k) - f(x_k + t_k * d_k)) < (- alpha * t_k * (d_f(x_k).T @ d_k))):
                t_k = beta * t_k

            x_k = x_k + (t_k * d_k)
            x1 = np.append(x1, x_k[0])
            x2 = np.append(x2, x_k[1])
            k = k + 1

            x_k_array = np.append(x_k_array, x_k)
            f_k_array = np.append(f_k_array, f(x_k))
            d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
            k_array = np.append(k_array, k)

            if (f_k_array[-1] - f_k_array[-2]) >= 0:
                print("Non-convergent, incrementing counter:", f_k_array[-1] - f_k_array[-2])
                counter += 1

        # Plot: f(x) vs iterations
        fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        # Plot: |f'(x)| vs iterations
        f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")

        # print(f"Damped Newton: {f.__name__}: final x_k: {x_k}")
        # print("")
        return x_k

    elif condition == "Levenberg-Marquardt":

        k = 0
        epsilon = 10**-6
        x_k = inital_point
        x1 = np.array([])
        x2 = np.array([])
        x1 = x_k[0].astype(np.float64)
        x2 = x_k[1].astype(np.float64)
        x_k_array = np.array([])
        f_k_array = np.array([])
        d_f_k_array = np.array([])
        k_array = np.array([])

        x_k_array = np.append(x_k_array, x_k)
        f_k_array = np.append(f_k_array, f(x_k))
        d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
        k_array = np.append(k_array, k)

        counter = 0
        while ((k <= 10**4 and lp.norm(d_f(x_k)) > epsilon) and counter<=20): # stopping condition
            det = np.linalg.det(d2_f(x_k)) # calculating the determinant of matrix
            if det:
                h_inverse = np.linalg.inv(d2_f(x_k))
            else:
                print(f"Levenberg: Inverse of the function {d2_f.__name__} doesn't exist!")
                break

            if det <= 10**-6:
                print("Determinant is very small, hence exiting the code.")
                break
            
            EVA, _ = lp.eigh(d2_f(x_k)) #calculating the eigenvalues and eigenvectors
    
            lambda_min = min(EVA)
            if (lambda_min <= 0):
                mu = - lambda_min + 0.1
                I = np.identity(d2_f(x_k).shape[0], dtype = float)
                h_mu_inverse = lp.inv(d2_f(x_k) + np.dot(mu, I))
                d_k = - (np.matmul(h_mu_inverse, d_f(x_k)))
            else:
                h_inverse = lp.inv(d2_f(x_k))
                d_k = - np.matmul(h_inverse, d_f(x_k))

            x_k = x_k + d_k
            x1 = np.append(x1, x_k[0])
            x2 = np.append(x2, x_k[1])
            k = k + 1

            x_k_array = np.append(x_k_array, x_k)
            f_k_array = np.append(f_k_array, f(x_k))
            d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
            k_array = np.append(k_array, k)

            if (f_k_array[-1] - f_k_array[-2]) >= 0:
                print("Non-convergent, incrementing counter:", f_k_array[-1] - f_k_array[-2])
                counter += 1

        # Plot: f(x) vs iterations
        fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        # Plot: |f'(x)| vs iterations
        f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")

        # print(f"Levenberg: {f.__name__}: final x_k: {x_k}")
        return x_k

    # elif condition == "Combined":

    #     k = 0
    #     epsilon = 10**-6
    #     x_k = inital_point
    #     x1 = np.array([])
    #     x2 = np.array([])
    #     x1 = x_k[0].astype(np.float64)
    #     x2 = x_k[1].astype(np.float64)
    #     x_k_array = np.array([])
    #     f_k_array = np.array([])
    #     d_f_k_array = np.array([])
    #     k_array = np.array([])

    #     step_size_0 = 1.0 # same as Newton
    #     rho = 0.75
    #     c1 = 0.001

    #     x_k_array = np.append(x_k_array, x_k)
    #     f_k_array = np.append(f_k_array, f(x_k))
    #     d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
    #     k_array = np.append(k_array, k)

    #     counter = 0
    #     while ((k <= 10**4 and lp.norm(d_f(x_k)) > epsilon) and counter<=20): # stopping condition

    #         det = np.linalg.det(d2_f(x_k)) # calculating the determinant of matrix
    #         if det:
    #             h_inverse = np.linalg.inv(d2_f(x_k))
    #         else:
    #             print(f"Combined: Inverse of the function {d2_f.__name__} doesn't exist!")
    #             break
            
    #         if det <= 10**-6:
    #             print("Determinant is very small, hence exiting the code.")
    #             break
           
    #         EVA = lp.eigvals(d2_f(x_k))  #calculating the eigenvalues and eigenvectors
            
    #         lambda_min = min(EVA)

    #         if (lambda_min <= 0):
    #             mu = - lambda_min + 0.1
    #             I = np.identity(d2_f(x_k).shape[0], dtype = float)
    #             h_mu_inverse = np.linalg.inv(d2_f(x_k) + mu*I)
    #             d_k = - np.dot(h_mu_inverse, d_f(x_k))
    #         else:
    #             d_k = - np.dot(h_inverse, d_f(x_k))


    #         step_size = step_size_0
    #         # Calculate alpha_k using Backtracking with armijo condition with descent direction as d_k
    #         while (f(x_k - step_size * d_f(x_k)) > f(x_k) - c1 * step_size * ((lp.norm(d_f(x_k)))**2)): # while armijo condition not satisfied
    #             step_size = rho * step_size
            
    #         x_k = x_k + step_size * d_k
    #         x1 = np.append(x1, x_k[0])
    #         x2 = np.append(x2, x_k[1])
    #         k = k + 1
            

    #         x_k_array = np.append(x_k_array, x_k)
    #         f_k_array = np.append(f_k_array, f(x_k))
    #         d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
    #         k_array = np.append(k_array, k)

    #         print("********************************")
    #         print(f"x_k: {x_k}")
    #         print(f"f(x_k): {f(x_k)}")
    #         print(f"d_f(x_k): {d_f(x_k)}")
    #         print(f"d2_f(x_k): {d2_f(x_k)}")
    #         print(f"step_size: {step_size}")
    #         print(f"d_k: {d_k}")
    #         print("********************************")


    #         if (f_k_array[-1] - f_k_array[-2]) >= 0:
    #             print("Non-convergent, incrementing counter:", f_k_array[-1] - f_k_array[-2])
    #             counter += 1


    #     # Plot: f(x) vs iterations
    #     # fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.png", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
    #     # Plot: |f'(x)| vs iterations
    #     # f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.png", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
    #     # contour_plot(x1, x2, inital_point, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.png", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")

    #     # print(f"Combined: {f.__name__}: final x_k: {x_k}")
    #     return x_k

    elif condition == "Combined":

        k = 0
        epsilon = 10**-6
        x_k = inital_point
        x1 = np.array([])
        x2 = np.array([])
        x1 = x_k[0].astype(np.float64)
        x2 = x_k[1].astype(np.float64)
        x_k_array = np.array([])
        f_k_array = np.array([])
        d_f_k_array = np.array([])
        k_array = np.array([])

        step_size_0 = 1.0 # same as Newton
        rho = 0.75
        c1 = 0.001

        x_k_array = np.append(x_k_array, x_k)
        f_k_array = np.append(f_k_array, f(x_k))
        d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
        k_array = np.append(k_array, k)

        counter = 0
        while ((k <= 10**4 and lp.norm(d_f(x_k)) > epsilon) and counter<=50): # stopping condition

            hess = d2_f(x_k)
            lambda_min = np.min(lp.eigvals(hess))  #calculating the min eigenvalues
            
            if (lambda_min <= 0):
                mu = - lambda_min + 0.1
                I = np.identity(x_k.shape[0], dtype = float)
                hess += mu*I

            det = np.linalg.det(hess) # calculating the determinant of matrix
                
            if det <= 10**-6:
                print("Determinant is very small, hence exiting the code.")
                break

            d_k = - np.dot(lp.inv(hess), d_f(x_k))

            step_size = 1
            # Calculate alpha_k using Backtracking with armijo condition with descent direction as d_k
            while ((f(x_k) - f(x_k + step_size * d_k)) < - (c1 * step_size * np.dot(d_f(x_k).T, d_k))): # while armijo condition not satisfied
                step_size = rho * step_size

            x_k = x_k + step_size * d_k
            x1 = np.append(x1, x_k[0])
            x2 = np.append(x2, x_k[1])
            k = k + 1            
            
            x_k_array = np.append(x_k_array, x_k)
            f_k_array = np.append(f_k_array, f(x_k))
            d_f_k_array = np.append(d_f_k_array, lp.norm(d_f(x_k)))
            k_array = np.append(k_array, k)

            # print("********************************")
            # print(f"x_k: {x_k}")
            # print(f"f(x_k): {f(x_k)}")
            # print(f"d_f(x_k): {d_f(x_k)}")
            # print(f"d2_f(x_k): {d2_f(x_k)}")
            # print(f"step_size: {step_size}")
            # print(f"d_k: {d_k}")
            # print("********************************")


            if (f_k_array[-1] - f_k_array[-2]) >= 0:
                print("Non-convergent, incrementing counter:", f_k_array[-1] - f_k_array[-2])
                counter += 1


        # Plot: f(x) vs iterations
        fx_iteration_plot(k_array, f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        # Plot: |f'(x)| vs iterations
        f2x_iteration_plot(k_array, d_f_k_array, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")
        
        contour_plot(x1, x2, f, f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.jpg", f"{f.__name__}_{np.array2string(inital_point)}_{condition}")

        # print(f"Combined: {f.__name__}: final x_k: {x_k}")
        return x_k
    
    else:
        pass