from typing import Callable, Literal

import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np

def projection(new_point, constraints, constraint_type):
    if(constraint_type == 'l_2'):
        centre, rad = constraints
        centre = np.array(centre)
        return ((new_point - centre) * rad) / (max(rad, np.linalg.norm(new_point - centre))) + centre
    elif(constraint_type == 'linear'):
        lb, ub = constraints
        return np.min([np.max([new_point, lb], 0), ub], 0)
    else:
        return None
    
def projected_gd(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    point: npt.NDArray[np.float64],
    constraint_type: Literal["linear", "l_2"],
    constraints: npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], np.float64]
) -> npt.NDArray[np.float64]:

    k = 0
    x_k = point.copy()
    initial_guess_s = 1
    alpha_ = 0.001
    beta = 0.9
    
    while k < 1000: # terminate after 1000 iterations

        t_k = initial_guess_s
        grad_fx = d_f(x_k)
        x_k_proj = projection(x_k - t_k * grad_fx, constraints, constraint_type)
        grad_map = (x_k - x_k_proj) / t_k

        while(f(x_k) - f(x_k_proj) < alpha_ * t_k * np.linalg.norm(grad_map)**2): # backtracking stopping condition given in PDF 
            t_k = beta * t_k
            x_k_proj = projection(x_k - t_k * grad_fx , constraints, constraint_type)
            grad_map = (x_k - x_k_proj)/t_k
        
        x_k = x_k_proj
        k = k + 1

    return x_k


def dual_ascent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    c: list[Callable[[npt.NDArray[np.float64]], np.float64 | float]],
    d_c: list[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    initial_point: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    alpha = 10**-3
    k = 0
    x_k = initial_point

    lambda_list = []
    # initializing lambdas with value "1"

    for each_constraint in range(len(c)):
        lambda_list.append(1.0)
    lambda_arr_k = np.asarray(lambda_list) # if 4 constraint equations are there, then lambda_arr_k = [1 1 1 1]


    while k < 100000:
        # print(f"lambda_arr_k: {lambda_arr_k}")
        deriv_lag_wrt_x = d_f(x_k)
        # print(f"d_f({x_k}): {d_f(x_k)}")
        
        for index in range(len(c)):
            # print(f"d_c[{index}]({x_k}): {d_c[index](x_k)}")
            scalar = lambda_arr_k[index]
            # print(f"scalar: {scalar}")
            deriv_lag_wrt_x = deriv_lag_wrt_x + (d_c[index](x_k) * scalar)

        # print(f"deriv_lag_wrt_x: {deriv_lag_wrt_x}")
        x_k = x_k - alpha * deriv_lag_wrt_x
        # print(f"x_k: {x_k}")

        for index in range(len(c)):
            # print(f"c[{index}]({x_k}): {c[index](x_k)}")
            deriv_lag_wrt_lambda = c[index](x_k)
            # print(f"deriv_lag_wrt_lambda: {deriv_lag_wrt_lambda}")
            lambda_arr_k[index] = lambda_arr_k[index] + alpha * deriv_lag_wrt_lambda
            lambda_arr_k[index] = max(0, lambda_arr_k[index])

        # print(f"lambda_arr_k: {lambda_arr_k}")

        k = k + 1

    # print("------------------------------------------------------------------------------------------")
    # print(f"For {f.__name__} with {initial_point} as starting point using Dual Ascent:")
    # print(f"Lambda value λ*: {lambda_arr_k}")
    # print(f"Optimal Value x*: {x_k}")
    # print("Verifying if obtained optimal value satisfies 1st Order KKT conditions:")
    # print(f"\t Grad of Lagrange function wrt optimal value --> {deriv_lag_wrt_x}")
    # print(f"\t Norm of Grad of Lagrange function wrt optimal value --> {np.linalg.norm(deriv_lag_wrt_x)}")
    # print("------------------------------------------------------------------------------------------")
    # print()

    final_kkt_point = (x_k, lambda_arr_k)
    # print(f"final_kkt_point: {final_kkt_point}")

    return final_kkt_point
        

