
import numpy as np
import pandas as pd
import sys

def main():

    def GradientDescent(alpha, X, y, means, stds, df, output, iterations = 100):
        n = len(X)
        beta = np.zeros(3)
        loss_list = np.zeros(iterations)
    
        # Iterate: update beta values and compute loss
        for i in range(iterations):
            beta = beta - alpha*(1.0/n)*np.transpose(X).dot(X.dot(beta)-y)
            loss_list[i] = (1.0/2*n) * np.sum(np.square(X.dot(beta)-y))
    
        # De-normalize weights
        weights = np.zeros(3)
        weights[1] = beta[1] / stds[0]
        weights[2] = beta[2] / stds[1]
        weights[0] = beta[0] - (weights[1] * means[0]) - (weights[2] * means[1])
        
        """
        # Plot model
        xlim = (min(df[0]), max(df[0]))
        ylim = (min(df[1]), max(df[1]))
        zlim = (min(df[2]), max(df[2]))
        visualize_3d(df, lin_reg_weights = weights, alpha = alpha, 
                     xlim = xlim, ylim = ylim, zlim = zlim)
    
        # Plot loss versus iterations
        plt.plot(loss_list)
        plt.title(f'Convergence for alpha = {alpha:.3f}')
        plt.xlabel('Iterations')
        plt.ylabel('MSE loss')
        plt.show()
        """
    
        # File content
        output.append([alpha, iterations, f"{weights[0]:0.8f}", f"{weights[1]:0.8f}", f"{weights[2]:0.8f}"])
    
        # Done
        return weights
    
    # Alpha values
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    
    """
    Custom alpha and iterations
    When alpha is too high it can skip the minima for convergence, but if alpha is too low then it will
    converge too slowly. We want to find a balance of the alpha and the number of iterations in order to 
    converge on the minima with the lowest error rates. 
    """
    
    my_alpha = 0.6
    my_iterations = 30
    
    # Import data and get output file
    df = pd.read_csv(sys.argv[1], header=None)
    output_file = str(sys.argv[2]) 
    
    # Convert to numpy
    data = df.to_numpy()
    X = data[:, [0, 1]]
    y = data[:, 2]
    
    # Normalize features
    # Track the means and std devs to de-normalize later when plotting
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    for i in range(2):
      X[:, i] = (X[:, i] - means[i]) / stds[i]
    
    # Add a column of ones at the start
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Process the pre-set alpha values
    output = []
    for alpha in alphas:
        GradientDescent(alpha, X, y, means, stds, df, output)
        
    # Process the custom alpha value and iterations count
    GradientDescent(my_alpha, X, y, means, stds, df, output, my_iterations)
    
    # Write output CSV file
    pd.DataFrame(output).to_csv(output_file, header=False, index=False)

if __name__ == "__main__":
    main()