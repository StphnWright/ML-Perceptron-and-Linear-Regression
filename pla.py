import pandas as pd
import numpy as np
import sys


def main():

    def perceptron(X, y, output_file):
        num_samples = X.shape[0]
        num_features = X.shape[1]
        output = []

        # Initialize weights and bias to zero
        w = np.zeros((num_features, 1))
        b = 0
    
        # Iterate until the model converges
        model_not_converged = True 
        while model_not_converged:
    
            # Assume convergence to start
            model_not_converged = False
    
            # Loop through each sample
            for i in range(num_samples):
                # Get the prediction
                y_pred = b + np.dot(X[[i],:], w)
                y_pred = 1 if y_pred > 0 else -1
    
                # Check if the true and predicted labels have different signs
                if y[i] * y_pred <= 0:
                    # Yes, update weights and bias, and flag model as not converged
                    w = w + y[i] * np.transpose(X[[i],:])
                    b = b + y[i]
                    model_not_converged = True
    
            # Record weights
            out_line = []
            for w_value in w:
                out_line.append(str(int(w_value[0])))
            out_line.append(str(int(b)))
            output.append(out_line)
        
        # Write output CSV file
        pd.DataFrame(output).to_csv(output_file, header=False, index=False)
        
        # Return combined weights vector
        weights = np.zeros((num_features + 1, 1))
        weights[:-1] = w
        weights[-1] = b
        return weights
      
    # Read data
    df = pd.read_csv(sys.argv[1], header=None)
    input = df.to_numpy()
    data = input[:, [0, 1]]
    bias = input[:, 2]
    
    # Output file
    output_file = str(sys.argv[2]) 
    weights = perceptron(data, bias, output_file)
    
    # Visualize
    # visualize_scatter(df, weights = weights)


if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()
