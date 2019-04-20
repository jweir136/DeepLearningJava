package DeepLearning;

public class Utils {
  /** Returns a 2D array of random values between 0 and 1
   *    to represent the weights between 2 layers in the 
   *    neural network.
   *  @param m
   *    An integer representing the number of nodes in the
   *      layer that the weights feed into.
   *  @param n
   *    An integer representing the number of nodes in the
   *      layer that the weights are being fed from.
   *  @return a 2D array representing the weights from one
   *    layer to another layer in a neural network.
   *  @precondition m and n are both integers greater than
   *    0.
  */
  public static double[][] initWeights(int m, int n) {
    double[][] weights = new double[m][n];
    
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        weights[i][j] = Math.random();
      }
    }

    return weights;
  }

  /** Returns the dot product of two 2D arrays.
   *  @param matrix1
   *    The first 2D matrix used in the dot product
   *      operation.
   *  @param matrix2
   *    The second 2D matrix used in the dot product
   *      operation.
   *  @return a 2D matrix that is the dot product of 
   *    the arguments matrix1 and matrix2.
   *  @precondition the length of matrix1 and matrix 2
   *    are both greater than 0.
  */
  public static double[][] dot(double[][] matrix1, double[][] matrix2) {
    double[][] resultantMatrix = new double[matrix1.length][matrix1[0].length];

    for (int i = 0; i < matrix1.length; i++) {
      for (int j = 0; j < matrix2[0].length; j++) {
        for (int k = 0; k < matrix1[0].length; k++) {
          resultantMatrix[i][j] = matrix1[i][k] * matrix2[j][k];
        }
      }
    }

    return resultantMatrix;
  }

  /** Subtracts a scalar value from each element of a 2D
   *    matrix.
   *  @param matrix1
   *    The 2D matrix which the scalar value is subtracted
   *      from.
   *  @param scalar
   *    The scalar value that is subtracted from the matrix
   *  @return a 2D matrix equal to matrix1 minus a scalar
   *    value.
   *  @precondition matrix1 has a length greater than 0.
  */
  public static double[][] subtract(double[][] matrix1, double scalar) {
    double[][] resulantMatrix = new double[matrix1.length][matrix1[0].length];

    for (int i = 0; i < matrix1.length; i++) {
      for (int j = 0; j < matrix1[0].length; j++) {
        resulantMatrix[i][j] = matrix1[i][j] - scalar;
      }
    }

    return resulantMatrix;
  }

  /** Computes the sigmoid function for each element in a
   *    2D matrix.
   *  @param matrix
   *    A 2D matrix. Each value in the matrix will have 
   *      the sigmoid function computed on it.
   *  @return a 2D array with its elements equal to the 
   *    inputed matrix's element with the sigmoid function
   *    computed on it.
   *  @precondition matrix must have a length greater than 
   *    0. The second dimension of matrix must be equal to
   *    1.
  */
  public static double[][] sigmoid(double[][] matrix) {
    double[][] resultantMatrix = new double[matrix.length][matrix[0].length];

    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++) {
        resultantMatrix[i][j] = (1.0 / (1.0 + Math.exp(-matrix[i][j])));
      }
    }

    return resultantMatrix;
  }

  /** Computes the mean squared error.
   *  @param predictions
   *    A 2D matrix containing the predictions made by the
   *      model.
   *  @param y
   *    A 2D matrix containing predictors.
   *  @return a double value representing the mean squared
   *    error value.
   *  @precondition both the predictions and the y matrices
   *    must have a length greater than 0, and the 2nd 
   *    dimension mmust be equal to 1. The predictors
   *    matrix must have the same length as the y matrix.
  */
  public static double meanSquaredError(double[][] predictions, double[][] y) {
    double error = 0.0;

    for (int i = 0; i < y.length; i++) {
      error += Math.pow(y[i][0] - predictions[i][0], 2);
    }

    return error / y.length;
  }

  /** Returns a new matrix where the rows are equal to the 
   *    columns of the original matrix, and the columns are
   *    equal to the rows of the original matrix.
   *  @param matrix
   *    The orginal matrix that is to transformed into the
   *      new transposed matrix.
   *  @return a new 2D matrix where the rows are the same
   *    as the columns in the inputed matrix, and the  
   *    columns are the same as the rows in the inputed
   *    matrix.
   *  @precondition the length of the inputed matrix must
   *    be greater than 0. The length of the second
   *    dimension must be greater than 0.
  */
  public static double[][] transpose(double[][] matrix) {
    double[][] transposedMatrix = new double[matrix[0].length][matrix.length];

    for (int i = 0; i < matrix[0].length; i++) {
      for (int j = 0; j < matrix.length; j++) {
        transposedMatrix[i][j] = matrix[j][i];
      }
    }

    return transposedMatrix;
  }

  /** Returns the max value between the inputed value and
   *    0.
   *  @param matrix
   *    A 2D matrix containing all the values to feed into
   *      the ReLU function.
   *  @return a 2D matrix contraining the output of the 
   *    ReLU function for each value in the inputed matrix.
   *  @precondition matrix must have a length greater than
   *    0. The length of matrix's second dimentions must be
   *    1.
  */
  public static double[][] relu(double[][] matrix) {
    double[][] resultantMatrix = new double[matrix.length][1];

    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++) {
        if (matrix[i][j] > 0.0) {
          resultantMatrix[i][j] = matrix[i][j];
        } else {
          resultantMatrix[i][j] = 0.0;
        }
      }
    }
    
    return resultantMatrix;
  }

  /** Computes the quotient of the corresponding elements
   *    in 2 matrices.
   *  @param matrix1
   *    The first 2D matrix. The elements of this matrix
   *      are divided by the elements in the second matrix.
   *  @param matrix2
   *    The second 2D matrix. The elements of the first 
   *      matrix are divided by the elements in this matrix
   *  @return a new 2D matrix where the elements are equal
   *    to the corresponding element in matrix1 divided by
   *    the corresponding element in matrix2.
   *  @precondition both matrix1 and matrix2 must have a
   *    length greater than 0. matrix1 and matrix2 must
   *    have the same shape.
  */
  public static double[][] divide(double[][] matrix1, double[][] matrix2) {
    double[][] resultantMatrix = new double[matrix1.length][matrix1[0].length];

    for (int i = 0; i < matrix1.length; i++) {
      for (int j = 0; j < matrix1[0].length; j++) {
        resultantMatrix[i][j] = matrix1[i][j] / matrix2[i][j];
      }
    }

    return resultantMatrix;
  }

  /** Computes the quotient of each element in a 2D matrix
   *    and a scalar value.
   *  @param matrix
   *    The matrix which each element is to be divided by a
   *      scalar value.
   *  @param scalar
   *    The scalar value to divide each element in matrix
   *      by.
   *  @return a new 2D matrix where each element is equal
   *    to the corresponding matrix value divided by scalar
   *  @precondition The value of scalar cannot equal 0. 
   *    The length of matrix must be greater than 0. The
   *    length of matrix's second dimension must be greater
   *    than 0.
  */
  public static double[][] divide(double[][] matrix, double scalar) {
    double[][] resultantMatrix = new double[matrix.length][matrix[0].length];

    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++) {
        resultantMatrix[i][j] = matrix[i][j] / scalar;
      }
    }

    return resultantMatrix;
  }

  /** Multiplies each element in a matrix by a scalar value
   *  @param matrix
   *    The matrix that each element is to be multipled by
   *      the scalar value.
   *  @param scalar
   *    The scalar value that each element of matrix is to
   *      be multiplied by.
   *  @return a new 2D array where each element is equal
   *    to the corresponding element in matrix times the
   *    scalar value.
   *  @precondition the length of matrix must be greater 
   *    than 0, and the length of the second dimension of
   *    matrix must be greater than 0.
  */
  public static double[][] multiply(double[][] matrix, double scalar) {
    double[][] resultantMatrix = new double[matrix.length][matrix[0].length];

    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++) {
        resultantMatrix[i][j] = matrix[i][j] * scalar;
      }
    }

    return resultantMatrix;
  }

  /** Computes each element in a 2D matrix to the power of
   *    a scalar.
   *  @param matrix
   *    A 2D matrix which each element is to be put to the
   *      power of a scalar value.
   *  @param scalar
   *    The scalar value that each element in matrix is to
   *      be put to the power of.
   *  @return a new 2D matrix where each element is equal
   *    to the corresponding element in matrix to the power
   *    of scalar.
   *  @precondition the length of matrix must be greater
   *    than 0, and the length of dimension 2 must be 
   *    greater than 0.
  */
  public static double[][] power(double[][] matrix, double scalar) {
    double[][] resultantMatrix = new double[matrix.length][matrix[0].length];

    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++) {
        resultantMatrix[i][j] = Math.pow(matrix[i][j], scalar);
      }
    }

    return resultantMatrix;
  }

  /** Returns the difference of corresponding elements in
   *   two matrices of equal size.
   *  @param matrix1
   *    The first matrix. This is the matrix whose elements
   *      will be subtracted from.
   *  @param matrix2
   *    The second matrix. This is the matrix that will be
   *      subtracted from the first matrix.
   *  @return a new 2D matrix whose elements are equal to 
   *    the corresponding elements in matrix1 minus the
   *    corresponding elements in matrix2.
   *  @precondition both matrix1 and matrix2 must have the 
   *    same size, and both have lengths greater than 0. 
   *    The length of the second dimension of both matrix1
   *    and matrix2 must be greater than 0.
  */
  public static double[][] subtract(double[][] matrix1, double[][] matrix2) {
    double[][] resultantMatrix = new double[matrix1.length][matrix1[0].length];

    for (int i = 0; i < matrix1.length; i++) {
      for (int j = 0; j < matrix1[0].length; j++) {
        resultantMatrix[i][j] = matrix1[i][j] - matrix2[i][j];
      }
    }

    return resultantMatrix;
  }

  /** Computes the cross entropy score of a model given its
   *    outputed predictions to a series of predictors.
   *    Only valid when used in binary classification.
   *  @param predictions
   *    A 2D matrix whose elements are equal to the output
   *      of a sigmoid or softmax function. Is in the form
   *      of a probability of a certain class.
   *  @param y
   *    A 2D matrix containing the correct labels for the 
   *      given matrix of predictions.
   *  @return a double value representing the accuracy of a
   *    model given its outputs.
   *  @precondition the length of both the predictions 
   *    argument and the y argument must be greater than
   *    0. The length of the second dimension of both
   *    inputs must equal 1.
  */
  public static double crossEntropy(double[][] predictions, double[][] y) {
    double error = 0.0;
    
    for (int i = 0; i < predictions.length; i++) {
      error += (-y[i][0] * Math.log(predictions[i][0])) - ((1.0 - y[i][0]) * Math.log(1.0 - predictions[i][0])); 
    }

    return error / y.length;
  }
}
