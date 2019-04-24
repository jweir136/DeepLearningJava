package DeepLearning;

public class Utils {
  /** Return the dot product between two matrices.
   *  @param matrix1:
   *    The first matrix to use in the dot product calculation. Since dot product is communitive, the order of the matrices does not matter.
   *  @param matrix2:
   *    The second matrix to use in the dot product.
   *  @Returns:
   *    Return the dot product of 2 matrices as a scalar value.
   *  @Preconditions:
   *    1). The length of matrix1 and matrix2 must be the same.
   *    2). The length of matrix1 and matrix2 must be greater than 0. 
  */
  public static double dot(double[] matrix1, double[] matrix2) {
    double result = 0.0;
    for (int i = 0; i < matrix1.length; i++)
      result += matrix1[i] * matrix2[i];
    return result;
  }

  /** Creates an array of random values between 0 and 1. To be used as the weights in a single layer of an ANN.
   *  @param nWeights:
   *    The number of weights in the ANN layer. To find, take the number of weights in the first layer time the number of nodes in the second layer.
   *  @Returns:
   *    Returns a new 1D array to be used as the weights in the ANN.
   *  @Preconditions:
   *    1). The value of nWeights must be greater than 0.
  */
  public static double[] initWeights(int n) {
    double[] weights = new double[n];
    for (int i = 0; i < n; i++)
      weights[i] = Math.random();
    return weights;
  }

  public static double sigmoid(double value) {
    return 1.0 / (1.0 + Math.exp(-value));
  }
}
