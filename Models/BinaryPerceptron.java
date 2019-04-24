import DeepLearning.Utils;

/** Creates a very simple perceptron ANN constiting of only 1 input layer, and 1 output layer containing a single node. Can only be used for binary classification.
*/
public class BinaryPerceptron {
  private double[][] X;
  private int[] y;
  public double[] weights;
  private int epochs;
  private double learningRate;
  private double bias;

  /** Sets the max number of epochs to use during training, the learning rate, and the bias constant.
   *  @param epochs:
   *    The max number of iterations throught the training data to be used when training. Training may use less than the number of epochs during training is model converges early.
   *  @param learningRate:
   *    The rate at which to update weights during training.
   *  @Precondition:
   *    1). epochs must be greater than or equal to 0. If set to zero, then no training will take place.
   *    2). learningRate must be greater than or equal to zero. If set to zero, then no training will take place.
  */
  public BinaryPerceptron(int epochs, double learningRate) {
    this.epochs = epochs;
    this.learningRate = learningRate;
    this.bias = 1.0;
  }

  /** Fits the model with the training features and predictors. Also inits the weights randomly.
   *  @param X:
   *    
  */
  public void fit(double[][] X, int[] y) {
    this.X = X;
    this.y = y;
    this.weights = Utils.initWeights(this.X[0].length);
  }

  public double[] predict(double[][] X) {
    double[] predictions = new double[X.length];
    int counter = 0;

    for (double[] x : X)
      predictions[counter++] = Utils.sigmoid(Utils.dot(this.weights, x) + this.bias);

    return predictions;
  }

  public double error(double[][] X, int[] y) {
    double error = 0.0;
    double[] predictions = predict(X);

    for (int i = 0; i < y.length; i++)
      error += Math.pow(y[i] - predictions[i], 2);
    
    return error;
  }

  private double weightDeriv(int index) {
    double result = 0.0;
    double[] predictions = predict(this.X);

    for (int i = 0; i < this.X.length; i++) {
      result += (this.y[i] - predictions[i]) * (-this.X[i][index]);
    }

    return (2.0 * result) / this.X.length;
  }

  public void train() {
    double loss, lastLoss;
    lastLoss = error(this.X, this.y);

    for (int i = 0; i < this.epochs; i++) {
      for (int j = 0; j < this.weights.length; j++)
        this.weights[j] -= this.learningRate * weightDeriv(j);
      loss = error(this.X, this.y);
      if (loss > lastLoss) {
        System.out.println("Converged at " + i);
        break;
      }
      System.out.println("Epoch: " + i + "\tLoss:" + loss);
      lastLoss = error(this.X, this.y);
    }
  }
}
