import DeepLearning.Utils;
import java.util.*;

public class FeedForwardBinaryClassifier {
  private double[][] X, y;
  private int nInputNodes, nHiddenNodes, nHiddenLayers, nOutputNodes, maxEpochs;
  private double learningRate;
  private List<double[][]> weights;

  /** @param maxEpochs
   *    The maximum number of iterations through the 
   *      training data to go through during training.
   *  @param learningRate
   *    A scalar value that determines the size of the
   *      steps to take during gradient descent.
   *  @param nHiddenLayers
   *    The number of hidden layers in the neural network.
   *  @param nHiddenNodes
   *    The number of nodes in the hidden layers.
   *  @preconditions maxEpochs must be greater than 0.
   *    learningRate argument must be greater than 0.
   *    If the value of the nHiddenLayers argument is 0,
   *    then the value of the nHiddenNodes argument must
   *    also be 0. If the value of nHiddenLayers is not 0, 
   *    then the value of nHiddenNodes must be greater than
   *    0.
  */
  public FeedForwardBinaryClassifier(int maxEpochs, double learningRate, int nHiddenLayers, int nHiddenNodes) {
    this.learningRate = learningRate;
    this.maxEpochs = maxEpochs;
    this.nHiddenNodes = nHiddenNodes;
    this.nHiddenLayers = nHiddenLayers;
    this.nOutputNodes = 2;
  }

  /** Fits the model to the training data.
   *  @param X
   *    A 2D array containing the training features.
   *  @param y
   *    A 2D array containing the training predictors.
   *  @precondition The number of unique classes in X must
   *    be equal to 2. The length of the second dimension
   *    of y must be equal to 1.
  */
  public void fit(double[][] X, double[][] y) {
    this.X = X;
    this.y = y;
    this.nInputNodes = X[0].length;
    newWeights();
  }

  /** Initializes the weights of the ANN to random values
   *    between 0 and 1.
  */
  private void newWeights() {
    this.weights = new ArrayList<double[][]>();
    int node1, node2;

    for (int i = 0; i < (2 + this.nHiddenNodes); i++) {
      if (i == 0) {
        node1 = this.nInputNodes;
        node2 = this.nHiddenNodes;
      } else if (i == 1 + this.nHiddenLayers) {
        node1 = this.nHiddenNodes;
        node2 = this.nOutputNodes;
      } else {
        node1 = this.nHiddenNodes;
        node2 = this.nHiddenNodes;
      }

      this.weights.add(Utils.initWeights(node2, node1));
    }
  }
}
