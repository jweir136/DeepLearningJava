import DeepLearning.Utils;
import java.util.*;

public class FeedForwardBinaryClassifier {
  private double[][] X, y;
  private int nInputNodes, nHiddenNodes, nHiddenLayers, nOutputNodes, maxEpochs;
  private double learningRate;
  private List<double[][]> weights;

  public FeedForwardBinaryClassifier(int maxEpochs, double learningRate, int nHiddenLayers, int nHiddenNodes) {
    this.learningRate = learningRate;
    this.maxEpochs = maxEpochs;
    this.nHiddenNodes = nHiddenNodes;
    this.nHiddenLayers = nHiddenLayers;
    this.nOutputNodes = 2;
  }

  public void fit(double[][] X, double[][] y) {
    this.X = X;
    this.y = y;
    this.nInputNodes = X[0].length;
    newWeights();
  }

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
