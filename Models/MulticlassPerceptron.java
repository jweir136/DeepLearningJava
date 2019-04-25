import DeepLearning.Utils;

public class MulticlassPerceptron implements Classifier {
  private double[][] X;
  private int[][] y;
  public double[] weights;
  private double learningRate;
  private int epochs;

  public MulticlassPerceptron(int epochs, double learningRate) {
    this.epochs = epochs;
    this.learningRate = learningRate;
  }

  public void fit(double[][] X, int[] y) {
    this.X = X;
    this.y = Utils.oneHotEncode(y);
    this.weights = Utils.initWeights(X[0].length);
  }

  public void train() {
    ;
  }

  public double error(double[][] X, int[] y) {
    int[][] encodedY = Utils.oneHotEncode(y);
    return 0.0;
  }

  public double[] predict(double[][] X) {
    double[] preds = new double[X.length];

    return preds;
  }
}
