using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FFNN
{
    using System;

    public class FeedForwardNetwork
    {
        private readonly int inputSize;
        private readonly int hiddenSize;
        private readonly int outputSize;
        private readonly double learningRate;

        // Network weights and biases
        private readonly double[,] weightsInputHidden;
        private readonly double[,] weightsHiddenOutput;
        private readonly double[] biasHidden;
        private readonly double[] biasOutput;

        // Layer state caches (needed for backpropagation)
        private readonly double[] hiddenOutputs;
        private readonly double[] finalOutputs;

        private readonly Random _random = new Random();

        public FeedForwardNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.learningRate = learningRate;

            weightsInputHidden = new double[inputSize, hiddenSize];
            weightsHiddenOutput = new double[hiddenSize, outputSize];
            biasHidden = new double[hiddenSize];
            biasOutput = new double[outputSize];

            hiddenOutputs = new double[hiddenSize];
            finalOutputs = new double[outputSize];

            InitializeWeights();
        }

        // Initialize parameters using Xavier/Glorot initialization.
        // Scaling by 1/sqrt(fan_in) keeps the weighted sums fed into Sigmoid
        // in a reasonable range at the start of training, instead of saturating
        // the activation (and killing the gradient) immediately.
        private void InitializeWeights()
        {
            double inputHiddenScale = 1.0 / Math.Sqrt(inputSize);
            double hiddenOutputScale = 1.0 / Math.Sqrt(hiddenSize);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    weightsInputHidden[i, j] = (_random.NextDouble() * 2.0 - 1.0) * inputHiddenScale;

            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < outputSize; j++)
                    weightsHiddenOutput[i, j] = (_random.NextDouble() * 2.0 - 1.0) * hiddenOutputScale;

            for (int i = 0; i < hiddenSize; i++) biasHidden[i] = 0.0;
            for (int i = 0; i < outputSize; i++) biasOutput[i] = 0.0;
        }

        // Activation function (Sigmoid)
        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        // Derivative of Sigmoid for backpropagation
        private double SigmoidDerivative(double sigmoidVal) => sigmoidVal * (1.0 - sigmoidVal);

        // 1. Forward Pass Strategy
        public double[] ComputeOutputs(double[] inputs)
        {
            // Compute hidden layer activations
            for (int j = 0; j < hiddenSize; j++)
            {
                // add the bias
                double sum = biasHidden[j];
                for (int i = 0; i < inputSize; i++)
                {
                    //sum wieghted inputs for the ith weights of the Jth hidden neuron
                    sum += inputs[i] * weightsInputHidden[i, j];
                }
                // compute f(net) using the Sigmoid Activation Function
                hiddenOutputs[j] = Sigmoid(sum);
            }

            // Compute output layer activations
            for (int k = 0; k < outputSize; k++)
            {
                double sum = biasOutput[k];
                for (int j = 0; j < hiddenSize; j++)
                {
                    //sum wieghted inputs for the ith weights of the Jth output neuron
                    sum += hiddenOutputs[j] * weightsHiddenOutput[j, k];
                }
                // compute f(net) using the Sigmoid Activation Function
                finalOutputs[k] = Sigmoid(sum);
            }

            return finalOutputs;
        }

        // 2. Backward Pass Correction
        public void Train(double[] inputs, double[] targets)
        {
            // First run the data forward to fill the state caches
            ComputeOutputs(inputs);

            // Step A: Calculate Output Layer Gradients (Deltas)
            double[] outputGradients = new double[outputSize];
            for (int k = 0; k < outputSize; k++)
            {
                double error = targets[k] - finalOutputs[k];
                outputGradients[k] = error * SigmoidDerivative(finalOutputs[k]);
            }

            // Step B: Calculate Hidden Layer Gradients (Deltas)
            double[] hiddenDeltas = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                double downstreamGradient = 0.0;
                for (int k = 0; k < outputSize; k++)
                {
                    downstreamGradient += outputGradients[k] * weightsHiddenOutput[j, k];
                }
                hiddenDeltas[j] = downstreamGradient * SigmoidDerivative(hiddenOutputs[j]);
            }

            // Step C: Update Hidden-to-Output Weights and Biases
            for (int j = 0; j < hiddenSize; j++)
            {
                for (int k = 0; k < outputSize; k++)
                {
                    weightsHiddenOutput[j, k] += learningRate * outputGradients[k] * hiddenOutputs[j];
                }
            }
            for (int k = 0; k < outputSize; k++)
            {
                biasOutput[k] += learningRate * outputGradients[k];
            }

            // Step D: Update Input-to-Hidden Weights and Biases
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    weightsInputHidden[i, j] += learningRate * hiddenDeltas[j] * inputs[i];
                }
            }
            for (int j = 0; j < hiddenSize; j++)
            {
                biasHidden[j] += learningRate * hiddenDeltas[j];
            }
        }

        public int Predict(double[] inputs)
        {
            double[] outputs = ComputeOutputs(inputs);

            int predictedClass = 0;
            double maxOutput = outputs[0];

            for (int i = 1; i < outputs.Length; i++)
            {
                if (outputs[i] > maxOutput)
                {
                    maxOutput = outputs[i];
                    predictedClass = i;
                }
            }

            return predictedClass;
        }
        public double CalculateSSE(double[][] inputs, double[][] targets)
        {
            double sse = 0.0;

            for (int i = 0; i < inputs.Length; i++)
            {
                double[] outputs = ComputeOutputs(inputs[i]);

                for (int j = 0; j < outputSize; j++)
                {
                    double error = targets[i][j] - outputs[j];
                    sse += error * error;
                }
            }

            return sse;
        }
        // Add near CalculateSSE
        public double CalculateAverageSSE(double[][] inputs, double[][] targets)
        {
            return CalculateSSE(inputs, targets) / inputs.Length;
        }

    }
}



