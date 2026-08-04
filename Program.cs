using System;
using System.Collections.Generic;
using System.IO;

namespace FFNN
{
    internal class Program
    {
        public static int NrEL = 12311;
        public static int P = (int)(NrEL * 0.6);       // number of training patterns
        public static int sizeV = (int)(NrEL * 0.2);   // size of validation
        public static int V = P;                        // start index of validation set (0-based)
        public static int T = V + sizeV;                 // start index of testing set
        public static int sizeT = NrEL - T;              // size of test = whatever remains, so P + sizeV + sizeT always == NrEL exactly (avoids rounding gaps/overflows)
        public static int I = 16;

        public double[,] z = new double[P, I];
        public string[] g = new string[P];
        public double[,] zValidation = new double[sizeV, I];
        public string[] gValidation = new string[sizeV];
        public double[,] zTest = new double[sizeT, I];
        public string[] gTest = new string[sizeT];
        public double[] sseHistory = new double[P];
        double bestSSE = double.MaxValue;
        int patienceCounter = 0;
        int patience = 10;          // how many epochs to tolerate no improvement
        double minDelta = 1e-5;     // minimum change to count as "improvement"
        int maxEpochs = 500;
        public DataSet unseen;
        public string path = "Demo.csv";

        

        public Dictionary<string, int> classIndex = new Dictionary<string, int>()
        {
            { "SEKER", 0 },
            { "BARBUNYA", 1 },
            { "BOMBAY", 2 },
            { "CALI", 3 },
            { "DERMASON", 4 },
            { "HOROZ", 5 },
            { "SIRA", 6 }
        };

        // reverse lookup: index -> class name
        public string[] labels = { "SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA" };

        public static void Main()
        {
            new Program();
        }

        public Program()
        {
            readData();
            NormalizeFeatures();

            FeedForwardNetwork ffnn = new FeedForwardNetwork(16, 7, 7, 0.1);// changed neurons to 32

            Console.WriteLine("Training network...");
            for (int epoch = 0; epoch < maxEpochs; epoch++)// changed epoch to 1000
            {
                for (int i = 0; i < P; i++)
                {
                    double[] rowInputs = new double[I];
                    for (int j = 0; j < I; j++)
                    {
                        rowInputs[j] = z[i, j];
                    }
                    ffnn.Train(rowInputs, EncodeLabel(g[i]));

                }
               
                double currentSSE = CalculateAverageSSE(ffnn); // use a held-out validation set if you have one, else trainInputs/trainTargets

                if (bestSSE - currentSSE > minDelta)
                {
                    bestSSE = currentSSE;
                    patienceCounter = 0;
                }
                else
                {
                    patienceCounter++;
                    if (patienceCounter >= patience)
                    {
                        Console.WriteLine($"Stopped early at epoch {epoch + 1}, Average SSE = {currentSSE:F6}");
                        break;
                    }
                }

                double sse = CalculateTrainingSSE(ffnn);
                sseHistory[epoch] = sse;
                if(epoch % 10 == 0) 
                    Console.WriteLine($"Epoch {epoch + 1}: SSE = {sse:F6}");

                using (StreamWriter writer = new StreamWriter("SSEHistory.csv"))
                {
                    writer.WriteLine("Epoch,SSE");

                    for (int i = 0; i < sseHistory.Length; i++)
                    {
                        writer.WriteLine($"{i + 1},{sseHistory[i]}");
                    }
                }
            }

            Console.WriteLine("Evaluating on validation set...");
            EvaluateSet(ffnn, zValidation, gValidation, sizeV, "Validation");

            Console.WriteLine("Evaluating on test set...");
            EvaluateSet(ffnn, zTest, gTest, sizeT, "Test");

            Console.WriteLine("Pridicting with Demo set");
            unseen = new DataSet().LoadAndNormalize(path, featureMean, featureStd, hasLabels: false);
            DisplayPredictions(ffnn, unseen);
        }

        public double CalculateTrainingSSE(FeedForwardNetwork ffnn)
        {
            double sse = 0.0;

            for (int i = 0; i < P; i++)
            {
                double[] inputs = new double[I];

                for (int j = 0; j < I; j++)
                    inputs[j] = z[i, j];

                double[] outputs = ffnn.ComputeOutputs(inputs);

                double[] target = EncodeLabel(g[i]);

                for (int k = 0; k < target.Length; k++)
                {
                    double error = target[k] - outputs[k];
                    sse += error * error;
                }
            }

            return sse;
        }
        public double CalculateAverageSSE(FeedForwardNetwork ffnn)
        {
            double sse = 0.0;
            for (int i = 0; i < P; i++)
            {
                double[] inputs = new double[I];
                for (int j = 0; j < I; j++)
                    inputs[j] = z[i, j];
                double[] outputs = ffnn.ComputeOutputs(inputs);
                double[] target = EncodeLabel(g[i]);
                for (int k = 0; k < target.Length; k++)
                {
                    double error = target[k] - outputs[k];
                    sse += error * error;
                }
            }
            return sse / P;
        }

        public void EvaluateSet(FeedForwardNetwork ffnn, double[,] data, string[] trueLabels, int size, string setName)
        {
            int correct = 0;

            for (int i = 0; i < size; i++)
            {
                double[] rowInputs = new double[I];
                for (int j = 0; j < I; j++)
                {
                    rowInputs[j] = data[i, j];
                }

                int predictedIndex = ffnn.Predict(rowInputs);
                string predictedClass = labels[predictedIndex];

                if (predictedClass == trueLabels[i])
                {
                    correct++;
                }
            }

            double accuracy = size > 0 ? (double)correct / size * 100.0 : 0.0;
            Console.WriteLine($"{setName} accuracy: {correct}/{size} ({accuracy:F2}%)");
        }

        public double[] featureMean = new double[I];
        public double[] featureStd = new double[I];
        private int inputSize;

        // Compute mean/std from the TRAINING set only, then apply to all three sets.
        // Val/test must reuse training stats -- recomputing stats on them would leak
        // information and make results inconsistent with what the model was trained on.
        /*public void NormalizeFeatures()
        {
            for (int j = 0; j < I; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < P; i++) sum += z[i, j];
                double mean = sum / P;

                double sqSum = 0.0;
                for (int i = 0; i < P; i++) sqSum += (z[i, j] - mean) * (z[i, j] - mean);
                double std = Math.Sqrt(sqSum / P);
                if (std < 1e-8) std = 1.0; // avoid divide-by-zero for constant columns

                featureMean[j] = mean;
                featureStd[j] = std;
            }

            for (int i = 0; i < P; i++)
                for (int j = 0; j < I; j++)
                    z[i, j] = (z[i, j] - featureMean[j]) / featureStd[j];

            for (int i = 0; i < sizeV; i++)
                for (int j = 0; j < I; j++)
                    zValidation[i, j] = (zValidation[i, j] - featureMean[j]) / featureStd[j];

            for (int i = 0; i < sizeT; i++)
                for (int j = 0; j < I; j++)
                    zTest[i, j] = (zTest[i, j] - featureMean[j]) / featureStd[j];
        }*/

        // Computes mean/std from the training set and stores them in featureMean/featureStd
        public void ComputeNormalizationParams()
        {
            for (int j = 0; j < I; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < P; i++) sum += z[i, j];
                double mean = sum / P;

                double sqSum = 0.0;
                for (int i = 0; i < P; i++) sqSum += (z[i, j] - mean) * (z[i, j] - mean);
                double std = Math.Sqrt(sqSum / P);

                if (std < 1e-8) std = 1.0; // avoid divide-by-zero for constant columns

                featureMean[j] = mean;
                featureStd[j] = std;
            }
        }

        // Applies already-computed featureMean/featureStd to any dataset
        public void ApplyNormalization(double[,] data, int numRows)
        {
            for (int i = 0; i < numRows; i++)
                for (int j = 0; j < I; j++)
                    data[i, j] = (data[i, j] - featureMean[j]) / featureStd[j];
        }

        // Orchestrates the original behaviour using the two generic pieces above
        public void NormalizeFeatures()
        {
            ComputeNormalizationParams();

            ApplyNormalization(z, P);
            ApplyNormalization(zValidation, sizeV);
            ApplyNormalization(zTest, sizeT);
        }

        public double[] EncodeLabel(string label)
        {
            double[] target = new double[7];
            target[classIndex[label]] = 1.0;
            return target;
        }

        public void readData()
        {
            using (StreamReader sr = new StreamReader("BeanData.csv"))
            {
                sr.ReadLine(); // skip header

                string line;
                int counter = 0;

                while ((line = sr.ReadLine()) != null)
                {
                    string[] tokens = line.Split(',');

                    double[] features = new double[I];
                    for (int i = 0; i < I; i++)
                    {
                        features[i] = Double.Parse(tokens[i]);
                    }
                    string label = tokens[16]; // 16 features (0-15) then class label at index 16

                    if (counter < P)
                    {
                        for (int i = 0; i < I; i++)
                        {
                            z[counter, i] = features[i];
                        }
                        g[counter] = label;
                    }
                    else if (counter < T)
                    {
                        int k = counter - P;
                        for (int i = 0; i < I; i++)
                        {
                            zValidation[k, i] = features[i];
                        }
                        gValidation[k] = label;
                    }
                    else if (counter < NrEL)
                    {
                        int t = counter - T;
                        for (int i = 0; i < I; i++)
                        {
                            zTest[t, i] = features[i];
                        }
                        gTest[t] = label;
                    }

                    counter++;
                }

                if (counter < NrEL)
                {
                    Console.WriteLine($"Warning: expected {NrEL} rows but only read {counter}.");
                }
            }
        }

        public string[] PredictAll(FeedForwardNetwork ffnn, DataSet data)
        {
            string[] predictedLabels = new string[data.P];

            for (int i = 0; i < data.P; i++)
            {
                double[] inputs = new double[data.I];
                for (int j = 0; j < data.I; j++)
                    inputs[j] = data.z[i, j];

                int predictedIndex = ffnn.Predict(inputs);
                predictedLabels[i] = DecodeLabel(predictedIndex);
            }

            return predictedLabels;
        }


        // Inverse of EncodeLabel — maps class index back to original label
        private string DecodeLabel(int index)
        {
            return labels[index];
        }
        public void DisplayPredictions(FeedForwardNetwork ffnn, DataSet data)
        {
            string[] predictions = PredictAll(ffnn, data);

            Console.WriteLine("=== Predictions on Unseen Data ===");
            for (int i = 0; i < predictions.Length; i++)
            {
                Console.WriteLine($"Pattern {i,4}: Predicted Class = {predictions[i]}");
            }
        }
    }
}
