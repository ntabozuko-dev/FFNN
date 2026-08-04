using System.Collections.Generic;
using System.IO;

internal class DataSet
{
    public double[,] z;
    public int[] g;
    public int P;
    public int I;

    public DataSet LoadAndNormalize(string path, double[] featureMeans, double[] featureStds, bool hasLabels)
    {
        ReadData(path, hasLabels);

        for (int i = 0; i < P; i++)
            for (int j = 0; j < I; j++)
                z[i, j] = (z[i, j] - featureMeans[j]) / featureStds[j];

        return this;
    }

    private void ReadData(string path, bool hasLabels)
    {
        var rows = new List<double[]>();
        var labels = new List<int>();

        using (StreamReader sr = new StreamReader(path))
        {
            string line = sr.ReadLine();
            while ((line = sr.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                string[] tokens = line.Split(',');
                int featureCount = hasLabels ? tokens.Length - 1 : tokens.Length;

                double[] row = new double[featureCount];
                for (int j = 0; j < featureCount; j++)
                    row[j] = double.Parse(tokens[j]);

                rows.Add(row);
                if (hasLabels)
                    labels.Add(int.Parse(tokens[tokens.Length - 1]));
            }
        }

        P = rows.Count;
        I = P > 0 ? rows[0].Length : 0;

        z = new double[P, I];
        for (int i = 0; i < P; i++)
            for (int j = 0; j < I; j++)
                z[i, j] = rows[i][j];

        g = hasLabels ? labels.ToArray() : null;
    }
}