using ILGPU.Runtime;
using ILGPU;

namespace GDvsCD.Services
{
    public static class Algo
    {
        // GPU kernel to multiply a matrix by a vector.
        // Each thread computes one row (dot product).
        static void MatrixVectorMultiplyKernel(Index1D rowIndex, ArrayView2D<double, Stride2D.DenseX> matrix, ArrayView<double> vector, ArrayView<double> result)
        {
            int numCols = (int)matrix.Extent.Y;
            double sum = 0;
            // Compute dot product of row rowIndex with the vector.
            for (int j = 0; j < numCols; j++)
            {
                sum += matrix[rowIndex, j] * vector[j];
            }
            result[rowIndex] = sum;
        }

        // Helper function: perform matrix-vector multiplication on the GPU.
        public static double[] GpuMatrixVectorMultiply(Accelerator accelerator, double[,] X, double[] theta)
        {
            int m = X.GetLength(0); // number of rows
            int n = X.GetLength(1); // number of columns

            // Allocate GPU memory for X (as a 2D buffer) and theta (1D buffer).
            using var dX = accelerator.Allocate2DDenseX<double>(new LongIndex2D(m, n));
            dX.CopyFromCPU(accelerator.DefaultStream, X);

            using var dTheta = accelerator.Allocate1D<double>(theta.Length);
            dTheta.CopyFromCPU(accelerator.DefaultStream, theta);

            // Allocate GPU memory for the result vector.
            using var dResult = accelerator.Allocate1D<double>(m);

            // Load and launch the kernel.
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<double, Stride2D.DenseX>, ArrayView<double>, ArrayView<double>>(MatrixVectorMultiplyKernel);
            kernel((int)m, dX.View, dTheta.View, dResult.View);
            accelerator.Synchronize();

            // Copy the result back to the host.
            double[] result = dResult.GetAsArray1D();
            return result;
        }

        // Cost function: J = 1/(2*m) * (Xθ - y)^T (Xθ - y)
        public static double CostFunction(Accelerator accelerator, double[,] X, double[] y, double[] theta)
        {
            int m = y.Length;
            double[] h = GpuMatrixVectorMultiply(accelerator, X, theta);
            double sumSqError = 0;
            for (int i = 0; i < m; i++)
            {
                double diff = h[i] - y[i];
                sumSqError += diff * diff;
            }
            return sumSqError / (2 * m);
        }

        // Coordinate descent for linear regression.
        // Returns the updated theta and histories of cost, theta0, and theta1.
        public static (double[] theta, List<double> costHistory)
            CoordinateDescent(double[] theta, double[,] X, double[] y, Accelerator accelerator, double alpha = 0.03, int numIters = 20)
        {
            int m = X.GetLength(0);
            int n = X.GetLength(1);

            // Histories to store cost and parameter values.
            var costHistory = new List<double>();
            //var thetashisotry = new Dictionary<int, List<double>>();
            //for (int j = 0; j < n; j++)
            //    thetashisotry[j] = new();
                // Perform coordinate descent.
                for (int iter = 0; iter < numIters; iter++)
            {
                for (int j = 0; j < n; j++)
                {
                    // Compute hypothesis: h = X * theta (using GPU).
                    double[] h = GpuMatrixVectorMultiply(accelerator, X, theta);

                    // Compute the gradient for coordinate j: gradient = sum_i X[i,j]*(h[i]-y[i])
                    double gradient = 0;
                    for (int i = 0; i < m; i++)
                    {
                        gradient += X[i, j] * (h[i] - y[i]);
                    }
                    gradient= gradient / (2 * m); 
                    // Update theta for coordinate j.
                    theta[j] = theta[j] - alpha * gradient;

                    // Record cost and (for plotting) the parameter values.
                  
                    //thetashisotry[j].Add(theta[j]);
                }
                double cost = CostFunction(accelerator, X, y, theta);
                costHistory.Add(cost);
            }

            return (theta, costHistory);
        }

        public static (double[] theta, List<double> costHistory)
    GradientDescent(double[] theta, double[,] X, double[] y, Accelerator accelerator, double alpha = 0.03, int numIters = 20)
        {
            int m = X.GetLength(0);
            int n = X.GetLength(1);

            // Histories to store cost and parameter values.
            var costHistory = new List<double>();


            // Perform gradient descent.
            for (int iter = 0; iter < numIters; iter++)
            {
                // Compute hypothesis: h = X * theta (using GPU).
                double[] h = GpuMatrixVectorMultiply(accelerator, X, theta);

                // Compute gradients for all coordinates simultaneously.
                double[] gradients = new double[n];
                for (int j = 0; j < n; j++)
                {
                    double gradient = 0;
                    for (int i = 0; i < m; i++)
                    {
                        gradient += X[i, j] * (h[i] - y[i]);
                    }
                    gradients[j] = gradient;
                }

                // Update theta for all coordinates simultaneously.
                for (int j = 0; j < n; j++)
                {
                    theta[j] = theta[j] - alpha * gradients[j];
                }

                // Record the cost and parameter values after this iteration.
                double cost = CostFunction(accelerator, X, y, theta);
                costHistory.Add(cost);
            
            }

            return (theta, costHistory);
        }
        public static double[,] ConvertJaggedTo2D(double[][] jagged)
        {
            if (jagged == null || jagged.Length == 0)
                throw new ArgumentException("Jagged array is null or empty.");

            int rows = jagged.Length;
            int cols = jagged[0].Length;

            // Optional: Validate all rows have the same column count
            for (int i = 1; i < rows; i++)
            {
                if (jagged[i].Length != cols)
                    throw new InvalidOperationException($"Row {i} does not have the same column count.");
            }

            double[,] result = new double[rows, cols];

            // Unrolled access pattern
            for (int i = 0; i < rows; i++)
            {
                var row = jagged[i];
                Buffer.BlockCopy(row, 0, result, i * cols * sizeof(double), cols * sizeof(double));
            }

            return result;
        }

    }
}
