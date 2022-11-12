using System;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator_Temporary_name__V2._0
{
    class Program
    {
        static void Main(string[] args) {
            NeuralNetwork nn = new NeuralNetwork();

            while (true) {
                Console.Write("Give a number to the machine!: ");
                string input = Console.ReadLine();
                float result;
                if (!float.TryParse(input, out result)) {
                    Console.WriteLine("You must give it a number!");
                    continue;    
                }

                float[,] r = {{result}, {1}, {2}, {4}};
                var brainInput = Matrix<float>.Build.DenseOfArray(r);
                var resMat = nn.BrainBatch(brainInput);
                Console.WriteLine("The machine gives you back: " + resMat[0, 0]);

            }
            
            Console.ReadLine();
        }
    }
}
