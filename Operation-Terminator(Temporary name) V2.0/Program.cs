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

                float[] r = {result};
                var brainInput = Vector<float>.Build.Dense(r);
                var resVec = nn.Brain(brainInput);
                Console.WriteLine("The machine gives you back: " + resVec.At(0));

            }
            
            Console.ReadLine();
        }
    }
}
