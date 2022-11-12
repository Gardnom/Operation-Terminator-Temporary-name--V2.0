using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Operation_Terminator_Temporary_name__V2._0
{
    public class NeuralNetwork
    {
        int[] m_NetworkShape = {1, 4, 4, 1};
        private List<Layer> m_HiddenLayers;

        public NeuralNetwork() {
            m_HiddenLayers = new List<Layer>();
            for (int i = 1; i < m_NetworkShape.Length; i++) {
                int numInputs = m_NetworkShape[i - 1];
                int numNodes = m_NetworkShape[i];
                Layer layer = new Layer(numInputs, numNodes);
                m_HiddenLayers.Add(layer);
            }
        }

        public Vector<float> Brain(Vector<float> inputs) {
            if (m_HiddenLayers.Count < 1) return Vector<float>.Build.Dense(0);

            Layer layerRef = null;
            for (int i = 0; i < m_HiddenLayers.Count; i++) {
                layerRef = m_HiddenLayers[i];
                if (i == 0) {
                    layerRef.ForwardPass(inputs);
                    layerRef.ActivationReLU();
                }
                else {
                    Layer lastLayer = m_HiddenLayers[i - 1];
                    layerRef.ForwardPass(lastLayer.nodes);
                    
                    // Don't use activationfunction on output layer
                    if(i != (m_HiddenLayers.Count - 1))
                        layerRef.ActivationReLU();
                }
            }

            return layerRef?.nodes ?? Vector<float>.Build.Dense(0);
        }
        
        public Matrix<float> BrainBatch(Matrix<float> inputs) {
            if (m_HiddenLayers.Count < 1) return Matrix<float>.Build.Dense(0, 0);

            Layer layerRef = null;
            Matrix<float> lastResult = null;
            
            for (int i = 0; i < m_HiddenLayers.Count; i++) {
                layerRef = m_HiddenLayers[i];
                if (i == 0) {
                    lastResult = layerRef.ForwardPassBatch(inputs);
                    layerRef.ActivationReLU();
                }
                else {
                    layerRef.ForwardPassBatch(lastResult);
                    
                    // Don't use activationfunction on output layer
                    if(i != (m_HiddenLayers.Count - 1))
                        layerRef.ActivationReLU();
                }
            }

            return lastResult ?? Matrix<float>.Build.Dense(0, 0);
        }
    }
}