use crate::network::matrix::Matrix;
use super::Layer;

pub struct Network {
    layers: Layer,
}

impl Network {
    pub fn create(nodes_in_hidden_layer: usize, _output_nodes: usize, input_nodes: usize) -> Network {
        let layer = Layer::create(nodes_in_hidden_layer, input_nodes);

        return Network {
            layers: layer,
        };
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        return self.layers.feed_forward(Matrix::from_vec(inputs)).elements;
    }

    pub fn train(&mut self, inputs: Vec<f32>, expected: Vec<f32>, learning_rate: f32) -> () {
        let guess = self.feed_forward(inputs.clone());
        let mut error = 0.;

        for (i, _) in guess.iter().enumerate() {
            error += expected[i] - guess[i];
        }

        let bias_input = 1.;

        for node_index in 0..self.layers.weights.cols {
            for weight_index in 0..self.layers.weights.rows {
                let weight = self.layers.weights.get(node_index, weight_index);
                self.layers.weights.set(node_index, weight_index, weight + (error * inputs.get(weight_index).unwrap_or(&bias_input) * learning_rate))
            }
        }
    }

    pub fn get_output_layer(&self) -> &Layer {
        return &self.layers;
    }
}
