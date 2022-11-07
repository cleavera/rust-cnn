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

    pub fn feed_forward(&self, inputs: &Vec<f32>) -> Vec<f32> {
        return self.layers.feed_forward(inputs);
    }

    pub fn train(&mut self, inputs: &Vec<f32>, expected: Vec<f32>, learning_rate: f32) -> () {
        let guess = self.feed_forward(inputs);
        let mut error = 0.;

        for (i, _) in guess.iter().enumerate() {
            error += expected[i] - guess[i];
        }

        let bias = 1.;

        for node in self.layers.nodes.iter_mut() {
            for (i, w) in node.weights.iter_mut().enumerate() {
                w.adjust(error * inputs.get(i).unwrap_or(&bias) * learning_rate);
            }
        }
    }

    pub fn get_output_layer(&self) -> &Layer {
        return &self.layers;
    }
}
