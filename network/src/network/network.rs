use crate::network::activation::relu::Relu;
use crate::network::matrix::Matrix;
use crate::network::TrainingBatch;
use super::Layer;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn create(network_shape: Vec<usize>, mut input_nodes: usize) -> Network {
        return Network {
            layers: network_shape.iter().map(|num_of_nodes| {
                let layer = Layer::create(*num_of_nodes, input_nodes);
                input_nodes = *num_of_nodes;

                return layer;
            }).collect::<Vec<Layer>>(),
        };
    }

    pub fn feed_forward(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut food = Matrix::from_vec(inputs);

        for layer in self.layers.iter() {
            food = layer.feed_forward(food);
        }

        return food.elements;
    }

    pub fn train(&mut self, batch: Vec<TrainingBatch>, learning_rate: f32) -> () {
        // Without activation the gradient for a layer is 2x(xm - t)
        // We can omit the 2 as we don't need the exact amount just a general direction
        // g = x(xm - t)
        // y = xm
        // e = y - t
        // g = xe

        // With an activation function it becomes 2xf'(xm)(f(xm)-t)
        // g = xf'(xm)(f(xm)-t)
        // y = xm
        // e = f(y) - t
        // df = f'(y)
        // r = df*e
        // g = x*r

        let layer = self.get_output_layer();
        let mut nudge: Option<Matrix> = None;
        let len = batch.len() as f32;

        for b in batch {
            let x = Matrix::extend_rows(&Matrix::from_vec(b.input), vec![1.]).unwrap();
            let t = Matrix::from_vec(b.expected);
            let y = Matrix::matrix_multiplication(&x, &layer.weights).unwrap();
            let fy = Relu::activate(&y);
            let e = Matrix::subtraction(&fy, &t).unwrap();
            let df = Relu::derivative(&y);
            let r = Matrix::hadamard(&df, &e).unwrap();
            let g = Matrix::matrix_multiplication(&Matrix::transposition(&x), &r).unwrap();

            nudge = match nudge {
                None => Some(g),
                Some(p) => Some(Matrix::addition(&p, &g).unwrap()),
            };
        }

        let average_nudge = Matrix::map(&nudge.unwrap(), |v| {
            return v / len;
        });

        self.layers[0].adjust_weights(&Matrix::scalar_multiplication(&average_nudge, -learning_rate));
    }

    pub fn get_output_layer(&self) -> &Layer {
        return &self.layers.last().unwrap();
    }
}
