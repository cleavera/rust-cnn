use rand::random;
use crate::network::matrix::Matrix;

pub struct Layer {
    pub weights: Matrix,
}

impl Layer {
    pub fn create(num_of_nodes: usize, num_of_inputs: usize) -> Layer {
        let weights = (0..((num_of_inputs + 1) * num_of_nodes)).map(|_| {
            return (random::<f32>() * 2.) - 1.;
        }).collect::<Vec<f32>>();

        return Layer {
            weights: Matrix::create(num_of_nodes, num_of_inputs + 1, weights),
        };
    }

    pub fn feed_forward(&self, inputs: Matrix) -> Matrix {
        let inputs_with_bias = Matrix::extend_rows(&inputs, vec![1.]).unwrap();

        return Matrix::matrix_multiplication(&inputs_with_bias, &self.weights).unwrap();
    }

    pub fn adjust_weights(&mut self, adjustment: &Matrix) -> () {
        self.weights = Matrix::addition(&self.weights, adjustment).unwrap();
    }
}
