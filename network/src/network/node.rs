use crate::network::{Weight};
use rand::random;

pub struct Node {
    pub weights: Vec<Weight>,
}

impl Node {
    pub fn create(num_of_inputs: usize) -> Node {
        let weights = (0..num_of_inputs + 1).map(|_| {
            return Weight::create((random::<f32>() * 2.) - 1.);
        }).collect::<Vec<Weight>>();

        return Node {
            weights,
        }
    }

    pub fn feed_forward(&self, inputs: &Vec<f32>) -> f32 {
        let mut sum = 0.;
        let bias = 1.;

        for (index, weight) in self.weights.iter().enumerate() {
            sum += weight * inputs.get(index).unwrap_or(&bias);
        }

        if sum < 0. {
            sum = -1.;
        } else {
            sum = 1.;
        }

        return sum;
    }
}
