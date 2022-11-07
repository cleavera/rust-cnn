use super::Node;

pub struct Layer {
    pub nodes: Vec<Node>,
}

impl Layer {
    pub fn create(num_of_nodes: usize, num_of_inputs: usize) -> Layer {
        let nodes = (0..num_of_nodes).map(|_| {
            return Node::create(num_of_inputs);
        }).collect::<Vec<Node>>();

        return Layer {
            nodes,
        };
    }

    pub fn feed_forward(&self, inputs: &Vec<f32>) -> Vec<f32> {
        return self.nodes.iter().map(|node| node.feed_forward(inputs)).collect::<Vec<f32>>();
    }
}
