use rand::random;
use crate::function::Function;
use super::position::Position;
use super::label::Label;

pub struct DataSet {
    pub points: Vec<DataPoint>,
}

impl DataSet {
    pub fn generate(length: usize, classifier: &Function) -> DataSet {
        let points = (0..length).map(|_| {
            return DataPoint::random(classifier);
        });

        return DataSet {
            points: points.collect::<Vec<DataPoint>>()
        };
    }

    pub fn extend(&mut self, num: usize, classifier: &Function) -> () {
        for _ in 0..num {
            self.points.push(DataPoint::random(classifier));
        }
    }
}

pub struct DataPoint {
    pub position: Position,
    pub label: Label,
    pub guess: Option<Label>,
}

impl DataPoint {
    pub fn random(classifier: &Function) -> DataPoint {
        let position = DataPoint::random_position();

        return DataPoint {
            position,
            label: classifier.classify(position),
            guess: None,
        };
    }

    fn random_position() -> Position {
        return (
            (random::<f32>() * 2.) - 1.,
            (random::<f32>() * 2.) - 1.,
        );
    }
}
