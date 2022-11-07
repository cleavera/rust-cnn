use rand::random;
use crate::label::Label;
use crate::position::Position;

pub struct Function {
    pub m: f32,
    pub c: f32,
}

impl Function {
    pub fn create(m: f32, c: f32) -> Function {
        return Function {
            m,
            c,
        };
    }

    pub fn random() -> Function {
        let seed = (random::<f32>() * 2.) - 1.;
        let m;

        if seed.abs() > 0.85 {
            m = seed.powi(15) * 10.
        } else {
            m = seed;
        }

        let c = random::<f32>() - 0.5;
        return Self::create(m, c);
    }

    pub fn get_y(&self, x: f32) -> f32 {
        return self.m * x + self.c;
    }

    pub fn classify(&self, position: Position) -> Label {
        if position.1 > self.get_y(position.0) {
            return Label::A;
        }

        return Label::B;
    }
}
