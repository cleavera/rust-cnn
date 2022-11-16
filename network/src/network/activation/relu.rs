use crate::network::matrix::Matrix;

pub struct Relu {

}

impl Relu {
    pub fn activate(m: &Matrix) -> Matrix {
        return Matrix::map(m, |v| {
            return if *v < 0. {
                0.
            } else {
                *v
            }
        });
    }

    pub fn derivative(m: &Matrix) -> Matrix {
        return Matrix::map(m, |v| {
            return if *v < 0. {
                0.
            } else {
                1.
            }
        });
    }
}