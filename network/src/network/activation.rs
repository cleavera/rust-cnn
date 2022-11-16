use crate::network::matrix::Matrix;

pub mod softmax;
pub mod relu;

pub trait ActivationFn {
    fn activate(val: Matrix) -> Matrix;
    fn derivative(val: Matrix) -> Matrix;
}
