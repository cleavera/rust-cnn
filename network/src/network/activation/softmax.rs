use crate::network::activation::ActivationFn;
use crate::network::matrix::Matrix;

pub struct SoftMax {

}

impl SoftMax {
    fn activate(m: &Matrix) -> Matrix {
        let sum = Matrix::map(m, |v| {
            return std::f32::consts::E.powf(*v);
        }).sum();

        return Matrix::map(m, |v| {
            return (std::f32::consts::E.powf(*v)) / sum;
        });
    }

    fn derivative(m: &Matrix) -> Matrix {
        let y = SoftMax::activate(m);
        let mut yt = Matrix::transposition(&SoftMax::activate(m));
        let extension_values = yt.elements.clone();

        for _ in 1..m.rows {
            yt = Matrix::extend_columns(&yt, extension_values.clone()).unwrap();
        }

        println!("{}\n\n", m);
        println!("{}\n\n", y);
        println!("{}\n\n", yt);
        println!("{}\n\n", Matrix::identity(m.rows));
        println!("{}", &Matrix::subtraction(&Matrix::identity(m.rows), &yt).unwrap());

        return Matrix::matrix_multiplication(&Matrix::transposition(&y), &Matrix::subtraction(&Matrix::identity(m.rows), &yt).unwrap()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::network::activation::softmax::SoftMax;
    use crate::network::matrix::Matrix;

    #[test]
    fn activate() {
        let s = SoftMax::activate(&Matrix::create(2, 2, vec![1.,2.,3.,4.]));
        let expect = Matrix::create(2, 2, vec![0.032058604, 0.08714432, 0.23688282, 0.6439143]);
        assert_eq!(s, expect);
    }

    #[test]
    fn derivative() {
        let s = SoftMax::derivative(&Matrix::create(1, 4, vec![1.,2.,3.,4.]));
        let expect = Matrix::create(4, 4, vec![0.0, 3.7252903e-9, 0.0, -1.4901161e-8]);
        assert_eq!(s, expect);
    }
}