use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub struct Matrix {
    pub cols: usize,
    pub rows: usize,
    pub elements: Vec<f32>,
}

impl Matrix {
    pub fn create(cols: usize, rows: usize, elements: Vec<f32>) -> Matrix {
        return Matrix {
            cols,
            rows,
            elements,
        };
    }

    pub fn from_vec(elements: Vec<f32>) -> Matrix {
        return Matrix::create(elements.len(), 1, elements);
    }

    pub fn map<F>(m: &Matrix, mapper: F) -> Matrix
    where
        F: Fn(&f32) -> f32 {

        return Matrix {
            rows: m.rows,
            cols: m.cols,
            elements: m.elements.iter().map(mapper).collect::<Vec<f32>>()
        };
    }

    pub fn addition(m1: &Matrix, m2: &Matrix) -> Result<Matrix, MatrixAdditionOperationError> {
        if m1.rows != m2.rows || m1.cols != m2.cols {
            return Err(MatrixAdditionOperationError::MatricesShapesDoNotMatch);
        }

        if m1.elements.len() != m2.elements.len() {
            return Err(MatrixAdditionOperationError::MatricesHaveDifferentNumberOfElements);
        }

        let mut result_elements = vec![];

        for i in 0..m1.elements.len() {
            result_elements.push(m1.elements[i] + m2.elements[i]);
        }

        return Ok(Matrix {
            cols: m1.cols,
            rows: m1.rows,
            elements: result_elements,
        });
    }

    pub fn subtraction(m1: &Matrix, m2: &Matrix) -> Result<Matrix, MatrixAdditionOperationError> {
        return Matrix::addition(m1, &Matrix::scalar_multiplication(m2, -1.));
    }

    pub fn scalar_multiplication(m: &Matrix, s: f32) -> Matrix {
        return Matrix::map(m, |e| e * s);
    }

    pub fn transposition(m: &Matrix) -> Matrix {
        let mut result_elements = vec![];

        for i in 0..m.cols {
            for j in 0..m.rows {
                let index = j * m.cols + i;

                result_elements.push(m.elements[index]);
            }
        }

        return Matrix {
            cols: m.rows,
            rows: m.cols,
            elements: result_elements,
        };
    }

    pub fn matrix_multiplication(m1: &Matrix, m2: &Matrix) -> Result<Matrix, MatrixMultiplicationOperationError> {
        let mut result_elements = vec![];

        if m1.cols != m2.rows {
            return Err(MatrixMultiplicationOperationError::LeftMatrixColumnsDoNotEqualRightMatrixRows);
        }

        for j in 0..m1.rows {
            for i in 0..m2.cols {
                let mut sum = 0.;

                for k in 0..m1.cols {
                    sum += m1.get(k, j) * m2.get(i, k);
                }

                result_elements.push(sum);
            }
        }

        return Ok(Matrix::create(
            m2.cols,
            m1.rows,
            result_elements,
        ));
    }

    pub fn extend_columns(m: &Matrix, values: Vec<f32>) -> Result<Matrix, MatrixExtendOperationError> {
        if values.len() < m.cols {
            return Err(MatrixExtendOperationError::NotEnoughValuesToExtend);
        }

        let mut result_elements = vec![];

        for e in m.elements.iter() {
            result_elements.push(*e);
        }

        for v in values {
            result_elements.push(v);
        }

        return Ok(Matrix {
            cols: m.cols,
            rows: m.rows + 1,
            elements: result_elements
        });
    }

    pub fn extend_rows(m: &Matrix, values: Vec<f32>) -> Result<Matrix, MatrixExtendOperationError> {
        if values.len() < m.rows {
            return Err(MatrixExtendOperationError::NotEnoughValuesToExtend);
        }

        let mut result_elements = vec![];
        let mut values_pointer = 0;

        for x in 0..m.elements.len() {
            if x != 0 && (x % m.cols) == 0 {
                result_elements.push(values[values_pointer]);
                values_pointer += 1;
            }

            result_elements.push(m.elements[x]);
        }

        result_elements.push(values[values_pointer]);

        return Ok(Matrix {
            cols: m.cols + 1,
            rows: m.rows,
            elements: result_elements
        });
    }

    pub fn hadamard(m1: &Matrix, m2: &Matrix) -> Result<Matrix, MatrixHadamardOperationError> {
        if m1.rows != m2.rows || m1.cols != m2.cols {
            return Err(MatrixHadamardOperationError::MatricesShapesDoNotMatch);
        }

        if m1.elements.len() != m2.elements.len() {
            return Err(MatrixHadamardOperationError::MatricesHaveDifferentNumberOfElements);
        }

        let mut result_elements = vec![];

        for i in 0..m1.elements.len() {
            result_elements.push(m1.elements[i] * m2.elements[i]);
        }

        return Ok(Matrix {
            cols: m1.cols,
            rows: m1.rows,
            elements: result_elements,
        });
    }

    pub fn get(&self, col: usize, row: usize) -> f32 {
        let index = row * self.cols + col;

        return self.elements[index];
    }

    pub fn set(&mut self, col: usize, row: usize, value: f32) -> () {
        let index = row * self.cols + col;

        self.elements[index] = value;
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        let mut result = self.cols == other.cols && self.rows == other.rows && self.elements.len() == other.elements.len();

        for i in 0..self.elements.len() {
            result = result && self.elements[i] == other.elements[i];
        }

        return result;
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut out = ("|").to_owned();

        for (i, e) in self.elements.iter().enumerate() {
            if i != 0 && (i % (self.cols)) == 0 {
                out += " |\n|";
            }

            out += &(format!(" {:.2}", e));
        }

        out += " |";

        return write!(f, "{}", out);
    }
}

#[derive(Debug, PartialEq)]
pub enum MatrixAdditionOperationError {
    MatricesShapesDoNotMatch,
    MatricesHaveDifferentNumberOfElements,
}

#[derive(Debug, PartialEq)]
pub enum MatrixMultiplicationOperationError {
    LeftMatrixColumnsDoNotEqualRightMatrixRows,
}

#[derive(Debug, PartialEq)]
pub enum MatrixExtendOperationError {
    NotEnoughValuesToExtend,
}

#[derive(Debug, PartialEq)]
pub enum MatrixHadamardOperationError {
    MatricesShapesDoNotMatch,
    MatricesHaveDifferentNumberOfElements,
}

#[cfg(test)]
mod tests {
    use crate::network::matrix::{Matrix, MatrixAdditionOperationError, MatrixExtendOperationError, MatrixHadamardOperationError, MatrixMultiplicationOperationError};

    #[test]
    fn addition() {
        let m1 = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let m2 = Matrix::create(3, 2, vec![5.2, 8., 3., 4., 0., 7.]);
        let expected = Matrix::create(3, 2, vec![5.2, 10., 6., 5., 4., 13.]);
        let result = Matrix::addition(&m1, &m2).expect("Could not add");

        assert_eq!(result, expected);
    }

    #[test]
    fn addition_mismatch_shapes() {
        let m1 = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let m2 = Matrix::create(2, 3, vec![5.2, 8., 3., 4., 0., 7.]);
        match Matrix::addition(&m1, &m2) {
            Ok(_) => panic!("Should error"),
            Err(e) => assert_eq!(e, MatrixAdditionOperationError::MatricesShapesDoNotMatch),
        };
    }

    #[test]
    fn addition_missing_elements() {
        let m1 = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let m2 = Matrix::create(3, 2, vec![5.2, 8., 3., 4., 0.]);
        match Matrix::addition(&m1, &m2) {
            Ok(_) => panic!("Should error"),
            Err(e) => assert_eq!(e, MatrixAdditionOperationError::MatricesHaveDifferentNumberOfElements),
        };
    }

    #[test]
    fn subtraction() {
        let m1 = Matrix::create(3, 2, vec![0., 2., 3., 4., 4., 7.]);
        let m2 = Matrix::create(3, 2, vec![5.2, 8., 3., 1., 0., 6.]);
        let expected = Matrix::create(3, 2, vec![-5.2, -6., 0., 3., 4., 1.]);
        let result = Matrix::subtraction(&m1, &m2).expect("Could not add");

        assert_eq!(result, expected);
    }

    #[test]
    fn subtraction_mismatch_shapes() {
        let m1 = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let m2 = Matrix::create(2, 3, vec![5.2, 8., 3., 4., 0., 7.]);
        match Matrix::subtraction(&m1, &m2) {
            Ok(_) => panic!("Should error"),
            Err(e) => assert_eq!(e, MatrixAdditionOperationError::MatricesShapesDoNotMatch),
        };
    }

    #[test]
    fn subtraction_missing_elements() {
        let m1 = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let m2 = Matrix::create(3, 2, vec![5.2, 8., 3., 4., 0.]);
        match Matrix::subtraction(&m1, &m2) {
            Ok(_) => panic!("Should error"),
            Err(e) => assert_eq!(e, MatrixAdditionOperationError::MatricesHaveDifferentNumberOfElements),
        };
    }

    #[test]
    fn scalar_multiplication() {
        let m = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let expected = Matrix::create(3, 2, vec![0., 4., 6., 2., 8., 12.]);
        let result = Matrix::scalar_multiplication(&m, 2.);

        assert_eq!(result, expected);
    }

    #[test]
    fn transposition() {
        let m = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let expected = Matrix::create(2, 3, vec![0., 1., 2., 4., 3., 6.]);
        let result = Matrix::transposition(&m);

        assert_eq!(result, expected);
    }

    #[test]
    fn matrix_multiplication() {
        let m1 = Matrix::create(3, 2, vec![2., 3., 4., 1., 0., 0.]);
        let m2 = Matrix::create(2, 3, vec![0., 1000., 1., 100., 0., 10.]);
        let expected = Matrix::create(2, 2, vec![3., 2340., 0., 1000.]);
        let result = Matrix::matrix_multiplication(&m1, &m2).expect("Could not multiply");

        assert_eq!(result, expected);
    }

    #[test]
    fn matrix_multiplication_2() {
        let m1 = Matrix::create(3, 1, vec![0.54, -0.64, 1.00]);
        let m2 = Matrix::create(1, 3, vec![0.68, 0.08, 0.38]);
        let expected = Matrix::create(1, 1, vec![0.696]);
        let result = Matrix::matrix_multiplication(&m1, &m2).expect("Could not multiply");

        assert_eq!(result, expected);
    }

    #[test]
    fn matrix_multiplication_shape_mismatch() {
        let m1 = Matrix::create(3, 2, vec![2., 3., 4., 1., 0., 0.]);
        let m2 = Matrix::create(3, 2, vec![2., 3., 4., 1., 0., 0.]);
        match Matrix::matrix_multiplication(&m1, &m2) {
            Ok(_) => panic!("Should error"),
            Err(e) => assert_eq!(e, MatrixMultiplicationOperationError::LeftMatrixColumnsDoNotEqualRightMatrixRows),
        };
    }

    #[test]
    fn extend_columns() {
        let m = Matrix::create(3, 2, vec![2., 3., 4., 1., 0., 0.]);
        let values = vec![9., 8., 7.];
        let expected = Matrix::create(3, 3, vec![2., 3., 4., 1., 0., 0., 9., 8., 7.]);
        let result = Matrix::extend_columns(&m, values).expect("Could not extend");

        assert_eq!(result, expected);
    }

    #[test]
    fn extend_columns_not_enough_values() {
        let m1 = Matrix::create(3, 2, vec![2., 3., 4., 1., 0., 0.]);
        let values = vec![9., 8.];
        match Matrix::extend_columns(&m1, values) {
            Ok(_) => panic!("Should error"),
            Err(e) => assert_eq!(e, MatrixExtendOperationError::NotEnoughValuesToExtend),
        };
    }

    #[test]
    fn extend_rows() {
        let m = Matrix::create(2, 3, vec![2., 3., 4., 1., 0., 0.]);
        let values = vec![9., 8., 7.];
        let expected = Matrix::create(3, 3, vec![2., 3., 9., 4., 1., 8., 0., 0., 7.]);
        let result = Matrix::extend_rows(&m, values).expect("Could not extend");

        assert_eq!(result, expected);
    }

    #[test]
    fn extend_rows_not_enough_values() {
        let m1 = Matrix::create(2, 3, vec![2., 3., 4., 1., 0., 0.]);
        let values = vec![9., 8.];
        match Matrix::extend_rows(&m1, values) {
            Ok(_) => panic!("Should error"),
            Err(e) => assert_eq!(e, MatrixExtendOperationError::NotEnoughValuesToExtend),
        };
    }

    #[test]
    fn hadamard() {
        let m1 = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let m2 = Matrix::create(3, 2, vec![5.2, 8., 3., 4., 0., 7.]);
        let expected = Matrix::create(3, 2, vec![0., 16., 9., 4., 0., 42.]);
        let result = Matrix::hadamard(&m1, &m2).expect("Could not hadamard");

        assert_eq!(result, expected);
    }

    #[test]
    fn hadamard_mismatch_shapes() {
        let m1 = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let m2 = Matrix::create(2, 3, vec![5.2, 8., 3., 4., 0., 7.]);
        match Matrix::hadamard(&m1, &m2) {
            Ok(_) => panic!("Should error"),
            Err(e) => assert_eq!(e, MatrixHadamardOperationError::MatricesShapesDoNotMatch),
        };
    }

    #[test]
    fn hadamard_missing_elements() {
        let m1 = Matrix::create(3, 2, vec![0., 2., 3., 1., 4., 6.]);
        let m2 = Matrix::create(3, 2, vec![5.2, 8., 3., 4., 0.]);
        match Matrix::hadamard(&m1, &m2) {
            Ok(_) => panic!("Should error"),
            Err(e) => assert_eq!(e, MatrixHadamardOperationError::MatricesHaveDifferentNumberOfElements),
        };
    }
}