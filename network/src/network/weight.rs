pub struct Weight {
    pub value: f32,
}

impl Weight {
    pub fn create(value: f32) -> Weight {
        return Weight {
            value,
        };
    }

    pub fn adjust(&mut self, adjustment: f32) -> () {
        self.value += adjustment;
    }
}

impl std::ops::Mul<f32> for Weight {
    type Output = f32;

    fn mul(self, rhs: f32) -> Self::Output {
        return self.value * rhs;
    }
}


impl std::ops::Mul<&f32> for &Weight {
    type Output = f32;

    fn mul(self, rhs: &f32) -> Self::Output {
        return self.value * rhs;
    }
}
