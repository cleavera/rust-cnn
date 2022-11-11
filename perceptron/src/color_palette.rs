use raqote::Color;

pub struct ColorPalette {}

impl ColorPalette {
    pub fn label_b() -> Color {
        Color::new(0xff, 0, 0, 0xff)
    }

    pub fn label_a() -> Color {
        Color::new(0xff, 0xff, 0x66, 0xff)
    }

    pub fn correct() -> Color {
        Color::new(0, 0, 0, 0)
    }

    pub fn incorrect() -> Color {
        Color::new(0xff, 0xff, 0, 0)
    }

    pub fn line_guess() -> Color {
        Color::new(0xff, 0, 0xff, 0)
    }

    pub fn background() -> Color {
        Color::new(0xff, 0xff, 0xff, 0xff)
    }

    pub fn line() -> Color {
        Color::new(0xff, 0, 0, 0)
    }
}