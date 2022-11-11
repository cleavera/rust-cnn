use fps_counter::FPSCounter;
use minifb::{Key, Menu, Window, WindowOptions};
use raqote::{Color, DrawOptions, DrawTarget, PathBuilder, SolidSource, Source, StrokeStyle};

use network::network::{Network, TrainingBatch};

use self::color_palette::ColorPalette;
use self::data::{DataPoint, DataSet};
use self::function::Function;
use self::label::Label;
use self::position::Position;

mod data;
mod position;
mod function;
mod label;
mod color_palette;

const WIDTH: usize = 800;
const HEIGHT: usize = 800;
const BATCH_SIZE: usize = 500;

fn main() {
    let mut window = Window::new("Perceptron", WIDTH, HEIGHT, WindowOptions {
        ..WindowOptions::default()
    }).unwrap();

    let f = Function::random();

    let mut menu = Menu::new("Train").unwrap();
    menu.add_item("Next", 0)
        .shortcut(Key::F1, 0)
        .build();
    menu.add_item("Autorun start", 1)
        .shortcut(Key::F5, 0)
        .build();
    menu.add_item("Autorun stop", 2)
        .shortcut(Key::F6, 0)
        .build();
    window.add_menu(&menu);

    let size = window.get_size();
    let mut dt = DrawTarget::new(size.0 as i32, size.1 as i32);
    let stroke = StrokeStyle {
        width: 3.,
        ..StrokeStyle::default()
    };

    let point_stroke = StrokeStyle {
        width: 5.,
        ..StrokeStyle::default()
    };

    let mut dataset = DataSet::generate(BATCH_SIZE, &f);
    let mut network = Network::create(vec![1], 2);
    let mut generation = -1;
    let mut fps = FPSCounter::new();
    let mut error = 0.;
    let mut auto_play = false;

    while window.is_open() {
        let do_training = window.is_menu_pressed().and_then(|id| {
            if id == 0 {
                return Some(true);
            }

            if id == 1 {
                auto_play = true;
            }

            if id == 2 {
                auto_play = false;
            }

            return None;
        }).unwrap_or(auto_play);

        if do_training {
            dataset.reset();
            dataset.extend(BATCH_SIZE, &f);

            generation += 1;
        }

        if do_training || generation == -1 {
            error = 0.;
            dt.clear(SolidSource::from(ColorPalette::background()));

            if do_training {
                let batch = dataset.points.iter().map(|p| {
                    let expected = match p.label {
                        Label::A => 1.,
                        Label::B => -1.
                    };

                    return TrainingBatch{ input: vec![p.position.0, p.position.1], expected: vec![expected]};
                }).collect::<Vec<TrainingBatch>>();

                network.train(batch, 0.1);
            }

            for p in dataset.points.iter_mut() {
                let guess_value = network.feed_forward(vec![p.position.0, p.position.1])[0];

                let guess = match network.feed_forward(vec![p.position.0, p.position.1])[0] > 0. {
                    true => Label::A,
                    false => Label::B,
                };

                p.guess = Some(guess);

                let expected = match p.label {
                    Label::A => 1.,
                    Label::B => -1.
                };

                error += (expected - guess_value).powi(2);

                point(&mut dt, &p, &point_stroke, p.guess.as_ref().unwrap());
            }

            line(&mut dt, (1., f.get_y(1.)), (-1., f.get_y(-1.)), &stroke, ColorPalette::line());
            line(&mut dt, (1., guess_y(1., &network)), (-1., guess_y(-1., &network)), &stroke, ColorPalette::line_guess());

            if generation == -1 {
                generation = 0;
            }

            window.update_with_buffer(dt.get_data(), size.0, size.1).unwrap();
            error /= dataset.points.len() as f32;
        } else {
            window.update();
        }

        let gb = guess_y(0., &network);
        let gm = guess_y(1., &network) - guess_y(0., &network);

        window.set_title(format!("y = {}x + {} | y = {}x + {} | Error: {} | Generation: {} | Points: {} | FPS: {}", f.m, f.c, gm, gb, error, generation.to_string(), dataset.points.len(), fps.tick()).as_str());
    }
}

fn point(dt: &mut DrawTarget, p: &DataPoint, stroke: &StrokeStyle, guess: &Label) -> () {
    let mut pb = PathBuilder::new();
    let normalised_position = normalise_position(p.position, dt.width(), dt.height());

    let color = match p.label {
        Label::A => ColorPalette::label_a(),
        Label::B => ColorPalette::label_b(),
    };

    let guess_color = match *guess == p.label {
        true => ColorPalette::correct(),
        false => ColorPalette::incorrect(),
    };

    pb.arc(normalised_position.0, normalised_position.1, 4., 0., 2. * std::f32::consts::PI);
    let path = pb.finish();
    dt.stroke(&path, &Source::Solid(SolidSource::from(guess_color)), stroke, &DrawOptions::new());
    dt.fill(&path, &Source::Solid(SolidSource::from(color)), &DrawOptions::new());
}

fn line(dt: &mut DrawTarget, start: Position, end: Position, stroke: &StrokeStyle, color: Color) -> () {
    let mut pb = PathBuilder::new();
    let normalised_start_position = normalise_position(start, dt.width(), dt.height());
    let normalised_end_position = normalise_position(end, dt.width(), dt.height());

    pb.move_to(normalised_start_position.0, normalised_start_position.1);
    pb.line_to(normalised_end_position.0, normalised_end_position.1);
    let path = pb.finish();
    dt.stroke(&path, &Source::Solid(SolidSource::from(color)), stroke, &DrawOptions::new());
}

fn normalise_position(position: Position, width: i32, height: i32) -> Position {
    return ((position.0 + 1.) * ((width / 2) as f32), (height as f32) - ((position.1 + 1.) * ((height / 2) as f32)));
}

fn guess_y(x: f32, network: &Network) -> f32 {
    let output_layer = network.get_output_layer();

    let w0 = output_layer.weights.get(0, 0);
    let w1 = output_layer.weights.get(0, 1);
    let w2 = output_layer.weights.get(0, 2);

    return -(w2 / w1) - (w0 / w1) * x;
}
