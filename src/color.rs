use std::io::Write;
use std::ops::{Add, AddAssign, Mul};
use crate::interval::Interval;
use crate::util::{random_f64, random_f64_range};
use crate::vec3::Vec3;

#[derive(Clone,Copy)]
pub struct Color{
    pub r:f64,
    pub g:f64,
    pub b:f64
}

impl From<Vec3> for Color{
    fn from(v: Vec3) -> Color {
        Color::new(v.x,v.y,v.z)
    }
}

impl Color {
    pub fn new(r:f64, g:f64, b:f64) -> Color {
        Color{r,g,b}
    }

    pub fn random() -> Color {
        Color::new(random_f64(), random_f64(), random_f64())
    }
    pub fn random_range(min : f64, max:f64) -> Color {
        Color::new(random_f64_range(min,max),
                  random_f64_range(min,max),
                  random_f64_range(min,max))
    }
}
impl Add for Color{
    type Output =Color;
    fn add(self, other: Color) -> Color {
        Color::new(self.r+other.r,
                  self.g+other.g,
                  self.b+other.b)
    }
}
impl Mul for Color {
    type Output = Color;
    fn mul(self, other: Color) -> Color {
        Color::new(self.r*other.r, self.g*other.g,self.b*other.b)
    }

}
impl Mul<Color> for f64{
    type Output = Color;
    fn mul(self, other: Color) -> Color{
        Color::new(self*other.r,
                  self*other.g,
                  self*other.b)
    }
}

impl AddAssign for Color {
    fn add_assign(&mut self, other: Color) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}

const INTENSITY: Interval = Interval::new(0.0, 0.999);

fn linear_to_gamma(linear_component : f64) -> f64 {
    if linear_component > 0.0 {
        linear_component.sqrt()
    } else {
        0.0
    }
}

pub fn write_color_ldr16(out : &mut impl Write, pixel_color : Color) {
    let r = linear_to_gamma(pixel_color.r);
    let g = linear_to_gamma(pixel_color.g);
    let b = linear_to_gamma(pixel_color.b);

    let rword = (65536.0 * INTENSITY.clamp(r)) as u16;
    let gword = (65536.0 * INTENSITY.clamp(g)) as u16;
    let bword = (65536.0 * INTENSITY.clamp(b)) as u16;

    out.write_all(&rword.to_be_bytes()).unwrap();
    out.write_all(&gword.to_be_bytes()).unwrap();
    out.write_all(&bword.to_be_bytes()).unwrap();
}

pub fn write_color_hdr(out : &mut impl Write, pixel_color : Color) {
    // PFM stores linear float values and can preserve values above 1.0 (HDR).
    let r = pixel_color.r.max(0.0) as f32;
    let g = pixel_color.g.max(0.0) as f32;
    let b = pixel_color.b.max(0.0) as f32;

    out.write_all(&r.to_le_bytes()).unwrap();
    out.write_all(&g.to_le_bytes()).unwrap();
    out.write_all(&b.to_le_bytes()).unwrap();
}
