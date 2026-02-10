use crate::util::{random_f64, random_f64_range};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, PartialEq, Clone,Copy,Default)]
pub struct Vec3{
    pub x: f64,
    pub y: f64,
    pub z: f64
}
impl Vec3{
    pub fn new(x:f64, y:f64, z:f64) -> Vec3 {
        Vec3 {x,y,z}
    }
    pub fn zero() -> Vec3 {
        Vec3::new(0.0,0.0,0.0)
    }

    pub fn random() -> Vec3 {
        Vec3::new(random_f64(), random_f64(), random_f64())
    }
    pub fn random_range(min : f64, max:f64) -> Vec3 {
        Vec3::new(random_f64_range(min,max),
                  random_f64_range(min,max),
                  random_f64_range(min,max))
    }
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }
    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn near_zero(&self) -> bool {
        const S: f64 = 1e-8;
        self.x.abs() < S && self.y.abs() < S && self.z.abs() < S
    }
    pub fn unit_vector(self) -> Vec3 {
        self/self.length()
    }

    pub fn random_unit_vector() -> Vec3 {
        loop {
            let p = Vec3::random_range(-1.0,1.0);
            let lensq = p.length_squared();
            if 1e-160 < lensq && lensq <= 1.0 {
                return p / lensq.sqrt();
            }
        }
    }
    pub fn random_on_hemisphere(normal : Vec3) -> Vec3 {
        let on_unit_sphere = Vec3::random_unit_vector();
        if Vec3::dot(on_unit_sphere, normal) > 0.0 {
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }
    pub fn reflect(v : Vec3, n : Vec3) ->Vec3 {
        v-2.0*Vec3::dot(v,n)*n
    }
    pub fn refract(uv : Vec3, n:Vec3,etai_over_etat: f64) -> Vec3{
        let cos_theta = Vec3::dot(-uv,n).min(1.0);
        let r_out_perp = etai_over_etat * (uv + cos_theta*n);
        let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt()*n;
        r_out_perp + r_out_parallel
    }
    pub fn cross(u : Vec3, v : Vec3) -> Vec3 {
        Vec3::new(u.y * v.z - u.z * v.y,
                  u.z * v.x - u.x * v.z,
                  u.x * v.y - u.y * v.x)
    }
    pub fn dot(a:Vec3, b:Vec3) -> f64 {

        a.x * b.x + a.y * b.y + a.z * b.z
    }
    pub fn random_in_unit_disk() -> Vec3 {
        loop {
            let p = Vec3::new(random_f64_range(-1.0,1.0),
                                    random_f64_range(-1.0,1.0),
                                    0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }
}

impl Add for Vec3{
    type Output =Vec3;
    fn add(self, other: Vec3) -> Self {
        Vec3::new(self.x+other.x,
                  self.y+other.y,
                  self.z+other.z)
    }
}
impl Sub for Vec3{
    type Output =Vec3;
    fn sub(self, other: Vec3) -> Self {
        Vec3::new(self.x-other.x,
                  self.y-other.y,
                  self.z-other.z)
    }
}
impl Mul<Vec3> for f64{
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3{
        Vec3::new(self*other.x,
                  self*other.y,
                  self*other.z)
    }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self,other: f64) -> Vec3{
        Vec3::new(self.x / other,
                  self.y/other,
                  self.z/other)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x,-self.y,-self.z)
    }
}

pub type Point3=Vec3;
