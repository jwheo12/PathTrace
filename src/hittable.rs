use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::test::Point3;
use crate::vec3::{Vec3};

#[derive(Clone, Copy)]
pub struct HitRecord {
    pub t: f64,
    pub p : Point3,
    pub normal : Vec3,
    pub front_face:bool,
    pub mat: Material,
}

impl HitRecord {
    pub fn new(t:f64, p:Point3, normal:Vec3, mat: Material) -> HitRecord {
        HitRecord{p,normal,t,front_face:true, mat}
    }
    pub fn set_face_normal(&mut self, r: &Ray) {
        self.front_face = Vec3::dot(r.direction, self.normal) <0.0;
        if !self.front_face{
            self.normal = -self.normal;
        }
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, r: &Ray, ray_t : Interval) -> Option<HitRecord>;
}
