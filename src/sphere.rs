use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

#[derive(Clone, Copy)]
pub struct Sphere {
    pub center : Point3,
    pub radius : f64,
    pub mat: Material,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, mat: impl Into<Material>) -> Sphere {
        Sphere {
            center,
            radius,
            mat: mat.into(),
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let oc = self.center - r.origin;
        let a = r.direction.length_squared();
        let h = Vec3::dot(r.direction,oc);
        let c = oc.length_squared()-self.radius*self.radius;
        let discriminant = h*h - a*c;

        if discriminant<0.0{
            return None
        }
        let sqrtd=discriminant.sqrt();
        let mut root = (h-sqrtd)/a;
        if !ray_t.surrounds(root){
            root = (h+sqrtd) /a;
            if !ray_t.surrounds(root) {
                return None;
            }
        }
        let t = root;
        let p = r.at(t);
        let normal = (p-self.center)/self.radius;
        let mut rec = HitRecord::new(t, p, normal, self.mat);
        rec.set_face_normal(r);
        Some(rec)
    }
}
