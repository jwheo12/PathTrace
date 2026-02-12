use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

#[derive(Clone, Copy)]
pub struct Plane {
    pub point: Point3,
    pub normal: Vec3,
    pub mat: Material,
}

impl Plane {
    pub fn new(point: Point3, normal: Vec3, mat: impl Into<Material>) -> Plane {
        Plane {
            point,
            normal: normal.unit_vector(),
            mat: mat.into(),
        }
    }
}

impl Hittable for Plane {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let denom = Vec3::dot(self.normal, r.direction);
        if denom.abs() < 1e-6 {
            return None;
        }

        let t = Vec3::dot(self.point - r.origin, self.normal) / denom;
        if !ray_t.surrounds(t) {
            return None;
        }

        let p = r.at(t);
        let mut rec = HitRecord::new(t, p, self.normal, self.mat);
        rec.set_face_normal(r);
        Some(rec)
    }
}
