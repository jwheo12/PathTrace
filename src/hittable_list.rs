use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::plane::Plane;
use crate::ray::Ray;
use crate::sphere::Sphere;

#[derive(Clone)]
pub struct HittableList {
    spheres: Vec<Sphere>,
    planes: Vec<Plane>,
}

impl HittableList {
    pub fn new() -> HittableList {
        HittableList {
            spheres: Vec::new(),
            planes: Vec::new(),
        }
    }
    pub fn add(&mut self, object: Sphere) {
        self.spheres.push(object)
    }

    pub fn add_plane(&mut self, object: Plane) {
        self.planes.push(object)
    }

    pub fn spheres(&self) -> &[Sphere] {
        &self.spheres
    }

    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let mut rec = None;
        let mut closet_so_far = ray_t.max;
        for object in self.spheres.iter() {
            if let Some(hrec) = object.hit(r, Interval::new(ray_t.min, closet_so_far)) {
                closet_so_far = hrec.t;
                rec = Some(hrec);
            }
        }
        for object in self.planes.iter() {
            if let Some(hrec) = object.hit(r, Interval::new(ray_t.min, closet_so_far)) {
                closet_so_far = hrec.t;
                rec = Some(hrec);
            }
        }
        rec
    }
}
