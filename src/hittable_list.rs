use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::ray::Ray;
use crate::sphere::Sphere;

#[derive(Clone)]
pub struct HittableList {
    objects: Vec<Sphere>,
}

impl HittableList {
    pub fn new() -> HittableList {
        HittableList {
            objects: Vec::new(),
        }
    }
    pub fn add(&mut self, object: Sphere) {
        self.objects.push(object)
    }

    pub fn spheres(&self) -> &[Sphere] {
        &self.objects
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let mut rec = None;
        let mut closet_so_far = ray_t.max;
        for object in self.objects.iter() {
            if let Some(hrec) = object.hit(r, Interval::new(ray_t.min, closet_so_far)) {
                closet_so_far = hrec.t;
                rec = Some(hrec);
            }
        }
        rec
    }
}
