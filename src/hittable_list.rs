use crate::box3::Box3;
use crate::cylinder::Cylinder;
use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::plane::Plane;
use crate::ray::Ray;
use crate::sphere::Sphere;

#[derive(Clone)]
pub struct HittableList {
    spheres: Vec<Sphere>,
    planes: Vec<Plane>,
    boxes: Vec<Box3>,
    cylinders: Vec<Cylinder>,
}

impl HittableList {
    pub fn new() -> HittableList {
        HittableList {
            spheres: Vec::new(),
            planes: Vec::new(),
            boxes: Vec::new(),
            cylinders: Vec::new(),
        }
    }
    pub fn add(&mut self, object: Sphere) {
        self.spheres.push(object)
    }

    pub fn add_plane(&mut self, object: Plane) {
        self.planes.push(object)
    }

    pub fn add_box(&mut self, object: Box3) {
        self.boxes.push(object)
    }

    pub fn add_cylinder(&mut self, object: Cylinder) {
        self.cylinders.push(object)
    }

    pub fn spheres(&self) -> &[Sphere] {
        &self.spheres
    }

    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }

    pub fn boxes(&self) -> &[Box3] {
        &self.boxes
    }

    pub fn cylinders(&self) -> &[Cylinder] {
        &self.cylinders
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
        for object in self.boxes.iter() {
            if let Some(hrec) = object.hit(r, Interval::new(ray_t.min, closet_so_far)) {
                closet_so_far = hrec.t;
                rec = Some(hrec);
            }
        }
        for object in self.cylinders.iter() {
            if let Some(hrec) = object.hit(r, Interval::new(ray_t.min, closet_so_far)) {
                closet_so_far = hrec.t;
                rec = Some(hrec);
            }
        }
        rec
    }
}
