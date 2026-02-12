use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

#[derive(Clone, Copy)]
pub struct Cylinder {
    pub center: Point3,
    pub radius: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub mat: Material,
}

impl Cylinder {
    pub fn new(
        center: Point3,
        radius: f64,
        y_min: f64,
        y_max: f64,
        mat: impl Into<Material>,
    ) -> Cylinder {
        Cylinder {
            center,
            radius,
            y_min,
            y_max,
            mat: mat.into(),
        }
    }
}

impl Hittable for Cylinder {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let ox = r.origin.x - self.center.x;
        let oz = r.origin.z - self.center.z;
        let dx = r.direction.x;
        let dz = r.direction.z;

        let mut best_t = ray_t.max;
        let mut best_normal = None::<Vec3>;

        let a = dx * dx + dz * dz;
        if a > 1e-9 {
            let b = 2.0 * (ox * dx + oz * dz);
            let c = ox * ox + oz * oz - self.radius * self.radius;
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                let inv2a = 0.5 / a;
                let roots = [(-b - sq) * inv2a, (-b + sq) * inv2a];
                for t in roots {
                    if !ray_t.surrounds(t) || t >= best_t {
                        continue;
                    }
                    let y = r.origin.y + t * r.direction.y;
                    if y < self.y_min || y > self.y_max {
                        continue;
                    }
                    let p = r.at(t);
                    best_t = t;
                    best_normal = Some(
                        Vec3::new(p.x - self.center.x, 0.0, p.z - self.center.z) / self.radius,
                    );
                }
            }
        }

        if r.direction.y.abs() > 1e-9 {
            let caps = [
                (self.y_min, Vec3::new(0.0, -1.0, 0.0)),
                (self.y_max, Vec3::new(0.0, 1.0, 0.0)),
            ];
            for (y_cap, cap_normal) in caps {
                let t = (y_cap - r.origin.y) / r.direction.y;
                if !ray_t.surrounds(t) || t >= best_t {
                    continue;
                }
                let p = r.at(t);
                let dx = p.x - self.center.x;
                let dz = p.z - self.center.z;
                if dx * dx + dz * dz <= self.radius * self.radius {
                    best_t = t;
                    best_normal = Some(cap_normal);
                }
            }
        }

        let normal = best_normal?;
        let p = r.at(best_t);
        let mut rec = HitRecord::new(best_t, p, normal, self.mat);
        rec.set_face_normal(r);
        Some(rec)
    }
}
