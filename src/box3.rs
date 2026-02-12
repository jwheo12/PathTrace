use crate::hittable::{HitRecord, Hittable};
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::vec3::{Point3, Vec3};

#[derive(Clone, Copy)]
pub struct Box3 {
    pub min: Point3,
    pub max: Point3,
    pub mat: Material,
}

impl Box3 {
    pub fn new(min: Point3, max: Point3, mat: impl Into<Material>) -> Box3 {
        Box3 {
            min,
            max,
            mat: mat.into(),
        }
    }
}

impl Hittable for Box3 {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let mut t_enter = f64::NEG_INFINITY;
        let mut t_exit = f64::INFINITY;

        let axes = [
            (r.origin.x, r.direction.x, self.min.x, self.max.x),
            (r.origin.y, r.direction.y, self.min.y, self.max.y),
            (r.origin.z, r.direction.z, self.min.z, self.max.z),
        ];

        for (origin, dir, minv, maxv) in axes {
            if dir.abs() < 1e-9 {
                if origin < minv || origin > maxv {
                    return None;
                }
                continue;
            }

            let inv = 1.0 / dir;
            let mut t0 = (minv - origin) * inv;
            let mut t1 = (maxv - origin) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_enter = t_enter.max(t0);
            t_exit = t_exit.min(t1);
            if t_exit < t_enter {
                return None;
            }
        }

        let t = if ray_t.surrounds(t_enter) {
            t_enter
        } else if ray_t.surrounds(t_exit) {
            t_exit
        } else {
            return None;
        };

        let p = r.at(t);
        let eps = 1e-5;
        let normal = if (p.x - self.min.x).abs() < eps {
            Vec3::new(-1.0, 0.0, 0.0)
        } else if (p.x - self.max.x).abs() < eps {
            Vec3::new(1.0, 0.0, 0.0)
        } else if (p.y - self.min.y).abs() < eps {
            Vec3::new(0.0, -1.0, 0.0)
        } else if (p.y - self.max.y).abs() < eps {
            Vec3::new(0.0, 1.0, 0.0)
        } else if (p.z - self.min.z).abs() < eps {
            Vec3::new(0.0, 0.0, -1.0)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };

        let mut rec = HitRecord::new(t, p, normal, self.mat);
        rec.set_face_normal(r);
        Some(rec)
    }
}
