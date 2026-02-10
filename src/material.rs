use crate::color::Color;
use crate::hittable::HitRecord;
use crate::ray::Ray;
use crate::util::random_f64;
use crate::vec3::Vec3;

#[derive(Clone, Copy)]
pub struct Lambertion {
    albedo: Color,
}

impl Lambertion {
    pub fn new(albedo: Color) -> Lambertion {
        Lambertion { albedo }
    }
}

#[derive(Clone, Copy)]
pub struct Metal {
    albedo: Color,
    fuzz: f64,
}

impl Metal {
    pub fn new(albedo: Color, fuzz: f64) -> Metal {
        Metal {
            albedo,
            fuzz: if fuzz < 1.0 { fuzz } else { 1.0 },
        }
    }
}

#[derive(Clone, Copy)]
pub struct Dialectric {
    refraction_index: f64,
}

impl Dialectric {
    pub fn new(refraction_index: f64) -> Dialectric {
        Dialectric { refraction_index }
    }
}

#[derive(Clone, Copy)]
pub struct DiffuseLight {
    emit: Color,
}

impl DiffuseLight {
    pub fn new(emit: Color) -> DiffuseLight {
        DiffuseLight { emit }
    }
}

#[derive(Clone, Copy)]
pub enum Material {
    Lambertian { albedo: Color },
    Metallic { albedo: Color, fuzz: f64 },
    Dielectric { refraction_index: f64 },
    DiffuseLight { emit: Color },
}

impl From<Lambertion> for Material {
    fn from(value: Lambertion) -> Self {
        Material::Lambertian {
            albedo: value.albedo,
        }
    }
}

impl From<Metal> for Material {
    fn from(value: Metal) -> Self {
        Material::Metallic {
            albedo: value.albedo,
            fuzz: value.fuzz,
        }
    }
}

impl From<Dialectric> for Material {
    fn from(value: Dialectric) -> Self {
        Material::Dielectric {
            refraction_index: value.refraction_index,
        }
    }
}

impl From<DiffuseLight> for Material {
    fn from(value: DiffuseLight) -> Self {
        Material::DiffuseLight { emit: value.emit }
    }
}

impl Material {
    fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
        let mut r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
        r0 = r0 * r0;
        r0 + (1.0 - r0) * ((1.0 - cosine).powf(5.0))
    }

    pub fn scatter(&self, r_in: &Ray, rec: &HitRecord) -> Option<(Color, Ray)> {
        match *self {
            Material::Lambertian { albedo } => {
                let mut scatter_direction = rec.normal + Vec3::random_unit_vector();
                if scatter_direction.near_zero() {
                    scatter_direction = rec.normal;
                }
                let scattered = Ray::new(rec.p, scatter_direction);
                Some((albedo, scattered))
            }
            Material::Metallic { albedo, fuzz } => {
                let mut reflected = Vec3::reflect(r_in.direction, rec.normal);
                reflected = reflected.unit_vector() + (fuzz * Vec3::random_unit_vector());
                let scattered = Ray::new(rec.p, reflected);
                if Vec3::dot(scattered.direction, rec.normal) > 0.0 {
                    Some((albedo, scattered))
                } else {
                    None
                }
            }
            Material::Dielectric { refraction_index } => {
                let attenuation = Color::new(1.0, 1.0, 1.0);
                let ri = if rec.front_face {
                    1.0 / refraction_index
                } else {
                    refraction_index
                };

                let unit_direction = r_in.direction.unit_vector();
                let cos_theta = Vec3::dot(-unit_direction, rec.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let cannot_refract = ri * sin_theta > 1.0;
                let direction = if cannot_refract || Material::reflectance(cos_theta, ri) > random_f64()
                {
                    Vec3::reflect(unit_direction, rec.normal)
                } else {
                    Vec3::refract(unit_direction, rec.normal, ri)
                };
                let scattered = Ray::new(rec.p, direction);
                Some((attenuation, scattered))
            }
            Material::DiffuseLight { .. } => None,
        }
    }

    pub fn emitted(&self) -> Color {
        match *self {
            Material::DiffuseLight { emit } => emit,
            _ => Color::new(0.0, 0.0, 0.0),
        }
    }
}
