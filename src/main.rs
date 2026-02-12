mod box3;
mod camera;
mod color;
mod cylinder;
mod hittable;
mod hittable_list;
mod interval;
mod material;
mod plane;
mod ray;
mod sphere;
mod test;
mod util;
mod vec3;

use box3::Box3;
use camera::Camera;
use color::Color;
use cylinder::Cylinder;
use hittable_list::HittableList;
use material::{Dialectric, DiffuseLight, Lambertion, Material, Metal};
use plane::Plane;
use sphere::Sphere;
use std::f64::consts::PI;
use vec3::{Point3, Vec3};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn hash01(seed: u32) -> f64 {
    hash_u32(seed) as f64 / u32::MAX as f64
}

fn hash_range(seed: u32, min: f64, max: f64) -> f64 {
    min + (max - min) * hash01(seed)
}

fn build_cinematic_world() -> HittableList {
    let mut world = HittableList::new();

    let floor_stone = Lambertion::new(Color::new(0.70, 0.69, 0.66));
    let grass = Lambertion::new(Color::new(0.28, 0.52, 0.26));
    let foliage_dark = Lambertion::new(Color::new(0.14, 0.34, 0.16));
    let foliage_light = Lambertion::new(Color::new(0.20, 0.45, 0.20));
    let white_plaster = Lambertion::new(Color::new(0.82, 0.80, 0.78));
    let terracotta = Lambertion::new(Color::new(0.66, 0.34, 0.22));
    let glass = Dialectric::new(1.5);
    let chrome = Metal::new(Color::new(0.95, 0.97, 0.99), 0.0);
    let brushed = Metal::new(Color::new(0.74, 0.78, 0.82), 0.06);
    let copper = Metal::new(Color::new(0.88, 0.56, 0.30), 0.03);

    let sky_light = DiffuseLight::new(Color::new(0.66, 0.80, 1.12));
    let sun_light = DiffuseLight::new(Color::new(118.0, 102.0, 76.0));
    let fill_light = DiffuseLight::new(Color::new(4.5, 6.5, 9.0));

    // Sky and sunlight.
    world.add(Sphere::new(Point3::new(0.0, 0.0, 0.0), 5200.0, sky_light));
    world.add(Sphere::new(
        Point3::new(185.0, 170.0, -250.0),
        56.0,
        sun_light,
    ));

    // Large ground layers.
    world.add_plane(Plane::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        floor_stone,
    ));
    world.add(Sphere::new(Point3::new(0.0, -1400.0, 0.0), 1400.0, grass));

    // Center hero cluster (mix of shapes).
    world.add(Sphere::new(Point3::new(0.0, 1.35, -1.4), 1.35, glass));
    world.add_cylinder(Cylinder::new(
        Point3::new(-2.8, 0.0, -0.2),
        0.86,
        0.0,
        2.15,
        copper,
    ));
    world.add_box(Box3::new(
        Point3::new(2.0, 0.0, -1.0),
        Point3::new(3.7, 2.2, 0.7),
        chrome,
    ));
    world.add_box(Box3::new(
        Point3::new(-0.8, 0.0, 1.8),
        Point3::new(0.9, 1.35, 3.2),
        brushed,
    ));
    world.add_cylinder(Cylinder::new(
        Point3::new(1.8, 0.0, 2.5),
        0.48,
        0.0,
        1.10,
        white_plaster,
    ));
    world.add(Sphere::new(Point3::new(0.0, 1.1, -14.0), 1.0, fill_light));

    // Pavilion columns and roofline.
    for i in -7..=7 {
        let x = i as f64 * 1.1;
        world.add_cylinder(Cylinder::new(
            Point3::new(x, 0.0, -10.5),
            0.33,
            0.0,
            4.0,
            white_plaster,
        ));
        world.add(Sphere::new(Point3::new(x, 4.2, -10.5), 0.24, terracotta));
    }
    for i in -18..=18 {
        let x = i as f64 * 0.48;
        world.add_box(Box3::new(
            Point3::new(x - 0.17, 4.25, -11.0),
            Point3::new(x + 0.17, 4.65, -10.0),
            terracotta,
        ));
    }

    // Distant background masses for depth.
    for i in 0..70 {
        let a = i as f64 / 70.0 * 2.0 * PI;
        world.add(Sphere::new(
            Point3::new(
                44.0 * a.cos(),
                -11.0 + 1.8 * (a * 3.0).sin(),
                -56.0 + 15.0 * a.sin(),
            ),
            8.5 + 3.8 * (a * 4.0).sin().abs(),
            if i % 2 == 0 {
                foliage_dark
            } else {
                foliage_light
            },
        ));
    }

    // Thousands of mixed objects (boxes/cylinders/spheres/lights).
    let grid_radius = env_usize("PATHTRACE_GRID_RADIUS", 28) as i32;
    let spacing = 0.72;
    for gx in -grid_radius..=grid_radius {
        for gz in -grid_radius..=grid_radius {
            let x = gx as f64 * spacing;
            let z = gz as f64 * spacing - 2.2;
            if x * x + z * z < 42.0 {
                continue;
            }

            let seed = ((gx + 3000) as u32)
                .wrapping_mul(127_481)
                .wrapping_add(((gz + 3000) as u32).wrapping_mul(596_057));
            let jx = hash_range(seed ^ 0x1111, -0.24, 0.24);
            let jz = hash_range(seed ^ 0x2222, -0.24, 0.24);
            let px = x + jx;
            let pz = z + jz;

            let pick = hash01(seed ^ 0x3333);
            if pick < 0.44 {
                let w = hash_range(seed ^ 0x4444, 0.14, 0.42);
                let d = hash_range(seed ^ 0x5555, 0.14, 0.42);
                let h = hash_range(seed ^ 0x6666, 0.22, 2.10);
                let c = Color::new(
                    hash_range(seed ^ 0x7100, 0.20, 0.92),
                    hash_range(seed ^ 0x7200, 0.20, 0.92),
                    hash_range(seed ^ 0x7300, 0.20, 0.92),
                );
                let mat: Material = if hash01(seed ^ 0x7777) < 0.12 {
                    Metal::new(c, hash_range(seed ^ 0x7888, 0.0, 0.16)).into()
                } else {
                    Lambertion::new(c).into()
                };
                world.add_box(Box3::new(
                    Point3::new(px - w, 0.0, pz - d),
                    Point3::new(px + w, h, pz + d),
                    mat,
                ));
            } else if pick < 0.82 {
                let r = hash_range(seed ^ 0x8888, 0.09, 0.24);
                let h = hash_range(seed ^ 0x9999, 0.24, 2.4);
                let c = Color::new(
                    hash_range(seed ^ 0xA100, 0.22, 0.95),
                    hash_range(seed ^ 0xA200, 0.22, 0.95),
                    hash_range(seed ^ 0xA300, 0.22, 0.95),
                );
                let mat: Material = if hash01(seed ^ 0xAAAA) < 0.20 {
                    Metal::new(c, hash_range(seed ^ 0xBBBB, 0.0, 0.14)).into()
                } else {
                    Lambertion::new(c).into()
                };
                world.add_cylinder(Cylinder::new(Point3::new(px, 0.0, pz), r, 0.0, h, mat));
            } else if pick < 0.96 {
                let r = hash_range(seed ^ 0xCCCC, 0.10, 0.30);
                if hash01(seed ^ 0xCDCD) < 0.45 {
                    world.add(Sphere::new(Point3::new(px, r, pz), r, glass));
                } else {
                    world.add(Sphere::new(
                        Point3::new(px, r, pz),
                        r,
                        Metal::new(
                            Color::new(
                                hash_range(seed ^ 0xDD01, 0.60, 0.98),
                                hash_range(seed ^ 0xDD02, 0.60, 0.98),
                                hash_range(seed ^ 0xDD03, 0.60, 0.98),
                            ),
                            hash_range(seed ^ 0xDD04, 0.0, 0.08),
                        ),
                    ));
                }
            } else {
                let r = hash_range(seed ^ 0xEEEE, 0.08, 0.20);
                world.add(Sphere::new(
                    Point3::new(px, r + 0.06, pz),
                    r,
                    DiffuseLight::new(Color::new(
                        hash_range(seed ^ 0xF100, 3.2, 10.0),
                        hash_range(seed ^ 0xF200, 2.8, 8.8),
                        hash_range(seed ^ 0xF300, 2.4, 7.6),
                    )),
                ));
            }
        }
    }

    world
}

fn main() {
    let world = build_cinematic_world();
    let sphere_count = world.spheres().len();
    let plane_count = world.planes().len();
    let box_count = world.boxes().len();
    let cylinder_count = world.cylinders().len();
    let object_count = sphere_count + plane_count + box_count + cylinder_count;

    let mut cam = Camera::new();
    cam.aspect_ratio = 2.39;
    cam.image_width = env_usize("PATHTRACE_WIDTH", 2560);
    cam.samples_per_pixel = env_usize("PATHTRACE_SPP", 128);
    cam.max_depth = env_usize("PATHTRACE_MAX_DEPTH", 96);

    cam.vfov = 31.0;
    cam.lookfrom = Point3::new(-14.0, 7.2, 26.0);
    cam.lookat = Point3::new(0.0, 1.8, -2.0);
    cam.vup = Vec3::new(0.0, 1.0, 0.0);
    cam.defocus_angle = 0.06;
    cam.focus_dist = 28.0;

    eprintln!(
        "cinematic still | objects={} (sphere={}, plane={}, box={}, cylinder={}) | spp={} | width={}",
        object_count,
        sphere_count,
        plane_count,
        box_count,
        cylinder_count,
        cam.samples_per_pixel,
        cam.image_width
    );

    if std::env::var("PATHTRACE_CPU_ONLY").is_ok() {
        cam.render(&world);
    } else if std::env::var("PATHTRACE_GPU_ONLY").is_ok() {
        if let Err(err) = cam.render_gpu(&world) {
            eprintln!("\nGPU rendering failed: {err}");
            eprintln!("Falling back to CPU renderer...");
            cam.render(&world);
        }
    } else if let Err(err) = cam.render_hybrid(&world) {
        eprintln!("\nHybrid rendering failed: {err}");
        eprintln!("Falling back to GPU renderer...");
        if let Err(gpu_err) = cam.render_gpu(&world) {
            eprintln!("\nGPU rendering failed: {gpu_err}");
            eprintln!("Falling back to CPU renderer...");
            cam.render(&world);
        }
    }
}
