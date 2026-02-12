mod camera;
mod color;
mod hittable;
mod hittable_list;
mod interval;
mod material;
mod ray;
mod sphere;
mod test;
mod util;
mod vec3;

use camera::Camera;
use color::Color;
use hittable_list::HittableList;
use material::{Dialectric, DiffuseLight, Lambertion, Metal};
use sphere::Sphere;
use vec3::{Point3, Vec3};

fn main() {
    let mut world = HittableList::new();

    let room_half = 6.5;
    let wall_radius = 1200.0;

    let ceiling_stone = Lambertion::new(Color::new(0.74, 0.72, 0.70));
    let left_panel = Lambertion::new(Color::new(0.36, 0.22, 0.16));
    let right_panel = Lambertion::new(Color::new(0.18, 0.24, 0.34));
    let back_stage = Lambertion::new(Color::new(0.08, 0.10, 0.18));
    let piano_black_floor = Metal::new(Color::new(0.02, 0.02, 0.025), 0.002);

    let marble = Lambertion::new(Color::new(0.83, 0.80, 0.76));
    let velvet = Lambertion::new(Color::new(0.56, 0.06, 0.08));
    let trim_gold = Metal::new(Color::new(0.88, 0.74, 0.35), 0.03);
    let chrome = Metal::new(Color::new(0.95, 0.97, 0.99), 0.0);

    let chandelier_light = DiffuseLight::new(Color::new(28.0, 24.0, 20.0));
    let sconce_light = DiffuseLight::new(Color::new(18.0, 12.0, 8.0));
    let cool_fill = DiffuseLight::new(Color::new(8.0, 12.0, 18.0));

    // New cinematic lounge shell.
    world.add(Sphere::new(
        Point3::new(0.0, -(wall_radius + room_half), 0.0),
        wall_radius,
        piano_black_floor,
    ));
    world.add(Sphere::new(
        Point3::new(0.0, wall_radius + room_half, 0.0),
        wall_radius,
        ceiling_stone,
    ));
    world.add(Sphere::new(
        Point3::new(-(wall_radius + room_half), 0.0, 0.0),
        wall_radius,
        left_panel,
    ));
    world.add(Sphere::new(
        Point3::new(wall_radius + room_half, 0.0, 0.0),
        wall_radius,
        right_panel,
    ));
    world.add(Sphere::new(
        Point3::new(0.0, 0.0, -(wall_radius + room_half)),
        wall_radius,
        back_stage,
    ));

    // Ceiling chandeliers.
    for &z in &[-6.0, 0.0, 6.0] {
        world.add(Sphere::new(Point3::new(0.0, 5.6, z), 0.95, chandelier_light));
        world.add(Sphere::new(Point3::new(-0.95, 5.1, z), 0.38, Dialectric::new(1.5)));
        world.add(Sphere::new(Point3::new(0.95, 5.1, z), 0.38, Dialectric::new(1.5)));
        world.add(Sphere::new(Point3::new(0.0, 4.75, z), 0.30, Dialectric::new(1.5)));
        world.add(Sphere::new(Point3::new(0.0, 6.25, z), 0.22, trim_gold));
    }

    // Side wall sconces.
    for &x in &[-5.2, 5.2] {
        for &z in &[-6.0, -2.2, 2.2, 6.0] {
            world.add(Sphere::new(Point3::new(x, 0.85, z), 0.33, sconce_light));
            world.add(Sphere::new(Point3::new(x * 0.97, 0.72, z), 0.24, trim_gold));
        }
    }

    // Decorative side columns.
    for &x in &[-4.4, 4.4] {
        for &z in &[-6.8, -3.4, 0.0, 3.4, 6.8] {
            world.add(Sphere::new(Point3::new(x, -5.95, z), 0.52, marble));
            world.add(Sphere::new(Point3::new(x, -4.95, z), 0.52, marble));
            world.add(Sphere::new(Point3::new(x, -3.95, z), 0.52, marble));
            world.add(Sphere::new(Point3::new(x, -3.10, z), 0.30, trim_gold));
        }
    }

    // Center stage hero objects.
    world.add(Sphere::new(Point3::new(0.0, -5.85, 0.0), 1.15, marble));
    world.add(Sphere::new(Point3::new(0.0, -4.10, 0.0), 1.35, Dialectric::new(1.5)));
    world.add(Sphere::new(Point3::new(-2.2, -4.6, -1.0), 0.95, chrome));
    world.add(Sphere::new(Point3::new(2.5, -4.7, 1.2), 0.85, trim_gold));
    world.add(Sphere::new(
        Point3::new(1.0, -5.5, -2.8),
        0.45,
        Lambertion::new(Color::new(0.24, 0.34, 0.78)),
    ));
    world.add(Sphere::new(
        Point3::new(-1.2, -5.45, 2.6),
        0.48,
        Lambertion::new(Color::new(0.72, 0.22, 0.18)),
    ));

    // Velvet bead-like center runway.
    for i in -10..=10 {
        let z = i as f64 * 0.75;
        let r = if i % 3 == 0 { 0.16 } else { 0.14 };
        world.add(Sphere::new(Point3::new(0.0, -6.42, z), r, velvet));
    }

    // Soft cool fill from rear.
    world.add(Sphere::new(Point3::new(0.0, -0.6, -7.9), 0.62, cool_fill));

    let mut cam = Camera::new();

    cam.aspect_ratio = 2.39;
    cam.image_width = 3840;
    cam.samples_per_pixel = 32768;
    cam.max_depth = 128;

    cam.vfov = 33.0;
    cam.lookfrom = Point3::new(0.0, -2.8, 24.0);
    cam.lookat = Point3::new(0.0, -4.6, 0.0);
    cam.vup = Vec3::new(0.0, 1.0, 0.0);

    cam.defocus_angle = 0.18;
    cam.focus_dist = 24.0;

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
