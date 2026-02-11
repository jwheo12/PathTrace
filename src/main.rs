mod vec3;
mod color;
mod ray;
mod test;
mod hittable;
mod sphere;
mod hittable_list;
mod interval;
mod camera;
mod util;
mod material;

use camera::Camera;
use hittable_list::HittableList;
use sphere::Sphere;
use vec3::{Vec3,Point3};
use color::Color;
use material::{Dialectric, DiffuseLight, Lambertion, Metal};

fn main() {
    let mut world = HittableList::new();

    let room_half = 5.0;
    let wall_radius = 1000.0;

    let white = Lambertion::new(Color::new(0.73, 0.73, 0.73));
    let red = Lambertion::new(Color::new(0.65, 0.05, 0.05));
    let green = Lambertion::new(Color::new(0.12, 0.45, 0.15));
    let blue = Lambertion::new(Color::new(0.10, 0.18, 0.70));
    let glossy_black_floor = Metal::new(Color::new(0.05, 0.05, 0.06), 0.01);
    let warm_light = DiffuseLight::new(Color::new(36.0, 32.0, 28.0));

    // Cornell-box style room from large spheres.
    world.add(Sphere::new(
        Point3::new(0.0, -(wall_radius + room_half), 0.0),
        wall_radius,
        glossy_black_floor,
    )); // floor
    world.add(Sphere::new(
        Point3::new(0.0, wall_radius + room_half, 0.0),
        wall_radius,
        white,
    )); // ceiling
    world.add(Sphere::new(
        Point3::new(-(wall_radius + room_half), 0.0, 0.0),
        wall_radius,
        red,
    )); // left wall
    world.add(Sphere::new(
        Point3::new(wall_radius + room_half, 0.0, 0.0),
        wall_radius,
        green,
    )); // right wall
    world.add(Sphere::new(
        Point3::new(0.0, 0.0, -(wall_radius + room_half)),
        wall_radius,
        blue,
    )); // back wall

    // Practical area-like ceiling light.
    world.add(Sphere::new(Point3::new(0.0, 4.2, 0.0), 1.6, warm_light));

    // Hero objects.
    world.add(Sphere::new(
        Point3::new(-1.6, -4.0, -0.8),
        1.0,
        Dialectric::new(1.5),
    ));
    world.add(Sphere::new(
        Point3::new(1.5, -4.0, 0.4),
        1.0,
        Metal::new(Color::new(0.92, 0.94, 0.98), 0.01),
    ));
    world.add(Sphere::new(
        Point3::new(0.0, -4.25, -2.0),
        0.75,
        Lambertion::new(Color::new(0.72, 0.72, 0.70)),
    ));

    let mut cam = Camera::new();


    cam.aspect_ratio= 3.0/4.0;
    cam.image_width = 2160;
    cam.samples_per_pixel = 16384;
    cam.max_depth = 200;

    cam.vfov = 38.0;
    cam.lookfrom = Point3::new(0.0, 0.3, 14.0);
    cam.lookat = Point3::new(0.0, -3.6, 0.0);
    cam.vup = Vec3::new(0.0,1.0,0.0);

    cam.defocus_angle = 0.0;
    cam.focus_dist = 14.0;

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
