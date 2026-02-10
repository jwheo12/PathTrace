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
use material::{Dialectric, Lambertion, Metal};
use util::{random_f64,random_f64_range};
fn main() {
    let mut world = HittableList::new();
    let ground_material = Lambertion::new(Color::new(0.5,0.5,0.5));
    world.add(Sphere::new(Point3::new(0.0,-1000.0,0.0), 1000.0, ground_material));
    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_f64();
            let center = Point3::new(a as f64 + 0.9*random_f64(),
                                            0.2,
                                            b as f64 + 0.9*random_f64());
            if (center-Point3::new(4.0,0.2,0.0)).length() >0.9 {
                if choose_mat < 0.8 {
                    // Shiny silver-like metals: neutral/cool albedo + very low fuzz.
                    let albedo = Color::new(
                        random_f64_range(0.80, 0.95),
                        random_f64_range(0.82, 0.97),
                        random_f64_range(0.86, 1.00),
                    );
                    let fuzz = random_f64_range(0.0,0.03);
                    let sphere_material = Metal::new(albedo, fuzz);
                    world.add(Sphere::new(center, 0.2, sphere_material));
                } else {
                    let sphere_material = Dialectric::new(1.5);
                    world.add(Sphere::new(center, 0.2, sphere_material));

                }
            }
        }
    }
    let material1 = Dialectric::new(1.5);
    world.add(Sphere::new(Point3::new(0.0,1.0,0.0),1.0,material1));

    let material2 = Lambertion::new(Color::new(0.4,0.2,0.1));
    world.add(Sphere::new(Point3::new(-4.0,1.0,0.0),1.0,material2));

    let material3 = Metal::new(Color::new(0.92,0.94,0.98),0.0);
    world.add(Sphere::new(Point3::new(4.0,1.0,0.0),1.0,material3));

    let mut cam = Camera::new();


    cam.aspect_ratio=16.0/9.0;
    cam.image_width = 3840;
    cam.samples_per_pixel = 4096;
    cam.max_depth = 200;

    cam.vfov = 20.0;
    cam.lookfrom = Point3::new(13.0,2.0,3.0);
    cam.lookat = Point3::new(0.0,0.0,0.0);
    cam.vup = Vec3::new(0.0,1.0,0.0);

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

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
