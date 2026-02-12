mod camera;
mod color;
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

use camera::Camera;
use color::Color;
use hittable_list::HittableList;
use material::{Dialectric, DiffuseLight, Lambertion, Metal};
use plane::Plane;
use sphere::Sphere;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;
use vec3::{Point3, Vec3};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
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

fn build_cinematic_world(frame_idx: usize, total_frames: usize) -> HittableList {
    let mut world = HittableList::new();
    let frame_t = if total_frames <= 1 {
        0.0
    } else {
        frame_idx as f64 / (total_frames - 1) as f64
    };

    let grass = Lambertion::new(Color::new(0.28, 0.52, 0.26));
    let plaza_stone = Lambertion::new(Color::new(0.72, 0.70, 0.66));
    let brushed_steel = Metal::new(Color::new(0.78, 0.82, 0.86), 0.03);
    let mirror_chrome = Metal::new(Color::new(0.95, 0.97, 0.99), 0.0);
    let polished_copper = Metal::new(Color::new(0.90, 0.58, 0.30), 0.02);

    let sky_light = DiffuseLight::new(Color::new(0.62, 0.76, 1.08));
    let sun_light = DiffuseLight::new(Color::new(110.0, 98.0, 72.0));
    let fill_light = DiffuseLight::new(Color::new(4.0, 6.0, 9.0));

    // Sky + sun + ground.
    world.add(Sphere::new(Point3::new(0.0, 0.0, 0.0), 5200.0, sky_light));
    world.add_plane(Plane::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        plaza_stone,
    ));
    world.add(Sphere::new(Point3::new(0.0, -1200.0, 0.0), 1200.0, grass));

    let sun_theta = (0.18 + 0.28 * frame_t) * PI;
    world.add(Sphere::new(
        Point3::new(
            220.0 * sun_theta.cos(),
            165.0 + 25.0 * frame_t,
            -240.0 + 80.0 * sun_theta.sin(),
        ),
        52.0,
        sun_light,
    ));

    // Hero objects.
    world.add(Sphere::new(
        Point3::new(0.0, 1.35, -1.1),
        1.35,
        Dialectric::new(1.5),
    ));
    world.add(Sphere::new(
        Point3::new(2.8, 1.05, 0.45),
        1.05,
        mirror_chrome,
    ));
    world.add(Sphere::new(
        Point3::new(-2.7, 1.10, 0.15),
        1.10,
        polished_copper,
    ));
    world.add(Sphere::new(
        Point3::new(0.7, 0.58, 2.7),
        0.58,
        brushed_steel,
    ));
    world.add(Sphere::new(Point3::new(0.0, 1.2, -16.0), 1.1, fill_light));

    // Glass-light arches around center.
    for i in 0..96 {
        let a = i as f64 / 96.0 * 2.0 * PI;
        let r = 8.2 + 0.8 * (3.0 * a + frame_t * 2.0 * PI).sin();
        let x = r * a.cos();
        let z = r * a.sin() - 1.2;
        let y = 0.24 + 0.10 * (a * 4.0).sin().abs();
        if i % 4 == 0 {
            world.add(Sphere::new(
                Point3::new(x, y, z),
                0.20,
                DiffuseLight::new(Color::new(8.5, 7.5, 6.0)),
            ));
        } else {
            world.add(Sphere::new(
                Point3::new(x, y, z),
                0.20,
                Dialectric::new(1.5),
            ));
        }
    }

    // Thousands of small objects: micro city field.
    let grid_radius = env_usize("PATHTRACE_GRID_RADIUS", 24) as i32;
    let spacing = env_f64("PATHTRACE_GRID_SPACING", 0.56);
    for gx in -grid_radius..=grid_radius {
        for gz in -grid_radius..=grid_radius {
            let x = gx as f64 * spacing;
            let z = gz as f64 * spacing;
            if x * x + z * z < 22.0 {
                continue;
            }

            let seed = ((gx + 2000) as u32)
                .wrapping_mul(93_113)
                .wrapping_add(((gz + 2000) as u32).wrapping_mul(689_287_499))
                .wrapping_add((frame_idx as u32).wrapping_mul(977));

            let jitter_x = hash_range(seed ^ 0xA2D3, -0.22, 0.22);
            let jitter_z = hash_range(seed ^ 0x7B1F, -0.22, 0.22);
            let radius = hash_range(seed ^ 0xE91C, 0.06, 0.18);
            let px = x + jitter_x;
            let pz = z + jitter_z - 1.8;
            let py = radius;
            let mat_pick = hash01(seed ^ 0x51FA);

            if mat_pick < 0.56 {
                let c = Color::new(
                    hash_range(seed ^ 0x1111, 0.18, 0.92),
                    hash_range(seed ^ 0x2222, 0.18, 0.92),
                    hash_range(seed ^ 0x3333, 0.18, 0.92),
                );
                world.add(Sphere::new(
                    Point3::new(px, py, pz),
                    radius,
                    Lambertion::new(c),
                ));
            } else if mat_pick < 0.86 {
                let c = Color::new(
                    hash_range(seed ^ 0x4444, 0.58, 0.98),
                    hash_range(seed ^ 0x5555, 0.58, 0.98),
                    hash_range(seed ^ 0x6666, 0.58, 0.98),
                );
                let fuzz = hash_range(seed ^ 0x7777, 0.0, 0.10);
                world.add(Sphere::new(
                    Point3::new(px, py, pz),
                    radius,
                    Metal::new(c, fuzz),
                ));
            } else if mat_pick < 0.96 {
                world.add(Sphere::new(
                    Point3::new(px, py, pz),
                    radius,
                    Dialectric::new(1.5),
                ));
            } else {
                let emit = Color::new(
                    hash_range(seed ^ 0x8888, 3.5, 9.5),
                    hash_range(seed ^ 0x9999, 3.0, 8.0),
                    hash_range(seed ^ 0xAAAA, 2.8, 7.4),
                );
                world.add(Sphere::new(
                    Point3::new(px, py + 0.04, pz),
                    radius,
                    DiffuseLight::new(emit),
                ));
            }
        }
    }

    // Distant background mass.
    for i in 0..60 {
        let a = i as f64 / 60.0 * 2.0 * PI;
        world.add(Sphere::new(
            Point3::new(
                42.0 * a.cos(),
                -10.0 + 1.8 * (a * 3.0).sin(),
                -58.0 + 14.0 * a.sin(),
            ),
            9.5 + 3.5 * (a * 5.0).sin().abs(),
            Lambertion::new(Color::new(0.16, 0.28, 0.14)),
        ));
    }

    world
}

fn build_camera(frame_idx: usize, total_frames: usize) -> Camera {
    let frame_t = if total_frames <= 1 {
        0.0
    } else {
        frame_idx as f64 / (total_frames - 1) as f64
    };
    let orbit_span_deg = env_f64("PATHTRACE_ORBIT_DEG", 62.0);
    let orbit_span = orbit_span_deg.to_radians();
    let theta = (-0.5 + frame_t) * orbit_span;
    let radius = env_f64("PATHTRACE_CAM_RADIUS", 21.5);
    let base_height = env_f64("PATHTRACE_CAM_HEIGHT", 3.2);

    let mut cam = Camera::new();
    cam.aspect_ratio = env_f64("PATHTRACE_ASPECT", 16.0 / 9.0);
    cam.image_width = env_usize("PATHTRACE_WIDTH", 2560);
    cam.samples_per_pixel = env_usize("PATHTRACE_SPP", if total_frames > 1 { 64 } else { 128 });
    cam.max_depth = env_usize("PATHTRACE_MAX_DEPTH", 96);

    cam.vfov = env_f64("PATHTRACE_VFOV", 34.0);
    cam.lookfrom = Point3::new(
        radius * theta.sin(),
        base_height + 0.55 * (frame_t * 2.0 * PI).sin(),
        radius * theta.cos() + 6.2,
    );
    cam.lookat = Point3::new(0.0, 1.1 + 0.25 * (frame_t * 2.0 * PI).cos(), -1.4);
    cam.vup = Vec3::new(0.0, 1.0, 0.0);

    cam.defocus_angle = env_f64("PATHTRACE_DEFOCUS", 0.07);
    cam.focus_dist = env_f64("PATHTRACE_FOCUS_DIST", 22.0);
    cam
}

fn render_with_mode(cam: &mut Camera, world: &HittableList) -> Result<(), String> {
    if std::env::var_os("PATHTRACE_CPU_ONLY").is_some() {
        cam.render(world);
        Ok(())
    } else if std::env::var_os("PATHTRACE_GPU_ONLY").is_some() {
        if let Err(err) = cam.render_gpu(world) {
            eprintln!("\nGPU rendering failed: {err}");
            eprintln!("Falling back to CPU renderer...");
            cam.render(world);
        }
        Ok(())
    } else if let Err(err) = cam.render_hybrid(world) {
        eprintln!("\nHybrid rendering failed: {err}");
        eprintln!("Falling back to GPU renderer...");
        if let Err(gpu_err) = cam.render_gpu(world) {
            eprintln!("\nGPU rendering failed: {gpu_err}");
            eprintln!("Falling back to CPU renderer...");
            cam.render(world);
        }
        Ok(())
    } else {
        Ok(())
    }
}

fn move_outputs_to_frame(output_dir: &str, frame_idx: usize) -> Result<(), String> {
    let ldr_dst = format!("{output_dir}/frame_{frame_idx:04}_16bit.ppm");
    let hdr_dst = format!("{output_dir}/frame_{frame_idx:04}_hdr.pfm");
    if Path::new(&ldr_dst).exists() {
        fs::remove_file(&ldr_dst).map_err(|e| format!("Failed removing {ldr_dst}: {e}"))?;
    }
    if Path::new(&hdr_dst).exists() {
        fs::remove_file(&hdr_dst).map_err(|e| format!("Failed removing {hdr_dst}: {e}"))?;
    }
    fs::rename("image_16bit.ppm", &ldr_dst).map_err(|e| format!("Failed moving LDR frame: {e}"))?;
    fs::rename("image_hdr.pfm", &hdr_dst).map_err(|e| format!("Failed moving HDR frame: {e}"))?;
    Ok(())
}

fn maybe_make_video(output_dir: &str, frames: usize) {
    if std::env::var_os("PATHTRACE_MAKE_MP4").is_none() {
        return;
    }
    let fps = env_usize("PATHTRACE_ANIM_FPS", 24);
    let output_mp4 = format!("{output_dir}/cinematic.mp4");
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg(fps.to_string())
        .arg("-i")
        .arg(format!("{output_dir}/frame_%04d_16bit.ppm"))
        .arg("-frames:v")
        .arg(frames.to_string())
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&output_mp4)
        .status();

    match status {
        Ok(s) if s.success() => eprintln!("MP4 generated: {output_mp4}"),
        Ok(s) => eprintln!("ffmpeg failed with status: {s}"),
        Err(e) => eprintln!("ffmpeg not available or failed to launch: {e}"),
    }
}

fn main() {
    let frames = env_usize("PATHTRACE_ANIM_FRAMES", 1);
    let output_dir = std::env::var("PATHTRACE_ANIM_DIR").unwrap_or_else(|_| "frames".to_string());

    if frames <= 1 {
        let world = build_cinematic_world(0, 1);
        let object_count = world.spheres().len() + world.planes().len();
        let mut cam = build_camera(0, 1);
        eprintln!(
            "Single frame render | objects={} (spheres={}, planes={}) | spp={}",
            object_count,
            world.spheres().len(),
            world.planes().len(),
            cam.samples_per_pixel
        );
        if let Err(err) = render_with_mode(&mut cam, &world) {
            eprintln!("Rendering failed: {err}");
        }
        return;
    }

    if let Err(e) = fs::create_dir_all(&output_dir) {
        eprintln!("Failed to create output directory {output_dir}: {e}");
        return;
    }

    let all_start = Instant::now();
    for frame_idx in 0..frames {
        let frame_start = Instant::now();
        let world = build_cinematic_world(frame_idx, frames);
        let object_count = world.spheres().len() + world.planes().len();
        let mut cam = build_camera(frame_idx, frames);

        eprintln!(
            "\n[Frame {}/{}] objects={} (spheres={}, planes={}) spp={}",
            frame_idx + 1,
            frames,
            object_count,
            world.spheres().len(),
            world.planes().len(),
            cam.samples_per_pixel
        );

        if let Err(err) = render_with_mode(&mut cam, &world) {
            eprintln!("Frame {} render failed: {err}", frame_idx);
            return;
        }
        if let Err(err) = move_outputs_to_frame(&output_dir, frame_idx) {
            eprintln!("Frame {} output move failed: {err}", frame_idx);
            return;
        }
        eprintln!(
            "[Frame {}/{}] done in {:.1}s",
            frame_idx + 1,
            frames,
            frame_start.elapsed().as_secs_f64()
        );
    }
    eprintln!(
        "\nAnimation render complete: {} frames in {:.1}s",
        frames,
        all_start.elapsed().as_secs_f64()
    );
    maybe_make_video(&output_dir, frames);
}
