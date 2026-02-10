use bytemuck::{Pod, Zeroable};
use crate::color::{write_color_hdr, write_color_ldr16, Color};
use crate::hittable::Hittable;
use crate::hittable_list::HittableList;
use crate::interval::Interval;
use crate::material::Material;
use crate::ray::Ray;
use crate::test::Point3;
use crate::util::random_f64;
use crate::vec3::Vec3;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const GPU_SHADER: &str = include_str!("pathtrace.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    dims: [u32; 4],     // width, height, samples, max_depth
    scene: [u32; 4],    // sphere_count, reserved...
    center: [f32; 4],
    pixel00: [f32; 4],
    delta_u: [f32; 4],
    delta_v: [f32; 4],
    defocus_u: [f32; 4], // xyz + enabled flag in w
    defocus_v: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSphere {
    center_radius: [f32; 4], // xyz + radius
    material: [f32; 4],      // rgb + (fuzz or refraction_index)
    meta: [f32; 4],          // x = material kind (0 lambert, 1 metal, 2 dielectric)
}

impl GpuSphere {
    fn dummy() -> Self {
        Self {
            center_radius: [0.0, -10000.0, 0.0, 1.0],
            material: [0.0, 0.0, 0.0, 0.0],
            meta: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

#[derive(Default, Clone)]
pub struct Camera {
    pub aspect_ratio : f64,
    pub image_width : usize,
    pub samples_per_pixel: usize,
    pub max_depth : usize,
    pub vfov : f64,
    pub lookfrom : Point3,
    pub lookat : Point3,
    pub vup : Vec3,
    pub defocus_angle : f64,
    pub focus_dist : f64,

    image_height : usize,
    center : Point3,
    pixel00_loc : Point3,
    pixel_delta_u : Vec3,
    pixel_delta_v : Vec3,
    pixel_samples_scale : f64,
    u : Vec3,
    v : Vec3,
    w : Vec3,
    defocus_disk_u : Vec3,
    defocus_disk_v : Vec3
}

impl Camera {
    pub fn new() -> Camera{
        Camera {
            aspect_ratio:1.0,
            image_width:100,
            samples_per_pixel: 10,
            max_depth : 10,
            vfov : 90.0,
            lookfrom : Point3::new(0.0,0.0,0.0),
            lookat : Point3::new(0.0,0.0,-1.0),
            vup : Vec3::new(0.0,1.0,0.0),
            ..Default::default()
        }
    }

    pub fn render(&mut self, world : & impl Hittable) {
        self.initialize();
        let rows = self.render_cpu_rows(world, Some("Scanlines remaining"), None);
        self.write_outputs(&rows).unwrap();
        eprint!("\rDone                                  \n");
    }

    pub fn render_gpu(&mut self, world: &HittableList) -> Result<(), String> {
        self.initialize();
        let rows = self.render_gpu_rows(world, true, 0, true)?;
        eprint!("\rGPU done, writing image files...                     ");
        self.write_outputs(&rows)
            .map_err(|e| format!("Failed to write rendered images: {e}"))?;
        eprint!("\rDone                                  \n");
        Ok(())
    }

    pub fn render_hybrid(&mut self, world: &HittableList) -> Result<(), String> {
        self.initialize();

        let total_samples = self.samples_per_pixel.max(1);
        if total_samples <= 1 {
            return self.render_gpu(world);
        }

        let chunk_samples = Camera::env_usize("PATHTRACE_HYBRID_CHUNK_SPP")
            .unwrap_or(64)
            .clamp(1, total_samples);

        let default_cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(2).max(1))
            .unwrap_or(1);
        let cpu_threads = Camera::env_usize("PATHTRACE_HYBRID_CPU_THREADS")
            .unwrap_or(default_cpu_threads);

        eprintln!(
            "\nHybrid render (dynamic): total {} spp, chunk {} spp, CPU threads {}",
            total_samples, chunk_samples, cpu_threads
        );

        let zero_row = vec![Color::new(0.0, 0.0, 0.0); self.image_width];
        let mut cpu_sum_rows = vec![zero_row.clone(); self.image_height];

        let mut gpu_cam = self.clone();
        gpu_cam.initialize();

        let mut cpu_cam = self.clone();
        cpu_cam.initialize();

        let next_sample = Arc::new(AtomicUsize::new(0));
        let completed_samples = Arc::new(AtomicUsize::new(0));
        let world_for_gpu = world.clone();
        let next_for_gpu = Arc::clone(&next_sample);
        let completed_for_gpu = Arc::clone(&completed_samples);
        let gpu_handle = std::thread::spawn(move || -> Result<Vec<Vec<Color>>, String> {
            let zero_row = vec![Color::new(0.0, 0.0, 0.0); gpu_cam.image_width];
            let mut gpu_sum_rows = vec![zero_row; gpu_cam.image_height];
            loop {
                let sample_base = next_for_gpu.fetch_add(chunk_samples, Ordering::Relaxed);
                if sample_base >= total_samples {
                    break;
                }
                let spp = (total_samples - sample_base).min(chunk_samples);
                gpu_cam.samples_per_pixel = spp;
                let rows = gpu_cam.render_gpu_rows(&world_for_gpu, false, sample_base as u32, false)?;
                Camera::accumulate_rows_scaled(&mut gpu_sum_rows, &rows, spp as f64);
                let done = completed_for_gpu.fetch_add(spp, Ordering::Relaxed) + spp;
                eprint!("\rHybrid spp done: {}/{} ", done.min(total_samples), total_samples);
            }
            Ok(gpu_sum_rows)
        });

        loop {
            let sample_base = next_sample.fetch_add(chunk_samples, Ordering::Relaxed);
            if sample_base >= total_samples {
                break;
            }
            let spp = (total_samples - sample_base).min(chunk_samples);
            cpu_cam.samples_per_pixel = spp;
            let rows = cpu_cam.render_cpu_rows(world, None, Some(cpu_threads));
            Camera::accumulate_rows_scaled(&mut cpu_sum_rows, &rows, spp as f64);
            let done = completed_samples.fetch_add(spp, Ordering::Relaxed) + spp;
            eprint!("\rHybrid spp done: {}/{} ", done.min(total_samples), total_samples);
        }

        let gpu_sum_rows = gpu_handle
            .join()
            .map_err(|_| "GPU worker thread panicked".to_string())??;

        Camera::accumulate_rows_scaled(&mut cpu_sum_rows, &gpu_sum_rows, 1.0);
        let inv_samples = 1.0 / total_samples as f64;
        for row in &mut cpu_sum_rows {
            for pixel in row {
                *pixel = inv_samples * *pixel;
            }
        }

        self.write_outputs(&cpu_sum_rows)
            .map_err(|e| format!("Failed to write rendered images: {e}"))?;
        eprint!("\rDone                                  \n");
        Ok(())
    }

    fn render_cpu_rows(
        &self,
        world: &impl Hittable,
        progress_label: Option<&str>,
        thread_count: Option<usize>,
    ) -> Vec<Vec<Color>> {
        let remaining = AtomicUsize::new(self.image_height);
        let build_rows = || {
            (0..self.image_height)
                .into_par_iter()
                .map(|j| {
                    let mut row = Vec::with_capacity(self.image_width);
                    for i in 0..self.image_width {
                        let mut pixel_color = Color::new(0.0, 0.0, 0.0);
                        for _ in 0..self.samples_per_pixel {
                            let r = self.get_ray(i, j);
                            pixel_color += self.ray_color(&r, self.max_depth, world);
                        }
                        row.push(self.pixel_samples_scale * pixel_color);
                    }
                    if let Some(label) = progress_label {
                        let scanlines_remaining = remaining.fetch_sub(1, Ordering::Relaxed) - 1;
                        eprint!("\r{label}: {} ", scanlines_remaining);
                    }
                    row
                })
                .collect()
        };

        match thread_count {
            Some(threads) if threads > 0 => {
                if let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                    pool.install(build_rows)
                } else {
                    build_rows()
                }
            }
            _ => build_rows(),
        }
    }

    fn render_gpu_rows(
        &mut self,
        world: &HittableList,
        show_progress: bool,
        sample_base_offset: u32,
        gpu_only_mode: bool,
    ) -> Result<Vec<Vec<Color>>, String> {
        if show_progress {
            eprint!("\rGPU render dispatching...            ");
        }

        let sphere_count = world.spheres().len() as u32;
        let mut gpu_spheres: Vec<GpuSphere> = world
            .spheres()
            .iter()
            .map(|sphere| {
                let (material, kind) = match sphere.mat {
                    Material::Lambertian { albedo } => {
                        ([albedo.r as f32, albedo.g as f32, albedo.b as f32, 0.0], 0.0)
                    }
                    Material::Metallic { albedo, fuzz } => (
                        [albedo.r as f32, albedo.g as f32, albedo.b as f32, fuzz as f32],
                        1.0,
                    ),
                    Material::Dielectric { refraction_index } => (
                        [1.0, 1.0, 1.0, refraction_index as f32],
                        2.0,
                    ),
                    Material::DiffuseLight { emit } => (
                        [emit.r as f32, emit.g as f32, emit.b as f32, 0.0],
                        3.0,
                    ),
                };

                GpuSphere {
                    center_radius: [
                        sphere.center.x as f32,
                        sphere.center.y as f32,
                        sphere.center.z as f32,
                        sphere.radius as f32,
                    ],
                    material,
                    meta: [kind, 0.0, 0.0, 0.0],
                }
            })
            .collect();

        if gpu_spheres.is_empty() {
            gpu_spheres.push(GpuSphere::dummy());
        }

        let total_samples = self.samples_per_pixel.max(1) as u32;
        let max_depth = self.max_depth.max(1) as u32;
        let defocus_enabled = if self.defocus_angle > 0.0 { 1.0 } else { 0.0 };
        let mut params = GpuParams {
            dims: [
                self.image_width as u32,
                self.image_height as u32,
                0,
                max_depth,
            ],
            scene: [sphere_count, 0, 0, 0],
            center: [
                self.center.x as f32,
                self.center.y as f32,
                self.center.z as f32,
                0.0,
            ],
            pixel00: [
                self.pixel00_loc.x as f32,
                self.pixel00_loc.y as f32,
                self.pixel00_loc.z as f32,
                0.0,
            ],
            delta_u: [
                self.pixel_delta_u.x as f32,
                self.pixel_delta_u.y as f32,
                self.pixel_delta_u.z as f32,
                0.0,
            ],
            delta_v: [
                self.pixel_delta_v.x as f32,
                self.pixel_delta_v.y as f32,
                self.pixel_delta_v.z as f32,
                0.0,
            ],
            defocus_u: [
                self.defocus_disk_u.x as f32,
                self.defocus_disk_u.y as f32,
                self.defocus_disk_u.z as f32,
                defocus_enabled,
            ],
            defocus_v: [
                self.defocus_disk_v.x as f32,
                self.defocus_disk_v.y as f32,
                self.defocus_disk_v.z as f32,
                0.0,
            ],
        };

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "No suitable GPU adapter found".to_string())?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("pathtrace-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .map_err(|e| format!("Failed to create GPU device: {e}"))?;

        let spheres_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu-spheres-buffer"),
            contents: bytemuck::cast_slice(&gpu_spheres),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_len = self.image_width * self.image_height;
        let output_buffer_size =
            (output_len * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress;

        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu-output-buffer"),
            contents: &vec![0u8; output_buffer_size as usize],
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-readback-buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let default_in_flight = if gpu_only_mode { 8 } else { 1 };
        let in_flight_dispatches = Camera::env_usize("PATHTRACE_GPU_INFLIGHT_DISPATCHES")
            .unwrap_or(default_in_flight)
            .max(1);

        let mut params_buffers = Vec::with_capacity(in_flight_dispatches);
        let mut bind_groups = Vec::with_capacity(in_flight_dispatches);
        for _ in 0..in_flight_dispatches {
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu-params-buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu-bind-group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: spheres_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });
            params_buffers.push(params_buffer);
            bind_groups.push(bind_group);
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-pathtrace-shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu-pathtrace-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let workgroup_x = (self.image_width as u32).div_ceil(8);
        let workgroup_y = (self.image_height as u32).div_ceil(8);
        let dispatch_base = if gpu_only_mode { 1024 } else { 256 };
        let dispatch_cap = if gpu_only_mode { 64 } else { 16 };
        let default_samples_per_dispatch = (dispatch_base / max_depth).clamp(1, dispatch_cap);
        let samples_per_dispatch = Camera::env_usize("PATHTRACE_GPU_SPP_PER_DISPATCH")
            .map(|v| v as u32)
            .unwrap_or(default_samples_per_dispatch)
            .clamp(1, total_samples.max(1));
        let target_outstanding_spp = Camera::env_usize("PATHTRACE_GPU_TARGET_OUTSTANDING_SPP")
            .map(|v| v as u32)
            .unwrap_or(if gpu_only_mode { 64 } else { 16 })
            .max(samples_per_dispatch);
        let dispatches_before_poll =
            ((target_outstanding_spp / samples_per_dispatch).max(1))
                .min(in_flight_dispatches as u32) as usize;

        if show_progress {
            eprint!(
                "\rGPU render dispatching... (spp/dispatch={}, inflight={}, poll_every={} dispatches) ",
                samples_per_dispatch, in_flight_dispatches, dispatches_before_poll
            );
        }

        let progress_start = if show_progress {
            Some(Instant::now())
        } else {
            None
        };

        let mut submitted_sample_base = 0u32;
        let mut dispatch_idx = 0usize;
        let mut in_flight_submissions: VecDeque<(wgpu::SubmissionIndex, u32)> = VecDeque::new();
        while submitted_sample_base < total_samples {
            let slot = dispatch_idx % in_flight_dispatches;

            let samples_this_pass =
                (total_samples - submitted_sample_base).min(samples_per_dispatch);
            params.dims[2] = samples_this_pass;
            params.scene[1] = sample_base_offset + submitted_sample_base;
            queue.write_buffer(&params_buffers[slot], 0, bytemuck::bytes_of(&params));

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu-command-encoder"),
            });
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu-pathtrace-pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_groups[slot], &[]);
                compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
            }
            let submission = queue.submit(Some(encoder.finish()));
            dispatch_idx += 1;

            submitted_sample_base += samples_this_pass;
            in_flight_submissions.push_back((submission, submitted_sample_base));

            while in_flight_submissions.len() >= dispatches_before_poll {
                if let Some((submission_to_wait, completed_samples)) = in_flight_submissions.pop_front()
                {
                    device.poll(wgpu::Maintain::wait_for(submission_to_wait));
                    if show_progress {
                        let elapsed_sec = progress_start
                            .as_ref()
                            .map(|start| start.elapsed().as_secs_f64())
                            .unwrap_or(0.0)
                            .max(1e-6);
                        let samples_per_sec =
                            (completed_samples as f64 / elapsed_sec).round() as u64;
                        eprint!(
                            "\rGPU sampling: {completed_samples}/{total_samples} ({samples_per_sec} samples/sec) "
                        );
                    }
                }
            }
        }

        while let Some((submission_to_wait, completed_samples)) = in_flight_submissions.pop_front()
        {
            device.poll(wgpu::Maintain::wait_for(submission_to_wait));
            if show_progress {
                let elapsed_sec = progress_start
                    .as_ref()
                    .map(|start| start.elapsed().as_secs_f64())
                    .unwrap_or(0.0)
                    .max(1e-6);
                let samples_per_sec = (completed_samples as f64 / elapsed_sec).round() as u64;
                eprint!(
                    "\rGPU sampling: {completed_samples}/{total_samples} ({samples_per_sec} samples/sec) "
                );
            }
        }

        if show_progress {
            eprint!("\rGPU compute complete, reading back...                ");
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu-readback-encoder"),
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_buffer_size);
        let copy_submission = queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::wait_for(copy_submission));

        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let map_timeout_secs = Camera::env_usize("PATHTRACE_GPU_MAP_TIMEOUT_SEC")
            .unwrap_or(120) as u64;
        let map_wait_start = Instant::now();
        loop {
            match receiver.try_recv() {
                Ok(map_result) => {
                    map_result
                        .map_err(|e| format!("Failed to map GPU output buffer: {e:?}"))?;
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err("GPU map callback channel disconnected".to_string());
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }

            if map_wait_start.elapsed() >= Duration::from_secs(map_timeout_secs) {
                return Err(format!(
                    "Timed out waiting for GPU readback map after {map_timeout_secs}s. \
Try lowering PATHTRACE_GPU_SPP_PER_DISPATCH or PATHTRACE_GPU_INFLIGHT_DISPATCHES."
                ));
            }
            device.poll(wgpu::Maintain::Poll);
            std::thread::sleep(Duration::from_millis(2));
        }

        let data = buffer_slice.get_mapped_range();
        let pixels: &[[f32; 4]] = bytemuck::cast_slice(&data);
        if pixels.len() != output_len {
            return Err(format!(
                "GPU output size mismatch: expected {} pixels, got {}",
                output_len,
                pixels.len()
            ));
        }

        let mut rows = Vec::with_capacity(self.image_height);
        let sample_scale = 1.0 / total_samples as f64;
        for j in 0..self.image_height {
            let mut row = Vec::with_capacity(self.image_width);
            for i in 0..self.image_width {
                let p = pixels[j * self.image_width + i];
                row.push(Color::new(
                    p[0] as f64 * sample_scale,
                    p[1] as f64 * sample_scale,
                    p[2] as f64 * sample_scale,
                ));
            }
            rows.push(row);
        }

        drop(data);
        readback_buffer.unmap();
        Ok(rows)
    }

    fn accumulate_rows_scaled(dst: &mut [Vec<Color>], src: &[Vec<Color>], scale: f64) {
        for (dst_row, src_row) in dst.iter_mut().zip(src.iter()) {
            for (dst_pixel, src_pixel) in dst_row.iter_mut().zip(src_row.iter()) {
                *dst_pixel += scale * *src_pixel;
            }
        }
    }

    fn env_usize(name: &str) -> Option<usize> {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0)
    }

    fn write_outputs(&self, rows: &[Vec<Color>]) -> std::io::Result<()> {
        let mut ldr_out = BufWriter::new(File::create("image_16bit.ppm")?);
        write!(
            ldr_out,
            "P6\n{} {}\n65535\n",
            self.image_width,
            self.image_height
        )?;

        for row in rows {
            for pixel_color in row {
                write_color_ldr16(&mut ldr_out, *pixel_color);
            }
        }

        let mut hdr_out = BufWriter::new(File::create("image_hdr.pfm")?);
        write!(
            hdr_out,
            "PF\n{} {}\n-1.0\n",
            self.image_width,
            self.image_height
        )?;

        // PFM expects scanlines from bottom to top.
        for row in rows.iter().rev() {
            for pixel_color in row {
                write_color_hdr(&mut hdr_out, *pixel_color);
            }
        }

        ldr_out.flush()?;
        hdr_out.flush()?;
        Ok(())
    }

    fn initialize(&mut self) {
        self.image_height = (self.image_width as f64
            / self.aspect_ratio) as usize;
        self.image_height = if self.image_height < 1 {1} else {self.image_height};

        self.pixel_samples_scale = 1.0 / self.samples_per_pixel.max(1) as f64;

        self.center = self.lookfrom;
        let theta = self.vfov.to_radians();
        let h = (theta/2.0).tan();

        let viewport_height = 2.0 * h * self.focus_dist;
        let viewport_width = viewport_height
            * (self.image_width as f64
            / self.image_height as f64);

        self.w = (self.lookfrom - self.lookat).unit_vector();
        self.u = Vec3::cross(self.vup, self.w).unit_vector();
        self.v = Vec3::cross(self.w, self.u);

        let viewport_u = viewport_width * self.u;
        let viewport_v = viewport_height * -self.v;

        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

        let viewport_upper_left = self.center - (self.focus_dist * self.w)
            -viewport_u / 2.0 - viewport_v / 2.0;
        self.pixel00_loc = viewport_upper_left+0.5*(self.pixel_delta_u+self.pixel_delta_v);

        let defocus_radius = self.focus_dist * (self.defocus_angle / 2.0).to_radians().tan();
        self.defocus_disk_u = defocus_radius * self.u;
        self.defocus_disk_v = defocus_radius * self.v;
    }

    fn get_ray(&self, i : usize, j : usize) -> Ray {
        let offset = self.sample_square();
        let pixel_sample = self.pixel00_loc
            + (i as f64 + offset.x) * self.pixel_delta_u
            +(j as f64 + offset.y)* self.pixel_delta_v;
        let ray_origin = if self.defocus_angle <= 0.0 {self.center} else {self.defocus_disk_sample()};

        let ray_direction = pixel_sample - ray_origin;
        Ray::new(ray_origin, ray_direction)
    }

    fn defocus_disk_sample(&self) -> Point3 {
        let p = Vec3::random_in_unit_disk();
        self.center + (p.x * self.defocus_disk_u) + (p.y * self.defocus_disk_v)
    }

    fn sample_square(&self) -> Vec3 {
        Vec3::new(random_f64()-0.5, random_f64()-0.5,0.0)
    }

    fn ray_color(&self, r :&Ray, depth : usize, world : &impl Hittable) -> Color {
        if depth == 0 {
            return Color::new(0.0, 0.0, 0.0);
        }

        let mut current_ray = Ray::new(r.origin, r.direction);
        let mut throughput = Color::new(1.0, 1.0, 1.0);
        let mut radiance = Color::new(0.0, 0.0, 0.0);

        for _ in 0..depth {
            if let Some(rec) = world.hit(&current_ray, Interval::new(0.001, f64::INFINITY)) {
                radiance += throughput * rec.mat.emitted();
                if let Some((attenuation, scattered)) = rec.mat.scatter(&current_ray, &rec) {
                    throughput = throughput * attenuation;
                    current_ray = scattered;
                    continue;
                }
                return radiance;
            }

            let background = Color::new(0.0, 0.0, 0.0);
            radiance += throughput * background;
            return radiance;
        }

        radiance
    }
}
