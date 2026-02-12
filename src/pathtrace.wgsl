struct Params {
    dims: vec4<u32>,      // width, height, samples, max_depth
    scene: vec4<u32>,     // sphere_count, sample_base, plane_count, reserved
    center: vec4<f32>,
    pixel00: vec4<f32>,
    delta_u: vec4<f32>,
    delta_v: vec4<f32>,
    defocus_u: vec4<f32>, // xyz + enabled flag in w
    defocus_v: vec4<f32>,
}

struct Sphere {
    center_radius: vec4<f32>, // xyz + radius
    material: vec4<f32>,      // rgb + (fuzz or refraction_index)
    kind_data: vec4<f32>,     // x = kind (0 lambert, 1 metal, 2 dielectric)
}

struct Plane {
    point: vec4<f32>,         // xyz + unused
    normal: vec4<f32>,        // xyz + unused
    material: vec4<f32>,      // rgb + (fuzz or refraction_index)
    kind_data: vec4<f32>,     // x = kind (0 lambert, 1 metal, 2 dielectric)
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct HitRecord {
    hit: u32,
    t: f32,
    p: vec3<f32>,
    normal: vec3<f32>,
    front_face: u32,
    object_kind: u32,  // 0 sphere, 1 plane
    object_index: u32,
}

struct ScatterResult {
    ok: u32,
    attenuation: vec3<f32>,
    origin: vec3<f32>,
    direction: vec3<f32>,
}

const RAY_T_MIN: f32 = 0.001;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read_write> output_pixels: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> planes: array<Plane>;

fn rng_next(state: ptr<function, u32>) -> f32 {
    (*state) = (*state) * 1664525u + 1013904223u;
    return f32((*state) & 0x00ffffffu) / 16777216.0;
}

fn rng_range(state: ptr<function, u32>, min_v: f32, max_v: f32) -> f32 {
    return min_v + (max_v - min_v) * rng_next(state);
}

fn near_zero(v: vec3<f32>) -> bool {
    let s = 1e-8;
    return abs(v.x) < s && abs(v.y) < s && abs(v.z) < s;
}

fn random_in_unit_sphere(state: ptr<function, u32>) -> vec3<f32> {
    var p = vec3<f32>(0.0);
    loop {
        p = vec3<f32>(
            rng_range(state, -1.0, 1.0),
            rng_range(state, -1.0, 1.0),
            rng_range(state, -1.0, 1.0),
        );
        if dot(p, p) < 1.0 {
            break;
        }
    }
    return p;
}

fn random_unit_vector(state: ptr<function, u32>) -> vec3<f32> {
    let p = random_in_unit_sphere(state);
    let len2 = dot(p, p);
    if len2 > 1e-12 {
        return p / sqrt(len2);
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

fn random_in_unit_disk(state: ptr<function, u32>) -> vec2<f32> {
    var p = vec2<f32>(0.0);
    loop {
        p = vec2<f32>(
            rng_range(state, -1.0, 1.0),
            rng_range(state, -1.0, 1.0),
        );
        if dot(p, p) < 1.0 {
            break;
        }
    }
    return p;
}

fn reflectance(cosine: f32, refraction_index: f32) -> f32 {
    var r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

fn get_ray(i: u32, j: u32, state: ptr<function, u32>) -> Ray {
    let offset = vec2<f32>(rng_next(state) - 0.5, rng_next(state) - 0.5);
    let pixel_sample =
        params.pixel00.xyz +
        (f32(i) + offset.x) * params.delta_u.xyz +
        (f32(j) + offset.y) * params.delta_v.xyz;

    var ray_origin = params.center.xyz;
    if params.defocus_u.w > 0.5 {
        let disk = random_in_unit_disk(state);
        ray_origin = params.center.xyz + disk.x * params.defocus_u.xyz + disk.y * params.defocus_v.xyz;
    }

    return Ray(ray_origin, pixel_sample - ray_origin);
}

fn world_hit(ray: Ray, t_min: f32) -> HitRecord {
    var rec = HitRecord(0u, 0.0, vec3<f32>(0.0), vec3<f32>(0.0), 0u, 0u, 0u);
    var closest = 1e30;
    let sphere_count = params.scene.x;
    let plane_count = params.scene.z;

    for (var i: u32 = 0u; i < sphere_count; i = i + 1u) {
        let sphere = spheres[i];
        let oc = sphere.center_radius.xyz - ray.origin;
        let a = dot(ray.direction, ray.direction);
        let h = dot(ray.direction, oc);
        let c = dot(oc, oc) - sphere.center_radius.w * sphere.center_radius.w;
        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            continue;
        }

        let sqrtd = sqrt(discriminant);
        var root = (h - sqrtd) / a;
        if !(root > t_min && root < closest) {
            root = (h + sqrtd) / a;
            if !(root > t_min && root < closest) {
                continue;
            }
        }

        let p = ray.origin + root * ray.direction;
        var normal = (p - sphere.center_radius.xyz) / sphere.center_radius.w;
        let front = dot(ray.direction, normal) < 0.0;
        if !front {
            normal = -normal;
        }

        rec.hit = 1u;
        rec.t = root;
        rec.p = p;
        rec.normal = normal;
        rec.front_face = select(0u, 1u, front);
        rec.object_kind = 0u;
        rec.object_index = i;
        closest = root;
    }

    for (var i: u32 = 0u; i < plane_count; i = i + 1u) {
        let plane = planes[i];
        let n = normalize(plane.normal.xyz);
        let denom = dot(n, ray.direction);
        if abs(denom) < 1e-5 {
            continue;
        }

        let root = dot(plane.point.xyz - ray.origin, n) / denom;
        if !(root > t_min && root < closest) {
            continue;
        }

        let p = ray.origin + root * ray.direction;
        var normal = n;
        let front = dot(ray.direction, normal) < 0.0;
        if !front {
            normal = -normal;
        }

        rec.hit = 1u;
        rec.t = root;
        rec.p = p;
        rec.normal = normal;
        rec.front_face = select(0u, 1u, front);
        rec.object_kind = 1u;
        rec.object_index = i;
        closest = root;
    }

    return rec;
}

fn material_kind(rec: HitRecord) -> u32 {
    if rec.object_kind == 0u {
        let sphere = spheres[rec.object_index];
        return u32(sphere.kind_data.x + 0.5);
    }
    let plane = planes[rec.object_index];
    return u32(plane.kind_data.x + 0.5);
}

fn material_data(rec: HitRecord) -> vec4<f32> {
    if rec.object_kind == 0u {
        return spheres[rec.object_index].material;
    }
    return planes[rec.object_index].material;
}

fn emitted_color(rec: HitRecord) -> vec3<f32> {
    let kind = material_kind(rec);
    if kind == 3u {
        return material_data(rec).xyz;
    }
    return vec3<f32>(0.0);
}

fn scatter(ray: Ray, rec: HitRecord, state: ptr<function, u32>) -> ScatterResult {
    let mat = material_data(rec);
    let kind = material_kind(rec);

    if kind == 0u {
        var scatter_direction = rec.normal + random_unit_vector(state);
        if near_zero(scatter_direction) {
            scatter_direction = rec.normal;
        }
        return ScatterResult(1u, mat.xyz, rec.p, scatter_direction);
    }

    if kind == 1u {
        var reflected = reflect(normalize(ray.direction), rec.normal);
        reflected = reflected + mat.w * random_unit_vector(state);
        if dot(reflected, rec.normal) <= 0.0 {
            return ScatterResult(0u, vec3<f32>(0.0), rec.p, reflected);
        }
        return ScatterResult(1u, mat.xyz, rec.p, reflected);
    }

    if kind == 2u {
        let ir = mat.w;
        let ri = select(ir, 1.0 / ir, rec.front_face == 1u);
        let unit_direction = normalize(ray.direction);
        let cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
        let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        let cannot_refract = ri * sin_theta > 1.0;

        var direction = vec3<f32>(0.0);
        if cannot_refract || reflectance(cos_theta, ri) > rng_next(state) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, ri);
        }
        return ScatterResult(1u, vec3<f32>(1.0), rec.p, direction);
    }

    if kind == 3u {
        return ScatterResult(0u, vec3<f32>(0.0), rec.p, vec3<f32>(0.0));
    }

    return ScatterResult(0u, vec3<f32>(0.0), rec.p, vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params.dims.x;
    let height = params.dims.y;
    let samples = max(params.dims.z, 1u);
    let max_depth = params.dims.w;
    let sample_base = params.scene.y;

    if gid.x >= width || gid.y >= height {
        return;
    }

    let idx = gid.y * width + gid.x;
    var state = (gid.x + gid.y * width + 1u + sample_base * 1597334677u) * 747796405u + 2891336453u;
    var accum = vec3<f32>(0.0);

    for (var sample_idx: u32 = 0u; sample_idx < samples; sample_idx = sample_idx + 1u) {
        let sample_id = sample_base + sample_idx;
        state = state ^ (sample_id * 374761393u + 668265263u);
        var ray = get_ray(gid.x, gid.y, &state);
        var throughput = vec3<f32>(1.0);
        var radiance = vec3<f32>(0.0);

        for (var depth: u32 = 0u; depth < max_depth; depth = depth + 1u) {
            let rec = world_hit(ray, RAY_T_MIN);
            if rec.hit == 1u {
                radiance = radiance + throughput * emitted_color(rec);
                let sc = scatter(ray, rec, &state);
                if sc.ok == 0u {
                    break;
                }
                throughput = throughput * sc.attenuation;
                ray = Ray(sc.origin, sc.direction);
            } else {
                let background = vec3<f32>(0.0, 0.0, 0.0);
                radiance = radiance + throughput * background;
                break;
            }
        }
        accum = accum + radiance;
    }

    let prev = output_pixels[idx].xyz;
    output_pixels[idx] = vec4<f32>(prev + accum, 1.0);
}
