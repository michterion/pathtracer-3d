#include "cuda_pt.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mutex>
#include <windows.h>
#include <cuda.h>
#include <nvrtc.h>

static const char* s_cuda_pt_source = R"(
struct Vec3 {
    float x, y, z;
    __device__ Vec3() : x(0), y(0), z(0) {}
    __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __device__ Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    __device__ Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    __device__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __device__ Vec3 operator*(float t) const { return Vec3(x * t, y * t, z * t); }
    __device__ Vec3 operator/(float t) const { return Vec3(x / t, y / t, z / t); }
    __device__ float length_squared() const { return x * x + y * y + z * z; }
    __device__ float length() const { return sqrtf(length_squared()); }
};

__device__ inline Vec3 operator*(float t, const Vec3& v) { return v * t; }
__device__ inline float dot(const Vec3& u, const Vec3& v) { return u.x * v.x + u.y * v.y + u.z * v.z; }
__device__ inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}
__device__ inline Vec3 unit_vector(const Vec3& v) {
    float l = v.length();
    return l > 1e-8f ? v / l : Vec3(0, 0, 0);
}
__device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}
__device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}
__device__ inline float schlick_reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

struct RngState {
    unsigned int state;
    unsigned int inc;
};

__device__ inline unsigned int pcg_hash(unsigned int& state, unsigned int inc) {
    unsigned int oldstate = state;
    state = oldstate * 747796405u + (inc | 1u);
    unsigned int word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ inline float rand01(RngState& rng) {
    return (float)pcg_hash(rng.state, rng.inc) * (1.0f / 4294967296.0f);
}

__device__ inline Vec3 random_unit_vector(RngState& rng) {
    float z = rand01(rng) * 2.0f - 1.0f;
    float a = rand01(rng) * 6.28318530718f;
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return Vec3(r * cosf(a), r * sinf(a), z);
}

struct Ray {
    Vec3 origin;
    Vec3 direction;
    __device__ Ray() {}
    __device__ Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}
    __device__ Vec3 at(float t) const { return origin + t * direction; }
};

struct HitRecord {
    Vec3 p;
    Vec3 normal;
    int mat_type;
    Vec3 color;
    float param;
    float dispersion;
    float t;
    bool front_face;
    __device__ void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// 1. Sfera
struct GPUSphere {
    Vec3 center;
    float radius;
    int mat_type; // 0=Lambertian, 1=Metal, 2=Dielectric, 3=Light
    Vec3 color;
    float param;
};

__device__ bool hit_sphere(const GPUSphere& s, const Ray& r, float t_min, float t_max, HitRecord& rec) {
    Vec3 oc = r.origin - s.center;
    float a = r.direction.length_squared();
    float half_b = dot(oc, r.direction);
    float c = oc.length_squared() - s.radius * s.radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0f) return false;
    float sqrtd = sqrtf(discriminant);

    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root) return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    Vec3 outward_normal = (rec.p - s.center) / s.radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_type = s.mat_type;
    rec.color = s.color;
    rec.param = s.param;
    rec.dispersion = 0.0f;
    return true;
}

// 2. Czworokąt (Quad / Panel ścienny / Lustro / Płytka)
struct GPUQuad {
    Vec3 q;
    Vec3 u;
    Vec3 v;
    Vec3 normal;
    float d;
    Vec3 w;
    int mat_type;
    Vec3 color;
    float param;
};

__device__ bool hit_quad(const GPUQuad& quad, const Ray& r, float t_min, float t_max, HitRecord& rec) {
    float denom = dot(quad.normal, r.direction);
    if (fabsf(denom) < 1e-8f) return false;

    float t = (quad.d - dot(quad.normal, r.origin)) / denom;
    if (t < t_min || t > t_max) return false;

    Vec3 intersection = r.at(t);
    Vec3 planar_hit = intersection - quad.q;
    float alpha = dot(quad.w, cross(planar_hit, quad.v));
    float beta  = dot(quad.w, cross(quad.u, planar_hit));

    if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) {
        return false;
    }

    rec.t = t;
    rec.p = intersection;
    rec.mat_type = quad.mat_type;
    rec.color = quad.color;
    rec.param = quad.param;
    rec.dispersion = 0.0f;
    rec.set_face_normal(r, quad.normal);
    return true;
}

// 3. Płaszczyzna i Szlifowany Kryształ Szklany (ConvexPolyhedron z 20 fasetami)
struct GPUPlane {
    Vec3 n;
    float d;
};

struct GPUCrystal {
    Vec3 center;
    float bounding_radius;
    int plane_start;
    int plane_count;
    int mat_type;
    Vec3 color;
    float param;
    float dispersion;
};

// Źródło światła dla Next Event Estimation (Direct Light Sampling w stylu Cycles)
struct GPULight {
    Vec3 center;
    float radius;
    Vec3 emission;
    float pad;
};

__device__ bool hit_crystal(
    const GPUCrystal& c, const GPUPlane* planes,
    const Ray& r, float t_min, float t_max, HitRecord& rec
) {
    // 1. Szybki test sfery otaczającej (Bounding Sphere Culling)
    Vec3 oc = r.origin - c.center;
    float a = r.direction.length_squared();
    float half_b = dot(oc, r.direction);
    float c_val = oc.length_squared() - c.bounding_radius * c.bounding_radius;
    if (c_val > 0.0f && half_b > 0.0f) return false;
    float discr = half_b * half_b - a * c_val;
    if (discr < 0.0f) return false;

    float sqrtd = sqrtf(discr);
    float t0 = (-half_b - sqrtd) / a;
    float t1 = (-half_b + sqrtd) / a;
    if (t1 < t_min || t0 > t_max) return false;

    // 2. Precyzyjny test obcinania półprzestrzeni (Slab Method)
    float t_enter = -1e9f;
    float t_exit  = 1e9f;
    Vec3 n_enter(0.0f, 1.0f, 0.0f);
    Vec3 n_exit(0.0f, 1.0f, 0.0f);

    for (int p = 0; p < c.plane_count; ++p) {
        GPUPlane plane = planes[c.plane_start + p];
        float denom = dot(plane.n, r.direction);
        float numer = plane.d - dot(plane.n, oc);

        if (fabsf(denom) < 1e-7f) {
            if (numer < 0.0f) return false;
        } else if (denom > 0.0f) {
            float t = numer / denom;
            if (t < t_exit) {
                t_exit = t;
                n_exit = plane.n;
            }
        } else {
            float t = numer / denom;
            if (t > t_enter) {
                t_enter = t;
                n_enter = plane.n;
            }
        }
        if (t_enter > t_exit) return false;
    }

    float t_hit = -1.0f;
    Vec3 hit_normal;

    if (t_enter >= t_min && t_enter <= t_max) {
        t_hit = t_enter;
        hit_normal = n_enter;
    } else if (t_exit >= t_min && t_exit <= t_max) {
        t_hit = t_exit;
        hit_normal = n_exit;
    } else {
        return false;
    }

    rec.t = t_hit;
    rec.p = r.at(t_hit);
    rec.mat_type = c.mat_type;
    rec.color = c.color;
    rec.param = c.param;
    rec.dispersion = c.dispersion;
    rec.set_face_normal(r, hit_normal);
    return true;
}

// 4. Trójkąt i Siatka Mesh (.OBJ z akceleracją Bounding Box AABB)
struct GPUTriangle {
    Vec3 v0, v1, v2;
    Vec3 normal;
};

struct GPUMesh {
    Vec3 bmin;
    Vec3 bmax;
    int tri_start;
    int tri_count;
    int mat_type;
    Vec3 color;
    float param;
};

__device__ inline bool hit_aabb_gpu(const Ray& r, float t_min, float t_max, const Vec3& b_min, const Vec3& b_max) {
    for (int a = 0; a < 3; ++a) {
        float r_dir  = (a == 0) ? r.direction.x : ((a == 1) ? r.direction.y : r.direction.z);
        float r_orig = (a == 0) ? r.origin.x    : ((a == 1) ? r.origin.y    : r.origin.z);
        float min_v  = (a == 0) ? b_min.x       : ((a == 1) ? b_min.y       : b_min.z);
        float max_v  = (a == 0) ? b_max.x       : ((a == 1) ? b_max.y       : b_max.z);
        float invD   = 1.0f / (fabsf(r_dir) > 1e-8f ? r_dir : (r_dir < 0.0f ? -1e-8f : 1e-8f));
        float t0 = (min_v - r_orig) * invD;
        float t1 = (max_v - r_orig) * invD;
        if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;
    }
    return true;
}

__device__ inline bool hit_triangle_gpu(
    const GPUTriangle& tri, const Ray& r,
    float t_min, float t_max, HitRecord& rec,
    int mat_type, const Vec3& color, float param
) {
    Vec3 edge1 = tri.v1 - tri.v0;
    Vec3 edge2 = tri.v2 - tri.v0;
    Vec3 h = cross(r.direction, edge2);
    float a = dot(edge1, h);
    if (fabsf(a) < 1e-7f) return false;
    float f = 1.0f / a;
    Vec3 s = r.origin - tri.v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    Vec3 q = cross(s, edge1);
    float v = f * dot(r.direction, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = f * dot(edge2, q);
    if (t < t_min || t > t_max) return false;

    rec.t = t;
    rec.p = r.at(t);
    rec.mat_type = mat_type;
    rec.color = color;
    rec.param = param;
    rec.dispersion = 0.0f;
    rec.set_face_normal(r, tri.normal);
    return true;
}

__device__ bool world_hit(
    const GPUSphere* spheres, int num_spheres,
    const GPUQuad* quads, int num_quads,
    const GPUCrystal* crystals, int num_crystals, const GPUPlane* crystal_planes,
    const GPUMesh* meshes, int num_meshes, const GPUTriangle* triangles,
    const Ray& r, float t_min, float t_max, HitRecord& rec
) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    // 1. Kule
    for (int i = 0; i < num_spheres; ++i) {
        if (hit_sphere(spheres[i], r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    // 2. Czworokąty (Quady)
    for (int i = 0; i < num_quads; ++i) {
        if (hit_quad(quads[i], r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    // 3. Szlifowane kryształy (ConvexPolyhedron)
    for (int i = 0; i < num_crystals; ++i) {
        if (hit_crystal(crystals[i], crystal_planes, r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    // 4. Siatki trójkątów (.OBJ Meshes)
    for (int m = 0; m < num_meshes; ++m) {
        if (hit_aabb_gpu(r, t_min, closest_so_far, meshes[m].bmin, meshes[m].bmax)) {
            for (int t = 0; t < meshes[m].tri_count; ++t) {
                if (hit_triangle_gpu(triangles[meshes[m].tri_start + t], r, t_min, closest_so_far, temp_rec,
                                     meshes[m].mat_type, meshes[m].color, meshes[m].param)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
        }
    }

    return hit_anything;
}

// Próbkowanie zmapowanej na GPU dwuwymiarowej siatki kaustyki z interpolacją bilinearną i płynnym wygaszaniem krawędzi
__device__ inline Vec3 sample_caustic(
    const float* c_r, const float* c_g, const float* c_b,
    int grid_w, int grid_h,
    float min_x, float max_x, float min_z, float max_z,
    float px, float pz
) {
    if (!c_r || !c_g || !c_b) return Vec3(0.0f, 0.0f, 0.0f);
    if (px <= min_x || px >= max_x || pz <= min_z || pz >= max_z) return Vec3(0.0f, 0.0f, 0.0f);

    float u = (px - min_x) / (max_x - min_x) * (float)(grid_w - 1);
    float v = (pz - min_z) / (max_z - min_z) * (float)(grid_h - 1);

    int x0 = (int)u;
    int z0 = (int)v;
    int x1 = min(x0 + 1, grid_w - 1);
    int z1 = min(z0 + 1, grid_h - 1);

    float fx = u - (float)x0;
    float fz = v - (float)z0;

    int idx00 = z0 * grid_w + x0;
    int idx10 = z0 * grid_w + x1;
    int idx01 = z1 * grid_w + x0;
    int idx11 = z1 * grid_w + x1;

    float r = (1.0f - fx) * (1.0f - fz) * c_r[idx00] + fx * (1.0f - fz) * c_r[idx10] +
              (1.0f - fx) * fz * c_r[idx01] + fx * fz * c_r[idx11];
    float g = (1.0f - fx) * (1.0f - fz) * c_g[idx00] + fx * (1.0f - fz) * c_g[idx10] +
              (1.0f - fx) * fz * c_g[idx01] + fx * fz * c_g[idx11];
    float b = (1.0f - fx) * (1.0f - fz) * c_b[idx00] + fx * (1.0f - fz) * c_b[idx10] +
              (1.0f - fx) * fz * c_b[idx01] + fx * fz * c_b[idx11];

    // Płynne wygaszanie krawędzi siatki eliminujące sztuczne odcięcia
    float margin = 0.6f;
    float fade_x = fminf(fmaxf((px - min_x) / margin, 0.0f), 1.0f) * fminf(fmaxf((max_x - px) / margin, 0.0f), 1.0f);
    float fade_z = fminf(fmaxf((pz - min_z) / margin, 0.0f), 1.0f) * fminf(fmaxf((max_z - pz) / margin, 0.0f), 1.0f);
    float fade = fade_x * fade_z;

    return Vec3(r * fade, g * fade, b * fade);
}

// Obliczanie współczynnika przepuszczalności promienia cienia NEE (Transparent Shadows)
// Eliminuje "czarną plamę" za szkłem, gdy kaustyka jest wyłączona lub w strefach rozproszonych
__device__ Vec3 eval_shadow_transmittance(
    const GPUSphere* spheres, int num_spheres,
    const GPUQuad* quads, int num_quads,
    const GPUCrystal* crystals, int num_crystals, const GPUPlane* crystal_planes,
    const GPUMesh* meshes, int num_meshes, const GPUTriangle* triangles,
    Ray shadow_ray, float dist, int enable_caustics
) {
    Vec3 transmittance(1.0f, 1.0f, 1.0f);
    float cur_t_min = 0.001f;
    float max_dist = dist - 0.002f;

    for (int step = 0; step < 4; ++step) {
        HitRecord srec;
        if (!world_hit(spheres, num_spheres, quads, num_quads, crystals, num_crystals, crystal_planes, meshes, num_meshes, triangles, shadow_ray, cur_t_min, max_dist, srec)) {
            return transmittance; // Światło dociera bez przeszkód
        }

        // Przeszkody całkowicie nieprzezroczyste (dyfuzyjne lub lustrzany metal)
        if (srec.mat_type == 0 || srec.mat_type == 1) {
            return Vec3(0.0f, 0.0f, 0.0f);
        }

        // Materiały dielektryczne (kryształy szklane, kule ze szkła)
        if (srec.mat_type == 2) {
            if (enable_caustics) {
                // Gdy kaustyka GPU jest WŁĄCZONA: bazowa transmisja 12% barwiona kolorem szkła,
                // dzięki czemu cień poza ostrymi wstęgami kaustyki nie jest smolistą czernią
                return transmittance * srec.color * 0.12f;
            } else {
                // Gdy kaustyka jest WYŁĄCZONA: TRANSPARENTNE CIENIE SZKŁA!
                // Szkło przepuszcza bezpośrednie światło (82%), całkowicie likwidując czarną plamę
                transmittance = transmittance * srec.color * 0.82f;
                shadow_ray.origin = srec.p + shadow_ray.direction * 0.002f;
                max_dist -= (srec.t + 0.002f);
                if (max_dist <= 0.001f) return transmittance;
                cur_t_min = 0.001f;
            }
        } else {
            break;
        }
    }
    return transmittance;
}

__device__ Vec3 trace_path(
    Ray ray,
    const GPUSphere* spheres, int num_spheres,
    const GPUQuad* quads, int num_quads,
    const GPUCrystal* crystals, int num_crystals, const GPUPlane* crystal_planes,
    const GPUMesh* meshes, int num_meshes, const GPUTriangle* triangles,
    const GPULight* lights, int num_lights,
    int enable_caustics,
    const float* caustic_r, const float* caustic_g, const float* caustic_b,
    int caustic_w, int caustic_h,
    float c_min_x, float c_max_x, float c_min_z, float c_max_z,
    int max_depth, float clamp_indirect, RngState& rng
) {
    Vec3 accumulated_color(0.0f, 0.0f, 0.0f);
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    bool specular_bounce = true; // True dla promienia kamery i odbić lustrzanych/szklanych

    for (int bounce = 0; bounce < max_depth; ++bounce) {
        HitRecord rec;
        if (!world_hit(spheres, num_spheres, quads, num_quads, crystals, num_crystals, crystal_planes, meshes, num_meshes, triangles, ray, 0.001f, 1e9f, rec)) {
            Vec3 unit_direction = unit_vector(ray.direction);
            float a = 0.5f * (unit_direction.y + 1.0f);
            Vec3 sky = (1.0f - a) * Vec3(0.04f, 0.05f, 0.08f) + a * Vec3(0.12f, 0.16f, 0.25f);
            accumulated_color = accumulated_color + throughput * sky;
            break;
        }

        // 1. Bezpośrednie trafienie w emiter światła (Cycles Direct Hit)
        if (rec.mat_type == 3) { // Light
            if (specular_bounce || bounce == 0) {
                Vec3 emission = rec.color * rec.param;
                if (bounce > 0 && clamp_indirect > 0.0f) {
                    float lum = fmaxf(emission.x, fmaxf(emission.y, emission.z));
                    if (lum > clamp_indirect) emission = emission * (clamp_indirect / lum);
                }
                accumulated_color = accumulated_color + throughput * emission;
            }
            break;
        }

        Ray scattered;
        Vec3 attenuation;
        bool is_specular = false;

        if (rec.mat_type == 0) { // Lambertian Diffuse (Cycles NEE + GPU Caustics)
            // Kaustyka GPU: pobranie promieniowania kaustycznego z siatki fotonów na podłożu
            if (enable_caustics) {
                if (rec.normal.y > 0.4f && fabsf(rec.p.y) < 0.25f && (bounce == 0 || specular_bounce)) {
                    Vec3 c_irr = sample_caustic(
                        caustic_r, caustic_g, caustic_b,
                        caustic_w, caustic_h,
                        c_min_x, c_max_x, c_min_z, c_max_z,
                        rec.p.x, rec.p.z
                    );
                    accumulated_color = accumulated_color + throughput * rec.color * c_irr;
                }
            }

            // A. Next Event Estimation (Próbkowanie bezpośrednie światła z transparentnym cieniem)
            if (num_lights > 0) {
                int light_idx = (int)(rand01(rng) * (float)num_lights);
                if (light_idx >= num_lights) light_idx = num_lights - 1;
                GPULight light = lights[light_idx];

                Vec3 p_light = light.center + light.radius * random_unit_vector(rng);
                Vec3 to_light = p_light - rec.p;
                float dist_sq = to_light.length_squared();
                float dist = sqrtf(dist_sq);

                if (dist > 1e-4f) {
                    Vec3 dir_to_light = to_light / dist;
                    float cos_surf = dot(rec.normal, dir_to_light);
                    Vec3 light_norm = (p_light - light.center) / light.radius;
                    float cos_light = dot(-dir_to_light, light_norm);

                    if (cos_surf > 1e-4f && cos_light > 1e-4f) {
                        Ray shadow_ray(rec.p + rec.normal * 0.001f, dir_to_light);
                        Vec3 shadow_trans = eval_shadow_transmittance(
                            spheres, num_spheres,
                            quads, num_quads,
                            crystals, num_crystals, crystal_planes,
                            meshes, num_meshes, triangles,
                            shadow_ray, dist, enable_caustics
                        );

                        if (shadow_trans.x > 1e-4f || shadow_trans.y > 1e-4f || shadow_trans.z > 1e-4f) {
                            float geom = (4.0f * light.radius * light.radius * cos_surf * cos_light) / dist_sq;
                            geom = fminf(geom, 2.5f);
                            Vec3 direct_contrib = throughput * rec.color * light.emission * geom * (float)num_lights * shadow_trans;
                            if (bounce > 0 && clamp_indirect > 0.0f) {
                                float lum = fmaxf(direct_contrib.x, fmaxf(direct_contrib.y, direct_contrib.z));
                                if (lum > clamp_indirect) direct_contrib = direct_contrib * (clamp_indirect / lum);
                            }
                            accumulated_color = accumulated_color + direct_contrib;
                        }
                    }
                }
            }

            // B. Pośrednie śledzenie ścieżki
            Vec3 scatter_dir = rec.normal + random_unit_vector(rng);
            if (scatter_dir.length_squared() < 1e-8f) scatter_dir = rec.normal;
            scattered = Ray(rec.p, unit_vector(scatter_dir));
            attenuation = rec.color;
            is_specular = false;
        }
        else if (rec.mat_type == 1) { // Metal / Conductor (Cycles Microfacet Metal)
            Vec3 reflected = reflect(unit_vector(ray.direction), rec.normal);
            float roughness = fminf(fmaxf(rec.param, 0.0f), 1.0f);
            Vec3 scatter_dir = reflected + roughness * random_unit_vector(rng);
            scattered = Ray(rec.p, unit_vector(scatter_dir));
            attenuation = rec.color;
            if (dot(scattered.direction, rec.normal) <= 0.0f) break;
            is_specular = (roughness < 0.08f);
        }
        else if (rec.mat_type == 2) { // Dielectric (Cycles Glass BSDF + Cauchy Spectral Dispersion)
            float effective_ior = rec.param;
            attenuation = rec.color;

            // Spektralna dyspersja Cauchy'ego (Pryzmatyczny podział światła na kolory tęczy)
            if (rec.dispersion > 1e-4f) {
                float lambda = rand01(rng);
                effective_ior = rec.param + (2.0f * lambda - 1.0f) * rec.dispersion;
                float wr = expf(-((lambda - 0.15f) * (lambda - 0.15f)) / 0.14f);
                float wg = expf(-((lambda - 0.50f) * (lambda - 0.50f)) / 0.14f);
                float wb = expf(-((lambda - 0.85f) * (lambda - 0.85f)) / 0.14f);
                Vec3 spec(wr, wg, wb);
                float avg = (spec.x + spec.y + spec.z) / 3.0f;
                if (avg > 1e-4f) spec = spec / avg;
                float blend = fminf(fmaxf(rec.dispersion * 10.0f, 0.0f), 1.0f);
                attenuation = (1.0f - blend) * rec.color + blend * (rec.color * spec);
            }

            float ref_ratio = rec.front_face ? (1.0f / effective_ior) : effective_ior;
            Vec3 unit_dir = unit_vector(ray.direction);
            float cos_theta = fminf(dot(-unit_dir, rec.normal), 1.0f);
            float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

            bool cannot_refract = ref_ratio * sin_theta > 1.0f;
            Vec3 direction;
            if (cannot_refract || schlick_reflectance(cos_theta, ref_ratio) > rand01(rng)) {
                direction = reflect(unit_dir, rec.normal);
            } else {
                direction = refract(unit_dir, rec.normal, ref_ratio);
            }
            scattered = Ray(rec.p, unit_vector(direction));
            is_specular = true;
        }

        specular_bounce = is_specular;
        throughput = throughput * attenuation;
        ray = scattered;

        // Cycles Russian Roulette Path Termination (po 3 odbiciu)
        if (bounce >= 3) {
            float max_c = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            float q = fminf(fmaxf(max_c, 0.05f), 0.95f);
            if (rand01(rng) > q) break;
            throughput = throughput / q;
        }
    }
    return accumulated_color;
}

struct CameraData {
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
};

extern "C" __global__ void render_accum_kernel(
    Vec3* accum_fb, int width, int height,
    int spp_this_pass, int total_spp_so_far, int max_depth, float clamp_indirect,
    CameraData cam,
    const GPUSphere* spheres, int num_spheres,
    const GPUQuad* quads, int num_quads,
    const GPUCrystal* crystals, int num_crystals, const GPUPlane* crystal_planes,
    const GPUMesh* meshes, int num_meshes, const GPUTriangle* triangles,
    const GPULight* lights, int num_lights,
    int enable_caustics,
    const float* caustic_r, const float* caustic_g, const float* caustic_b,
    int caustic_w, int caustic_h,
    float c_min_x, float c_max_x, float c_min_z, float c_max_z,
    unsigned int base_seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    RngState rng;
    rng.state = base_seed + (unsigned int)pixel_index * 1973u;
    rng.inc = (unsigned int)(pixel_index * 2 + 1);

    Vec3 pass_color(0.0f, 0.0f, 0.0f);
    for (int s = 0; s < spp_this_pass; ++s) {
        float u = (float(i) + rand01(rng)) / float(width - 1);
        float v = (float(height - 1 - j) + rand01(rng)) / float(height - 1);

        Ray r(cam.origin, cam.lower_left_corner + u * cam.horizontal + v * cam.vertical - cam.origin);
        pass_color = pass_color + trace_path(
            r,
            spheres, num_spheres,
            quads, num_quads,
            crystals, num_crystals, crystal_planes,
            meshes, num_meshes, triangles,
            lights, num_lights,
            enable_caustics,
            caustic_r, caustic_g, caustic_b,
            caustic_w, caustic_h,
            c_min_x, c_max_x, c_min_z, c_max_z,
            max_depth, clamp_indirect, rng
        );
    }

    Vec3 prev = (total_spp_so_far > 0) ? accum_fb[pixel_index] : Vec3(0.0f, 0.0f, 0.0f);
    accum_fb[pixel_index] = prev + pass_color;
}

// Błyskawiczny kernel mapowania tonów i konwersji do 32-bit BGRA bezpośrednio na GPU (0.05 ms!)
extern "C" __global__ void display_tonemap_kernel(
    const Vec3* accum_fb,
    unsigned int* out_bgra,
    int width, int height,
    int done_spp,
    float exposure_scale,
    int tonemap_mode
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    int idx = j * width + i;
    Vec3 rad = accum_fb[idx] * (1.0f / (float)done_spp);

    float scale = exposure_scale;
    float x = fmaxf(0.0f, rad.x * scale);
    float y = fmaxf(0.0f, rad.y * scale);
    float z = fmaxf(0.0f, rad.z * scale);

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (tonemap_mode == 0) { // ACES Filmic HDR
        float a = 2.51f, bb = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
        r = fminf(fmaxf((x * (a * x + bb)) / (x * (cc * x + d) + e), 0.0f), 1.0f);
        g = fminf(fmaxf((y * (a * y + bb)) / (y * (cc * y + d) + e), 0.0f), 1.0f);
        b = fminf(fmaxf((z * (a * z + bb)) / (z * (cc * z + d) + e), 0.0f), 1.0f);
    } else if (tonemap_mode == 1) { // Reinhard
        float white_sq = 16.0f;
        r = fminf(fmaxf((x * (1.0f + x / white_sq)) / (1.0f + x), 0.0f), 1.0f);
        g = fminf(fmaxf((y * (1.0f + y / white_sq)) / (1.0f + y), 0.0f), 1.0f);
        b = fminf(fmaxf((z * (1.0f + z / white_sq)) / (1.0f + z), 0.0f), 1.0f);
    } else { // Linear clamp
        r = fminf(fmaxf(x, 0.0f), 1.0f);
        g = fminf(fmaxf(y, 0.0f), 1.0f);
        b = fminf(fmaxf(z, 0.0f), 1.0f);
    }

    // sRGB Gamma 2.2
    float inv_gamma = 1.0f / 2.2f;
    r = powf(r, inv_gamma);
    g = powf(g, inv_gamma);
    b = powf(b, inv_gamma);

    unsigned int ir = (unsigned int)(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
    unsigned int ig = (unsigned int)(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
    unsigned int ib = (unsigned int)(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));

    out_bgra[idx] = (ib) | (ig << 8) | (ir << 16) | (0xFF000000u);
}

// ============================================================================
// DEDYKOWANY KERNEL EMISJI FOTONÓW KAUSTYKI Z DYSPERSJĄ SPEKTRALNĄ CAUCHY'EGO (GPU)
// Śledzi załamania światła przez fasetki kryształu i kule szklane w ułamku milisekundy
// ============================================================================
extern "C" __global__ void emit_caustic_photons_kernel(
    float* caustic_r, float* caustic_g, float* caustic_b,
    int grid_w, int grid_h,
    float min_x, float max_x, float min_z, float max_z,
    int num_photons, int scene_mode,
    const GPULight* lights, int num_lights,
    const GPUCrystal* crystals, int num_crystals, const GPUPlane* crystal_planes,
    const GPUSphere* spheres, int num_spheres,
    const GPUQuad* quads, int num_quads,
    unsigned int base_seed,
    float light_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_photons) return;
    if (num_lights <= 0) return;

    RngState rng;
    rng.state = base_seed + (unsigned int)tid * 2654435761u;
    rng.inc = (unsigned int)(tid * 2 + 1);

    // Wybór źródła światła
    int l_idx = (int)(rand01(rng) * (float)num_lights);
    if (l_idx >= num_lights) l_idx = num_lights - 1;
    GPULight light = lights[l_idx];

    // Sprawdzenie dostępnych celów dielektrycznych (kryształy oraz szklane kule)
    int num_dielectric_spheres = 0;
    int diel_sphere_idx[8];
    for (int s = 0; s < num_spheres && num_dielectric_spheres < 8; ++s) {
        if (spheres[s].mat_type == 2) {
            diel_sphere_idx[num_dielectric_spheres++] = s;
        }
    }

    int total_targets = num_crystals + num_dielectric_spheres;
    if (total_targets <= 0) return;

    int target_sel = (int)(rand01(rng) * (float)total_targets);
    if (target_sel >= total_targets) target_sel = total_targets - 1;

    bool is_crystal_target = (target_sel < num_crystals);
    int crystal_idx = is_crystal_target ? target_sel : 0;
    int ds_idx = !is_crystal_target ? diel_sphere_idx[target_sel - num_crystals] : 0;

    Vec3 target_center = is_crystal_target ? crystals[crystal_idx].center : spheres[ds_idx].center;
    float target_radius = is_crystal_target ? (crystals[crystal_idx].bounding_radius * 1.15f) : (spheres[ds_idx].radius * 1.05f);
    float base_ior = is_crystal_target ? crystals[crystal_idx].param : spheres[ds_idx].param;
    float dispersion = is_crystal_target ? crystals[crystal_idx].dispersion : 0.0f;
    Vec3 obj_color = is_crystal_target ? crystals[crystal_idx].color : spheres[ds_idx].color;

    // Skupiona emisja z rdzenia źródła światła (eliminuje 24-stopniowy blur wielkiego emitera o średnicy 1.7 m)
    Vec3 p_light = light.center + (light.radius * light_size) * random_unit_vector(rng);

    // Promień początkowy w stożku skierowanym na cel
    Vec3 light_to_target = target_center - light.center;
    float dist_lt = light_to_target.length();
    Vec3 dir_lt = light_to_target / (dist_lt > 1e-4f ? dist_lt : 1.0f);

    Vec3 up(0.0f, 1.0f, 0.0f);
    if (fabsf(dir_lt.y) > 0.9f) up = Vec3(1.0f, 0.0f, 0.0f);
    Vec3 right = unit_vector(cross(dir_lt, up));
    Vec3 v_up = cross(right, dir_lt);

    float r_disk = target_radius * sqrtf(rand01(rng));
    float th_angle = rand01(rng) * 6.28318530718f;
    Vec3 disk_target = target_center + right * (r_disk * cosf(th_angle)) + v_up * (r_disk * sinf(th_angle));
    Vec3 init_dir = unit_vector(disk_target - p_light);
    Ray cur_ray(p_light, init_dir);

    if (scene_mode == 1) {
        // Scena pokoju z zewnętrznym lustrem przy Z=3.2
        float mx = 0.8f + rand01(rng) * 1.1f;
        float my = 1.2f + rand01(rng) * 0.9f;
        Vec3 p_mir(mx, my, 3.2f);
        Vec3 dir_to_mir = unit_vector(p_mir - p_light);
        Vec3 dir_refl(dir_to_mir.x, dir_to_mir.y, -fabsf(dir_to_mir.z));
        cur_ray = Ray(p_mir + dir_refl * 0.002f, unit_vector(dir_refl));
    }

    // Płynna dyspersja spektralna Cauchy'ego (rozszczepienie pryzmatyczne)
    float eff_ior = base_ior;
    Vec3 ch_tint(1.0f, 1.0f, 1.0f);
    if (dispersion > 1e-4f) {
        float lambda = rand01(rng);
        eff_ior = base_ior + (2.0f * lambda - 1.0f) * dispersion;
        float wr = expf(-((lambda - 0.15f) * (lambda - 0.15f)) / 0.14f);
        float wg = expf(-((lambda - 0.50f) * (lambda - 0.50f)) / 0.14f);
        float wb = expf(-((lambda - 0.85f) * (lambda - 0.85f)) / 0.14f);
        Vec3 spec(wr, wg, wb);
        float avg = (spec.x + spec.y + spec.z) / 3.0f;
        if (avg > 1e-4f) spec = spec / avg;
        float blend = fminf(fmaxf(dispersion * 35.0f, 0.0f), 1.0f);
        ch_tint = (1.0f - blend) * Vec3(1.0f, 1.0f, 1.0f) + blend * spec;
    }

    // Obliczenie strumienia fotonu z zachowaniem energii
    float solid_angle = (3.14159265f * target_radius * target_radius) / (dist_lt * dist_lt + 0.01f);
    float cone_fraction = solid_angle / (4.0f * 3.14159265f);
    if (scene_mode == 1) cone_fraction = 0.015f;

    Vec3 total_cone_flux = light.emission * (4.0f * 3.14159265f * 3.14159265f * light.radius * light.radius * cone_fraction * (float)num_lights * (float)total_targets);
    Vec3 photon_flux = (total_cone_flux / (float)num_photons) * ch_tint * obj_color;

    Ray r = cur_ray;
    bool refracted_through = false;

    if (is_crystal_target) {
        GPUCrystal c = crystals[crystal_idx];
        for (int b = 0; b < 8; ++b) {
            HitRecord hit;
            if (!hit_crystal(c, crystal_planes, r, 0.001f, 1e9f, hit)) break;
            refracted_through = true;

            float ref_idx = hit.front_face ? (1.0f / eff_ior) : eff_ior;
            Vec3 unit_d = unit_vector(r.direction);
            float cos_theta = fminf(dot(-unit_d, hit.normal), 1.0f);
            float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

            bool cannot_refract = ref_idx * sin_theta > 1.0f;
            float refl_prob = schlick_reflectance(cos_theta, ref_idx);

            Vec3 next_dir;
            if (cannot_refract || rand01(rng) < refl_prob) {
                next_dir = reflect(unit_d, hit.normal);
            } else {
                next_dir = refract(unit_d, hit.normal, ref_idx);
            }
            r = Ray(hit.p + next_dir * 0.0015f, unit_vector(next_dir));
        }
    } else {
        GPUSphere s = spheres[ds_idx];
        HitRecord h1;
        if (hit_sphere(s, r, 0.001f, 1e9f, h1)) {
            float ref_idx1 = 1.0f / eff_ior;
            Vec3 unit_d1 = unit_vector(r.direction);
            float cos_t1 = fminf(dot(-unit_d1, h1.normal), 1.0f);
            Vec3 dir1 = refract(unit_d1, h1.normal, ref_idx1);
            Ray r1(h1.p + dir1 * 0.0015f, unit_vector(dir1));

            HitRecord h2;
            if (hit_sphere(s, r1, 0.001f, 1e9f, h2)) {
                float ref_idx2 = eff_ior;
                Vec3 unit_d2 = unit_vector(r1.direction);
                float cos_t2 = fminf(dot(-unit_d2, h2.normal), 1.0f);
                float sin_t2 = sqrtf(fmaxf(0.0f, 1.0f - cos_t2 * cos_t2));
                if (ref_idx2 * sin_t2 <= 1.0f) {
                    Vec3 dir2 = refract(unit_d2, h2.normal, ref_idx2);
                    r = Ray(h2.p + dir2 * 0.0015f, unit_vector(dir2));
                    refracted_through = true;
                }
            }
        }
    }

    if (!refracted_through) return;

    // Przecięcie z płaszczyzną podłoża Y = 0
    if (r.direction.y >= -1e-4f) return;
    float t_floor = (0.0f - r.origin.y) / r.direction.y;
    if (t_floor <= 0.001f) return;

    Vec3 p_floor = r.origin + t_floor * r.direction;
    p_floor.y = 0.0f;

    // Sprawdzenie czy promień kaustyki nie został zasłonięty przez nieprzezroczystą kulę
    for (int s = 0; s < num_spheres; ++s) {
        if (spheres[s].mat_type != 2 && spheres[s].mat_type != 3) {
            HitRecord srec;
            if (hit_sphere(spheres[s], r, 0.001f, t_floor - 0.002f, srec)) {
                return;
            }
        }
    }

    if (p_floor.x < min_x || p_floor.x > max_x || p_floor.z < min_z || p_floor.z > max_z) return;

    float cell_w = (max_x - min_x) / (float)grid_w;
    float cell_h = (max_z - min_z) / (float)grid_h;
    float cell_area = cell_w * cell_h;

    Vec3 deposit = photon_flux * (1.0f / (3.14159265f * cell_area));
    // Limit pojedynczego fotonu skalowany dla siatek gęstszych niż 2048x2048 (zachowanie jasności ostrych skupień)
    float ref_area = (11.0f / 2048.0f) * (11.0f / 2048.0f);
    float max_deposit = 120.0f * fmaxf(1.0f, ref_area / cell_area);
    float dep_lum = fmaxf(deposit.x, fmaxf(deposit.y, deposit.z));
    if (dep_lum > max_deposit) deposit = deposit * (max_deposit / dep_lum);

    float u = (p_floor.x - min_x) / (max_x - min_x) * (float)(grid_w - 1);
    float v = (p_floor.z - min_z) / (max_z - min_z) * (float)(grid_h - 1);

    int x0 = (int)u;
    int z0 = (int)v;
    int x1 = min(x0 + 1, grid_w - 1);
    int z1 = min(z0 + 1, grid_h - 1);

    float fx = u - (float)x0;
    float fz = v - (float)z0;

    float w00 = (1.0f - fx) * (1.0f - fz);
    float w10 = fx * (1.0f - fz);
    float w01 = (1.0f - fx) * fz;
    float w11 = fx * fz;

    atomicAdd(&caustic_r[z0 * grid_w + x0], deposit.x * w00);
    atomicAdd(&caustic_g[z0 * grid_w + x0], deposit.y * w00);
    atomicAdd(&caustic_b[z0 * grid_w + x0], deposit.z * w00);

    atomicAdd(&caustic_r[z0 * grid_w + x1], deposit.x * w10);
    atomicAdd(&caustic_g[z0 * grid_w + x1], deposit.y * w10);
    atomicAdd(&caustic_b[z0 * grid_w + x1], deposit.z * w10);

    atomicAdd(&caustic_r[z1 * grid_w + x0], deposit.x * w01);
    atomicAdd(&caustic_g[z1 * grid_w + x0], deposit.y * w01);
    atomicAdd(&caustic_b[z1 * grid_w + x0], deposit.z * w01);

    atomicAdd(&caustic_r[z1 * grid_w + x1], deposit.x * w11);
    atomicAdd(&caustic_g[z1 * grid_w + x1], deposit.y * w11);
    atomicAdd(&caustic_b[z1 * grid_w + x1], deposit.z * w11);
}

// Filtr Gaussa siatki kaustyki o regulowanej sigmie (w komórkach siatki) i wzmocnieniu jasności.
// sigma_cells <= 0.05 -> brak rozmycia (surowa, najostrzejsza kaustyka)
extern "C" __global__ void filter_caustic_map_kernel(
    const float* in_r, const float* in_g, const float* in_b,
    float* out_r, float* out_g, float* out_b,
    int grid_w, int grid_h,
    float sigma_cells, float gain
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= grid_w || z >= grid_h) return;

    int out_idx = z * grid_w + x;
    if (sigma_cells <= 0.05f) {
        out_r[out_idx] = in_r[out_idx] * gain;
        out_g[out_idx] = in_g[out_idx] * gain;
        out_b[out_idx] = in_b[out_idx] * gain;
        return;
    }

    int rad = (int)ceilf(2.5f * sigma_cells);
    if (rad < 1) rad = 1;
    if (rad > 8) rad = 8;
    float inv_2s2 = 1.0f / (2.0f * sigma_cells * sigma_cells);

    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
    float w_total = 0.0f;

    for (int dz = -rad; dz <= rad; ++dz) {
        int cz = z + dz;
        if (cz < 0 || cz >= grid_h) continue;
        for (int dx = -rad; dx <= rad; ++dx) {
            int cx = x + dx;
            if (cx < 0 || cx >= grid_w) continue;

            float w = expf(-(float)(dx * dx + dz * dz) * inv_2s2);
            int idx = cz * grid_w + cx;
            r_sum += in_r[idx] * w;
            g_sum += in_g[idx] * w;
            b_sum += in_b[idx] * w;
            w_total += w;
        }
    }

    float norm = (w_total > 1e-6f) ? (gain / w_total) : 0.0f;
    out_r[out_idx] = r_sum * norm;
    out_g[out_idx] = g_sum * norm;
    out_b[out_idx] = b_sum * norm;
}
)";

bool CudaPT_IsAvailable(std::string& out_device_name) {
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) return false;

    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count == 0) return false;

    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return false;

    char name[256] = {0};
    if (cuDeviceGetName(name, sizeof(name), dev) == CUDA_SUCCESS) {
        out_device_name = name;
    } else {
        out_device_name = "NVIDIA CUDA Device";
    }
    return true;
}

// Struktura kamery dla kernela GPU
struct KernelCameraData {
    CudaVec3 origin;
    CudaVec3 lower_left_corner;
    CudaVec3 horizontal;
    CudaVec3 vertical;
};

static KernelCameraData ComputeCameraData(
    const CudaVec3& lookfrom, const CudaVec3& lookat, const CudaVec3& vup,
    float vfov_deg, float aspect_ratio
) {
    auto dot_v = [](const CudaVec3& a, const CudaVec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    };
    auto cross_v = [](const CudaVec3& u, const CudaVec3& v) {
        return CudaVec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
    };
    auto sub_v = [](const CudaVec3& a, const CudaVec3& b) {
        return CudaVec3(a.x - b.x, a.y - b.y, a.z - b.z);
    };
    auto add_v = [](const CudaVec3& a, const CudaVec3& b) {
        return CudaVec3(a.x + b.x, a.y + b.y, a.z + b.z);
    };
    auto mul_v = [](const CudaVec3& a, float t) {
        return CudaVec3(a.x * t, a.y * t, a.z * t);
    };
    auto unit_v = [&](const CudaVec3& v) {
        float l = std::sqrt(dot_v(v, v));
        return l > 1e-8f ? mul_v(v, 1.0f / l) : CudaVec3(0, 0, 0);
    };

    float theta = vfov_deg * 3.14159265358979323846f / 180.0f;
    float h = std::tan(theta / 2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    CudaVec3 w = unit_v(sub_v(lookfrom, lookat));
    CudaVec3 u = unit_v(cross_v(vup, w));
    CudaVec3 v = cross_v(w, u);

    KernelCameraData cam;
    cam.origin = lookfrom;
    cam.horizontal = mul_v(u, viewport_width);
    cam.vertical = mul_v(v, viewport_height);
    cam.lower_left_corner = sub_v(sub_v(sub_v(cam.origin, mul_v(cam.horizontal, 0.5f)), mul_v(cam.vertical, 0.5f)), w);
    return cam;
}

// Struktura źródła światła po stronie hosta C++
struct HostGPULight {
    CudaVec3 center;
    float radius = 0.0f;
    CudaVec3 emission;
    float pad = 0.0f;
};

static bool GetCompiledPTX(std::vector<char>& out_ptx, CUdevice dev) {
    static std::vector<char> s_cached_ptx;
    static std::mutex s_ptx_mutex;
    std::lock_guard<std::mutex> lock(s_ptx_mutex);
    if (!s_cached_ptx.empty()) {
        out_ptx = s_cached_ptx;
        return true;
    }

    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    char arch_str[64];
    snprintf(arch_str, sizeof(arch_str), "--gpu-architecture=compute_%d%d", major, minor);

    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, s_cuda_pt_source, "cuda_pt_kernel.cu", 0, NULL, NULL) != NVRTC_SUCCESS) {
        return false;
    }

    const char* opts[] = { arch_str, "-use_fast_math" };
    nvrtcResult comp_res = nvrtcCompileProgram(prog, 2, opts);
    if (comp_res != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size + 1);
        nvrtcGetProgramLog(prog, log.data());
        std::cerr << "NVRTC Compilation Error:\n" << log.data() << std::endl;
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    s_cached_ptx.resize(ptx_size);
    nvrtcGetPTX(prog, s_cached_ptx.data());
    nvrtcDestroyProgram(&prog);

    out_ptx = s_cached_ptx;
    return true;
}

bool CudaPT_Render(
    int width, int height, int spp, int max_depth,
    const CudaVec3& cam_origin, const CudaVec3& cam_lookat, const CudaVec3& cam_vup, float cam_fov,
    const CudaSceneData& scene,
    std::vector<CudaVec3>& out_pixels,
    const CudaRenderOptions& options,
    std::atomic<bool>* cancel_flag,
    CudaProgressCallback progress_cb
) {
    if (width <= 0 || height <= 0 || spp <= 0) return false;

    // Zwolnienie zasobów viewportu przed pełnym renderem
    CudaPT_CleanupViewport();

    // 1. Inicjalizacja CUDA
    if (cuInit(0) != CUDA_SUCCESS) return false;

    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return false;

    // 2. Pobranie skompilowanego kodu PTX (zbuforowanego w pamięci RAM)
    std::vector<char> ptx;
    if (!GetCompiledPTX(ptx, dev)) return false;

    // 3. Utworzenie kontekstu CUDA i załadowanie modułu
    CUcontext ctx;
    if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) return false;

    CUmodule module;
    if (cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0) != CUDA_SUCCESS) {
        cuCtxDestroy(ctx);
        return false;
    }

    CUfunction render_kernel;
    if (cuModuleGetFunction(&render_kernel, module, "render_accum_kernel") != CUDA_SUCCESS) {
        cuModuleUnload(module);
        cuCtxDestroy(ctx);
        return false;
    }

    CUfunction tonemap_kernel;
    if (cuModuleGetFunction(&tonemap_kernel, module, "display_tonemap_kernel") != CUDA_SUCCESS) {
        cuModuleUnload(module);
        cuCtxDestroy(ctx);
        return false;
    }

    // 4. Przygotowanie danych kamery
    float aspect = (float)width / (float)height;
    KernelCameraData cam_data = ComputeCameraData(cam_origin, cam_lookat, cam_vup, cam_fov, aspect);

    size_t num_pixels = (size_t)width * height;
    size_t fb_size = num_pixels * sizeof(CudaVec3);
    size_t display_size = num_pixels * sizeof(uint32_t);

    CUdeviceptr d_fb = 0;
    CUdeviceptr d_display_bgra = 0;
    CUdeviceptr d_spheres = 0;
    CUdeviceptr d_quads = 0;
    CUdeviceptr d_crystals = 0;
    CUdeviceptr d_crystal_planes = 0;
    CUdeviceptr d_meshes = 0;
    CUdeviceptr d_triangles = 0;
    CUdeviceptr d_lights = 0;

    cuMemAlloc(&d_fb, fb_size);
    cuMemAlloc(&d_display_bgra, display_size);

    if (!scene.spheres.empty()) {
        size_t sz = scene.spheres.size() * sizeof(CudaSphere);
        cuMemAlloc(&d_spheres, sz);
        cuMemcpyHtoD(d_spheres, scene.spheres.data(), sz);
    }
    if (!scene.quads.empty()) {
        size_t sz = scene.quads.size() * sizeof(CudaQuad);
        cuMemAlloc(&d_quads, sz);
        cuMemcpyHtoD(d_quads, scene.quads.data(), sz);
    }
    if (!scene.crystals.empty()) {
        size_t sz = scene.crystals.size() * sizeof(CudaCrystal);
        cuMemAlloc(&d_crystals, sz);
        cuMemcpyHtoD(d_crystals, scene.crystals.data(), sz);
    }
    if (!scene.crystal_planes.empty()) {
        size_t sz = scene.crystal_planes.size() * sizeof(CudaPlane);
        cuMemAlloc(&d_crystal_planes, sz);
        cuMemcpyHtoD(d_crystal_planes, scene.crystal_planes.data(), sz);
    }
    if (!scene.meshes.empty()) {
        size_t sz = scene.meshes.size() * sizeof(CudaMesh);
        cuMemAlloc(&d_meshes, sz);
        cuMemcpyHtoD(d_meshes, scene.meshes.data(), sz);
    }
    if (!scene.triangles.empty()) {
        size_t sz = scene.triangles.size() * sizeof(CudaTriangle);
        cuMemAlloc(&d_triangles, sz);
        cuMemcpyHtoD(d_triangles, scene.triangles.data(), sz);
    }

    // Wykrywanie źródeł światła dla Next Event Estimation (Direct Light Sampling w stylu Cycles)
    std::vector<HostGPULight> lights;
    for (const auto& s : scene.spheres) {
        if (s.mat_type == CUDA_MAT_LIGHT) {
            HostGPULight l;
            l.center = s.center;
            l.radius = s.radius;
            l.emission = CudaVec3(s.color.x * s.param, s.color.y * s.param, s.color.z * s.param);
            lights.push_back(l);
        }
    }
    if (!lights.empty()) {
        size_t sz = lights.size() * sizeof(HostGPULight);
        cuMemAlloc(&d_lights, sz);
        cuMemcpyHtoD(d_lights, lights.data(), sz);
    }

    int num_spheres = static_cast<int>(scene.spheres.size());
    int num_quads = static_cast<int>(scene.quads.size());
    int num_crystals = static_cast<int>(scene.crystals.size());
    int num_meshes = static_cast<int>(scene.meshes.size());
    int num_lights = static_cast<int>(lights.size());

    // Alokacja i obliczenie mapy kaustyki GPU z dyspersją Cauchy'ego
    int enable_caustics = options.enable_caustics;
    int caustic_w = (options.caustic_grid_res > 0) ? std::clamp(options.caustic_grid_res, 256, 4096) : (options.caustic_hq ? 2048 : 1024);
    int caustic_h = caustic_w;
    float c_min_x = -5.5f, c_max_x = 5.5f;
    float c_min_z = -5.5f, c_max_z = 5.5f;
    size_t caustic_size = (size_t)caustic_w * caustic_h * sizeof(float);
    CUdeviceptr d_caustic_raw_r = 0, d_caustic_raw_g = 0, d_caustic_raw_b = 0;
    CUdeviceptr d_caustic_filtered_r = 0, d_caustic_filtered_g = 0, d_caustic_filtered_b = 0;

    bool has_dielectric = (num_crystals > 0);
    for (const auto& s : scene.spheres) {
        if (s.mat_type == CUDA_MAT_DIELECTRIC) {
            has_dielectric = true;
            break;
        }
    }

    if (enable_caustics) {
        cuMemAlloc(&d_caustic_raw_r, caustic_size);
        cuMemAlloc(&d_caustic_raw_g, caustic_size);
        cuMemAlloc(&d_caustic_raw_b, caustic_size);
        cuMemAlloc(&d_caustic_filtered_r, caustic_size);
        cuMemAlloc(&d_caustic_filtered_g, caustic_size);
        cuMemAlloc(&d_caustic_filtered_b, caustic_size);

        cuMemsetD8(d_caustic_raw_r, 0, caustic_size);
        cuMemsetD8(d_caustic_raw_g, 0, caustic_size);
        cuMemsetD8(d_caustic_raw_b, 0, caustic_size);
        cuMemsetD8(d_caustic_filtered_r, 0, caustic_size);
        cuMemsetD8(d_caustic_filtered_g, 0, caustic_size);
        cuMemsetD8(d_caustic_filtered_b, 0, caustic_size);

        if (has_dielectric && num_lights > 0) {
            CUfunction emit_caustic_kernel = nullptr;
            CUfunction filter_caustic_kernel = nullptr;
            if (cuModuleGetFunction(&emit_caustic_kernel, module, "emit_caustic_photons_kernel") == CUDA_SUCCESS &&
                cuModuleGetFunction(&filter_caustic_kernel, module, "filter_caustic_map_kernel") == CUDA_SUCCESS) {

                int n_photons = (options.caustic_photons > 0) ? options.caustic_photons : (options.caustic_hq ? 1048576 : 524288);
                int sc_mode = options.scene_mode;
                unsigned int c_seed = 987654321u;
                float c_light_size = std::clamp(options.caustic_light_size, 0.0f, 1.0f);
                float c_sigma_cells = std::max(0.0f, options.caustic_blur_cm) * 0.01f / ((c_max_x - c_min_x) / (float)caustic_w);
                float c_gain = std::max(0.0f, options.caustic_gain);

                void* emit_args[] = {
                    &d_caustic_raw_r,
                    &d_caustic_raw_g,
                    &d_caustic_raw_b,
                    &caustic_w,
                    &caustic_h,
                    &c_min_x,
                    &c_max_x,
                    &c_min_z,
                    &c_max_z,
                    &n_photons,
                    &sc_mode,
                    &d_lights,
                    &num_lights,
                    &d_crystals,
                    &num_crystals,
                    &d_crystal_planes,
                    &d_spheres,
                    &num_spheres,
                    &d_quads,
                    &num_quads,
                    &c_seed,
                    &c_light_size
                };

                cuLaunchKernel(
                    emit_caustic_kernel,
                    (n_photons + 255) / 256, 1, 1,
                    256, 1, 1,
                    0, 0, emit_args, 0
                );

                void* filter_args[] = {
                    &d_caustic_raw_r,
                    &d_caustic_raw_g,
                    &d_caustic_raw_b,
                    &d_caustic_filtered_r,
                    &d_caustic_filtered_g,
                    &d_caustic_filtered_b,
                    &caustic_w,
                    &caustic_h,
                    &c_sigma_cells,
                    &c_gain
                };

                cuLaunchKernel(
                    filter_caustic_kernel,
                    (caustic_w + 15) / 16, (caustic_h + 15) / 16, 1,
                    16, 16, 1,
                    0, 0, filter_args, 0
                );
                cuCtxSynchronize();
            }
        }
    }

    int block_dim_x = 16;
    int block_dim_y = 16;
    int grid_dim_x = (width + block_dim_x - 1) / block_dim_x;
    int grid_dim_y = (height + block_dim_y - 1) / block_dim_y;

    // 5. Wieloprzebiegowy rendering GPU z pełną saturacją rdzeni CUDA (Cycles Throughput)
    int total_done_spp = 0;
    unsigned int pass_count = 0;

    out_pixels.assign(num_pixels, CudaVec3(0, 0, 0));
    std::vector<uint32_t> host_display_bgra(num_pixels);
    std::vector<CudaVec3> temp_accum(num_pixels);
    auto last_cb_time = std::chrono::high_resolution_clock::now();

    while (total_done_spp < spp) {
        if (cancel_flag && cancel_flag->load()) {
            break;
        }

        // Adaptacyjne porcje SPP: szybki start dla natychmiastowego podglądu UI,
        // a następnie duże porcje (32-64 SPP) wysycające 60 SM karty RTX 4070 Ti do 100%!
        int this_pass_spp;
        if (total_done_spp == 0) {
            this_pass_spp = std::min(8, spp);
        } else if (total_done_spp < 32) {
            this_pass_spp = std::min(16, spp - total_done_spp);
        } else if (total_done_spp < 96) {
            this_pass_spp = std::min(32, spp - total_done_spp);
        } else {
            this_pass_spp = std::min(64, spp - total_done_spp);
        }

        unsigned int seed = 1337u + pass_count * 7919u;
        float clamp_ind = options.clamp_indirect;

        void* args[] = {
            &d_fb,
            &width,
            &height,
            &this_pass_spp,
            &total_done_spp,
            &max_depth,
            &clamp_ind,
            &cam_data,
            &d_spheres,
            &num_spheres,
            &d_quads,
            &num_quads,
            &d_crystals,
            &num_crystals,
            &d_crystal_planes,
            &d_meshes,
            &num_meshes,
            &d_triangles,
            &d_lights,
            &num_lights,
            &enable_caustics,
            &d_caustic_filtered_r,
            &d_caustic_filtered_g,
            &d_caustic_filtered_b,
            &caustic_w,
            &caustic_h,
            &c_min_x,
            &c_max_x,
            &c_min_z,
            &c_max_z,
            &seed
        };

        CUresult launch_res = cuLaunchKernel(
            render_kernel,
            grid_dim_x, grid_dim_y, 1,
            block_dim_x, block_dim_y, 1,
            0, 0,
            args, 0
        );

        if (launch_res != CUDA_SUCCESS) {
            std::cerr << "cuLaunchKernel error: " << launch_res << std::endl;
            break;
        }

        cuCtxSynchronize();
        total_done_spp += this_pass_spp;
        pass_count++;

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_since_cb = now - last_cb_time;
        bool is_final = (total_done_spp >= spp);
        // Odświeżanie podglądu z płynnością ~15-20 FPS (co min. 50 ms), bez blokowania kolejki GPU!
        bool should_refresh_preview = (pass_count <= 2 || is_final || elapsed_since_cb.count() >= 0.05);

        if (progress_cb) {
            if (should_refresh_preview) {
                // Wykonanie tone mappingu bezpośrednio na GPU w 0.05 ms (zamiast 65 ms pętli CPU!)
                float exposure_scale = std::pow(2.0f, options.exposure_ev);
                int tm_mode = options.tonemap_mode;
                void* tm_args[] = {
                    &d_fb,
                    &d_display_bgra,
                    &width,
                    &height,
                    &total_done_spp,
                    &exposure_scale,
                    &tm_mode
                };
                cuLaunchKernel(
                    tonemap_kernel,
                    grid_dim_x, grid_dim_y, 1,
                    block_dim_x, block_dim_y, 1,
                    0, 0,
                    tm_args, 0
                );
                // Błyskawiczny transfer 32-bit BGRA (zajmuje 0.4 ms)
                cuMemcpyDtoH(host_display_bgra.data(), d_display_bgra, display_size);
                progress_cb(total_done_spp, spp, nullptr, host_display_bgra.data());
                last_cb_time = now;
            } else {
                progress_cb(total_done_spp, spp, nullptr, nullptr);
            }
        }
    }

    // 6. Pobranie ostatecznie zsumowanych pikseli HDR z GPU (wykonywane tylko RAZ na koniec!)
    if (total_done_spp > 0) {
        cuMemcpyDtoH(temp_accum.data(), d_fb, fb_size);

        float inv_spp = 1.0f / (float)total_done_spp;
        for (size_t i = 0; i < num_pixels; ++i) {
            out_pixels[i] = CudaVec3(temp_accum[i].x * inv_spp, temp_accum[i].y * inv_spp, temp_accum[i].z * inv_spp);
        }
    }

    // 7. Sprzątanie zasobów GPU
    if (d_caustic_raw_r) cuMemFree(d_caustic_raw_r);
    if (d_caustic_raw_g) cuMemFree(d_caustic_raw_g);
    if (d_caustic_raw_b) cuMemFree(d_caustic_raw_b);
    if (d_caustic_filtered_r) cuMemFree(d_caustic_filtered_r);
    if (d_caustic_filtered_g) cuMemFree(d_caustic_filtered_g);
    if (d_caustic_filtered_b) cuMemFree(d_caustic_filtered_b);
    if (d_fb) cuMemFree(d_fb);
    if (d_display_bgra) cuMemFree(d_display_bgra);
    if (d_spheres) cuMemFree(d_spheres);
    if (d_quads) cuMemFree(d_quads);
    if (d_crystals) cuMemFree(d_crystals);
    if (d_crystal_planes) cuMemFree(d_crystal_planes);
    if (d_meshes) cuMemFree(d_meshes);
    if (d_triangles) cuMemFree(d_triangles);
    if (d_lights) cuMemFree(d_lights);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    return total_done_spp > 0;
}

bool CudaPT_Render(
    int width, int height, int spp, int max_depth,
    const CudaVec3& cam_origin, const CudaVec3& cam_lookat, const CudaVec3& cam_vup, float cam_fov,
    const std::vector<CudaSphere>& spheres,
    std::vector<CudaVec3>& out_pixels,
    const CudaRenderOptions& options,
    std::atomic<bool>* cancel_flag,
    CudaProgressCallback progress_cb
) {
    CudaSceneData scene;
    scene.spheres = spheres;
    return CudaPT_Render(width, height, spp, max_depth, cam_origin, cam_lookat, cam_vup, cam_fov, scene, out_pixels, options, cancel_flag, progress_cb);
}

// ============================================================================
// Interaktywny podgląd Path Tracingu w viewporcie na żywo (GPU CUDA / RTX)
// ============================================================================
static CUcontext    s_vp_ctx = nullptr;
static CUmodule     s_vp_module = nullptr;
static CUfunction   s_vp_render_kernel = nullptr;
static CUfunction   s_vp_tonemap_kernel = nullptr;
static CUfunction   s_vp_emit_caustic_kernel = nullptr;
static CUfunction   s_vp_filter_caustic_kernel = nullptr;

static int          s_vp_width = 0;
static int          s_vp_height = 0;
static CUdeviceptr  s_vp_fb = 0;
static CUdeviceptr  s_vp_display_bgra = 0;

static CUdeviceptr  s_vp_spheres = 0;
static CUdeviceptr  s_vp_quads = 0;
static CUdeviceptr  s_vp_crystals = 0;
static CUdeviceptr  s_vp_crystal_planes = 0;
static CUdeviceptr  s_vp_meshes = 0;
static CUdeviceptr  s_vp_triangles = 0;
static CUdeviceptr  s_vp_lights = 0;

static CUdeviceptr  s_vp_caustic_raw_r = 0;
static CUdeviceptr  s_vp_caustic_raw_g = 0;
static CUdeviceptr  s_vp_caustic_raw_b = 0;
static CUdeviceptr  s_vp_caustic_filtered_r = 0;
static CUdeviceptr  s_vp_caustic_filtered_g = 0;
static CUdeviceptr  s_vp_caustic_filtered_b = 0;

static int          s_vp_num_spheres = 0;
static int          s_vp_num_quads = 0;
static int          s_vp_num_crystals = 0;
static int          s_vp_num_meshes = 0;
static int          s_vp_num_lights = 0;

static unsigned int s_vp_seed_counter = 1337;
static std::mutex   s_vp_mutex;

void CudaPT_CleanupViewport() {
    std::lock_guard<std::mutex> lock(s_vp_mutex);
    if (s_vp_ctx) {
        cuCtxSetCurrent(s_vp_ctx);

        if (s_vp_fb) { cuMemFree(s_vp_fb); s_vp_fb = 0; }
        if (s_vp_display_bgra) { cuMemFree(s_vp_display_bgra); s_vp_display_bgra = 0; }
        if (s_vp_spheres) { cuMemFree(s_vp_spheres); s_vp_spheres = 0; }
        if (s_vp_quads) { cuMemFree(s_vp_quads); s_vp_quads = 0; }
        if (s_vp_crystals) { cuMemFree(s_vp_crystals); s_vp_crystals = 0; }
        if (s_vp_crystal_planes) { cuMemFree(s_vp_crystal_planes); s_vp_crystal_planes = 0; }
        if (s_vp_meshes) { cuMemFree(s_vp_meshes); s_vp_meshes = 0; }
        if (s_vp_triangles) { cuMemFree(s_vp_triangles); s_vp_triangles = 0; }
        if (s_vp_lights) { cuMemFree(s_vp_lights); s_vp_lights = 0; }

        if (s_vp_caustic_raw_r) { cuMemFree(s_vp_caustic_raw_r); s_vp_caustic_raw_r = 0; }
        if (s_vp_caustic_raw_g) { cuMemFree(s_vp_caustic_raw_g); s_vp_caustic_raw_g = 0; }
        if (s_vp_caustic_raw_b) { cuMemFree(s_vp_caustic_raw_b); s_vp_caustic_raw_b = 0; }
        if (s_vp_caustic_filtered_r) { cuMemFree(s_vp_caustic_filtered_r); s_vp_caustic_filtered_r = 0; }
        if (s_vp_caustic_filtered_g) { cuMemFree(s_vp_caustic_filtered_g); s_vp_caustic_filtered_g = 0; }
        if (s_vp_caustic_filtered_b) { cuMemFree(s_vp_caustic_filtered_b); s_vp_caustic_filtered_b = 0; }

        if (s_vp_module) { cuModuleUnload(s_vp_module); s_vp_module = nullptr; }
        cuCtxDestroy(s_vp_ctx);
        s_vp_ctx = nullptr;
        s_vp_render_kernel = nullptr;
        s_vp_tonemap_kernel = nullptr;
        s_vp_emit_caustic_kernel = nullptr;
        s_vp_filter_caustic_kernel = nullptr;
        s_vp_width = 0;
        s_vp_height = 0;
        cuCtxSetCurrent(NULL);
    }
}

void CudaPT_ResetViewportAccum() {
    std::lock_guard<std::mutex> lock(s_vp_mutex);
    if (s_vp_ctx && s_vp_fb && s_vp_width > 0 && s_vp_height > 0) {
        cuCtxSetCurrent(s_vp_ctx);
        size_t fb_size = (size_t)s_vp_width * s_vp_height * sizeof(CudaVec3);
        cuMemsetD8(s_vp_fb, 0, fb_size);
    }
}

bool CudaPT_RenderViewportStep(
    int width, int height,
    int spp_step, int& inout_accum_spp,
    const CudaVec3& cam_origin, const CudaVec3& cam_lookat, const CudaVec3& cam_vup, float cam_fov,
    const CudaSceneData& scene,
    const CudaRenderOptions& options,
    uint32_t* out_display_bgra
) {
    if (width <= 0 || height <= 0 || !out_display_bgra || spp_step <= 0) return false;

    std::lock_guard<std::mutex> lock(s_vp_mutex);

    if (cuInit(0) != CUDA_SUCCESS) return false;
    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return false;

    // 1. Inicjalizacja kontekstu i modułu, jeśli nie istnieją
    if (!s_vp_ctx) {
        if (cuCtxCreate(&s_vp_ctx, 0, dev) != CUDA_SUCCESS) return false;
        std::vector<char> ptx;
        if (!GetCompiledPTX(ptx, dev)) {
            cuCtxDestroy(s_vp_ctx);
            s_vp_ctx = nullptr;
            return false;
        }
        if (cuModuleLoadDataEx(&s_vp_module, ptx.data(), 0, 0, 0) != CUDA_SUCCESS) {
            cuCtxDestroy(s_vp_ctx);
            s_vp_ctx = nullptr;
            return false;
        }
        cuModuleGetFunction(&s_vp_render_kernel, s_vp_module, "render_accum_kernel");
        cuModuleGetFunction(&s_vp_tonemap_kernel, s_vp_module, "display_tonemap_kernel");
        cuModuleGetFunction(&s_vp_emit_caustic_kernel, s_vp_module, "emit_caustic_photons_kernel");
        cuModuleGetFunction(&s_vp_filter_caustic_kernel, s_vp_module, "filter_caustic_map_kernel");
    } else {
        cuCtxSetCurrent(s_vp_ctx);
    }

    size_t num_pixels = (size_t)width * height;
    size_t fb_size = num_pixels * sizeof(CudaVec3);
    size_t display_size = num_pixels * sizeof(uint32_t);

    // 2. Alokacja lub zmiana rozmiaru buforów ekranowych
    if (width != s_vp_width || height != s_vp_height) {
        if (s_vp_fb) { cuMemFree(s_vp_fb); s_vp_fb = 0; }
        if (s_vp_display_bgra) { cuMemFree(s_vp_display_bgra); s_vp_display_bgra = 0; }

        cuMemAlloc(&s_vp_fb, fb_size);
        cuMemAlloc(&s_vp_display_bgra, display_size);
        s_vp_width = width;
        s_vp_height = height;
        inout_accum_spp = 0;
    }

    // Alokacja buforów siatki kaustyki w viewporcie
    int caustic_w = (options.caustic_grid_res > 0) ? std::clamp(options.caustic_grid_res, 256, 4096) : 1024;
    int caustic_h = caustic_w;
    float c_min_x = -5.5f, c_max_x = 5.5f;
    float c_min_z = -5.5f, c_max_z = 5.5f;
    size_t caustic_size = (size_t)caustic_w * caustic_h * sizeof(float);
    static int s_vp_caustic_res = 0;
    if (s_vp_caustic_raw_r && s_vp_caustic_res != caustic_w) {
        // Zmiana rozdzielczości siatki kaustyki -> realokacja buforów i restart akumulacji
        cuMemFree(s_vp_caustic_raw_r); cuMemFree(s_vp_caustic_raw_g); cuMemFree(s_vp_caustic_raw_b);
        cuMemFree(s_vp_caustic_filtered_r); cuMemFree(s_vp_caustic_filtered_g); cuMemFree(s_vp_caustic_filtered_b);
        s_vp_caustic_raw_r = s_vp_caustic_raw_g = s_vp_caustic_raw_b = 0;
        s_vp_caustic_filtered_r = s_vp_caustic_filtered_g = s_vp_caustic_filtered_b = 0;
        inout_accum_spp = 0;
    }
    s_vp_caustic_res = caustic_w;
    if (!s_vp_caustic_raw_r) {
        cuMemAlloc(&s_vp_caustic_raw_r, caustic_size);
        cuMemAlloc(&s_vp_caustic_raw_g, caustic_size);
        cuMemAlloc(&s_vp_caustic_raw_b, caustic_size);
        cuMemAlloc(&s_vp_caustic_filtered_r, caustic_size);
        cuMemAlloc(&s_vp_caustic_filtered_g, caustic_size);
        cuMemAlloc(&s_vp_caustic_filtered_b, caustic_size);
    }

    // 3. Jeśli akumulacja została zresetowana (ruch kamery / zmiana pozycji lub mocy świateł)
    if (inout_accum_spp == 0) {
        cuMemsetD8(s_vp_fb, 0, fb_size);
        cuMemsetD8(s_vp_caustic_raw_r, 0, caustic_size);
        cuMemsetD8(s_vp_caustic_raw_g, 0, caustic_size);
        cuMemsetD8(s_vp_caustic_raw_b, 0, caustic_size);
        cuMemsetD8(s_vp_caustic_filtered_r, 0, caustic_size);
        cuMemsetD8(s_vp_caustic_filtered_g, 0, caustic_size);
        cuMemsetD8(s_vp_caustic_filtered_b, 0, caustic_size);

        // Zwolnienie starych buforów geometrii
        if (s_vp_spheres) { cuMemFree(s_vp_spheres); s_vp_spheres = 0; }
        if (s_vp_quads) { cuMemFree(s_vp_quads); s_vp_quads = 0; }
        if (s_vp_crystals) { cuMemFree(s_vp_crystals); s_vp_crystals = 0; }
        if (s_vp_crystal_planes) { cuMemFree(s_vp_crystal_planes); s_vp_crystal_planes = 0; }
        if (s_vp_meshes) { cuMemFree(s_vp_meshes); s_vp_meshes = 0; }
        if (s_vp_triangles) { cuMemFree(s_vp_triangles); s_vp_triangles = 0; }
        if (s_vp_lights) { cuMemFree(s_vp_lights); s_vp_lights = 0; }

        s_vp_num_spheres = static_cast<int>(scene.spheres.size());
        s_vp_num_quads = static_cast<int>(scene.quads.size());
        s_vp_num_crystals = static_cast<int>(scene.crystals.size());
        s_vp_num_meshes = static_cast<int>(scene.meshes.size());

        if (!scene.spheres.empty()) {
            size_t sz = scene.spheres.size() * sizeof(CudaSphere);
            cuMemAlloc(&s_vp_spheres, sz);
            cuMemcpyHtoD(s_vp_spheres, scene.spheres.data(), sz);
        }
        if (!scene.quads.empty()) {
            size_t sz = scene.quads.size() * sizeof(CudaQuad);
            cuMemAlloc(&s_vp_quads, sz);
            cuMemcpyHtoD(s_vp_quads, scene.quads.data(), sz);
        }
        if (!scene.crystals.empty()) {
            size_t sz = scene.crystals.size() * sizeof(CudaCrystal);
            cuMemAlloc(&s_vp_crystals, sz);
            cuMemcpyHtoD(s_vp_crystals, scene.crystals.data(), sz);
        }
        if (!scene.crystal_planes.empty()) {
            size_t sz = scene.crystal_planes.size() * sizeof(CudaPlane);
            cuMemAlloc(&s_vp_crystal_planes, sz);
            cuMemcpyHtoD(s_vp_crystal_planes, scene.crystal_planes.data(), sz);
        }
        if (!scene.meshes.empty()) {
            size_t sz = scene.meshes.size() * sizeof(CudaMesh);
            cuMemAlloc(&s_vp_meshes, sz);
            cuMemcpyHtoD(s_vp_meshes, scene.meshes.data(), sz);
        }
        if (!scene.triangles.empty()) {
            size_t sz = scene.triangles.size() * sizeof(CudaTriangle);
            cuMemAlloc(&s_vp_triangles, sz);
            cuMemcpyHtoD(s_vp_triangles, scene.triangles.data(), sz);
        }

        // Źródła światła dla NEE (Direct Light Sampling)
        std::vector<HostGPULight> lights;
        for (const auto& s : scene.spheres) {
            if (s.mat_type == CUDA_MAT_LIGHT) {
                HostGPULight l;
                l.center = s.center;
                l.radius = s.radius;
                l.emission = CudaVec3(s.color.x * s.param, s.color.y * s.param, s.color.z * s.param);
                lights.push_back(l);
            }
        }
        s_vp_num_lights = static_cast<int>(lights.size());
        if (!lights.empty()) {
            size_t sz = lights.size() * sizeof(HostGPULight);
            cuMemAlloc(&s_vp_lights, sz);
            cuMemcpyHtoD(s_vp_lights, lights.data(), sz);
        }

        // Emisja fotonów kaustyki GPU na żywo w viewporcie (<1 ms)
        bool vp_has_dielectric = (s_vp_num_crystals > 0);
        for (const auto& s : scene.spheres) {
            if (s.mat_type == CUDA_MAT_DIELECTRIC) {
                vp_has_dielectric = true;
                break;
            }
        }

        if (options.enable_caustics && vp_has_dielectric && s_vp_num_lights > 0 && s_vp_emit_caustic_kernel && s_vp_filter_caustic_kernel) {
            int n_photons = (options.caustic_photons > 0) ? options.caustic_photons : 262144;
            int sc_mode = options.scene_mode;
            unsigned int c_seed = 987654321u + s_vp_seed_counter;
            float c_light_size = std::clamp(options.caustic_light_size, 0.0f, 1.0f);
            float c_sigma_cells = std::max(0.0f, options.caustic_blur_cm) * 0.01f / ((c_max_x - c_min_x) / (float)caustic_w);
            float c_gain = std::max(0.0f, options.caustic_gain);

            void* emit_args[] = {
                &s_vp_caustic_raw_r,
                &s_vp_caustic_raw_g,
                &s_vp_caustic_raw_b,
                &caustic_w,
                &caustic_h,
                &c_min_x,
                &c_max_x,
                &c_min_z,
                &c_max_z,
                &n_photons,
                &sc_mode,
                &s_vp_lights,
                &s_vp_num_lights,
                &s_vp_crystals,
                &s_vp_num_crystals,
                &s_vp_crystal_planes,
                &s_vp_spheres,
                &s_vp_num_spheres,
                &s_vp_quads,
                &s_vp_num_quads,
                &c_seed,
                &c_light_size
            };

            cuLaunchKernel(
                s_vp_emit_caustic_kernel,
                (n_photons + 255) / 256, 1, 1,
                256, 1, 1,
                0, 0, emit_args, 0
            );

            void* filter_args[] = {
                &s_vp_caustic_raw_r,
                &s_vp_caustic_raw_g,
                &s_vp_caustic_raw_b,
                &s_vp_caustic_filtered_r,
                &s_vp_caustic_filtered_g,
                &s_vp_caustic_filtered_b,
                &caustic_w,
                &caustic_h,
                &c_sigma_cells,
                &c_gain
            };

            cuLaunchKernel(
                s_vp_filter_caustic_kernel,
                (caustic_w + 15) / 16, (caustic_h + 15) / 16, 1,
                16, 16, 1,
                0, 0, filter_args, 0
            );
            cuCtxSynchronize();
        }
    }

    // 4. Parametry kamery
    float aspect = (float)width / (float)height;
    KernelCameraData cam_data = ComputeCameraData(cam_origin, cam_lookat, cam_vup, cam_fov, aspect);

    int block_dim_x = 16;
    int block_dim_y = 16;
    int grid_dim_x = (width + block_dim_x - 1) / block_dim_x;
    int grid_dim_y = (height + block_dim_y - 1) / block_dim_y;

    int max_depth = 8;
    float clamp_ind = options.clamp_indirect;
    int enable_caustics = options.enable_caustics;
    s_vp_seed_counter += 2347u;
    unsigned int seed = s_vp_seed_counter;

    void* render_args[] = {
        &s_vp_fb,
        &width,
        &height,
        &spp_step,
        &inout_accum_spp,
        &max_depth,
        &clamp_ind,
        &cam_data,
        &s_vp_spheres,
        &s_vp_num_spheres,
        &s_vp_quads,
        &s_vp_num_quads,
        &s_vp_crystals,
        &s_vp_num_crystals,
        &s_vp_crystal_planes,
        &s_vp_meshes,
        &s_vp_num_meshes,
        &s_vp_triangles,
        &s_vp_lights,
        &s_vp_num_lights,
        &enable_caustics,
        &s_vp_caustic_filtered_r,
        &s_vp_caustic_filtered_g,
        &s_vp_caustic_filtered_b,
        &caustic_w,
        &caustic_h,
        &c_min_x,
        &c_max_x,
        &c_min_z,
        &c_max_z,
        &seed
    };

    CUresult launch_res = cuLaunchKernel(
        s_vp_render_kernel,
        grid_dim_x, grid_dim_y, 1,
        block_dim_x, block_dim_y, 1,
        0, 0,
        render_args, 0
    );

    if (launch_res != CUDA_SUCCESS) {
        std::cerr << "Viewport render cuLaunchKernel error: " << launch_res << std::endl;
        return false;
    }

    cuCtxSynchronize();
    inout_accum_spp += spp_step;

    // 5. Tone mapping bezpośrednio na GPU (0.02 ms)
    float exposure_scale = std::pow(2.0f, options.exposure_ev);
    int tm_mode = options.tonemap_mode;
    void* tm_args[] = {
        &s_vp_fb,
        &s_vp_display_bgra,
        &width,
        &height,
        &inout_accum_spp,
        &exposure_scale,
        &tm_mode
    };

    cuLaunchKernel(
        s_vp_tonemap_kernel,
        grid_dim_x, grid_dim_y, 1,
        block_dim_x, block_dim_y, 1,
        0, 0,
        tm_args, 0
    );

    cuMemcpyDtoH(out_display_bgra, s_vp_display_bgra, display_size);
    return true;
}
