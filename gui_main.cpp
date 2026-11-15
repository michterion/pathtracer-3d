#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <commctrl.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdint>
#include <queue>
#include <condition_variable>
#include <functional>
#include <OpenImageDenoise/oidn.hpp>
#include "cuda_pt.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "comdlg32.lib")
#include <commdlg.h>


// ============================================================================
// 1. MATEMATYKA PATH TRACINGU (Vec3, Ray, HitRecord, Material)
// ============================================================================

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 operator*(float t) const { return Vec3(x * t, y * t, z * t); }
    Vec3 operator/(float t) const { return Vec3(x / t, y / t, z / t); }

    float length_squared() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(length_squared()); }

    bool near_zero() const {
        constexpr float eps = 1e-8f;
        return (std::fabs(x) < eps) && (std::fabs(y) < eps) && (std::fabs(z) < eps);
    }
};

inline Vec3 operator*(float t, const Vec3& v) { return v * t; }
inline float dot(const Vec3& u, const Vec3& v) { return u.x * v.x + u.y * v.y + u.z * v.z; }
inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x);
}
inline Vec3 unit_vector(const Vec3& v) {
    float l = v.length();
    return l > 1e-8f ? v / l : Vec3(0, 0, 0);
}

inline Vec3 random_unit_vector(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    while (true) {
        Vec3 p(dist(rng), dist(rng), dist(rng));
        float len_sq = p.length_squared();
        if (len_sq > 1e-6f && len_sq <= 1.0f) {
            return p / std::sqrt(len_sq);
        }
    }
}

inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = std::fmin(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -std::sqrt(std::fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

inline float schlick_reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * std::pow(1.0f - cosine, 5.0f);
}

struct Ray {
    Vec3 origin;
    Vec3 direction;
    Ray() = default;
    Ray(const Vec3& o, const Vec3& d) : origin(o), direction(d) {}
    Vec3 at(float t) const { return origin + t * direction; }
};

class Material;

struct HitRecord {
    Vec3 p;
    Vec3 normal;
    std::shared_ptr<Material> mat;
    float t = 0.0f;
    bool front_face = true;

    void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Material {
public:
    virtual ~Material() = default;
    virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, std::mt19937& rng) const {
        return false;
    }
    virtual Vec3 emitted(const Vec3& p) const { return Vec3(0, 0, 0); }
    virtual bool is_specular() const { return false; }
    virtual bool is_dielectric() const { return false; }
    virtual Vec3 get_albedo() const { return Vec3(0, 0, 0); }
};

class Lambertian : public Material {
public:
    Vec3 albedo;
    explicit Lambertian(const Vec3& a) : albedo(a) {}

    bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, std::mt19937& rng) const override {
        Vec3 scatter_direction = rec.normal + random_unit_vector(rng);
        if (scatter_direction.near_zero()) scatter_direction = rec.normal;
        scattered = Ray(rec.p, unit_vector(scatter_direction));
        attenuation = albedo;
        return true;
    }
    Vec3 get_albedo() const override { return albedo; }
};

class Metal : public Material {
public:
    Vec3 albedo;
    float roughness;
    Metal(const Vec3& a, float r) : albedo(a), roughness(r < 1.0f ? r : 1.0f) {}

    bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, std::mt19937& rng) const override {
        Vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
        scattered = Ray(rec.p, unit_vector(reflected + roughness * random_unit_vector(rng)));
        attenuation = albedo;
        return (dot(scattered.direction, rec.normal) > 0);
    }
    bool is_specular() const override { return roughness < 0.15f; }
    Vec3 get_albedo() const override { return albedo; }
};

class Dielectric : public Material {
public:
    float ir;
    float dispersion = 0.0f;
    explicit Dielectric(float index_of_refraction, float disp = 0.0f)
        : ir(index_of_refraction), dispersion(disp) {}

    bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, std::mt19937& rng) const override {
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
        float effective_ir = ir;
        attenuation = Vec3(1.0f, 1.0f, 1.0f);

        if (dispersion > 1e-4f) {
            float lambda = dist01(rng);
            effective_ir = ir + (2.0f * lambda - 1.0f) * dispersion;
            float wr = std::exp(-((lambda - 0.15f) * (lambda - 0.15f)) / 0.14f);
            float wg = std::exp(-((lambda - 0.50f) * (lambda - 0.50f)) / 0.14f);
            float wb = std::exp(-((lambda - 0.85f) * (lambda - 0.85f)) / 0.14f);
            Vec3 spec(wr, wg, wb);
            float avg = (spec.x + spec.y + spec.z) / 3.0f;
            if (avg > 1e-4f) spec = spec / avg;
            float blend = std::clamp(dispersion * 10.0f, 0.0f, 1.0f);
            attenuation = (1.0f - blend) * Vec3(1.0f, 1.0f, 1.0f) + blend * spec;
        }

        float refraction_ratio = rec.front_face ? (1.0f / effective_ir) : effective_ir;

        Vec3 unit_direction = unit_vector(r_in.direction);
        float cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

        Vec3 direction;
        if (cannot_refract || schlick_reflectance(cos_theta, refraction_ratio) > dist01(rng)) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = Ray(rec.p, direction);
        return true;
    }
    bool is_specular() const override { return true; }
    bool is_dielectric() const override { return true; }
    Vec3 get_albedo() const override { return Vec3(1.0f, 1.0f, 1.0f); }
};

class DiffuseLight : public Material {
public:
    Vec3 emit_color;
    explicit DiffuseLight(const Vec3& c) : emit_color(c) {}
    Vec3 emitted(const Vec3& p) const override { return emit_color; }
    Vec3 get_albedo() const override { return emit_color; }
};

// ============================================================================
// 2. GEOMETRIA: SFERA ORAZ SZLIFOWANY KRYSZTAŁ SZKLANY (CONVEX POLYHEDRON)
// ============================================================================

class Hittable {
public:
    virtual ~Hittable() = default;
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};

class Sphere : public Hittable {
public:
    Vec3 center;
    float radius;
    std::shared_ptr<Material> mat;

    Sphere(const Vec3& c, float r, std::shared_ptr<Material> m)
        : center(c), radius(r), mat(std::move(m)) {}

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
        Vec3 oc = r.origin - center;
        float a = r.direction.length_squared();
        float half_b = dot(oc, r.direction);
        float c = oc.length_squared() - radius * radius;
        float discriminant = half_b * half_b - a * c;

        if (discriminant < 0) return false;
        float sqrtd = std::sqrt(discriminant);

        float root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root) return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        Vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;
        return true;
    }
};

struct PolyPlane {
    Vec3 n;
    float d;
};

class ConvexPolyhedron : public Hittable {
public:
    Vec3 center;
    float bounding_radius = 1.0f;
    std::vector<PolyPlane> planes;
    std::shared_ptr<Material> mat;

    ConvexPolyhedron(const Vec3& c, std::shared_ptr<Material> m, float b_rad = 1.2f)
        : center(c), bounding_radius(b_rad), mat(std::move(m)) {}

    void add_plane(const Vec3& normal, float dist) {
        planes.push_back({unit_vector(normal), dist});
    }

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
        // 1. Szybki test sfery otaczającej (Bounding Sphere Culling)
        Vec3 oc = r.origin - center;
        float a = dot(r.direction, r.direction);
        float half_b = dot(oc, r.direction);
        float c = dot(oc, oc) - bounding_radius * bounding_radius;
        if (c > 0.0f && half_b > 0.0f) return false;
        float discr = half_b * half_b - a * c;
        if (discr < 0.0f) return false;

        float sqrtd = std::sqrt(discr);
        float t0 = (-half_b - sqrtd) / a;
        float t1 = (-half_b + sqrtd) / a;
        if (t1 < t_min || t0 > t_max) return false;

        // 2. Precyzyjny test slabów wielościanu wypukłego
        float t_enter = -1e9f;
        float t_exit  = 1e9f;
        Vec3 n_enter(0.0f, 1.0f, 0.0f);
        Vec3 n_exit(0.0f, 1.0f, 0.0f);

        for (const auto& plane : planes) {
            float denom = dot(plane.n, r.direction);
            float numer = plane.d - dot(plane.n, oc);

            if (std::fabs(denom) < 1e-7f) {
                if (numer < 0.0f) {
                    return false;
                }
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
        rec.set_face_normal(r, hit_normal);
        rec.mat = mat;
        return true;
    }
};

std::shared_ptr<ConvexPolyhedron> CreateFacetedCrystal(const Vec3& center, float radius, float height, std::shared_ptr<Material> mat) {
    float half_h = height * 0.5f;
    float b_rad = std::sqrt(radius * radius + half_h * half_h) * 1.18f;
    auto crystal = std::make_shared<ConvexPolyhedron>(center, mat, b_rad);

    crystal->add_plane(Vec3(0.0f, -1.0f, 0.0f), half_h);
    crystal->add_plane(Vec3(0.0f, 1.0f, 0.0f), half_h);

    for (int i = 0; i < 6; ++i) {
        float angle = i * (3.14159265f / 3.0f);
        crystal->add_plane(Vec3(std::cos(angle), 0.0f, std::sin(angle)), radius);
    }

    for (int i = 0; i < 6; ++i) {
        float angle = (i + 0.5f) * (3.14159265f / 3.0f);
        Vec3 slanted_norm = unit_vector(Vec3(std::cos(angle) * 0.85f, 1.15f, std::sin(angle) * 0.85f));
        crystal->add_plane(slanted_norm, radius * 1.08f);
    }

    for (int i = 0; i < 6; ++i) {
        float angle = i * (3.14159265f / 3.0f);
        Vec3 slanted_norm = unit_vector(Vec3(std::cos(angle) * 0.80f, -1.25f, std::sin(angle) * 0.80f));
        crystal->add_plane(slanted_norm, radius * 1.18f);
    }

    return crystal;
}

class Quad : public Hittable {
public:
    Vec3 q;
    Vec3 u;
    Vec3 v;
    Vec3 normal;
    float d;
    Vec3 w;
    std::shared_ptr<Material> mat;

    Quad(const Vec3& corner, const Vec3& u_, const Vec3& v_, std::shared_ptr<Material> m)
        : q(corner), u(u_), v(v_), mat(std::move(m)) {
        Vec3 n = cross(u, v);
        normal = unit_vector(n);
        d = dot(normal, q);
        float len_sq = dot(n, n);
        w = len_sq > 1e-8f ? (n / len_sq) : Vec3(0, 0, 0);
    }

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
        float denom = dot(normal, r.direction);
        if (std::fabs(denom) < 1e-8f) return false;

        float t = (d - dot(normal, r.origin)) / denom;
        if (t < t_min || t > t_max) return false;

        Vec3 intersection = r.at(t);
        Vec3 planar_hit = intersection - q;
        float alpha = dot(w, cross(planar_hit, v));
        float beta  = dot(w, cross(u, planar_hit));

        if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) {
            return false;
        }

        rec.t = t;
        rec.p = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, normal);
        return true;
    }
};

struct Triangle {
    Vec3 v0, v1, v2;
    Vec3 normal;
};

inline bool hit_aabb(const Ray& r, float t_min, float t_max, const Vec3& b_min, const Vec3& b_max) {
    for (int a = 0; a < 3; ++a) {
        float r_dir  = (a == 0) ? r.direction.x : ((a == 1) ? r.direction.y : r.direction.z);
        float r_orig = (a == 0) ? r.origin.x    : ((a == 1) ? r.origin.y    : r.origin.z);
        float min_v  = (a == 0) ? b_min.x       : ((a == 1) ? b_min.y       : b_min.z);
        float max_v  = (a == 0) ? b_max.x       : ((a == 1) ? b_max.y       : b_max.z);
        float invD   = 1.0f / (std::fabs(r_dir) > 1e-8f ? r_dir : (r_dir < 0 ? -1e-8f : 1e-8f));
        float t0 = (min_v - r_orig) * invD;
        float t1 = (max_v - r_orig) * invD;
        if (invD < 0.0f) std::swap(t0, t1);
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) return false;
    }
    return true;
}

inline bool hit_triangle(const Ray& r, float t_min, float t_max, const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& normal, HitRecord& rec) {
    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 h = cross(r.direction, edge2);
    float a = dot(edge1, h);
    if (std::fabs(a) < 1e-7f) return false;
    float f = 1.0f / a;
    Vec3 s = r.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    Vec3 q = cross(s, edge1);
    float v = f * dot(r.direction, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = f * dot(edge2, q);
    if (t < t_min || t > t_max) return false;
    rec.t = t;
    rec.p = r.at(t);
    rec.set_face_normal(r, normal);
    return true;
}

class TriangleMesh : public Hittable {
public:
    Vec3 center;
    float scale = 1.0f;
    std::vector<Triangle> triangles;
    Vec3 local_bmin;
    Vec3 local_bmax;
    std::shared_ptr<Material> mat;

    TriangleMesh(const Vec3& c, float s, std::vector<Triangle> tris, const Vec3& bmin, const Vec3& bmax, std::shared_ptr<Material> m)
        : center(c), scale(s), triangles(std::move(tris)), local_bmin(bmin), local_bmax(bmax), mat(std::move(m)) {}

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
        if (scale <= 1e-6f || triangles.empty()) return false;
        float inv_s = 1.0f / scale;
        Ray r_local((r.origin - center) * inv_s, r.direction * inv_s);

        if (!hit_aabb(r_local, t_min, t_max, local_bmin, local_bmax)) {
            return false;
        }

        HitRecord temp_rec;
        bool hit_any = false;
        float closest = t_max;

        for (const auto& tri : triangles) {
            if (hit_triangle(r_local, t_min, closest, tri.v0, tri.v1, tri.v2, tri.normal, temp_rec)) {
                hit_any = true;
                closest = temp_rec.t;
                rec = temp_rec;
            }
        }

        if (hit_any) {
            rec.p = r.at(rec.t);
            rec.mat = mat;
        }
        return hit_any;
    }
};

inline bool LoadOBJGeometry(const std::wstring& filename, std::vector<Triangle>& out_triangles, Vec3& out_bmin, Vec3& out_bmax) {
    FILE* fp = _wfopen(filename.c_str(), L"r");
    if (!fp) return false;

    std::vector<Vec3> temp_vertices;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3) {
                temp_vertices.push_back(Vec3(x, y, z));
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::vector<int> face_indices;
            char* ptr = line + 2;
            while (*ptr) {
                while (*ptr && isspace((unsigned char)*ptr)) ptr++;
                if (!*ptr) break;
                int vi = 0;
                if (sscanf(ptr, "%d", &vi) == 1) {
                    if (vi > 0) vi -= 1;
                    else if (vi < 0) vi = static_cast<int>(temp_vertices.size()) + vi;
                    face_indices.push_back(vi);
                }
                while (*ptr && !isspace((unsigned char)*ptr)) ptr++;
            }
            for (size_t i = 1; i + 1 < face_indices.size(); ++i) {
                int i0 = face_indices[0];
                int i1 = face_indices[i];
                int i2 = face_indices[i + 1];
                if (i0 >= 0 && i0 < static_cast<int>(temp_vertices.size()) &&
                    i1 >= 0 && i1 < static_cast<int>(temp_vertices.size()) &&
                    i2 >= 0 && i2 < static_cast<int>(temp_vertices.size())) {
                    Vec3 v0 = temp_vertices[i0];
                    Vec3 v1 = temp_vertices[i1];
                    Vec3 v2 = temp_vertices[i2];
                    Vec3 norm = unit_vector(cross(v1 - v0, v2 - v0));
                    if (norm.length_squared() < 1e-6f) norm = Vec3(0, 1, 0);
                    out_triangles.push_back({v0, v1, v2, norm});
                }
            }
        }
    }
    fclose(fp);

    if (out_triangles.empty()) return false;

    Vec3 bmin(1e9f, 1e9f, 1e9f), bmax(-1e9f, -1e9f, -1e9f);
    for (const auto& tri : out_triangles) {
        bmin.x = std::min({bmin.x, tri.v0.x, tri.v1.x, tri.v2.x});
        bmin.y = std::min({bmin.y, tri.v0.y, tri.v1.y, tri.v2.y});
        bmin.z = std::min({bmin.z, tri.v0.z, tri.v1.z, tri.v2.z});
        bmax.x = std::max({bmax.x, tri.v0.x, tri.v1.x, tri.v2.x});
        bmax.y = std::max({bmax.y, tri.v0.y, tri.v1.y, tri.v2.y});
        bmax.z = std::max({bmax.z, tri.v0.z, tri.v1.z, tri.v2.z});
    }

    Vec3 center = 0.5f * (bmin + bmax);
    Vec3 extents = bmax - bmin;
    float max_dim = std::max({extents.x, extents.y, extents.z});
    if (max_dim < 1e-6f) max_dim = 1.0f;
    float norm_scale = 1.0f / max_dim;

    out_bmin = Vec3(1e9f, 1e9f, 1e9f);
    out_bmax = Vec3(-1e9f, -1e9f, -1e9f);

    for (auto& tri : out_triangles) {
        tri.v0 = (tri.v0 - center) * norm_scale;
        tri.v1 = (tri.v1 - center) * norm_scale;
        tri.v2 = (tri.v2 - center) * norm_scale;

        out_bmin.x = std::min({out_bmin.x, tri.v0.x, tri.v1.x, tri.v2.x});
        out_bmin.y = std::min({out_bmin.y, tri.v0.y, tri.v1.y, tri.v2.y});
        out_bmin.z = std::min({out_bmin.z, tri.v0.z, tri.v1.z, tri.v2.z});
        out_bmax.x = std::max({out_bmax.x, tri.v0.x, tri.v1.x, tri.v2.x});
        out_bmax.y = std::max({out_bmax.y, tri.v0.y, tri.v1.y, tri.v2.y});
        out_bmax.z = std::max({out_bmax.z, tri.v0.z, tri.v1.z, tri.v2.z});
    }
    out_bmin = out_bmin - Vec3(0.01f, 0.01f, 0.01f);
    out_bmax = out_bmax + Vec3(0.01f, 0.01f, 0.01f);

    return true;
}

class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;

    void add(std::shared_ptr<Hittable> object) {
        objects.push_back(std::move(object));
    }

    void clear() { objects.clear(); }

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
        HitRecord temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (const auto& object : objects) {
            if (object->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
};

class Camera {
public:
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 lookat;
    Vec3 vup;

    Camera() = default;

    Camera(Vec3 lookfrom, Vec3 lookat_in, Vec3 vup_in, float vfov_deg, float aspect_ratio) {
        lookat = lookat_in;
        vup = vup_in;
        float theta = vfov_deg * 3.14159265358979323846f / 180.0f;
        float h = std::tan(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        Vec3 w = unit_vector(lookfrom - lookat);
        Vec3 u = unit_vector(cross(vup, w));
        Vec3 v = cross(w, u);

        origin = lookfrom;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - w;
    }

    Ray get_ray(float s, float t) const {
        return Ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
    }

    bool project_direction(const Vec3& dir, float& out_s, float& out_t) const {
        Vec3 w = unit_vector(origin - lookat);
        float alpha = dot(dir, -w);
        if (alpha <= 1e-5f) return false;

        Vec3 p_plane = (dir / alpha) - (-w);
        float u_len = horizontal.length();
        float v_len = vertical.length();
        if (u_len <= 1e-6f || v_len <= 1e-6f) return false;

        Vec3 u_dir = horizontal / u_len;
        Vec3 v_dir = vertical / v_len;

        out_s = 0.5f + dot(p_plane, u_dir) / u_len;
        out_t = 0.5f + dot(p_plane, v_dir) / v_len;
        return (out_s >= 0.0f && out_s <= 1.0f && out_t >= 0.0f && out_t <= 1.0f);
    }
};

// Parametry ostrości kaustyki (regulowane z GUI)
std::atomic<float> g_caustic_blur_cm{0.25f};     // GPU: sigma filtru Gaussa siatki kaustyki [cm]; 0 = brak rozmycia
std::atomic<float> g_caustic_light_size{0.12f};  // GPU: ułamek promienia źródła emitujący fotony (półcień kaustyki)
std::atomic<float> g_caustic_gain{1.0f};         // GPU: mnożnik jasności kaustyki
std::atomic<int>   g_caustic_grid_res{2048};     // GPU: rozdzielczość siatki fotonów (1024 / 2048 / 4096)
std::atomic<int>   g_manifold_sequences{0};     // CPU (Manifold NEE): liczba sekwencji faset w tabeli bieżącego renderu
std::atomic<float> g_caustic_cpu_radius{0.028f}; // CPU (BDPT / MLT / MEMLT): promień filtru stożkowego Jensena [m]

// ============================================================================
// 3. GLOBALNA MAPA FOTONÓW KAUSTYKI (Eliminacja szwów kafelkowania)
// ============================================================================

struct Photon {
    Vec3 pos;
    Vec3 flux;
};

class PhotonMap {
public:
    std::vector<Photon> photons;
    float cell_size = 0.02f;
    float min_x = -6.0f;
    float max_x = 6.0f;
    float min_z = -6.0f;
    float max_z = 6.0f;
    int grid_w = 0;
    int grid_h = 0;
    std::vector<int32_t> grid_head;
    std::vector<int32_t> photon_next;

    void init(float c_size, float x0, float x1, float z0, float z1) {
        cell_size = c_size;
        min_x = x0; max_x = x1;
        min_z = z0; max_z = z1;
        grid_w = static_cast<int>(std::ceil((max_x - min_x) / cell_size));
        grid_h = static_cast<int>(std::ceil((max_z - min_z) / cell_size));
        grid_head.assign(grid_w * grid_h, -1);
        photons.clear();
        photon_next.clear();
    }

    void add_photon(const Vec3& p, const Vec3& flux) {
        int32_t idx = static_cast<int32_t>(photons.size());
        photons.push_back({p, flux});
        photon_next.push_back(-1);

        int gx = static_cast<int>((p.x - min_x) / cell_size);
        int gz = static_cast<int>((p.z - min_z) / cell_size);
        if (gx >= 0 && gx < grid_w && gz >= 0 && gz < grid_h) {
            int cell_idx = gz * grid_w + gx;
            photon_next[idx] = grid_head[cell_idx];
            grid_head[cell_idx] = idx;
        }
    }

    // Dokładne oszacowanie natężenia oświetlenia filtrem stożkowym Jensena (Jensen Cone Filter)
    Vec3 estimate_radiance(const Vec3& p, const Vec3& normal, float search_radius) const {
        if (photons.empty()) return Vec3(0, 0, 0);

        int gx0 = std::max(0, static_cast<int>((p.x - search_radius - min_x) / cell_size));
        int gx1 = std::min(grid_w - 1, static_cast<int>((p.x + search_radius - min_x) / cell_size));
        int gz0 = std::max(0, static_cast<int>((p.z - search_radius - min_z) / cell_size));
        int gz1 = std::min(grid_h - 1, static_cast<int>((p.z + search_radius - min_z) / cell_size));

        float r2 = search_radius * search_radius;
        Vec3 accum(0, 0, 0);
        int hit_count = 0;

        // Normalizacja stożkowa Jensena:
        // Całka po dysku (1 - d/r) wynosi (pi * r^2) / 3.
        // Odbita luminancja: L_r = (albedo/pi) * E.
        // Współczynnik dla irradiance E / pi wynosi: 3.0f / (pi^2 * r^2).
        constexpr float PI = 3.14159265f;
        float norm_factor = 3.0f / (PI * PI * r2);

        for (int gz = gz0; gz <= gz1; ++gz) {
            int row = gz * grid_w;
            for (int gx = gx0; gx <= gx1; ++gx) {
                int cell_idx = row + gx;
                for (int32_t p_idx = grid_head[cell_idx]; p_idx >= 0; p_idx = photon_next[p_idx]) {
                    const auto& ph = photons[p_idx];
                    float dx = p.x - ph.pos.x;
                    float dy = p.y - ph.pos.y;
                    float dz = p.z - ph.pos.z;
                    float dist_sq = dx * dx + dy * dy + dz * dz;
                    if (dist_sq < r2) {
                        float dist = std::sqrt(dist_sq);
                        float w = (1.0f - dist / search_radius);
                        accum = accum + ph.flux * (w * norm_factor);
                        hit_count++;
                    }
                }
            }
        }

        // Regularyzacja chromatyczna (Chroma Regularization):
        // Zapobiega pojawianiu się pojedynczych kolorowych artefaktów/plamek (fireflies)
        // przy niskiej gęstości fotonów, płynnie dopuszczając czyste widmo w gęstych skupieniach kaustyki.
        if (hit_count > 0 && accum.length_squared() > 1e-8f) {
            float lum = (accum.x + accum.y + accum.z) / 3.0f;
            float chroma_trust = std::clamp((static_cast<float>(hit_count) - 1.0f) / 6.0f, 0.0f, 1.0f);
            accum = (1.0f - chroma_trust) * Vec3(lum, lum, lum) + chroma_trust * accum;
        }

        return accum;
    }
};

Vec3 ray_color_standard(const Ray& r, const HittableList& world, int depth, std::mt19937& rng) {
    if (depth <= 0) return Vec3(0, 0, 0);

    HitRecord rec;
    if (!world.hit(r, 0.001f, 1e9f, rec)) {
        Vec3 unit_direction = unit_vector(r.direction);
        float a = 0.5f * (unit_direction.y + 1.0f);
        return (1.0f - a) * Vec3(0.04f, 0.05f, 0.08f) + a * Vec3(0.12f, 0.16f, 0.25f);
    }

    Vec3 emitted = rec.mat->emitted(rec.p);
    Ray scattered;
    Vec3 attenuation;

    if (!rec.mat->scatter(r, rec, attenuation, scattered, rng)) {
        return emitted;
    }

    return emitted + attenuation * ray_color_standard(scattered, world, depth - 1, rng);
}

Vec3 ray_color_bdpt(const Ray& r, const HittableList& world, const std::vector<Sphere>& light_sources,
                    const PhotonMap& photon_map, int depth, std::mt19937& rng) {
    Vec3 accumulated_color(0.0f, 0.0f, 0.0f);
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    Ray current_ray = r;
    bool last_was_specular = true;

    for (int bounce = 0; bounce < depth; ++bounce) {
        HitRecord rec;
        if (!world.hit(current_ray, 0.001f, 1e9f, rec)) {
            Vec3 unit_direction = unit_vector(current_ray.direction);
            float a = 0.5f * (unit_direction.y + 1.0f);
            Vec3 sky = (1.0f - a) * Vec3(0.04f, 0.05f, 0.08f) + a * Vec3(0.12f, 0.16f, 0.25f);
            accumulated_color = accumulated_color + throughput * sky;
            break;
        }

        Vec3 emitted = rec.mat->emitted(rec.p);
        if (!emitted.near_zero()) {
            if (last_was_specular) {
                accumulated_color = accumulated_color + throughput * emitted;
            }
            break;
        }

        Ray scattered;
        Vec3 albedo;
        if (!rec.mat->scatter(current_ray, rec, albedo, scattered, rng)) {
            break;
        }

        bool is_spec = rec.mat->is_specular();

        if (!is_spec && !light_sources.empty()) {
            const auto& light = light_sources[0];
            Vec3 light_pt = light.center + light.radius * random_unit_vector(rng);
            Vec3 to_light = light_pt - rec.p;
            float dist = to_light.length();
            Vec3 dir_light = to_light / dist;

            float cos_theta = dot(rec.normal, dir_light);
            if (cos_theta > 0.0f) {
                HitRecord shadow_rec;
                if (!world.hit(Ray(rec.p + rec.normal * 0.002f, dir_light), 0.001f, dist - 0.005f, shadow_rec)) {
                    Vec3 light_norm = unit_vector(light_pt - light.center);
                    float cos_light = std::fmax(0.0f, dot(light_norm, -dir_light));
                    float area = 4.0f * 3.14159f * light.radius * light.radius;
                    float G = (cos_theta * cos_light) / (dist * dist + 0.05f);

                    Vec3 direct = light.mat->emitted(light_pt) * albedo * (G * area * 0.159f);
                    direct.x = std::fmin(direct.x, 25.0f);
                    direct.y = std::fmin(direct.y, 25.0f);
                    direct.z = std::fmin(direct.z, 25.0f);
                    accumulated_color = accumulated_color + throughput * direct;
                }
            }
        }

        if (!is_spec && !photon_map.photons.empty() && (bounce == 0 || last_was_specular)) {
            Vec3 caustic_flux = photon_map.estimate_radiance(rec.p, rec.normal, g_caustic_cpu_radius.load());
            accumulated_color = accumulated_color + throughput * albedo * caustic_flux;
        }

        throughput = throughput * albedo;
        current_ray = scattered;
        last_was_specular = is_spec;

        if (bounce > 3) {
            float p = std::fmax(throughput.x, std::fmax(throughput.y, throughput.z));
            std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
            if (dist01(rng) > p) break;
            throughput = throughput / p;
        }
    }

    return accumulated_color;
}

// ============================================================================
// 3b. METROPOLIS LIGHT TRANSPORT (MLT) - David Cline & Parris Egbert (2005)
// ============================================================================

constexpr size_t MLT_DIM = 128;

struct PrimarySampler {
    const std::vector<float>& u;
    mutable size_t ptr = 0;
    std::mt19937* fallback_rng = nullptr;

    float get_1d() const {
        if (ptr < u.size()) return u[ptr++];
        if (fallback_rng) {
            std::uniform_real_distribution<float> d(0.0f, 1.0f);
            return d(*fallback_rng);
        }
        return 0.5f;
    }

    Vec3 get_unit_vector() const {
        float z = 1.0f - 2.0f * get_1d();
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        float phi = 2.0f * 3.14159265f * get_1d();
        return Vec3(r * std::cos(phi), r * std::sin(phi), z);
    }
};

inline float color_luminance(const Vec3& c) {
    // Sekcja 2.2 z artykułu Cline & Egbert (2005): Luminance = 0.299 R + 0.587 G + 0.114 B
    return 0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
}

Vec3 evaluate_path_mlt(const Ray& initial_ray, const HittableList& world,
                       const std::vector<Sphere>& light_sources, const PhotonMap& photon_map,
                       int max_depth, const PrimarySampler& sampler) {
    Vec3 accumulated_color(0.0f, 0.0f, 0.0f);
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    Ray current_ray = initial_ray;
    bool last_was_specular = true;
    float path_lambda = -1.0f;

    for (int bounce = 0; bounce < max_depth; ++bounce) {
        HitRecord rec;
        if (!world.hit(current_ray, 0.001f, 1e9f, rec)) {
            Vec3 unit_direction = unit_vector(current_ray.direction);
            float a = 0.5f * (unit_direction.y + 1.0f);
            Vec3 sky = (1.0f - a) * Vec3(0.04f, 0.05f, 0.08f) + a * Vec3(0.12f, 0.16f, 0.25f);
            accumulated_color = accumulated_color + throughput * sky;
            break;
        }

        Vec3 emitted = rec.mat->emitted(rec.p);
        if (!emitted.near_zero()) {
            if (last_was_specular) {
                accumulated_color = accumulated_color + throughput * emitted;
            }
            break;
        }

        if (rec.mat->is_dielectric()) {
            auto diel = static_cast<const Dielectric*>(rec.mat.get());
            float effective_ir = diel->ir;

            if (diel->dispersion > 1e-4f) {
                if (path_lambda < 0.0f) {
                    path_lambda = sampler.get_1d();
                    float lambda = path_lambda;
                    float wr = std::exp(-((lambda - 0.15f) * (lambda - 0.15f)) / 0.14f);
                    float wg = std::exp(-((lambda - 0.50f) * (lambda - 0.50f)) / 0.14f);
                    float wb = std::exp(-((lambda - 0.85f) * (lambda - 0.85f)) / 0.14f);
                    Vec3 spec(wr, wg, wb);
                    float avg = (spec.x + spec.y + spec.z) / 3.0f;
                    if (avg > 1e-4f) spec = spec / avg;
                    float blend = std::clamp(diel->dispersion * 10.0f, 0.0f, 1.0f);
                    Vec3 attenuation = (1.0f - blend) * Vec3(1.0f, 1.0f, 1.0f) + blend * spec;
                    throughput = throughput * attenuation;
                }
                effective_ir = diel->ir + (2.0f * path_lambda - 1.0f) * diel->dispersion;
            }

            float ref_ratio = rec.front_face ? (1.0f / effective_ir) : effective_ir;

            Vec3 unit_dir = unit_vector(current_ray.direction);
            float cos_theta = std::fmin(dot(-unit_dir, rec.normal), 1.0f);
            float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));

            bool cannot_refract = ref_ratio * sin_theta > 1.0f;
            float refl_prob = schlick_reflectance(cos_theta, ref_ratio);

            Vec3 next_dir;
            if (cannot_refract || refl_prob > sampler.get_1d()) {
                next_dir = reflect(unit_dir, rec.normal);
            } else {
                next_dir = refract(unit_dir, rec.normal, ref_ratio);
            }

            current_ray = Ray(rec.p, unit_vector(next_dir));
            last_was_specular = true;
        }
        else if (rec.mat->is_specular()) {
            auto metal = static_cast<const Metal*>(rec.mat.get());
            Vec3 unit_dir = unit_vector(current_ray.direction);
            Vec3 reflected = reflect(unit_dir, rec.normal);
            Vec3 scatter_dir = reflected + metal->roughness * sampler.get_unit_vector();
            if (dot(scatter_dir, rec.normal) <= 0.0f) break;

            current_ray = Ray(rec.p, unit_vector(scatter_dir));
            throughput = throughput * metal->albedo;
            last_was_specular = (metal->roughness < 0.15f);
        }
        else {
            Vec3 albedo = rec.mat->get_albedo();

            // Direct light connection
            if (!light_sources.empty()) {
                const auto& light = light_sources[0];
                Vec3 light_pt = light.center + light.radius * sampler.get_unit_vector();
                Vec3 to_light = light_pt - rec.p;
                float dist = to_light.length();
                Vec3 dir_light = to_light / dist;

                float cos_theta = dot(rec.normal, dir_light);
                if (cos_theta > 0.0f) {
                    HitRecord shadow_rec;
                    if (!world.hit(Ray(rec.p + rec.normal * 0.002f, dir_light), 0.001f, dist - 0.005f, shadow_rec)) {
                        Vec3 light_norm = unit_vector(light_pt - light.center);
                        float cos_light = std::fmax(0.0f, dot(light_norm, -dir_light));
                        float area = 4.0f * 3.14159f * light.radius * light.radius;
                        float G = (cos_theta * cos_light) / (dist * dist + 0.05f);

                        Vec3 direct = light.mat->emitted(light_pt) * albedo * (G * area * 0.159f);
                        direct.x = std::fmin(direct.x, 25.0f);
                        direct.y = std::fmin(direct.y, 25.0f);
                        direct.z = std::fmin(direct.z, 25.0f);
                        accumulated_color = accumulated_color + throughput * direct;
                    }
                }
            }

            // Global Photon Map caustic estimation (Tylko dla bezpośredniego trafienia oka lub odbicia lustrzanego!)
            if (!photon_map.photons.empty() && (bounce == 0 || last_was_specular)) {
                Vec3 caustic_flux = photon_map.estimate_radiance(rec.p, rec.normal, g_caustic_cpu_radius.load());
                accumulated_color = accumulated_color + throughput * albedo * caustic_flux;
            }

            // Diffuse outgoing scatter
            Vec3 rand_v = sampler.get_unit_vector();
            Vec3 scatter_dir = rec.normal + rand_v;
            if (scatter_dir.near_zero()) scatter_dir = rec.normal;

            current_ray = Ray(rec.p, unit_vector(scatter_dir));
            throughput = throughput * albedo;
            last_was_specular = false;
        }

        // Russian roulette dla głębszych odbić
        if (bounce > 3) {
            float p = std::fmax(throughput.x, std::fmax(throughput.y, throughput.z));
            p = std::clamp(p, 0.05f, 0.95f);
            if (sampler.get_1d() > p) break;
            throughput = throughput / p;
        }
    }

    return accumulated_color;
}

// ============================================================================
// 3c. ROZMAITOŚĆ ZWIERCIADLANA: MANIFOLD NEXT EVENT ESTIMATION (MNEE)
// Jakob & Marschner 2012 "Manifold Exploration" (więzy i spacer po rozmaitości),
// Hanika i in. 2015 "Manifold Next Event Estimation", Zeltner i in. 2020 "Specular Manifold Sampling"
// ============================================================================

// Moduł rozmaitości zwierciadlanej (Specular Manifold) dla szlifowanego kryształu.
//
// Zastosowanie obecne  -> Manifold Next Event Estimation (MNEE): w każdym wierzchołku dyfuzyjnym x szukamy
//                         łańcucha zwierciadlanego x -> kryształ -> y (punkt na źródle światła), czyli ścieżek
//                         kaustycznych L S+ D E, których zwykły path tracing praktycznie nie znajduje.
// Rozbudowa do MEMLT   -> te same trzy klocki wystarczają do perturbacji rozmaitości z pracy Jakoba i Marschnera:
//     1. SpecularManifold::Walk(x', y, seed = kierunek bieżącego łańcucha)  - po przesunięciu wierzchołka
//        dyfuzyjnego x -> x' (perturbacja soczewki) ponownie łączy ścieżkę z TYM SAMYM punktem światła y,
//     2. ManifoldSolution::signature                                         - test odwracalności: spacer powrotny
//        x' -> x musi wrócić do łańcucha o tej samej sygnaturze faset,
//     3. ManifoldSolution::GeometryTerm() * chain.transmittance              - uogólniony człon geometryczny;
//        jego stosunek dla ścieżki proponowanej i bieżącej wchodzi do prawdopodobieństwa akceptacji.

constexpr int MANIFOLD_MAX_TIR   = 6;                      // maks. liczba całkowitych odbić wewnętrznych w łańcuchu
constexpr int MANIFOLD_MAX_VERTS = MANIFOLD_MAX_TIR + 2;   // wejście + odbicia wewnętrzne + wyjście

// Łańcuch zwierciadlany: załamanie na wejściu, k odbić TIR, załamanie na wyjściu
struct SpecularChain {
    int      n = 0;                        // liczba wierzchołków zwierciadlanych
    Vec3     p[MANIFOLD_MAX_VERTS];        // pozycje wierzchołków
    Vec3     nrm[MANIFOLD_MAX_VERTS];      // zewnętrzne normalne faset
    uint64_t signature = 0;                // zakodowana sekwencja indeksów faset (tożsamość rozwiązania)
    Vec3     dir_in;                       // kierunek jednostkowy z punktu startowego do p[0]
    Vec3     dir_exit;                     // kierunek jednostkowy promienia opuszczającego kryształ
    float    t_enter = 0.0f;               // odległość od punktu startowego do p[0]
    float    transmittance = 1.0f;         // iloczyn (1 - F) na obu załamaniach (Schlick, jak w path tracerze)
};

// Rozwiązanie równania więzów: łańcuch łączący punkt startowy z punktem docelowym
struct ManifoldSolution {
    SpecularChain chain;
    float t_exit = 0.0f;         // odległość od wierzchołka wyjściowego do punktu docelowego
    float area_per_sr = 0.0f;    // |dP/dω|: pole na płaszczyźnie celu przypadające na jednostkowy kąt bryłowy w punkcie startowym
    float cos_plane = 0.0f;      // |dir_exit · normalna płaszczyzny celu|
    uint64_t signature() const { return chain.signature; }

    // Uogólniony człon geometryczny G(x <-> y) = cosθx * cosθy * dω_x / dA_y.
    // Bez kryształu redukuje się do klasycznego cosθx * cosθy / r².
    float GeometryTerm(float cos_x, float cos_y) const {
        float denom = area_per_sr * cos_plane;
        return (denom > 1e-12f) ? (cos_x * cos_y / denom) : 0.0f;
    }
};

class SpecularManifold {
public:
    const ConvexPolyhedron* crystal = nullptr;
    float ior = 1.5f;

    SpecularManifold(const ConvexPolyhedron* c, float index) : crystal(c), ior(index) {}

    // Deterministyczne śledzenie łańcucha zwierciadlanego (zawsze załamanie, gdy jest możliwe; TIR w przeciwnym razie).
    // Dokładnie ta sama klasa ścieżek jest wyłączana z path tracera w ray_color_manifold (brak podwójnego liczenia).
    bool Trace(const Vec3& origin, const Vec3& dir, SpecularChain& c) const {
        c = SpecularChain();
        Vec3 d = unit_vector(dir);
        Vec3 oc = origin - crystal->center;
        const auto& planes = crystal->planes;
        const int np = static_cast<int>(planes.size());

        // Wejście z zewnątrz (test slabów z indeksem fasety)
        float t_en = -1e30f, t_ex = 1e30f;
        int i_en = -1;
        for (int k = 0; k < np; ++k) {
            float denom = dot(planes[k].n, d);
            float numer = planes[k].d - dot(planes[k].n, oc);
            if (std::fabs(denom) < 1e-9f) {
                if (numer < 0.0f) return false;
                continue;
            }
            float t = numer / denom;
            if (denom < 0.0f) { if (t > t_en) { t_en = t; i_en = k; } }
            else              { if (t < t_ex) t_ex = t; }
        }
        if (i_en < 0 || t_en > t_ex || t_en < 1e-4f) return false;

        Vec3 n_in = planes[i_en].n;
        float cos_i = std::fmin(dot(-d, n_in), 1.0f);
        if (cos_i <= 0.0f) return false;
        c.transmittance *= (1.0f - schlick_reflectance(cos_i, 1.0f / ior));

        Vec3 cur_p = origin + d * t_en;
        Vec3 cur_d = unit_vector(refract(d, n_in, 1.0f / ior));
        int cur_idx = i_en;

        c.dir_in = d;
        c.t_enter = t_en;
        c.p[0] = cur_p;
        c.nrm[0] = n_in;
        c.n = 1;
        c.signature = static_cast<uint64_t>(i_en + 1);

        for (int tir = 0; tir <= MANIFOLD_MAX_TIR; ++tir) {
            // Najbliższa faseta wyjściowa (wielościan wypukły: wystarczy minimum po płaszczyznach skierowanych zgodnie z promieniem)
            Vec3 pc = cur_p - crystal->center;
            float t_best = 1e30f;
            int i_best = -1;
            for (int k = 0; k < np; ++k) {
                if (k == cur_idx) continue;
                float denom = dot(planes[k].n, cur_d);
                if (denom <= 1e-9f) continue;
                float t = (planes[k].d - dot(planes[k].n, pc)) / denom;
                if (t < t_best) { t_best = t; i_best = k; }
            }
            if (i_best < 0 || t_best <= 0.0f) return false;

            Vec3 n_out = planes[i_best].n;
            cur_p = cur_p + cur_d * t_best;
            cur_idx = i_best;

            c.p[c.n] = cur_p;
            c.nrm[c.n] = n_out;
            c.n += 1;
            c.signature = c.signature * 64u + static_cast<uint64_t>(i_best + 1);

            float cos_o = std::fmin(dot(cur_d, n_out), 1.0f);
            float sin_o = std::sqrt(std::max(0.0f, 1.0f - cos_o * cos_o));
            if (ior * sin_o > 1.0f) {
                // Całkowite odbicie wewnętrzne
                if (tir == MANIFOLD_MAX_TIR) return false;
                cur_d = unit_vector(reflect(cur_d, n_out));
                continue;
            }

            c.transmittance *= (1.0f - schlick_reflectance(cos_o, ior));
            c.dir_exit = unit_vector(refract(cur_d, -n_out, ior));
            return true;
        }
        return false;
    }

    // Spacer po rozmaitości (metoda Newtona): szuka kierunku startowego, dla którego promień po przejściu przez
    // kryształ trafia w punkt `target`. `plane_n` to normalna dowolnej stałej płaszczyzny przechodzącej przez cel.
    // `seed_dir` to punkt startowy iteracji: promień wskazany przez model wiązki dla MNEE, kierunek bieżącego
    // łańcucha dla perturbacji MEMLT. Niezerowe `required_signature` ogranicza spacer do jednej sekwencji faset.
    bool Walk(const Vec3& origin, const Vec3& target, const Vec3& plane_n, const Vec3& seed_dir,
              ManifoldSolution& out, uint64_t required_signature = 0,
              float tolerance = 2e-4f, int max_iters = 10) const {
        Vec3 w = unit_vector(seed_dir);
        SpecularChain c;
        Vec3 P;
        float t_exit = 0.0f, cos_pl = 0.0f;
        if (!Shoot(origin, w, target, plane_n, c, P, t_exit, cos_pl)) return false;
        // Spacer ograniczony do jednej składowej rozmaitości (jednej sekwencji faset)
        if (required_signature != 0 && c.signature != required_signature) return false;
        float err = (target - P).length();

        for (int iter = 0; iter < max_iters; ++iter) {
            Vec3 up = (std::fabs(w.y) > 0.9f) ? Vec3(1.0f, 0.0f, 0.0f) : Vec3(0.0f, 1.0f, 0.0f);
            Vec3 U = unit_vector(cross(w, up));
            Vec3 V = cross(w, U);

            // Jacobian dP/d(a,b) w bazie stycznej (U, V), liczony w obrębie tej samej sekwencji faset
            Vec3 Ju, Jv;
            if (!Differentiate(origin, w, U, target, plane_n, c.signature, P, Ju)) return false;
            if (!Differentiate(origin, w, V, target, plane_n, c.signature, P, Jv)) return false;

            if (err < tolerance) {
                out.chain = c;
                out.t_exit = t_exit;
                out.cos_plane = cos_pl;
                out.area_per_sr = cross(Ju, Jv).length();
                return out.area_per_sr > 1e-9f;
            }

            // Krok Gaussa-Newtona: J * [a, b]^T = target - P
            Vec3 diff = target - P;
            float a11 = dot(Ju, Ju), a12 = dot(Ju, Jv), a22 = dot(Jv, Jv);
            float r1 = dot(diff, Ju), r2 = dot(diff, Jv);
            float det = a11 * a22 - a12 * a12;
            if (det < 1e-14f * (a11 * a22 + 1e-20f)) return false;
            float da = (a22 * r1 - a12 * r2) / det;
            float db = (a11 * r2 - a12 * r1) / det;

            float step_len = std::sqrt(da * da + db * db);
            if (step_len > 0.5f) { da *= 0.5f / step_len; db *= 0.5f / step_len; }

            // Przeszukiwanie liniowe: krok jest przyjmowany tylko wtedy, gdy zmniejsza błąd
            bool advanced = false;
            float s = 1.0f;
            for (int ls = 0; ls < 5; ++ls) {
                Vec3 w2 = unit_vector(w + s * (da * U + db * V));
                SpecularChain c2;
                Vec3 P2;
                float t2 = 0.0f, cp2 = 0.0f;
                if (Shoot(origin, w2, target, plane_n, c2, P2, t2, cp2) &&
                    (required_signature == 0 || c2.signature == required_signature)) {
                    float err2 = (target - P2).length();
                    if (err2 < err) {
                        w = w2; c = c2; P = P2; t_exit = t2; cos_pl = cp2; err = err2;
                        advanced = true;
                        break;
                    }
                }
                s *= 0.5f;
            }
            if (!advanced) return false;
        }
        return false;
    }

    // Losowy kierunek startowy: jednorodnie po dysku sfery otaczającej kryształ widzianym z punktu `origin`
    Vec3 SampleSeedDirection(const Vec3& origin, std::mt19937& rng) const {
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
        Vec3 axis = unit_vector(crystal->center - origin);
        Vec3 up = (std::fabs(axis.y) > 0.9f) ? Vec3(1.0f, 0.0f, 0.0f) : Vec3(0.0f, 1.0f, 0.0f);
        Vec3 A = unit_vector(cross(axis, up));
        Vec3 B = cross(axis, A);
        float r = crystal->bounding_radius * std::sqrt(dist01(rng));
        float phi = 6.2831853f * dist01(rng);
        Vec3 pt = crystal->center + A * (r * std::cos(phi)) + B * (r * std::sin(phi));
        return unit_vector(pt - origin);
    }

private:
    bool Shoot(const Vec3& origin, const Vec3& w, const Vec3& target, const Vec3& plane_n,
               SpecularChain& c, Vec3& P, float& t_exit, float& cos_plane) const {
        if (!Trace(origin, w, c)) return false;
        float denom = dot(c.dir_exit, plane_n);
        if (std::fabs(denom) < 1e-6f) return false;
        float t = dot(target - c.p[c.n - 1], plane_n) / denom;
        if (t <= 1e-4f) return false;
        P = c.p[c.n - 1] + c.dir_exit * t;
        t_exit = t;
        cos_plane = std::fabs(denom);
        return true;
    }

    // Pochodna kierunkowa punktu trafienia; różnica skończona brana po tej stronie, po której sekwencja faset się nie zmienia
    bool Differentiate(const Vec3& origin, const Vec3& w, const Vec3& axis, const Vec3& target, const Vec3& plane_n,
                       uint64_t signature, const Vec3& P, Vec3& J) const {
        for (float h : {1e-3f, 2e-4f, 5e-5f}) {
            for (float sgn : {1.0f, -1.0f}) {
                SpecularChain c2;
                Vec3 P2;
                float t2 = 0.0f, cp2 = 0.0f;
                if (Shoot(origin, unit_vector(w + (sgn * h) * axis), target, plane_n, c2, P2, t2, cp2) && c2.signature == signature) {
                    J = (P2 - P) / (sgn * h);
                    return true;
                }
            }
        }
        return false;
    }
};

// Liniowy model fragmentu wiązki opuszczającej kryształ: w płaszczyźnie odniesienia sekwencji promień
// przechodzący przez punkt q = (qu, qv) ma nachylenie względem osi wiązki
//     s = A*q + b + C*(y - środek źródła) + d*(ior - ior bazowy),
// czyli model uwzględnia konkretny punkt na źródle światła i długość fali danej próbki.
struct ManifoldBeam {
    float a11 = 0.0f, a12 = 0.0f, a21 = 0.0f, a22 = 0.0f;
    float b1 = 0.0f, b2 = 0.0f;
    Vec3  c1, c2;                                                 // zależność nachylenia od położenia punktu na źródle
    float d1 = 0.0f, d2 = 0.0f;                                   // zależność nachylenia od współczynnika załamania
    float residual = 0.0f;                                        // typowa odchyłka nachylenia od modelu (3 x RMS)
    float u_min = 0.0f, u_max = 0.0f, v_min = 0.0f, v_max = 0.0f; // zakres q objęty tym fragmentem

    // Punkt q płaszczyzny odniesienia, przez który według modelu przechodzi promień do punktu (vu, vv, z)
    // podanego w układzie sekwencji. Zwraca też normę odwrotności odwzorowania (do tolerancji).
    bool Locate(float vu, float vv, float z, const Vec3& y_off, float d_ior, float& qu, float& qv, float& inv_norm) const {
        float lu = vu - z * (b1 + dot(c1, y_off) + d1 * d_ior);
        float lv = vv - z * (b2 + dot(c2, y_off) + d2 * d_ior);
        float m11 = 1.0f + z * a11, m12 = z * a12, m21 = z * a21, m22 = 1.0f + z * a22;
        float det = m11 * m22 - m12 * m21;
        if (std::fabs(det) <= 1e-4f) return false;
        qu = (m22 * lu - m12 * lv) / det;
        qv = (m11 * lv - m21 * lu) / det;
        inv_norm = (std::fabs(m11) + std::fabs(m12) + std::fabs(m21) + std::fabs(m22)) / std::fabs(det);
        return true;
    }

    // Czy fragment wiązki może obejmować ten punkt? Jeśli tak, (qu, qv) jest punktem startowym dla spaceru.
    bool Covers(float vu, float vv, float z, const Vec3& y_off, float d_ior, float& qu, float& qv) const {
        float inv_norm = 0.0f;
        if (!Locate(vu, vv, z, y_off, d_ior, qu, qv, inv_norm)) {
            // Ognisko wiązki: model nie rozstrzyga, przyjmujemy środek fragmentu
            qu = 0.5f * (u_min + u_max);
            qv = 0.5f * (v_min + v_max);
            return true;
        }
        float tol = inv_norm * (1.5f * residual * z + 0.002f);
        float pad_u = 0.05f * (u_max - u_min) + 0.003f + tol;
        float pad_v = 0.05f * (v_max - v_min) + 0.003f + tol;
        if (qu < u_min - pad_u || qu > u_max + pad_u || qv < v_min - pad_v || qv > v_max + pad_v) return false;
        qu = std::clamp(qu, u_min, u_max);
        qv = std::clamp(qv, v_min, v_max);
        return true;
    }
};

struct ManifoldBeamRay {
    float qu, qv, su, sv;   // położenie w płaszczyźnie odniesienia i nachylenie promienia
    Vec3  y_off;            // położenie punktu na źródle względem jego środka
    float d_ior;            // odchyłka współczynnika załamania (dyspersja)
};

// Dopasowanie modelu liniowego metodą najmniejszych kwadratów do zbioru promieni wiązki
inline ManifoldBeam FitManifoldBeam(const std::vector<ManifoldBeamRay>& rays) {
    ManifoldBeam m;
    constexpr int NF = 7;   // cechy: qu, qv, 1, y_off.x, y_off.y, y_off.z, d_ior
    const double N = static_cast<double>(rays.size());

    bool fitted = false;
    if (rays.size() >= 24) {
        double M[NF][NF] = {}, Ru[NF] = {}, Rv[NF] = {};
        for (const auto& r : rays) {
            double f[NF] = {r.qu, r.qv, 1.0, r.y_off.x, r.y_off.y, r.y_off.z, r.d_ior};
            for (int a = 0; a < NF; ++a) {
                Ru[a] += f[a] * r.su;
                Rv[a] += f[a] * r.sv;
                for (int b = 0; b < NF; ++b) M[a][b] += f[a] * f[b];
            }
        }
        // Regularyzacja grzbietowa: cechy o zerowej wariancji (źródło punktowe, brak dyspersji) dostają współczynnik 0
        double trace = 0.0;
        for (int a = 0; a < NF; ++a) trace += M[a][a];
        for (int a = 0; a < NF; ++a) M[a][a] += 1e-9 * trace + 1e-14;

        // Eliminacja Gaussa z wyborem elementu głównego (dwie prawe strony jednocześnie)
        bool ok = true;
        for (int col = 0; col < NF && ok; ++col) {
            int piv = col;
            for (int r = col + 1; r < NF; ++r) if (std::fabs(M[r][col]) > std::fabs(M[piv][col])) piv = r;
            if (std::fabs(M[piv][col]) < 1e-18) { ok = false; break; }
            if (piv != col) {
                for (int k = 0; k < NF; ++k) std::swap(M[piv][k], M[col][k]);
                std::swap(Ru[piv], Ru[col]);
                std::swap(Rv[piv], Rv[col]);
            }
            for (int r = col + 1; r < NF; ++r) {
                double fct = M[r][col] / M[col][col];
                for (int k = col; k < NF; ++k) M[r][k] -= fct * M[col][k];
                Ru[r] -= fct * Ru[col];
                Rv[r] -= fct * Rv[col];
            }
        }
        if (ok) {
            double wu[NF], wv[NF];
            for (int r = NF - 1; r >= 0; --r) {
                double su = Ru[r], sv = Rv[r];
                for (int k = r + 1; k < NF; ++k) { su -= M[r][k] * wu[k]; sv -= M[r][k] * wv[k]; }
                wu[r] = su / M[r][r];
                wv[r] = sv / M[r][r];
            }
            m.a11 = static_cast<float>(wu[0]); m.a12 = static_cast<float>(wu[1]); m.b1 = static_cast<float>(wu[2]);
            m.a21 = static_cast<float>(wv[0]); m.a22 = static_cast<float>(wv[1]); m.b2 = static_cast<float>(wv[2]);
            m.c1 = Vec3(static_cast<float>(wu[3]), static_cast<float>(wu[4]), static_cast<float>(wu[5]));
            m.c2 = Vec3(static_cast<float>(wv[3]), static_cast<float>(wv[4]), static_cast<float>(wv[5]));
            m.d1 = static_cast<float>(wu[6]);
            m.d2 = static_cast<float>(wv[6]);
            fitted = true;
        }
    }
    if (!fitted) {
        // Za mało promieni na pełne dopasowanie: stałe nachylenie równe średniej
        double su = 0.0, sv = 0.0;
        for (const auto& r : rays) { su += r.su; sv += r.sv; }
        m.b1 = static_cast<float>(su / N);
        m.b2 = static_cast<float>(sv / N);
    }

    m.u_min = m.v_min = 1e30f;
    m.u_max = m.v_max = -1e30f;
    double sq_sum = 0.0;
    for (const auto& r : rays) {
        float eu = m.a11 * r.qu + m.a12 * r.qv + m.b1 + dot(m.c1, r.y_off) + m.d1 * r.d_ior - r.su;
        float ev = m.a21 * r.qu + m.a22 * r.qv + m.b2 + dot(m.c2, r.y_off) + m.d2 * r.d_ior - r.sv;
        sq_sum += static_cast<double>(eu) * eu + static_cast<double>(ev) * ev;
        m.u_min = std::fmin(m.u_min, r.qu); m.u_max = std::fmax(m.u_max, r.qu);
        m.v_min = std::fmin(m.v_min, r.qv); m.v_max = std::fmax(m.v_max, r.qv);
    }
    // Odporna miara błędu modelu: 3 x RMS. Promienie tuż przy kącie granicznym mają nieograniczony błąd nachylenia,
    // ale ich przepuszczalność Fresnela dąży do zera, więc nie mogą dyktować tolerancji.
    m.residual = 3.0f * static_cast<float>(std::sqrt(sq_sum / N));
    return m;
}

// Sekwencja faset, przez którą światło danego źródła faktycznie przechodzi przez kryształ, wraz z modelem
// wiązki opuszczającej kryształ w stronę odbiornika. Przy płaskich fasetach każdej sekwencji odpowiada co najwyżej
// jedno rozwiązanie równania więzów, więc zamiast losować punkty startowe wyliczamy wszystkie sekwencje.
struct ManifoldSequence {
    uint64_t signature = 0;      // sekwencja faset w kolejności od odbiornika (zgodna z SpecularChain::signature)
    int      count = 0;          // liczba próbek (miara energii niesionej przez sekwencję)
    Vec3     origin, axis, U, V; // układ odniesienia wiązki: punkt, oś i baza płaszczyzny odniesienia
    float    reach_radius = 0.0f; // stożek zasięgu wiązki: promień przekroju przy krysztale...
    float    reach_slope = 0.0f;  // ...i maksymalne nachylenie promieni względem osi
    ManifoldBeam whole;           // zgrubny model całej wiązki (pierwsze przybliżenie położenia)
    std::vector<ManifoldBeam> cells; // dokładne modele fragmentów wiązki
    int      grid = 1;            // siatka grid x grid komórek w płaszczyźnie odniesienia
    std::vector<int> cell_index;  // komórka siatki -> indeks w `cells` (pusta komórka wskazuje najbliższą niepustą)

    int CellAt(float qu, float qv) const {
        float du = std::max(1e-6f, whole.u_max - whole.u_min);
        float dv = std::max(1e-6f, whole.v_max - whole.v_min);
        int cu = std::clamp(static_cast<int>((qu - whole.u_min) / du * grid), 0, grid - 1);
        int cv = std::clamp(static_cast<int>((qv - whole.v_min) / dv * grid), 0, grid - 1);
        return cell_index[static_cast<size_t>(cv) * grid + cu];
    }

    // Wyznacza punkty startowe spaceru dla punktu odbiornika x: najpierw z komórki wskazanej przez model,
    // potem z trzech najbliższych komórek sąsiednich (ich modele dają nieco inne promienie, co ratuje rozwiązania leżące tuż przy
    // krawędziach faset). Zwraca liczbę punktów; 0 oznacza, że wiązka nie obejmuje punktu x.
    static constexpr int MAX_SEEDS = 4;
    int FindSeeds(const Vec3& x, const Vec3& y_off, float d_ior, Vec3* seeds) const {
        Vec3 v = x - origin;
        float z = dot(v, axis);
        if (z <= 0.0f) return 0;
        // 1. Stożek zasięgu (bardzo tani test)
        float rho2 = v.length_squared() - z * z;
        float lim = reach_radius + z * reach_slope;
        if (rho2 > lim * lim) return 0;

        // 2. Zgrubne położenie z modelu całej wiązki, potem doprecyzowanie modelami komórek
        float vu = dot(v, U), vv = dot(v, V);
        float qu = 0.5f * (whole.u_min + whole.u_max), qv = 0.5f * (whole.v_min + whole.v_max), inv_norm = 0.0f;
        whole.Locate(vu, vv, z, y_off, d_ior, qu, qv, inv_norm);
        int ci = CellAt(qu, qv);
        int prev = ci;
        for (int hop = 0; hop < 6; ++hop) {
            float qu2 = qu, qv2 = qv;
            if (!cells[ci].Locate(vu, vv, z, y_off, d_ior, qu2, qv2, inv_norm)) break;
            int cj = CellAt(qu2, qv2);
            if (cj == ci) break;
            prev = ci;
            ci = cj;
        }

        // 3. Test dokładny w znalezionej komórce (oraz w poprzedniej, gdy iteracja oscyluje na granicy komórek)
        float q0u = qu, q0v = qv;
        if (!cells[ci].Covers(vu, vv, z, y_off, d_ior, qu, qv)) {
            if (prev == ci || !cells[prev].Covers(vu, vv, z, y_off, d_ior, qu, qv)) return 0;
            ci = prev;
        }
        int n = 0;
        seeds[n++] = origin + U * qu + V * qv;

        // 4. Punkty zapasowe z sąsiednich komórek siatki
        if (grid > 1) {
            float du = std::max(1e-6f, whole.u_max - whole.u_min);
            float dv = std::max(1e-6f, whole.v_max - whole.v_min);
            int cu = std::clamp(static_cast<int>((q0u - whole.u_min) / du * grid), 0, grid - 1);
            int cv = std::clamp(static_cast<int>((q0v - whole.v_min) / dv * grid), 0, grid - 1);
            // Trzej sąsiedzi po tej stronie komórki, po której leży punkt q
            float fu = (q0u - whole.u_min) / du * grid - static_cast<float>(cu);
            float fv = (q0v - whole.v_min) / dv * grid - static_cast<float>(cv);
            int ou = (fu >= 0.5f) ? 1 : -1;
            int ov = (fv >= 0.5f) ? 1 : -1;
            const int offs[3][2] = {{ou, 0}, {0, ov}, {ou, ov}};
            int used[MAX_SEEDS];
            int n_used = 0;
            used[n_used++] = ci;
            for (const auto& off : offs) {
                int nu = cu + off[0], nv = cv + off[1];
                if (nu < 0 || nv < 0 || nu >= grid || nv >= grid) continue;
                int cj = cell_index[static_cast<size_t>(nv) * grid + nu];
                bool seen = false;
                for (int k = 0; k < n_used; ++k) seen = seen || (used[k] == cj);
                if (seen) continue;
                used[n_used++] = cj;
                float nqu = 0.0f, nqv = 0.0f;
                if (cells[cj].Covers(vu, vv, z, y_off, d_ior, nqu, nqv)) seeds[n++] = origin + U * nqu + V * nqv;
            }
        }
        return n;
    }
};

// Kontekst kaustyki manifoldowej przekazywany do ewaluatora ścieżek
struct ManifoldCausticContext {
    const ConvexPolyhedron* crystal = nullptr;
    const Material* crystal_material = nullptr;   // do rozpoznawania trafień w kryształ w path tracerze
    float base_ior = 1.5f;
    float dispersion = 0.0f;
    float light_size = 1.0f;     // ułamek promienia źródła emitujący światło kaustyki (0 = punktowe, 1 = fizyczny emiter)
    std::vector<std::vector<ManifoldSequence>> light_sequences;   // tabela sekwencji dla każdego źródła światła
};


inline uint64_t ReverseChainSignature(uint64_t s) {
    uint64_t r = 0;
    while (s) { r = r * 64u + (s % 64u); s /= 64u; }
    return r;
}

// Wyznacza tabelę sekwencji faset dla źródła światła, śledząc promienie od światła przez kryształ.
// Wykonywane raz przed renderem (ułamek sekundy).
inline std::vector<ManifoldSequence> BuildManifoldSequences(const ManifoldCausticContext& ctx, const Sphere& light,
                                                            int samples = 1000000, size_t max_sequences = 1024) {
    struct Sample { uint64_t sig; Vec3 p; Vec3 d; Vec3 y_off; float d_ior; };
    std::vector<Sample> smp;
    smp.reserve(samples / 2);
    std::mt19937 rng(20240611u);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    bool point_light = (ctx.light_size < 1e-3f);

    for (int i = 0; i < samples; ++i) {
        float ior = ctx.base_ior;
        if (ctx.dispersion > 1e-4f) ior += (2.0f * dist01(rng) - 1.0f) * ctx.dispersion;
        SpecularManifold manifold(ctx.crystal, ior);

        Vec3 y = light.center;
        if (!point_light) y = y + (light.radius * ctx.light_size) * random_unit_vector(rng);

        SpecularChain c;
        if (!manifold.Trace(y, manifold.SampleSeedDirection(y, rng), c)) continue;
        smp.push_back({ReverseChainSignature(c.signature), c.p[c.n - 1], c.dir_exit, y - light.center, ior - ctx.base_ior});
    }

    std::sort(smp.begin(), smp.end(), [](const Sample& a, const Sample& b) { return a.sig < b.sig; });

    std::vector<ManifoldSequence> out;
    for (size_t i = 0; i < smp.size();) {
        size_t j = i;
        Vec3 sum_p(0.0f, 0.0f, 0.0f), sum_d(0.0f, 0.0f, 0.0f);
        while (j < smp.size() && smp[j].sig == smp[i].sig) { sum_p = sum_p + smp[j].p; sum_d = sum_d + smp[j].d; ++j; }
        int count = static_cast<int>(j - i);
        if (count >= 2 && sum_d.length_squared() > 1e-12f) {
            ManifoldSequence s;
            s.signature = smp[i].sig;
            s.count = count;
            s.origin = sum_p / static_cast<float>(count);
            s.axis = unit_vector(sum_d);
            Vec3 up = (std::fabs(s.axis.y) > 0.9f) ? Vec3(1.0f, 0.0f, 0.0f) : Vec3(0.0f, 1.0f, 0.0f);
            s.U = unit_vector(cross(s.axis, up));
            s.V = cross(s.axis, s.U);

            // Próbki sprowadzone do płaszczyzny odniesienia: położenie q = (qu, qv) i nachylenie (su, sv)
            std::vector<ManifoldBeamRay> rays;
            rays.reserve(count);
            for (size_t k = i; k < j; ++k) {
                float dz = dot(smp[k].d, s.axis);
                if (dz < 0.05f) continue;
                Vec3 rel = smp[k].p - s.origin;
                float su = dot(smp[k].d, s.U) / dz, sv = dot(smp[k].d, s.V) / dz;
                float pz = dot(rel, s.axis);
                rays.push_back({dot(rel, s.U) - pz * su, dot(rel, s.V) - pz * sv, su, sv, smp[k].y_off, smp[k].d_ior});
            }
            if (rays.size() >= 2) {
                s.whole = FitManifoldBeam(rays);

                // Stożek zasięgu: promień przekroju i 99,5 percentyl nachylenia (promienie skrajne, tuż przy kącie
                // granicznym, mają dowolnie duże nachylenie i znikomą energię)
                std::vector<float> slopes;
                slopes.reserve(rays.size());
                for (const auto& r : rays) {
                    s.reach_radius = std::fmax(s.reach_radius, std::sqrt(r.qu * r.qu + r.qv * r.qv));
                    slopes.push_back(std::sqrt(r.su * r.su + r.sv * r.sv));
                }
                size_t nth = static_cast<size_t>(0.995 * static_cast<double>(slopes.size() - 1));
                std::nth_element(slopes.begin(), slopes.begin() + nth, slopes.end());
                s.reach_slope = slopes[nth] * 1.10f + 0.02f;
                s.reach_radius = s.reach_radius * 1.05f + 0.01f;

                // Siatka komórek: załamanie jest silnie nieliniowe na dużych fasetach, więc model liniowy
                // dopasowujemy lokalnie. Im więcej próbek, tym gęstsza siatka.
                s.grid = std::clamp(static_cast<int>(std::sqrt(static_cast<float>(rays.size()) / 64.0f)), 1, 8);
                std::vector<std::vector<ManifoldBeamRay>> buckets(static_cast<size_t>(s.grid) * s.grid);
                float du = std::max(1e-6f, s.whole.u_max - s.whole.u_min);
                float dv = std::max(1e-6f, s.whole.v_max - s.whole.v_min);
                for (const auto& r : rays) {
                    int cu = std::clamp(static_cast<int>((r.qu - s.whole.u_min) / du * s.grid), 0, s.grid - 1);
                    int cv = std::clamp(static_cast<int>((r.qv - s.whole.v_min) / dv * s.grid), 0, s.grid - 1);
                    buckets[static_cast<size_t>(cv) * s.grid + cu].push_back(r);
                }
                s.cell_index.assign(buckets.size(), -1);
                for (size_t bi = 0; bi < buckets.size(); ++bi) {
                    if (buckets[bi].empty()) continue;
                    s.cell_index[bi] = static_cast<int>(s.cells.size());
                    s.cells.push_back(FitManifoldBeam(buckets[bi]));
                }
                // Puste komórki siatki wskazują model najbliższej niepustej, żeby wyszukiwanie mogło przez nie przejść
                std::vector<int> filled = s.cell_index;
                for (int gv = 0; gv < s.grid; ++gv) {
                    for (int gu = 0; gu < s.grid; ++gu) {
                        if (s.cell_index[static_cast<size_t>(gv) * s.grid + gu] >= 0) continue;
                        int best = -1, best_d = 1 << 30;
                        for (int hv = 0; hv < s.grid; ++hv) {
                            for (int hu = 0; hu < s.grid; ++hu) {
                                int idx = s.cell_index[static_cast<size_t>(hv) * s.grid + hu];
                                int d2 = (hu - gu) * (hu - gu) + (hv - gv) * (hv - gv);
                                if (idx >= 0 && d2 < best_d) { best_d = d2; best = idx; }
                            }
                        }
                        filled[static_cast<size_t>(gv) * s.grid + gu] = best;
                    }
                }
                s.cell_index = filled;
                out.push_back(s);
            }
        }
        i = j;
    }

    std::sort(out.begin(), out.end(), [](const ManifoldSequence& a, const ManifoldSequence& b) { return a.count > b.count; });
    if (out.size() > max_sequences) out.resize(max_sequences);
    return out;
}

// Manifold Next Event Estimation: suma po wszystkich sekwencjach faset, których wiązka obejmuje punkt x.
// Zwraca napromienienie kaustyczne podzielone przez π: wynik mnoży się przez throughput * albedo.
inline Vec3 EstimateManifoldCaustic(const ManifoldCausticContext& ctx, const HittableList& world,
                                    const std::vector<Sphere>& light_sources,
                                    const Vec3& x, const Vec3& n_x, std::mt19937& rng) {
    Vec3 result(0.0f, 0.0f, 0.0f);
    if (!ctx.crystal) return result;
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    // Kryształ w całości pod horyzontem powierzchni -> brak kaustyki
    if (dot(ctx.crystal->center - x, n_x) < -ctx.crystal->bounding_radius) return result;

    for (size_t li = 0; li < light_sources.size() && li < ctx.light_sequences.size(); ++li) {
        const auto& light = light_sources[li];
        const auto& sequences = ctx.light_sequences[li];
        if (sequences.empty()) continue;

        // Długość fali ścieżki (dyspersja Cauchy'ego, identycznie jak w materiale Dielectric)
        float ior = ctx.base_ior;
        Vec3 spec_color(1.0f, 1.0f, 1.0f);
        if (ctx.dispersion > 1e-4f) {
            float lambda = dist01(rng);
            ior = ctx.base_ior + (2.0f * lambda - 1.0f) * ctx.dispersion;
            float wr = std::exp(-((lambda - 0.15f) * (lambda - 0.15f)) / 0.14f);
            float wg = std::exp(-((lambda - 0.50f) * (lambda - 0.50f)) / 0.14f);
            float wb = std::exp(-((lambda - 0.85f) * (lambda - 0.85f)) / 0.14f);
            Vec3 spec(wr, wg, wb);
            float avg = (spec.x + spec.y + spec.z) / 3.0f;
            if (avg > 1e-4f) spec = spec / avg;
            float blend = std::clamp(ctx.dispersion * 10.0f, 0.0f, 1.0f);
            spec_color = (1.0f - blend) * Vec3(1.0f, 1.0f, 1.0f) + blend * spec;
        }
        SpecularManifold manifold(ctx.crystal, ior);

        // Punkt na źródle światła. Emiter zmniejszony do ułamka promienia zachowuje pełną moc źródła.
        // Próbkujemy tylko czaszę sfery, z której kryształ może być widoczny (reszta ma zerowy wkład).
        bool point_light = (ctx.light_size < 1e-3f);
        Vec3 n_y(0.0f, 1.0f, 0.0f);
        Vec3 y = light.center;
        float cap_fraction = 1.0f;
        if (!point_light) {
            float r_emit = light.radius * ctx.light_size;
            Vec3 to_crystal = ctx.crystal->center - light.center;
            float d_lc = to_crystal.length();
            if (d_lc < 1e-4f) continue;
            Vec3 w_axis = to_crystal / d_lc;
            float cos_min = std::clamp((r_emit - ctx.crystal->bounding_radius) / d_lc, -1.0f, 1.0f);
            cap_fraction = 0.5f * (1.0f - cos_min);
            float cos_t = 1.0f - dist01(rng) * (1.0f - cos_min);
            float sin_t = std::sqrt(std::max(0.0f, 1.0f - cos_t * cos_t));
            float phi = 6.2831853f * dist01(rng);
            Vec3 up_l = (std::fabs(w_axis.y) > 0.9f) ? Vec3(1.0f, 0.0f, 0.0f) : Vec3(0.0f, 1.0f, 0.0f);
            Vec3 a_l = unit_vector(cross(w_axis, up_l));
            Vec3 b_l = cross(w_axis, a_l);
            n_y = w_axis * cos_t + a_l * (sin_t * std::cos(phi)) + b_l * (sin_t * std::sin(phi));
            y = light.center + r_emit * n_y;
        }
        Vec3 to_light = y - ctx.crystal->center;
        if (to_light.length_squared() < 1e-8f) continue;
        Vec3 plane_n = unit_vector(to_light);

        float r2 = light.radius * light.radius;
        float light_weight = point_light ? (3.14159265f * r2) : (4.0f * 3.14159265f * r2 * cap_fraction);
        Vec3 Le = light.mat->emitted(y);
        Vec3 y_off = y - light.center;
        float d_ior = ior - ctx.base_ior;

        for (const auto& seq : sequences) {
            // 1. Punkt startowy z modelu wiązki; sekwencje, których wiązka nie obejmuje punktu x, odpadają tutaj
            Vec3 seeds[ManifoldSequence::MAX_SEEDS];
            int n_seeds = seq.FindSeeds(x, y_off, d_ior, seeds);
            if (n_seeds == 0) continue;

            // 2. Spacer po rozmaitości ograniczony do tej sekwencji faset (co najwyżej jedno rozwiązanie)
            ManifoldSolution sol;
            bool found = false;
            for (int k = 0; k < n_seeds && !found; ++k) {
                found = manifold.Walk(x, y, plane_n, seeds[k] - x, sol, seq.signature);
            }
            if (!found) continue;

            float cos_x = dot(n_x, sol.chain.dir_in);
            if (cos_x <= 0.0f) continue;
            float cos_y = point_light ? 1.0f : dot(n_y, -sol.chain.dir_exit);
            if (cos_y <= 0.0f) continue;

            // 3. Widoczność obu odcinków poza kryształem
            HitRecord occ;
            if (world.hit(Ray(x + n_x * 0.002f, sol.chain.dir_in), 0.001f, sol.chain.t_enter - 0.004f, occ)) continue;
            const Vec3& p_exit = sol.chain.p[sol.chain.n - 1];
            if (world.hit(Ray(p_exit + sol.chain.dir_exit * 0.002f, sol.chain.dir_exit), 0.001f, sol.t_exit - 0.006f, occ)) {
                if (occ.mat->emitted(occ.p).near_zero()) continue;   // zasłonięte przez coś innego niż samo źródło
            }

            // 4. Wkład: Le * T * G / p_A(y) / π
            float G = sol.GeometryTerm(cos_x, cos_y);
            float scale = sol.chain.transmittance * G * light_weight / 3.14159265f;
            result = result + Le * spec_color * scale;
        }
    }
    return result;
}

// Path tracer z bezpośrednim próbkowaniem światła (NEE) i kaustyką przez rozmaitość zwierciadlaną (MNEE).
// Podział transportu światła bez podwójnego liczenia:
//   - światło bezpośrednie na powierzchniach dyfuzyjnych          -> NEE,
//   - dyfuzja -> [kryształ: załamanie, k x TIR, załamanie] -> źródło -> MNEE (EstimateManifoldCaustic),
//   - pozostałe łańcuchy zwierciadlane (lustra, szklane kule, odbicia Fresnela) -> zwykłe trafienie emitera.
Vec3 ray_color_manifold(const Ray& r, const HittableList& world, const std::vector<Sphere>& light_sources,
                        const ManifoldCausticContext* mctx, int depth, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    Vec3 accumulated_color(0.0f, 0.0f, 0.0f);
    Vec3 throughput(1.0f, 1.0f, 1.0f);
    Ray current_ray = r;
    bool last_was_specular = true;
    float path_lambda = -1.0f;

    // Stan łańcucha zwierciadlanego od ostatniego wierzchołka dyfuzyjnego, w którym wykonano MNEE
    enum { CHAIN_NONE, CHAIN_START, CHAIN_INSIDE, CHAIN_EXITED };
    int chain_state = CHAIN_NONE;
    int chain_tir = 0;

    for (int bounce = 0; bounce < depth; ++bounce) {
        HitRecord rec;
        if (!world.hit(current_ray, 0.001f, 1e9f, rec)) {
            Vec3 unit_direction = unit_vector(current_ray.direction);
            float a = 0.5f * (unit_direction.y + 1.0f);
            Vec3 sky = (1.0f - a) * Vec3(0.04f, 0.05f, 0.08f) + a * Vec3(0.12f, 0.16f, 0.25f);
            accumulated_color = accumulated_color + throughput * sky;
            break;
        }

        Vec3 emitted = rec.mat->emitted(rec.p);
        if (!emitted.near_zero()) {
            // Ścieżka klasy MNEE została już policzona w wierzchołku dyfuzyjnym
            if (last_was_specular && chain_state != CHAIN_EXITED) {
                accumulated_color = accumulated_color + throughput * emitted;
            }
            break;
        }

        if (rec.mat->is_dielectric()) {
            auto diel = static_cast<const Dielectric*>(rec.mat.get());
            float effective_ir = diel->ir;
            if (diel->dispersion > 1e-4f) {
                if (path_lambda < 0.0f) {
                    path_lambda = dist01(rng);
                    float wr = std::exp(-((path_lambda - 0.15f) * (path_lambda - 0.15f)) / 0.14f);
                    float wg = std::exp(-((path_lambda - 0.50f) * (path_lambda - 0.50f)) / 0.14f);
                    float wb = std::exp(-((path_lambda - 0.85f) * (path_lambda - 0.85f)) / 0.14f);
                    Vec3 spec(wr, wg, wb);
                    float avg = (spec.x + spec.y + spec.z) / 3.0f;
                    if (avg > 1e-4f) spec = spec / avg;
                    float blend = std::clamp(diel->dispersion * 10.0f, 0.0f, 1.0f);
                    throughput = throughput * ((1.0f - blend) * Vec3(1.0f, 1.0f, 1.0f) + blend * spec);
                }
                effective_ir = diel->ir + (2.0f * path_lambda - 1.0f) * diel->dispersion;
            }

            float ref_ratio = rec.front_face ? (1.0f / effective_ir) : effective_ir;
            Vec3 unit_dir = unit_vector(current_ray.direction);
            float cos_theta = std::fmin(dot(-unit_dir, rec.normal), 1.0f);
            float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
            bool cannot_refract = ref_ratio * sin_theta > 1.0f;
            bool reflected = cannot_refract || schlick_reflectance(cos_theta, ref_ratio) > dist01(rng);
            Vec3 next_dir = reflected ? reflect(unit_dir, rec.normal) : refract(unit_dir, rec.normal, ref_ratio);

            // Klasyfikacja łańcucha: załamanie na wejściu, odbicia TIR, załamanie na wyjściu
            bool is_crystal = (mctx && rec.mat.get() == mctx->crystal_material);
            if (!is_crystal) {
                chain_state = CHAIN_NONE;
            } else if (chain_state == CHAIN_START && rec.front_face && !reflected) {
                chain_state = CHAIN_INSIDE;
                chain_tir = 0;
            } else if (chain_state == CHAIN_INSIDE && !rec.front_face) {
                if (cannot_refract) {
                    if (++chain_tir > MANIFOLD_MAX_TIR) chain_state = CHAIN_NONE;
                } else {
                    chain_state = reflected ? CHAIN_NONE : CHAIN_EXITED;
                }
            } else {
                chain_state = CHAIN_NONE;
            }

            current_ray = Ray(rec.p, unit_vector(next_dir));
            last_was_specular = true;
        }
        else if (rec.mat->is_specular()) {
            Ray scattered;
            Vec3 attenuation;
            if (!rec.mat->scatter(current_ray, rec, attenuation, scattered, rng)) break;
            throughput = throughput * attenuation;
            current_ray = scattered;
            last_was_specular = true;
            chain_state = CHAIN_NONE;
        }
        else {
            Ray scattered;
            Vec3 albedo;
            if (!rec.mat->scatter(current_ray, rec, albedo, scattered, rng)) break;

            // A. Światło bezpośrednie (NEE) ze wszystkich źródeł
            for (const auto& light : light_sources) {
                Vec3 light_norm = random_unit_vector(rng);
                Vec3 light_pt = light.center + light.radius * light_norm;
                Vec3 to_light = light_pt - rec.p;
                float dist = to_light.length();
                if (dist < 1e-4f) continue;
                Vec3 dir_light = to_light / dist;
                float cos_theta = dot(rec.normal, dir_light);
                float cos_light = dot(light_norm, -dir_light);
                if (cos_theta <= 0.0f || cos_light <= 0.0f) continue;

                HitRecord shadow_rec;
                if (world.hit(Ray(rec.p + rec.normal * 0.002f, dir_light), 0.001f, dist - 0.005f, shadow_rec)) continue;

                // Le * (albedo / π) * G / p_A, gdzie p_A = 1 / (4πR²)
                float G = (cos_theta * cos_light) / (dist * dist);
                Vec3 direct = light.mat->emitted(light_pt) * albedo * (G * 4.0f * light.radius * light.radius);
                accumulated_color = accumulated_color + throughput * direct;
            }

            // B. Kaustyka przez kryształ (MNEE) w wierzchołkach widzianych bezpośrednio lub przez powierzchnie zwierciadlane;
            //    w pozostałych wierzchołkach te same ścieżki liczy zwykłe trafienie emitera (chain_state == CHAIN_NONE)
            if (mctx && (bounce == 0 || last_was_specular) && !light_sources.empty()) {
                Vec3 caustic = EstimateManifoldCaustic(*mctx, world, light_sources, rec.p, rec.normal, rng);
                accumulated_color = accumulated_color + throughput * albedo * caustic;
                chain_state = CHAIN_START;
            } else {
                chain_state = CHAIN_NONE;
            }

            throughput = throughput * albedo;
            current_ray = scattered;
            last_was_specular = false;
        }

        if (bounce > 3) {
            float p = std::fmax(throughput.x, std::fmax(throughput.y, throughput.z));
            p = std::clamp(p, 0.05f, 0.95f);
            if (dist01(rng) > p) break;
            throughput = throughput / p;
        }
    }

    return accumulated_color;
}

// ============================================================================
// 4. STRUKTURY DANYCH I ZMIENNE GLOBALNE
// ============================================================================

struct Tile {
    int x0, y0;
    int x1, y1;
};

struct EditableSceneObject {
    std::wstring name;
    Vec3 center;
    float radius = 1.0f;
    float secondary_param = 0.0f;
    bool is_crystal = false;
    bool is_mesh = false;
    bool is_quad = false;

    // For Quad
    Vec3 quad_u = Vec3(1, 0, 0);
    Vec3 quad_v = Vec3(0, 1, 0);

    // For Mesh
    std::shared_ptr<std::vector<Triangle>> mesh_triangles;
    Vec3 mesh_bmin = Vec3(-1, -1, -1);
    Vec3 mesh_bmax = Vec3(1, 1, 1);

    // Material definition & properties
    std::shared_ptr<Material> material;
    int mat_type = 0; // 0=Lambertian, 1=Metal, 2=Dielectric, 3=DiffuseLight
    Vec3 mat_color = Vec3(0.8f, 0.8f, 0.8f);
    float mat_param = 0.05f; // Roughness or IOR or Light intensity
    float mat_dispersion = 0.0f; // Cauchy Dispersion (0.00 = neutral, 0.01 - 0.08 = prism rainbow)
};

inline void SetObjectMaterial(EditableSceneObject& obj, int type, const Vec3& col, float param, float dispersion = 0.0f) {
    obj.mat_type = type;
    obj.mat_color = col;
    obj.mat_param = param;
    obj.mat_dispersion = dispersion;
    if (type == 0) {
        obj.material = std::make_shared<Lambertian>(col);
    } else if (type == 1) {
        obj.material = std::make_shared<Metal>(col, std::clamp(param, 0.001f, 1.0f));
    } else if (type == 2) {
        float ior = (param > 0.1f) ? param : 1.55f;
        obj.material = std::make_shared<Dielectric>(ior, dispersion);
    } else if (type == 3) {
        float intensity = (param > 0.01f) ? param : 1.0f;
        obj.material = std::make_shared<DiffuseLight>(col * intensity);
    }
}

inline EditableSceneObject make_sphere(const std::wstring& name, const Vec3& center, float r, int mat_type, const Vec3& col, float param = 0.05f) {
    EditableSceneObject obj;
    obj.name = name;
    obj.center = center;
    obj.radius = r;
    SetObjectMaterial(obj, mat_type, col, param, 0.0f);
    return obj;
}

inline EditableSceneObject make_crystal(const std::wstring& name, const Vec3& center, float r, float h, float ior = 1.55f, float dispersion = 0.02f) {
    EditableSceneObject obj;
    obj.name = name;
    obj.center = center;
    obj.radius = r;
    obj.secondary_param = h;
    obj.is_crystal = true;
    obj.mat_dispersion = dispersion;
    SetObjectMaterial(obj, 2, Vec3(1.0f, 1.0f, 1.0f), ior, dispersion);
    return obj;
}

inline EditableSceneObject make_quad(const std::wstring& name, const Vec3& corner, const Vec3& u, const Vec3& v, int mat_type, const Vec3& col, float param = 0.05f) {
    EditableSceneObject obj;
    obj.name = name;
    obj.center = corner;
    obj.radius = 1.0f;
    obj.is_quad = true;
    obj.quad_u = u;
    obj.quad_v = v;
    SetObjectMaterial(obj, mat_type, col, param, 0.0f);
    return obj;
}

std::atomic<float> g_global_light_mult(1.0f);

inline std::shared_ptr<Hittable> CreateHittableFromObject(const EditableSceneObject& obj) {
    std::shared_ptr<Material> mat = obj.material;
    if (obj.mat_type == 3) {
        float mult = g_global_light_mult.load();
        mat = std::make_shared<DiffuseLight>(obj.mat_color * obj.mat_param * mult);
    }

    if (obj.is_mesh && obj.mesh_triangles) {
        return std::make_shared<TriangleMesh>(obj.center, obj.radius, *obj.mesh_triangles, obj.mesh_bmin, obj.mesh_bmax, mat);
    } else if (obj.is_quad) {
        return std::make_shared<Quad>(obj.center, obj.quad_u * obj.radius, obj.quad_v * obj.radius, mat);
    } else if (obj.is_crystal) {
        float h = obj.secondary_param > 0.1f ? obj.secondary_param : 1.40f;
        return CreateFacetedCrystal(obj.center, obj.radius, h, mat);
    } else {
        return std::make_shared<Sphere>(obj.center, obj.radius, mat);
    }
}

int g_canvas_view_width  = 800;
int g_canvas_view_height = 450;
constexpr int TIMER_UPDATE_ID = 101;

constexpr int IDC_BTN_RENDER     = 1001;
constexpr int IDC_BTN_STOP       = 1002;
constexpr int IDC_BTN_SAVE_PNG   = 1003;
constexpr int IDC_COMBO_OBJECTS  = 1004;
constexpr int IDC_EDIT_SPP       = 1005;
constexpr int IDC_EDIT_POS_X     = 1006;
constexpr int IDC_EDIT_POS_Y     = 1007;
constexpr int IDC_EDIT_POS_Z     = 1008;
constexpr int IDC_EDIT_RADIUS    = 1009;
constexpr int IDC_BTN_APPLY_POS  = 1010;
constexpr int IDC_PROGRESS_BAR   = 1011;
constexpr int IDC_STATUS_LABEL   = 1012;
constexpr int IDC_COMBO_RES      = 1013;
constexpr int IDC_EDIT_RES_W     = 1014;
constexpr int IDC_EDIT_RES_H     = 1015;
constexpr int IDC_COMBO_TILE     = 1016;
constexpr int IDC_EDIT_DEPTH     = 1017;
constexpr int IDC_COMBO_ALGO     = 1018;
constexpr int IDC_EDIT_PHOTONS   = 1019;
constexpr int IDC_EDIT_FILENAME  = 1020;
constexpr int IDC_BTN_GRAB       = 1021;
constexpr int IDC_BTN_RESET_CAM  = 1022;
constexpr int IDC_COMBO_SCENE    = 1023;
constexpr int IDC_BTN_IMPORT_OBJ = 1024;
constexpr int IDC_COMBO_MAT_TYPE = 1025;
constexpr int IDC_EDIT_MAT_R     = 1026;
constexpr int IDC_EDIT_MAT_G     = 1027;
constexpr int IDC_EDIT_MAT_B     = 1028;
constexpr int IDC_EDIT_MAT_PARAM = 1029;
constexpr int IDC_BTN_APPLY_MAT  = 1030;
constexpr int IDC_BTN_EXPORT_OBJ = 1031;
constexpr int IDC_EDIT_MAT_DISP  = 1032;
constexpr int IDC_CHK_DENOISE    = 1033;
constexpr int IDC_BTN_DENOISE_NOW= 1034;
constexpr int IDC_EDIT_EXPOSURE  = 1035;
constexpr int IDC_BTN_EV_DOWN    = 1036;
constexpr int IDC_BTN_EV_UP      = 1037;
constexpr int IDC_BTN_EV_RESET   = 1038;
constexpr int IDC_COMBO_TONEMAP  = 1039;
constexpr int IDC_BTN_EV_APPLY   = 1040;
constexpr int IDC_CHK_LIVE_VIEWPORT_PT = 1045;
constexpr int IDC_EDIT_LIGHT_POWER     = 1046;
constexpr int IDC_BTN_LIGHT_DOWN       = 1047;
constexpr int IDC_BTN_LIGHT_UP         = 1048;
constexpr int IDC_BTN_LIGHT_HALF       = 1049;
constexpr int IDC_BTN_LIGHT_DOUBLE     = 1050;
constexpr int IDC_BTN_LIGHT_RESET      = 1051;
constexpr int IDC_BTN_LIGHT_APPLY      = 1052;
constexpr int IDC_CHK_CUDA_CAUSTICS    = 1053;
constexpr int IDC_CHK_CAUSTICS_HQ      = 1054;
constexpr int IDC_EDIT_CAUSTIC_BLUR    = 1060;
constexpr int IDC_EDIT_CAUSTIC_LIGHT   = 1061;
constexpr int IDC_EDIT_CAUSTIC_GAIN    = 1062;
constexpr int IDC_EDIT_CAUSTIC_RADIUS  = 1063;
constexpr int IDC_COMBO_CAUSTIC_GRID   = 1064;
constexpr int IDC_BTN_CAUSTIC_APPLY    = 1065;

std::atomic<bool> g_cuda_caustics{true};
std::atomic<bool> g_cuda_caustics_hq{true};

enum ToneMapType {
    TONEMAP_ACES_FILMIC = 0,
    TONEMAP_REINHARD    = 1,
    TONEMAP_FILMIC_SOFT = 2,
    TONEMAP_LINEAR      = 3
};

// Domyślnie -2.0 EV (zmniejszenie o 2 F-stopy zapobiega przepaleniom bieli)
std::atomic<float> g_exposure_ev(-2.0f);
std::atomic<int>   g_tonemap_mode(TONEMAP_ACES_FILMIC);

// Profesjonalny Tone Mapping i przeliczanie F-stopów (HDR)
inline Vec3 ApplyExposureAndToneMap(Vec3 rad, float exposure_ev, int tonemap_type) {
    float scale = std::pow(2.0f, exposure_ev);
    float x = std::max(0.0f, rad.x * scale);
    float y = std::max(0.0f, rad.y * scale);
    float z = std::max(0.0f, rad.z * scale);

    float r = 0.0f, g = 0.0f, b = 0.0f;

    switch (tonemap_type) {
        case TONEMAP_ACES_FILMIC: {
            // ACES Filmic Tone Mapping (Krzysztof Narkowicz / Stephen Hill fit)
            // Szeroki zakres dynamiki (16+ F-stopów), filmowy roll-off bieli i zachowanie nasycenia
            auto aces = [](float c) {
                float a = 2.51f, bb = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
                return std::clamp((c * (a * c + bb)) / (c * (cc * c + d) + e), 0.0f, 1.0f);
            };
            r = aces(x);
            g = aces(y);
            b = aces(z);
            break;
        }
        case TONEMAP_REINHARD: {
            // Rozszerzony Reinhard z kompresją bieli
            static constexpr float white_sq = 16.0f;
            auto reinhard = [](float c) {
                return std::clamp((c * (1.0f + c / white_sq)) / (1.0f + c), 0.0f, 1.0f);
            };
            r = reinhard(x);
            g = reinhard(y);
            b = reinhard(z);
            break;
        }
        case TONEMAP_FILMIC_SOFT: {
            // Filmic Soft / AgX-style
            auto filmic = [](float c) {
                if (c <= 0.0f) return 0.0f;
                return std::clamp(c / (c + 0.65f), 0.0f, 1.0f);
            };
            r = filmic(x);
            g = filmic(y);
            b = filmic(z);
            break;
        }
        case TONEMAP_LINEAR:
        default: {
            // Standardowy liniowy z twardym obcięciem do 1.0
            r = std::clamp(x, 0.0f, 1.0f);
            g = std::clamp(y, 0.0f, 1.0f);
            b = std::clamp(z, 0.0f, 1.0f);
            break;
        }
    }

    constexpr float inv_gamma = 1.0f / 2.2f;
    return Vec3(
        std::pow(r, inv_gamma),
        std::pow(g, inv_gamma),
        std::pow(b, inv_gamma)
    );
}

int g_current_scene  = 0;
int g_render_width   = 800;
int g_render_height  = 450;
int g_tile_size      = 32;
int g_render_algo    = 1;
int g_photon_count   = 50000;

// Dokładne śledzenie wymiarów bufora w pamięci zapobiegające rozjeżdżaniu skanlinii DIB!
std::atomic<int> g_cur_buf_w(800);
std::atomic<int> g_cur_buf_h(450);

std::vector<uint32_t> g_display_buffer(800 * 450, 0xFF18181B);
std::vector<Vec3>     g_raw_image(800 * 450, Vec3(0, 0, 0));
std::vector<Vec3>     g_unfiltered_raw_image;
bool                  g_is_currently_denoised = false;
std::vector<int>      g_sample_counts(800 * 450, 0);

// Flagi stanu renderowania i odszumiania Intel OIDN:
std::atomic<bool>     g_enable_oidn(true);
std::atomic<bool>     g_has_rendered_image(false);

// Struktury dla Metropolis Light Transport (MLT):
std::atomic<size_t> g_mlt_total_mutations(0);
std::atomic<size_t> g_mlt_mutations_done(0);
std::vector<Vec3>   g_mlt_accum;
float               g_mlt_fave = 1.0f;
std::mutex          g_mlt_mutex;

// Stan aktywnego renderera NVIDIA CUDA:
std::atomic<int>    g_cuda_done_spp(0);
std::atomic<int>    g_cuda_total_spp(0);
std::string         g_cuda_device_name;
bool                g_cuda_available = false;

// Stan podglądu Path Tracingu w viewporcie na żywo i mocy świateł:
std::atomic<bool>   g_live_viewport_pt(true);
int                 g_viewport_pt_spp = 0;

std::atomic<bool>   g_is_rendering(false);
std::atomic<bool>   g_cancel_render(false);
std::atomic<size_t> g_completed_tiles(0);
size_t              g_total_tiles = 0;
std::chrono::high_resolution_clock::time_point g_render_start_time;

// Funkcja odszumiania obrazu z wykorzystaniem Intel Open Image Denoise (OIDN v2.5.1)
inline bool RunIntelOIDN(std::vector<Vec3>& film, int width, int height) {
    if (film.size() < static_cast<size_t>(width * height) || width <= 0 || height <= 0) {
        return false;
    }

    try {
        oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
        device.commit();

        std::vector<float> color(width * height * 3);
        for (int i = 0; i < width * height; ++i) {
            color[i * 3 + 0] = std::max(0.0f, film[i].x);
            color[i * 3 + 1] = std::max(0.0f, film[i].y);
            color[i * 3 + 2] = std::max(0.0f, film[i].z);
        }

        std::vector<float> output(width * height * 3);

        oidn::FilterRef filter = device.newFilter("RT");
        filter.setImage("color", color.data(), oidn::Format::Float3, width, height);
        filter.setImage("output", output.data(), oidn::Format::Float3, width, height);
        filter.set("hdr", true);
        filter.set("cleanAux", true);
        filter.commit();

        filter.execute();

        const char* errorMessage = nullptr;
        if (device.getError(errorMessage) != oidn::Error::None) {
            return false;
        }

        for (int i = 0; i < width * height; ++i) {
            film[i] = Vec3(std::max(0.0f, output[i * 3 + 0]),
                           std::max(0.0f, output[i * 3 + 1]),
                           std::max(0.0f, output[i * 3 + 2]));
        }
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<EditableSceneObject> g_scene_objects = {
    make_crystal(L"1. Krysztal Szklany (Faceted Crystal)", Vec3(-0.5f, 0.70f, 0.1f), 0.65f, 1.40f, 1.55f),
    make_sphere(L"2. Zrodlo Swiatla (Kat boczno-tylny)",   Vec3(3.2f, 2.4f, -1.2f),   0.85f, 3, Vec3(35.0f, 32.0f, 28.0f), 1.0f),
    make_sphere(L"3. Kula Zlota (Prawa)",                 Vec3(1.7f, 0.80f, 0.6f),   0.80f, 1, Vec3(0.95f, 0.78f, 0.35f), 0.02f),
    make_sphere(L"4. Kula Rubinowa (Lewa-Tyl)",           Vec3(-2.0f, 0.65f, -0.6f), 0.65f, 0, Vec3(0.85f, 0.15f, 0.15f), 0.0f),
    make_sphere(L"5. Kula Chromowana (Przod)",            Vec3(0.7f, 0.35f, 1.7f),   0.35f, 1, Vec3(0.92f, 0.95f, 0.98f), 0.05f)
};

// ============================================================================
// 5. INTERAKTYWNA NAWIGACJA 3D (BLENDER ORBIT, PAN, ZOOM & TRANSFORMACJE)
// ============================================================================

enum AxisLock {
    AXIS_NONE = 0,
    AXIS_X,
    AXIS_Y,
    AXIS_Z
};

Vec3  g_cam_target(-0.3f, 0.5f, 0.0f);
float g_cam_distance = 4.6f;
float g_cam_yaw      = 0.05f;
float g_cam_pitch    = 0.45f;
float g_cam_fov      = 38.0f;

int      g_selected_object_idx = 0;
bool     g_is_grabbing         = false;
bool     g_is_scaling          = false;
bool     g_is_orbiting         = false;
bool     g_is_panning          = false;
bool     g_show_floor_grid     = true;

Vec3     g_grab_initial_pos;
float    g_scale_initial_r     = 1.0f;
POINT    g_grab_start_mouse    = {0, 0};
POINT    g_nav_last_mouse      = {0, 0};
bool     g_grab_by_drag        = false;  // Przesuwanie obiektu przeciąganiem ŚPM (zatwierdzane puszczeniem przycisku)
float    g_pan_depth           = 4.6f;   // Głębokość (wzdłuż osi widoku) punktu złapanego kursorem przy przesuwaniu widoku
Vec3     g_grab_plane_hit0;
Vec3     g_grab_plane_normal;
Vec3     g_grab_plane_point;
AxisLock g_axis_lock           = AXIS_NONE;

HWND g_hWnd          = nullptr;
HWND g_hCanvas       = nullptr;

HWND g_hGroupScene   = nullptr;
HWND g_hLblScene     = nullptr;
HWND g_hComboScene   = nullptr;
HWND g_hBtnImportObj = nullptr;
HWND g_hBtnExportObj = nullptr;

HWND g_hGroupRender  = nullptr;
HWND g_hLblAlgo      = nullptr;
HWND g_hComboAlgo    = nullptr;
HWND g_hLblPhotons   = nullptr;
HWND g_hEditPhotons  = nullptr;
HWND g_hLblCausticBlur = nullptr, g_hEditCausticBlur = nullptr;
HWND g_hLblCausticLight = nullptr, g_hEditCausticLight = nullptr;
HWND g_hLblCausticGain = nullptr, g_hEditCausticGain = nullptr;
HWND g_hLblCausticGrid = nullptr, g_hComboCausticGrid = nullptr;
HWND g_hLblCausticRadius = nullptr, g_hEditCausticRadius = nullptr;
HWND g_hBtnCausticApply = nullptr;
HWND g_hLblPhotHint  = nullptr;
HWND g_hLblRes       = nullptr;
HWND g_hComboRes     = nullptr;
HWND g_hLblW         = nullptr;
HWND g_hEditResW     = nullptr;
HWND g_hLblH         = nullptr;
HWND g_hEditResH     = nullptr;
HWND g_hLblTile      = nullptr;
HWND g_hComboTile    = nullptr;
HWND g_hLblSPP       = nullptr;
HWND g_hEditSPP      = nullptr;
HWND g_hLblDepth     = nullptr;
HWND g_hEditDepth    = nullptr;
HWND g_hBtnRender    = nullptr;
HWND g_hBtnStop      = nullptr;
HWND g_hChkDenoise   = nullptr;
HWND g_hBtnDenoiseNow= nullptr;

HWND g_hGroupExposure= nullptr;
HWND g_hLblExposure  = nullptr;
HWND g_hEditExposure = nullptr;
HWND g_hBtnEvDown    = nullptr;
HWND g_hBtnEvUp      = nullptr;
HWND g_hBtnEvReset   = nullptr;
HWND g_hBtnEvApply   = nullptr;
HWND g_hLblFStopHint = nullptr;
HWND g_hLblToneMap   = nullptr;
HWND g_hComboToneMap = nullptr;

HWND g_hChkLiveViewportPT = nullptr;
HWND g_hChkCudaCaustics   = nullptr;
HWND g_hChkCausticsHQ     = nullptr;
HWND g_hLblLightPower     = nullptr;
HWND g_hEditLightPower    = nullptr;
HWND g_hBtnLightDown      = nullptr;
HWND g_hBtnLightUp        = nullptr;
HWND g_hBtnLightHalf      = nullptr;
HWND g_hBtnLightDouble    = nullptr;
HWND g_hBtnLightReset     = nullptr;
HWND g_hLblVpSpp          = nullptr;

void RefreshDisplayBufferFromRaw() {
    size_t total_pixels = static_cast<size_t>(g_render_width) * g_render_height;
    if (total_pixels == 0 || g_raw_image.size() < total_pixels || g_display_buffer.size() < total_pixels) return;

    float ev = g_exposure_ev.load();
    int tm_mode = g_tonemap_mode.load();

    for (size_t i = 0; i < total_pixels; ++i) {
        int count = std::max(1, g_sample_counts[i]);
        Vec3 rad = g_raw_image[i] / static_cast<float>(count);
        Vec3 srgb = ApplyExposureAndToneMap(rad, ev, tm_mode);

        uint8_t ir = static_cast<uint8_t>(std::clamp(srgb.x * 255.0f, 0.0f, 255.0f));
        uint8_t ig = static_cast<uint8_t>(std::clamp(srgb.y * 255.0f, 0.0f, 255.0f));
        uint8_t ib = static_cast<uint8_t>(std::clamp(srgb.z * 255.0f, 0.0f, 255.0f));

        g_display_buffer[i] = (ib) | (ig << 8) | (ir << 16) | (0xFF << 24);
    }
    if (g_hCanvas) InvalidateRect(g_hCanvas, NULL, FALSE);
}

void UpdateFStopLabel() {
    if (!g_hLblFStopHint) return;
    float ev = g_exposure_ev.load();
    float f_number = 2.8f * std::pow(2.0f, -ev * 0.5f);
    float light_factor = std::pow(2.0f, ev);

    wchar_t buf[140];
    if (std::abs(ev) < 0.01f) {
        swprintf(buf, 140, L"Przysłona: f/%.1f (EV 0.0 | 100%% światła)", f_number);
    } else if (ev < 0.0f) {
        swprintf(buf, 140, L"Przysłona: f/%.1f (%.1f stopów ciemniej | %.1f%% światła)", f_number, -ev, light_factor * 100.0f);
    } else {
        swprintf(buf, 140, L"Przysłona: f/%.1f (+%.1f stopów jaśniej | %.0f%% światła)", f_number, ev, light_factor * 100.0f);
    }
    SetWindowTextW(g_hLblFStopHint, buf);
}

HWND g_hGroupObj     = nullptr;
HWND g_hLblObj       = nullptr;
HWND g_hComboObjects = nullptr;
HWND g_hLblPosX      = nullptr;
HWND g_hEditPosX     = nullptr;
HWND g_hLblPosY      = nullptr;
HWND g_hEditPosY     = nullptr;
HWND g_hLblPosZ      = nullptr;
HWND g_hEditPosZ     = nullptr;
HWND g_hLblRadius    = nullptr;
HWND g_hEditRadius   = nullptr;
HWND g_hBtnApplyPos  = nullptr;
HWND g_hBtnGrab      = nullptr;
HWND g_hBtnResetCam  = nullptr;

HWND g_hLblMatType   = nullptr;
HWND g_hComboMatType = nullptr;
HWND g_hLblMatCol    = nullptr;
HWND g_hEditMatR     = nullptr;
HWND g_hEditMatG     = nullptr;
HWND g_hEditMatB     = nullptr;
HWND g_hLblMatParam  = nullptr;
HWND g_hEditMatParam = nullptr;
HWND g_hLblMatDisp   = nullptr;
HWND g_hEditMatDisp  = nullptr;
HWND g_hLblMatDispHint = nullptr;
HWND g_hBtnApplyMat  = nullptr;

HWND g_hGroupSave    = nullptr;
HWND g_hLblFile      = nullptr;
HWND g_hEditFilename = nullptr;
HWND g_hBtnSave      = nullptr;
HWND g_hLblSaveStatus = nullptr;  // komunikat o wyniku zapisu PNG obok przycisku
HWND g_hProgressBar  = nullptr;
HWND g_hStatusLabel  = nullptr;

Camera GetCameraForRender(int width, int height) {
    float aspect_ratio = static_cast<float>(width) / static_cast<float>(std::max(1, height));

    float cy = std::cos(g_cam_yaw);
    float sy = std::sin(g_cam_yaw);
    float cp = std::cos(g_cam_pitch);
    float sp = std::sin(g_cam_pitch);

    Vec3 lookfrom(
        g_cam_target.x + g_cam_distance * cp * sy,
        g_cam_target.y + g_cam_distance * sp,
        g_cam_target.z + g_cam_distance * cp * cy
    );

    Vec3 vup(0.0f, 1.0f, 0.0f);
    return Camera(lookfrom, g_cam_target, vup, g_cam_fov, aspect_ratio);
}

Camera GetCurrentCamera() {
    return GetCameraForRender(g_canvas_view_width, g_canvas_view_height);
}

bool ProjectToScreen(const Vec3& p, const Camera& cam, float viewport_w, float viewport_h,
                     int screen_w, int screen_h, int& out_x, int& out_y) {
    Vec3 v = p - cam.origin;
    Vec3 w_dir = unit_vector(cam.lookat - cam.origin);
    float dz = dot(v, w_dir);
    if (dz <= 0.05f) return false;

    Vec3 u_dir = unit_vector(cam.horizontal);
    Vec3 v_dir = unit_vector(cam.vertical);

    float px = dot(v, u_dir) / dz;
    float py = dot(v, v_dir) / dz;

    float u = 0.5f + px / viewport_w;
    float v_norm = 0.5f + py / viewport_h;

    out_x = static_cast<int>(u * static_cast<float>(screen_w));
    out_y = static_cast<int>((1.0f - v_norm) * static_cast<float>(screen_h));
    return (out_x >= -300 && out_x <= screen_w + 300 && out_y >= -300 && out_y <= screen_h + 300);
}

int RayPickObject(int mx, int my) {
    if (mx < 0 || mx >= g_canvas_view_width || my < 0 || my >= g_canvas_view_height) return -1;
    Camera cam = GetCurrentCamera();

    float u = (static_cast<float>(mx) + 0.5f) / static_cast<float>(g_canvas_view_width);
    float v = (static_cast<float>(g_canvas_view_height - 1 - my) + 0.5f) / static_cast<float>(g_canvas_view_height);
    Ray r = cam.get_ray(u, v);

    int best_idx = -1;
    float closest_t = 1e9f;

    for (int i = 0; i < static_cast<int>(g_scene_objects.size()); ++i) {
        auto h = CreateHittableFromObject(g_scene_objects[i]);
        HitRecord rec;
        if (h->hit(r, 0.001f, closest_t, rec)) {
            closest_t = rec.t;
            best_idx = i;
        }
    }
    return best_idx;
}

// Głębokość sceny pod kursorem mierzona wzdłuż osi widoku kamery (promień kamery ma składową osiową = 1, więc t = głębokość).
// Gdy kursor nie trafia w geometrię, zwracana jest odległość kamery od punktu orbitowania.
float RayPickDepth(int mx, int my) {
    if (mx < 0 || mx >= g_canvas_view_width || my < 0 || my >= g_canvas_view_height) return g_cam_distance;
    Camera cam = GetCurrentCamera();
    float u = (static_cast<float>(mx) + 0.5f) / static_cast<float>(g_canvas_view_width);
    float v = (static_cast<float>(g_canvas_view_height - 1 - my) + 0.5f) / static_cast<float>(g_canvas_view_height);
    Ray r = cam.get_ray(u, v);

    float closest_t = 1e9f;
    for (int i = 0; i < static_cast<int>(g_scene_objects.size()); ++i) {
        auto h = CreateHittableFromObject(g_scene_objects[i]);
        HitRecord rec;
        if (h->hit(r, 0.001f, closest_t, rec)) closest_t = rec.t;
    }
    if (closest_t > 1e8f) return g_cam_distance;
    return std::clamp(closest_t, 0.05f, 200.0f);
}

class ViewportThreadPool {
public:
    ViewportThreadPool(size_t threads = std::clamp(std::thread::hardware_concurrency(), 2u, 8u))
        : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        cv.wait(lock, [this]() { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        active_tasks--;
                        if (active_tasks == 0 && tasks.empty()) {
                            done_cv.notify_all();
                        }
                    }
                }
            });
        }
    }

    void parallel_for(int start, int end, const std::function<void(int, int)>& func) {
        if (start >= end) return;
        int total = end - start;
        int num_chunks = std::min(static_cast<int>(workers.size()), total);
        int chunk_size = (total + num_chunks - 1) / num_chunks;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            active_tasks = num_chunks;
            for (int i = 0; i < num_chunks; ++i) {
                int c_start = start + i * chunk_size;
                int c_end = std::min(end, c_start + chunk_size);
                tasks.push([func, c_start, c_end]() {
                    func(c_start, c_end);
                });
            }
        }
        cv.notify_all();

        std::unique_lock<std::mutex> lock(queue_mutex);
        done_cv.wait(lock, [this]() { return active_tasks == 0 && tasks.empty(); });
    }

    ~ViewportThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        cv.notify_all();
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::condition_variable done_cv;
    size_t active_tasks = 0;
    bool stop = false;
};

std::unique_ptr<ViewportThreadPool> g_viewport_pool;
std::atomic<bool> g_viewport_dirty(false);

void RenderFastPreview() {
    if (g_is_rendering.load()) return;

    Camera cam = GetCurrentCamera();
    bool is_interacting = (g_is_orbiting || g_is_panning || g_is_grabbing || g_is_scaling);

    // Dynamiczna rozdzielczość podczas interaktywnego obracania/przesuwania (płynne 200+ FPS)
    int pw = is_interacting ? std::clamp(g_canvas_view_width / 3, 160, 360) : std::max(100, g_canvas_view_width / 2);
    int ph = is_interacting ? std::clamp(g_canvas_view_height / 3, 90, 200) : std::max(60, g_canvas_view_height / 2);

    std::vector<uint32_t> small_buf(pw * ph);

    HittableList world;
    for (int i = 0; i < static_cast<int>(g_scene_objects.size()); ++i) {
        world.add(CreateHittableFromObject(g_scene_objects[i]));
    }
    if (g_current_scene == 0) {
        world.add(std::make_shared<Sphere>(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f,
                                           std::make_shared<Lambertian>(Vec3(0.85f, 0.85f, 0.88f))));
    }

    Vec3 light_pos(3.2f, 2.4f, -1.2f);
    for (const auto& obj : g_scene_objects) {
        if (obj.material && !obj.material->emitted(obj.center).near_zero()) {
            light_pos = obj.center;
            break;
        }
    }

    if (!g_viewport_pool) {
        g_viewport_pool = std::make_unique<ViewportThreadPool>();
    }

    g_viewport_pool->parallel_for(0, ph, [&](int y_start, int y_end) {
        for (int py = y_start; py < y_end; ++py) {
            float v = (static_cast<float>(ph - 1 - py) + 0.5f) / static_cast<float>(ph);
            int row_offset = py * pw;

            for (int px = 0; px < pw; ++px) {
                float u = (static_cast<float>(px) + 0.5f) / static_cast<float>(pw);
                Ray r = cam.get_ray(u, v);

                HitRecord rec;
                Vec3 col;
                if (world.hit(r, 0.001f, 1e9f, rec)) {
                    Vec3 to_light = light_pos - rec.p;
                    float dist_l = to_light.length();
                    Vec3 dir_l = to_light / (dist_l > 1e-4f ? dist_l : 1.0f);
                    float diff = std::fmax(0.15f, dot(rec.normal, dir_l));

                    if (!rec.mat->emitted(rec.p).near_zero()) {
                        col = Vec3(1.0f, 0.95f, 0.85f);
                    } else if (rec.mat->is_dielectric()) {
                        float rim = std::pow(1.0f - std::fabs(dot(rec.normal, -unit_vector(r.direction))), 2.5f);
                        col = Vec3(0.12f, 0.18f, 0.28f) + Vec3(0.85f, 0.95f, 1.0f) * (rim * 0.9f);
                    } else if (rec.mat->is_specular()) {
                        Vec3 refl = reflect(unit_vector(r.direction), rec.normal);
                        float spec = std::pow(std::fmax(0.0f, dot(refl, dir_l)), 16.0f);
                        col = rec.mat->get_albedo() * diff + Vec3(1.0f, 1.0f, 1.0f) * (spec * 0.6f);
                    } else {
                        col = rec.mat->get_albedo() * diff;

                        if (g_show_floor_grid && std::fabs(rec.p.y) < 0.05f) {
                            float fx = std::fabs(rec.p.x - std::round(rec.p.x));
                            float fz = std::fabs(rec.p.z - std::round(rec.p.z));
                            if (fx < 0.035f || fz < 0.035f) {
                                col = col * 0.82f + Vec3(0.40f, 0.45f, 0.55f) * 0.35f;
                            }
                            if (std::fabs(rec.p.z) < 0.045f && std::fabs(rec.p.x) < 8.0f) {
                                col = Vec3(0.85f, 0.25f, 0.25f);
                            } else if (std::fabs(rec.p.x) < 0.045f && std::fabs(rec.p.z) < 8.0f) {
                                col = Vec3(0.25f, 0.45f, 0.90f);
                            }
                        }
                    }
                } else {
                    Vec3 unit_dir = unit_vector(r.direction);
                    float a = 0.5f * (unit_dir.y + 1.0f);
                    col = (1.0f - a) * Vec3(0.06f, 0.08f, 0.12f) + a * Vec3(0.18f, 0.22f, 0.32f);
                }

                float cr = std::clamp(std::sqrt(col.x), 0.0f, 0.999f);
                float cg = std::clamp(std::sqrt(col.y), 0.0f, 0.999f);
                float cb = std::clamp(std::sqrt(col.z), 0.0f, 0.999f);

                uint8_t ir = static_cast<uint8_t>(256.0f * cr);
                uint8_t ig = static_cast<uint8_t>(256.0f * cg);
                uint8_t ib = static_cast<uint8_t>(256.0f * cb);

                small_buf[row_offset + px] = (ib) | (ig << 8) | (ir << 16) | (0xFF << 24);
            }
        }
    });

    g_display_buffer = std::move(small_buf);
    g_cur_buf_w.store(pw);
    g_cur_buf_h.store(ph);
}

// ============================================================================
// BUDOWA DANYCH SCENY GPU CUDA ZE SKALOWANIEM MOCY ŹRÓDEŁ ŚWIATŁA
// ============================================================================
CudaSceneData BuildCudaScene(float light_multiplier = 1.0f) {
    CudaSceneData gpu_scene;
    if (g_current_scene == 0) {
        gpu_scene.spheres.push_back({
            CudaVec3(0.0f, -1000.0f, 0.0f), 1000.0f,
            CUDA_MAT_LAMBERTIAN, CudaVec3(0.85f, 0.85f, 0.88f), 0.0f
        });
    }

    for (const auto& obj : g_scene_objects) {
        float effective_param = obj.mat_param;
        if (obj.mat_type == 3) {
            effective_param *= light_multiplier;
        }

        if (obj.is_crystal) {
            float radius = obj.radius;
            float height = obj.secondary_param > 0.1f ? obj.secondary_param : 1.40f;
            float half_h = height * 0.5f;
            float b_rad = std::sqrt(radius * radius + half_h * half_h) * 1.18f;

            CudaCrystal cc;
            cc.center = CudaVec3(obj.center.x, obj.center.y, obj.center.z);
            cc.bounding_radius = b_rad;
            cc.plane_start = static_cast<int>(gpu_scene.crystal_planes.size());
            cc.plane_count = 20;
            cc.mat_type = obj.mat_type;
            cc.color = CudaVec3(obj.mat_color.x, obj.mat_color.y, obj.mat_color.z);
            cc.param = (effective_param > 0.1f) ? effective_param : 1.55f;
            cc.dispersion = obj.mat_dispersion;

            // 20 faset kryształu (identycznie jak w CreateFacetedCrystal):
            // Dno i szczyt
            gpu_scene.crystal_planes.push_back({ CudaVec3(0.0f, -1.0f, 0.0f), half_h });
            gpu_scene.crystal_planes.push_back({ CudaVec3(0.0f, 1.0f, 0.0f), half_h });

            // 6 pasów pionowych
            for (int i = 0; i < 6; ++i) {
                float angle = i * (3.14159265f / 3.0f);
                gpu_scene.crystal_planes.push_back({ CudaVec3(std::cos(angle), 0.0f, std::sin(angle)), radius });
            }

            // 6 faset korony
            for (int i = 0; i < 6; ++i) {
                float angle = (i + 0.5f) * (3.14159265f / 3.0f);
                Vec3 slanted_norm = unit_vector(Vec3(std::cos(angle) * 0.85f, 1.15f, std::sin(angle) * 0.85f));
                gpu_scene.crystal_planes.push_back({ CudaVec3(slanted_norm.x, slanted_norm.y, slanted_norm.z), radius * 1.08f });
            }

            // 6 faset podstawy
            for (int i = 0; i < 6; ++i) {
                float angle = i * (3.14159265f / 3.0f);
                Vec3 slanted_norm = unit_vector(Vec3(std::cos(angle) * 0.80f, -1.25f, std::sin(angle) * 0.80f));
                gpu_scene.crystal_planes.push_back({ CudaVec3(slanted_norm.x, slanted_norm.y, slanted_norm.z), radius * 1.18f });
            }

            gpu_scene.crystals.push_back(cc);
        }
        else if (obj.is_quad) {
            Vec3 corner = obj.center;
            Vec3 u = obj.quad_u * obj.radius;
            Vec3 v = obj.quad_v * obj.radius;
            Vec3 n = cross(u, v);
            Vec3 normal = unit_vector(n);
            float d = dot(normal, corner);
            float len_sq = dot(n, n);
            Vec3 w = len_sq > 1e-8f ? (n / len_sq) : Vec3(0, 0, 0);

            CudaQuad cq;
            cq.q = CudaVec3(corner.x, corner.y, corner.z);
            cq.u = CudaVec3(u.x, u.y, u.z);
            cq.v = CudaVec3(v.x, v.y, v.z);
            cq.normal = CudaVec3(normal.x, normal.y, normal.z);
            cq.d = d;
            cq.w = CudaVec3(w.x, w.y, w.z);
            cq.mat_type = obj.mat_type;
            cq.color = CudaVec3(obj.mat_color.x, obj.mat_color.y, obj.mat_color.z);
            cq.param = effective_param;

            gpu_scene.quads.push_back(cq);
        }
        else if (obj.is_mesh && obj.mesh_triangles && !obj.mesh_triangles->empty()) {
            CudaMesh cm;
            float scale = obj.radius;
            Vec3 bmin = obj.center + obj.mesh_bmin * scale;
            Vec3 bmax = obj.center + obj.mesh_bmax * scale;
            if (bmin.x > bmax.x) std::swap(bmin.x, bmax.x);
            if (bmin.y > bmax.y) std::swap(bmin.y, bmax.y);
            if (bmin.z > bmax.z) std::swap(bmin.z, bmax.z);

            cm.bmin = CudaVec3(bmin.x, bmin.y, bmin.z);
            cm.bmax = CudaVec3(bmax.x, bmax.y, bmax.z);
            cm.tri_start = static_cast<int>(gpu_scene.triangles.size());
            cm.tri_count = static_cast<int>(obj.mesh_triangles->size());
            cm.mat_type = obj.mat_type;
            cm.color = CudaVec3(obj.mat_color.x, obj.mat_color.y, obj.mat_color.z);
            cm.param = effective_param;

            for (const auto& tri : *obj.mesh_triangles) {
                CudaTriangle ct;
                Vec3 v0 = obj.center + tri.v0 * scale;
                Vec3 v1 = obj.center + tri.v1 * scale;
                Vec3 v2 = obj.center + tri.v2 * scale;
                ct.v0 = CudaVec3(v0.x, v0.y, v0.z);
                ct.v1 = CudaVec3(v1.x, v1.y, v1.z);
                ct.v2 = CudaVec3(v2.x, v2.y, v2.z);
                ct.normal = CudaVec3(tri.normal.x, tri.normal.y, tri.normal.z);
                gpu_scene.triangles.push_back(ct);
            }

            gpu_scene.meshes.push_back(cm);
        }
        else {
            CudaSphere cs;
            cs.center = CudaVec3(obj.center.x, obj.center.y, obj.center.z);
            cs.radius = obj.radius;
            cs.mat_type = obj.mat_type;
            cs.color = CudaVec3(obj.mat_color.x, obj.mat_color.y, obj.mat_color.z);
            cs.param = effective_param;
            gpu_scene.spheres.push_back(cs);
        }
    }
    return gpu_scene;
}

// ============================================================================
// PODGLĄD PATH TRACINGU NA ŻYWO W VIEWPORCIE (GPU CUDA / RTX)
// ============================================================================
void RenderLiveViewportFrame(bool is_interactive) {
    if (g_is_rendering.load() || g_has_rendered_image.load()) return;

    if (!g_live_viewport_pt.load() || !g_cuda_available) {
        RenderFastPreview();
        return;
    }

    int vw = g_canvas_view_width;
    int vh = g_canvas_view_height;
    if (vw <= 0 || vh <= 0) return;

    // Podczas ruchu wykonujemy 1 SPP dla 100+ FPS, w bezruchu akumulujemy po 2 SPP
    int spp_step = is_interactive ? 1 : 2;
    if (g_viewport_pt_spp >= 256) return;

    Camera cam = GetCurrentCamera();
    CudaVec3 c_orig(cam.origin.x, cam.origin.y, cam.origin.z);
    CudaVec3 c_look(cam.lookat.x, cam.lookat.y, cam.lookat.z);
    CudaVec3 c_vup(cam.vup.x, cam.vup.y, cam.vup.z);
    float c_fov = g_cam_fov;

    CudaSceneData scene = BuildCudaScene(g_global_light_mult.load());

    CudaRenderOptions render_opts;
    render_opts.exposure_ev = g_exposure_ev.load();
    render_opts.tonemap_mode = g_tonemap_mode.load();
    render_opts.clamp_indirect = 12.0f;
    render_opts.enable_caustics = g_cuda_caustics.load() ? 1 : 0;
    render_opts.caustic_hq = 0;
    render_opts.caustic_grid_res = g_caustic_grid_res.load();
    render_opts.caustic_blur_cm = g_caustic_blur_cm.load();
    render_opts.caustic_light_size = g_caustic_light_size.load();
    render_opts.caustic_gain = g_caustic_gain.load();
    render_opts.caustic_photons = (g_photon_count > 50000) ? g_photon_count : 262144;
    render_opts.scene_mode = g_current_scene;

    if (g_display_buffer.size() < static_cast<size_t>(vw * vh)) {
        g_display_buffer.resize(vw * vh);
    }

    bool ok = CudaPT_RenderViewportStep(
        vw, vh,
        spp_step, g_viewport_pt_spp,
        c_orig, c_look, c_vup, c_fov,
        scene, render_opts,
        g_display_buffer.data()
    );

    if (ok) {
        g_cur_buf_w.store(vw);
        g_cur_buf_h.store(vh);
        if (g_hLblVpSpp) {
            wchar_t spp_text[64];
            if (g_viewport_pt_spp >= 256) {
                swprintf(spp_text, 64, L"VP: 256 SPP (Czysty podgląd)");
            } else {
                swprintf(spp_text, 64, L"VP: %d / 256 SPP", g_viewport_pt_spp);
            }
            SetWindowTextW(g_hLblVpSpp, spp_text);
        }
    } else {
        RenderFastPreview();
    }
}

void ToggleLiveViewportPT() {
    bool newState = !g_live_viewport_pt.load();
    g_live_viewport_pt.store(newState);
    if (g_hChkLiveViewportPT) {
        SendMessage(g_hChkLiveViewportPT, BM_SETCHECK, newState ? BST_CHECKED : BST_UNCHECKED, 0);
    }
    if (newState && g_cuda_available) {
        g_viewport_pt_spp = 0;
        RenderLiveViewportFrame(true);
        SetWindowTextW(g_hStatusLabel, L"Włączono podgląd Path Tracing na żywo w Viewporcie (GPU RTX). Klawisz [Z] przełącza tryb.");
    } else {
        CudaPT_ResetViewportAccum();
        RenderFastPreview();
        SetWindowTextW(g_hStatusLabel, L"Przełączono na szybki podgląd bryłowy (Workbench). Klawisz [Z] włącza Path Tracing GPU.");
        if (g_hLblVpSpp) SetWindowTextW(g_hLblVpSpp, L"VP: Wyłączony");
    }
    if (g_hCanvas) InvalidateRect(g_hCanvas, NULL, FALSE);
}

void UpdateGlobalLightMultiplier(float new_mult) {
    new_mult = std::clamp(new_mult, 0.05f, 20.0f);
    g_global_light_mult.store(new_mult);
    if (g_hEditLightPower) {
        wchar_t buf[32];
        swprintf(buf, 32, L"%.2fx", new_mult);
        SetWindowTextW(g_hEditLightPower, buf);
    }
    g_viewport_pt_spp = 0;
    if (g_live_viewport_pt.load() && g_cuda_available && !g_is_rendering.load() && !g_has_rendered_image.load()) {
        RenderLiveViewportFrame(false);
    } else if (!g_is_rendering.load() && !g_has_rendered_image.load()) {
        RenderFastPreview();
    }
    if (g_hCanvas) InvalidateRect(g_hCanvas, NULL, FALSE);
}

void ExportSceneToOBJ() {
    if (g_is_rendering.load()) {
        MessageBoxW(g_hWnd, L"Renderowanie jest w toku! Przerwij renderowanie, aby wyeksportować scenę.", L"Uwaga", MB_OK | MB_ICONWARNING);
        return;
    }

    wchar_t filename[MAX_PATH] = L"scena_eksport.obj";
    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof(OPENFILENAMEW);
    ofn.hwndOwner = g_hWnd;
    ofn.lpstrFilter = L"Modele Wavefront OBJ (*.obj)\0*.obj\0Wszystkie pliki (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;

    if (GetSaveFileNameW(&ofn)) {
        FILE* fp = _wfopen(filename, L"w");
        if (!fp) {
            MessageBoxW(g_hWnd, L"Nie można utworzyć pliku docelowego do zapisu sceny.", L"Błąd zapisu OBJ", MB_OK | MB_ICONERROR);
            return;
        }

        fprintf(fp, "# Path Tracer 3D - Scene Export Wavefront OBJ\n");
        fprintf(fp, "# Liczba obiektow: %zu\n\n", g_scene_objects.size());

        int v_offset = 1;

        for (size_t i = 0; i < g_scene_objects.size(); ++i) {
            const auto& obj = g_scene_objects[i];
            char obj_name[128];
            WideCharToMultiByte(CP_UTF8, 0, obj.name.c_str(), -1, obj_name, sizeof(obj_name), nullptr, nullptr);
            for (char* p = obj_name; *p; ++p) {
                if (*p == ' ' || *p == '/' || *p == '\\' || *p == '(' || *p == ')') *p = '_';
            }

            fprintf(fp, "o Object_%zu_%s\n", i + 1, obj_name);
            fprintf(fp, "# Typ: %d, Kolor RGB: (%.2f, %.2f, %.2f), Param: %.2f, Dyspersja: %.3f\n",
                    obj.mat_type, obj.mat_color.x, obj.mat_color.y, obj.mat_color.z, obj.mat_param, obj.mat_dispersion);

            if (obj.is_mesh && obj.mesh_triangles) {
                for (const auto& tri : *obj.mesh_triangles) {
                    Vec3 v0 = obj.center + tri.v0 * obj.radius;
                    Vec3 v1 = obj.center + tri.v1 * obj.radius;
                    Vec3 v2 = obj.center + tri.v2 * obj.radius;
                    fprintf(fp, "v %.6f %.6f %.6f\n", v0.x, v0.y, v0.z);
                    fprintf(fp, "v %.6f %.6f %.6f\n", v1.x, v1.y, v1.z);
                    fprintf(fp, "v %.6f %.6f %.6f\n", v2.x, v2.y, v2.z);
                    fprintf(fp, "f %d %d %d\n", v_offset, v_offset + 1, v_offset + 2);
                    v_offset += 3;
                }
            } else if (obj.is_quad) {
                Vec3 p0 = obj.center;
                Vec3 p1 = obj.center + obj.quad_u * obj.radius;
                Vec3 p2 = obj.center + (obj.quad_u + obj.quad_v) * obj.radius;
                Vec3 p3 = obj.center + obj.quad_v * obj.radius;
                fprintf(fp, "v %.6f %.6f %.6f\n", p0.x, p0.y, p0.z);
                fprintf(fp, "v %.6f %.6f %.6f\n", p1.x, p1.y, p1.z);
                fprintf(fp, "v %.6f %.6f %.6f\n", p2.x, p2.y, p2.z);
                fprintf(fp, "v %.6f %.6f %.6f\n", p3.x, p3.y, p3.z);
                fprintf(fp, "f %d %d %d\n", v_offset, v_offset + 1, v_offset + 2);
                fprintf(fp, "f %d %d %d\n", v_offset, v_offset + 2, v_offset + 3);
                v_offset += 4;
            } else if (obj.is_crystal) {
                float h = obj.secondary_param > 0.1f ? obj.secondary_param : 1.40f;
                float half_h = h * 0.5f;
                float r = obj.radius;
                int start_v = v_offset;

                fprintf(fp, "v %.6f %.6f %.6f\n", obj.center.x, obj.center.y + half_h, obj.center.z);
                fprintf(fp, "v %.6f %.6f %.6f\n", obj.center.x, obj.center.y - half_h, obj.center.z);

                for (int k = 0; k < 6; ++k) {
                    float a = k * (3.14159265f / 3.0f);
                    fprintf(fp, "v %.6f %.6f %.6f\n",
                            obj.center.x + std::cos(a) * r * 0.85f,
                            obj.center.y + half_h * 0.65f,
                            obj.center.z + std::sin(a) * r * 0.85f);
                }
                for (int k = 0; k < 6; ++k) {
                    float a = (k + 0.5f) * (3.14159265f / 3.0f);
                    fprintf(fp, "v %.6f %.6f %.6f\n",
                            obj.center.x + std::cos(a) * r,
                            obj.center.y,
                            obj.center.z + std::sin(a) * r);
                }
                for (int k = 0; k < 6; ++k) {
                    float a = k * (3.14159265f / 3.0f);
                    fprintf(fp, "v %.6f %.6f %.6f\n",
                            obj.center.x + std::cos(a) * r * 0.85f,
                            obj.center.y - half_h * 0.65f,
                            obj.center.z + std::sin(a) * r * 0.85f);
                }

                int top_idx = start_v;
                int bot_idx = start_v + 1;
                int ring_top = start_v + 2;
                int ring_mid = start_v + 8;
                int ring_bot = start_v + 14;

                for (int k = 0; k < 6; ++k) {
                    int nxt = (k + 1) % 6;
                    fprintf(fp, "f %d %d %d\n", top_idx, ring_top + k, ring_top + nxt);
                }
                for (int k = 0; k < 6; ++k) {
                    int nxt = (k + 1) % 6;
                    fprintf(fp, "f %d %d %d\n", ring_top + k, ring_mid + k, ring_top + nxt);
                    fprintf(fp, "f %d %d %d\n", ring_top + nxt, ring_mid + k, ring_mid + nxt);
                }
                for (int k = 0; k < 6; ++k) {
                    int nxt = (k + 1) % 6;
                    fprintf(fp, "f %d %d %d\n", ring_mid + k, ring_bot + k, ring_mid + nxt);
                    fprintf(fp, "f %d %d %d\n", ring_mid + nxt, ring_bot + k, ring_bot + nxt);
                }
                for (int k = 0; k < 6; ++k) {
                    int nxt = (k + 1) % 6;
                    fprintf(fp, "f %d %d %d\n", bot_idx, ring_bot + nxt, ring_bot + k);
                }
                v_offset += 20;
            } else {
                int rings = 12;
                int sectors = 18;
                int start_v = v_offset;

                for (int r = 0; r <= rings; ++r) {
                    float theta = r * (3.14159265f / rings);
                    float sin_t = std::sin(theta);
                    float cos_t = std::cos(theta);
                    for (int s = 0; s <= sectors; ++s) {
                        float phi = s * (6.2831853f / sectors);
                        float x = obj.center.x + obj.radius * sin_t * std::cos(phi);
                        float y = obj.center.y + obj.radius * cos_t;
                        float z = obj.center.z + obj.radius * sin_t * std::sin(phi);
                        fprintf(fp, "v %.6f %.6f %.6f\n", x, y, z);
                    }
                }

                for (int r = 0; r < rings; ++r) {
                    for (int s = 0; s < sectors; ++s) {
                        int i0 = start_v + r * (sectors + 1) + s;
                        int i1 = i0 + 1;
                        int i2 = start_v + (r + 1) * (sectors + 1) + s;
                        int i3 = i2 + 1;
                        fprintf(fp, "f %d %d %d\n", i0, i2, i1);
                        fprintf(fp, "f %d %d %d\n", i1, i2, i3);
                    }
                }
                v_offset += (rings + 1) * (sectors + 1);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        wchar_t msg[300];
        swprintf(msg, 300, L"Wyeksportowano pomyślnie %zu obiektów do pliku Wavefront OBJ!\nPlik docelowy:\n%s",
                 g_scene_objects.size(), filename);
        MessageBoxW(g_hWnd, msg, L"Eksport sceny 3D zakończony", MB_OK | MB_ICONINFORMATION);
    }
}

void UpdateObjectFieldsFromSelection(int index) {
    if (index < 0 || index >= static_cast<int>(g_scene_objects.size())) return;
    const auto& obj = g_scene_objects[index];

    wchar_t buf[64];
    swprintf(buf, 64, L"%.2f", obj.center.x);
    SetWindowTextW(g_hEditPosX, buf);

    swprintf(buf, 64, L"%.2f", obj.center.y);
    SetWindowTextW(g_hEditPosY, buf);

    swprintf(buf, 64, L"%.2f", obj.center.z);
    SetWindowTextW(g_hEditPosZ, buf);

    swprintf(buf, 64, L"%.2f", obj.radius);
    SetWindowTextW(g_hEditRadius, buf);

    if (g_hComboMatType) {
        SendMessage(g_hComboMatType, CB_SETCURSEL, obj.mat_type, 0);
    }
    if (g_hEditMatR) {
        swprintf(buf, 64, L"%.2f", obj.mat_color.x);
        SetWindowTextW(g_hEditMatR, buf);
    }
    if (g_hEditMatG) {
        swprintf(buf, 64, L"%.2f", obj.mat_color.y);
        SetWindowTextW(g_hEditMatG, buf);
    }
    if (g_hEditMatB) {
        swprintf(buf, 64, L"%.2f", obj.mat_color.z);
        SetWindowTextW(g_hEditMatB, buf);
    }
    if (g_hEditMatParam) {
        swprintf(buf, 64, L"%.2f", obj.mat_param);
        SetWindowTextW(g_hEditMatParam, buf);
    }
    if (g_hEditMatDisp) {
        swprintf(buf, 64, L"%.3f", obj.mat_dispersion);
        SetWindowTextW(g_hEditMatDisp, buf);
    }
    if (g_hLblMatCol) {
        if (obj.mat_type == 3) {
            SetWindowTextW(g_hLblMatCol, L"RGB/Moc światła:");
        } else {
            SetWindowTextW(g_hLblMatCol, L"RGB / parametr:");
        }
    }
}

void ApplyMaterialFromFields() {
    int sel = static_cast<int>(SendMessage(g_hComboObjects, CB_GETCURSEL, 0, 0));
    if (sel < 0 || sel >= static_cast<int>(g_scene_objects.size())) return;

    int mat_type = static_cast<int>(SendMessage(g_hComboMatType, CB_GETCURSEL, 0, 0));
    wchar_t bufR[32], bufG[32], bufB[32], bufP[32], bufDisp[32];
    GetWindowTextW(g_hEditMatR, bufR, 32);
    GetWindowTextW(g_hEditMatG, bufG, 32);
    GetWindowTextW(g_hEditMatB, bufB, 32);
    GetWindowTextW(g_hEditMatParam, bufP, 32);
    if (g_hEditMatDisp) GetWindowTextW(g_hEditMatDisp, bufDisp, 32);
    else wcscpy(bufDisp, L"0.00");

    float r = wcstof(bufR, nullptr);
    float g = wcstof(bufG, nullptr);
    float b = wcstof(bufB, nullptr);
    float param = wcstof(bufP, nullptr);
    float disp = wcstof(bufDisp, nullptr);
    if (disp < 0.0f) disp = 0.0f;
    if (disp > 0.20f) disp = 0.20f;

    SetObjectMaterial(g_scene_objects[sel], mat_type, Vec3(r, g, b), param, disp);

    const wchar_t* typeNames[] = { L"Lambertian (matowy)", L"Metal (zwierciadlany)", L"Dielectric (szkło)", L"DiffuseLight (światło)" };
    wchar_t msg[200];
    swprintf(msg, 200, L"Zmieniono materiał [%s] na: %s (R: %.2f, G: %.2f, B: %.2f, parametr: %.2f, dyspersja: %.3f)",
             g_scene_objects[sel].name.c_str(), typeNames[std::clamp(mat_type, 0, 3)], r, g, b, param, disp);
    SetWindowTextW(g_hStatusLabel, msg);

    g_viewport_pt_spp = 0;
    if (g_live_viewport_pt.load() && g_cuda_available && !g_is_rendering.load() && !g_has_rendered_image.load()) {
        RenderLiveViewportFrame(false);
    } else {
        RenderFastPreview();
    }
    if (g_hCanvas) InvalidateRect(g_hCanvas, NULL, FALSE);
}

void ImportOBJModel() {
    if (g_is_rendering.load()) {
        MessageBoxW(g_hWnd, L"Renderowanie jest w toku! Przerwij renderowanie, aby zaimportować model 3D.", L"Uwaga", MB_OK | MB_ICONWARNING);
        return;
    }

    wchar_t filename[MAX_PATH] = L"";
    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof(OPENFILENAMEW);
    ofn.hwndOwner = g_hWnd;
    ofn.lpstrFilter = L"Modele Wavefront OBJ (*.obj)\0*.obj\0Wszystkie pliki (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (GetOpenFileNameW(&ofn)) {
        std::vector<Triangle> tris;
        Vec3 bmin, bmax;
        if (!LoadOBJGeometry(filename, tris, bmin, bmax)) {
            MessageBoxW(g_hWnd, L"Nie udało się wczytać geometrii z pliku .OBJ lub plik nie zawiera trójkątów.", L"Błąd importu OBJ", MB_OK | MB_ICONERROR);
            return;
        }

        std::wstring pathStr(filename);
        size_t lastSlash = pathStr.find_last_of(L"\\/");
        std::wstring baseName = (lastSlash != std::wstring::npos) ? pathStr.substr(lastSlash + 1) : pathStr;

        EditableSceneObject mesh_obj;
        mesh_obj.name = L"Model: " + baseName;
        mesh_obj.center = (g_current_scene == 0) ? Vec3(0.0f, 0.7f, 0.0f) : Vec3(0.0f, 0.5f, 0.2f);
        mesh_obj.radius = 0.8f;
        mesh_obj.is_mesh = true;
        mesh_obj.mesh_triangles = std::make_shared<std::vector<Triangle>>(std::move(tris));
        mesh_obj.mesh_bmin = bmin;
        mesh_obj.mesh_bmax = bmax;
        SetObjectMaterial(mesh_obj, 0, Vec3(0.85f, 0.75f, 0.65f), 0.05f);

        g_scene_objects.push_back(mesh_obj);
        int new_idx = static_cast<int>(g_scene_objects.size()) - 1;

        if (g_hComboObjects) {
            SendMessage(g_hComboObjects, CB_ADDSTRING, 0, (LPARAM)mesh_obj.name.c_str());
            SendMessage(g_hComboObjects, CB_SETCURSEL, new_idx, 0);
        }
        g_selected_object_idx = new_idx;
        UpdateObjectFieldsFromSelection(new_idx);

        wchar_t status[240];
        swprintf(status, 240, L"Zaimportowano pomyślnie [%s] (%zu trójkątów). Użyj 'G', aby przesunąć, 'S', aby skalować, lub zmień materiał!",
                 baseName.c_str(), mesh_obj.mesh_triangles->size());
        SetWindowTextW(g_hStatusLabel, status);

        g_has_rendered_image.store(false);
        RenderFastPreview();
        if (g_hCanvas) InvalidateRect(g_hCanvas, NULL, FALSE);
    }
}

void SwitchScene(int scene_id) {
    g_has_rendered_image.store(false);
    g_current_scene = scene_id;

    if (scene_id == 0) {
        // Scena 1: Oryginalna scena z szlifowanym krysztalem i kulami
        g_scene_objects = {
            make_crystal(L"1. Kryształ szklany (Faceted Crystal)", Vec3(-0.5f, 0.70f, 0.1f), 0.65f, 1.40f, 1.55f),
            make_sphere(L"2. Źródło światła (kąt boczno-tylny)",   Vec3(3.2f, 2.4f, -1.2f),   0.85f, 3, Vec3(35.0f, 32.0f, 28.0f), 1.0f),
            make_sphere(L"3. Kula złota (prawa)",                 Vec3(1.7f, 0.80f, 0.6f),   0.80f, 1, Vec3(0.95f, 0.78f, 0.35f), 0.02f),
            make_sphere(L"4. Kula rubinowa (lewa-tył)",           Vec3(-2.0f, 0.65f, -0.6f), 0.65f, 0, Vec3(0.85f, 0.15f, 0.15f), 0.0f),
            make_sphere(L"5. Kula chromowana (przód)",            Vec3(0.7f, 0.35f, 1.7f),   0.35f, 1, Vec3(0.92f, 0.95f, 0.98f), 0.05f)
        };
        g_cam_target   = Vec3(-0.3f, 0.5f, 0.0f);
        g_cam_distance = 4.6f;
        g_cam_yaw      = 0.05f;
        g_cam_pitch    = 0.45f;
        g_cam_fov      = 38.0f;
    } else {
        // Scena 2: Zamknięty pokój z podłużnym oknem, zewnętrznym źródłem światła i zewnętrznym zwierciadłem
        g_scene_objects.clear();

        // 1. Kryształ szklany wewnątrz pokoju
        g_scene_objects.push_back(
            make_crystal(L"1. Kryształ w pokoju (refrakcja)", Vec3(0.0f, 0.55f, 0.2f), 0.50f, 1.10f, 1.55f)
        );

        // 2. Kula złota wewnątrz pokoju
        g_scene_objects.push_back(
            make_sphere(L"2. Kula złota w pokoju", Vec3(0.75f, 0.35f, 0.4f), 0.35f, 1, Vec3(0.95f, 0.78f, 0.35f), 0.03f)
        );

        // 3. Zewnętrzne źródło światła (za przesłoną ochronną)
        g_scene_objects.push_back(
            make_sphere(L"3. Zewnętrzne źródło światła", Vec3(2.2f, 1.15f, 2.1f), 0.40f, 3, Vec3(70.0f, 65.0f, 55.0f), 1.0f)
        );

        // 4. Zewnętrzne lustro odbijające światło do wnętrza przez okno
        g_scene_objects.push_back(
            make_quad(L"4. Zewnętrzne lustro (reflektor)", Vec3(-2.0f, 0.0f, 3.2f), Vec3(4.0f, 0.0f, 0.0f), Vec3(0.0f, 2.4f, 0.0f), 1, Vec3(0.98f, 0.98f, 0.98f), 0.01f)
        );

        // 5. Zewnętrzna przesłona (baffle uniemożliwiający oświetlenie bezpośrednie)
        g_scene_objects.push_back(
            make_quad(L"5. Zewnętrzna przesłona (baffle)", Vec3(1.35f, 0.0f, 1.4f), Vec3(0.0f, 0.0f, 1.4f), Vec3(0.0f, 2.4f, 0.0f), 0, Vec3(0.2f, 0.2f, 0.25f))
        );

        // 6-9. Ściana z podłużnym oknem (dół, góra, lewo, prawo)
        g_scene_objects.push_back(
            make_quad(L"6. Ściana z oknem (panel dolny)", Vec3(-2.0f, 0.0f, 1.4f), Vec3(4.0f, 0.0f, 0.0f), Vec3(0.0f, 0.95f, 0.0f), 0, Vec3(0.72f, 0.70f, 0.68f))
        );
        g_scene_objects.push_back(
            make_quad(L"7. Ściana z oknem (panel górny)", Vec3(-2.0f, 1.35f, 1.4f), Vec3(4.0f, 0.0f, 0.0f), Vec3(0.0f, 1.05f, 0.0f), 0, Vec3(0.72f, 0.70f, 0.68f))
        );
        g_scene_objects.push_back(
            make_quad(L"8. Ściana z oknem (panel lewy)", Vec3(-2.0f, 0.95f, 1.4f), Vec3(0.8f, 0.0f, 0.0f), Vec3(0.0f, 0.40f, 0.0f), 0, Vec3(0.72f, 0.70f, 0.68f))
        );
        g_scene_objects.push_back(
            make_quad(L"9. Ściana z oknem (panel prawy)", Vec3(1.2f, 0.95f, 1.4f), Vec3(0.8f, 0.0f, 0.0f), Vec3(0.0f, 0.40f, 0.0f), 0, Vec3(0.72f, 0.70f, 0.68f))
        );

        // 10-14. Ściany zamkniętego pokoju (podłoga, sufit, lewa, prawa, tylna za kamerą)
        g_scene_objects.push_back(
            make_quad(L"10. Podłoga pokoju", Vec3(-2.0f, 0.0f, -2.0f), Vec3(4.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 3.4f), 0, Vec3(0.75f, 0.72f, 0.68f))
        );
        g_scene_objects.push_back(
            make_quad(L"11. Sufit pokoju", Vec3(-2.0f, 2.4f, -2.0f), Vec3(4.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 3.4f), 0, Vec3(0.82f, 0.82f, 0.82f))
        );
        g_scene_objects.push_back(
            make_quad(L"12. Ściana lewa", Vec3(-2.0f, 0.0f, -2.0f), Vec3(0.0f, 0.0f, 3.4f), Vec3(0.0f, 2.4f, 0.0f), 0, Vec3(0.75f, 0.35f, 0.32f))
        );
        g_scene_objects.push_back(
            make_quad(L"13. Ściana prawa", Vec3(2.0f, 0.0f, 1.4f), Vec3(0.0f, 0.0f, -3.4f), Vec3(0.0f, 2.4f, 0.0f), 0, Vec3(0.32f, 0.65f, 0.42f))
        );
        g_scene_objects.push_back(
            make_quad(L"14. Ściana tylna", Vec3(2.0f, 0.0f, -2.0f), Vec3(-4.0f, 0.0f, 0.0f), Vec3(0.0f, 2.4f, 0.0f), 0, Vec3(0.68f, 0.68f, 0.72f))
        );

        // Kamera wewnatrz zamknietego pokoju skierowana na okno
        g_cam_target   = Vec3(0.0f, 0.85f, 0.6f);
        g_cam_distance = 1.9f;
        g_cam_yaw      = 3.14159f;
        g_cam_pitch    = -0.05f;
        g_cam_fov      = 52.0f;
    }

    if (g_hComboObjects) {
        SendMessage(g_hComboObjects, CB_RESETCONTENT, 0, 0);
        for (const auto& obj : g_scene_objects) {
            SendMessage(g_hComboObjects, CB_ADDSTRING, 0, (LPARAM)obj.name.c_str());
        }
        SendMessage(g_hComboObjects, CB_SETCURSEL, 0, 0);
    }
    g_selected_object_idx = 0;
    UpdateObjectFieldsFromSelection(0);

    g_viewport_pt_spp = 0;
    if (g_live_viewport_pt.load() && g_cuda_available && !g_is_rendering.load()) {
        RenderLiveViewportFrame(false);
    } else {
        RenderFastPreview();
    }
    if (g_hCanvas) InvalidateRect(g_hCanvas, NULL, FALSE);
}

void StartGrabMode(int start_mx = -1, int start_my = -1) {
    if (g_selected_object_idx < 0 || g_selected_object_idx >= static_cast<int>(g_scene_objects.size())) {
        SetWindowTextW(g_hStatusLabel, L"Najpierw kliknij obiekt środkowym przyciskiem myszy (ŚPM), aby go wybrać!");
        return;
    }
    if (g_is_rendering.load()) {
        SetWindowTextW(g_hStatusLabel, L"Renderowanie jest w toku! Kliknij 'Przerwij', aby edytować pozycję.");
        return;
    }

    g_is_grabbing = true;
    g_is_scaling  = false;
    g_grab_initial_pos = g_scene_objects[g_selected_object_idx].center;
    g_axis_lock = AXIS_NONE;

    Camera cam = GetCurrentCamera();
    g_grab_plane_point = g_grab_initial_pos;
    g_grab_plane_normal = unit_vector(cam.origin - cam.lookat);

    POINT pt;
    GetCursorPos(&pt);
    ScreenToClient(g_hCanvas, &pt);
    if (start_mx >= 0 && start_my >= 0) pt = {start_mx, start_my};
    g_grab_start_mouse = pt;

    float u0 = (static_cast<float>(pt.x) + 0.5f) / static_cast<float>(g_canvas_view_width);
    float v0 = (static_cast<float>(g_canvas_view_height - 1 - pt.y) + 0.5f) / static_cast<float>(g_canvas_view_height);
    Ray r0 = cam.get_ray(u0, v0);

    float denom0 = dot(g_grab_plane_normal, r0.direction);
    if (std::fabs(denom0) > 1e-6f) {
        float t0 = dot(g_grab_plane_point - r0.origin, g_grab_plane_normal) / denom0;
        g_grab_plane_hit0 = r0.at(t0);
    } else {
        g_grab_plane_hit0 = g_grab_initial_pos;
    }

    SetCapture(g_hCanvas);
    SetFocus(g_hCanvas);

    wchar_t status[200];
    swprintf(status, 200, L"[BLENDER GRAB] %s | ruch myszą | [X/Y/Z]: blokada osi | [LPM/Enter]: zatwierdź | [PPM/Esc]: anuluj",
             g_scene_objects[g_selected_object_idx].name.c_str());
    SetWindowTextW(g_hStatusLabel, status);

    InvalidateRect(g_hCanvas, NULL, FALSE);
}

void ConfirmGrabMode() {
    if (!g_is_grabbing) return;
    g_is_grabbing = false;
    ReleaseCapture();
    g_axis_lock = AXIS_NONE;

    UpdateObjectFieldsFromSelection(g_selected_object_idx);

    wchar_t status[200];
    const auto& obj = g_scene_objects[g_selected_object_idx];
    swprintf(status, 200, L"Zatwierdzono pozycję: [%s] (X: %.2f, Y: %.2f, Z: %.2f)",
             obj.name.c_str(), obj.center.x, obj.center.y, obj.center.z);
    SetWindowTextW(g_hStatusLabel, status);

    RenderFastPreview();
    InvalidateRect(g_hCanvas, NULL, FALSE);
}

void CancelGrabMode() {
    if (!g_is_grabbing) return;
    g_scene_objects[g_selected_object_idx].center = g_grab_initial_pos;
    g_is_grabbing = false;
    ReleaseCapture();
    g_axis_lock = AXIS_NONE;

    UpdateObjectFieldsFromSelection(g_selected_object_idx);
    SetWindowTextW(g_hStatusLabel, L"Anulowano przesuwanie (przywrócono pozycję wyjściową).");

    RenderFastPreview();
    InvalidateRect(g_hCanvas, NULL, FALSE);
}

void ProcessGrabMove(int mx, int my) {
    if (!g_is_grabbing) return;

    Camera cam = GetCurrentCamera();
    float u = (static_cast<float>(mx) + 0.5f) / static_cast<float>(g_canvas_view_width);
    float v = (static_cast<float>(g_canvas_view_height - 1 - my) + 0.5f) / static_cast<float>(g_canvas_view_height);
    Ray r = cam.get_ray(u, v);

    float denom = dot(g_grab_plane_normal, r.direction);
    if (std::fabs(denom) > 1e-6f) {
        float t = dot(g_grab_plane_point - r.origin, g_grab_plane_normal) / denom;
        Vec3 cur_hit = r.at(t);
        Vec3 delta = cur_hit - g_grab_plane_hit0;

        if (g_axis_lock == AXIS_X) {
            delta.y = 0.0f;
            delta.z = 0.0f;
        } else if (g_axis_lock == AXIS_Y) {
            delta.x = 0.0f;
            delta.z = 0.0f;
        } else if (g_axis_lock == AXIS_Z) {
            delta.x = 0.0f;
            delta.y = 0.0f;
        }

        Vec3 new_pos = g_grab_initial_pos + delta;
        g_scene_objects[g_selected_object_idx].center = new_pos;

        UpdateObjectFieldsFromSelection(g_selected_object_idx);
        g_viewport_dirty.store(true);
        InvalidateRect(g_hCanvas, NULL, FALSE);
    }
}

void StartScaleMode() {
    if (g_selected_object_idx < 0 || g_selected_object_idx >= static_cast<int>(g_scene_objects.size())) return;
    if (g_is_rendering.load()) return;

    g_is_scaling  = true;
    g_is_grabbing = false;
    g_scale_initial_r = g_scene_objects[g_selected_object_idx].radius;

    POINT pt; GetCursorPos(&pt); ScreenToClient(g_hCanvas, &pt);
    g_grab_start_mouse = pt;

    SetCapture(g_hCanvas);
    SetFocus(g_hCanvas);

    wchar_t status[200];
    swprintf(status, 200, L"[BLENDER SCALE] %s | oddal/przybliż kursor | [LPM/Enter]: zatwierdź | [PPM/Esc]: anuluj",
             g_scene_objects[g_selected_object_idx].name.c_str());
    SetWindowTextW(g_hStatusLabel, status);

    InvalidateRect(g_hCanvas, NULL, FALSE);
}

void ConfirmScaleMode() {
    if (!g_is_scaling) return;
    g_is_scaling = false;
    ReleaseCapture();

    UpdateObjectFieldsFromSelection(g_selected_object_idx);
    wchar_t status[200];
    swprintf(status, 200, L"Zatwierdzono nowy promień/skalę: R = %.2f", g_scene_objects[g_selected_object_idx].radius);
    SetWindowTextW(g_hStatusLabel, status);

    RenderFastPreview();
    InvalidateRect(g_hCanvas, NULL, FALSE);
}

void CancelScaleMode() {
    if (!g_is_scaling) return;
    g_scene_objects[g_selected_object_idx].radius = g_scale_initial_r;
    g_is_scaling = false;
    ReleaseCapture();

    UpdateObjectFieldsFromSelection(g_selected_object_idx);
    SetWindowTextW(g_hStatusLabel, L"Anulowano skalowanie obiektu.");

    RenderFastPreview();
    InvalidateRect(g_hCanvas, NULL, FALSE);
}

void ProcessScaleMove(int mx, int my) {
    if (!g_is_scaling) return;

    Camera cam = GetCurrentCamera();
    float v_w = 2.0f * std::tan(g_cam_fov * 3.14159f / 360.0f) * (static_cast<float>(g_canvas_view_width) / g_canvas_view_height);
    float v_h = 2.0f * std::tan(g_cam_fov * 3.14159f / 360.0f);

    int sx = 0, sy = 0;
    if (ProjectToScreen(g_scene_objects[g_selected_object_idx].center, cam, v_w, v_h,
                        g_canvas_view_width, g_canvas_view_height, sx, sy)) {
        float d0 = std::hypot(static_cast<float>(g_grab_start_mouse.x - sx), static_cast<float>(g_grab_start_mouse.y - sy));
        float d1 = std::hypot(static_cast<float>(mx - sx), static_cast<float>(my - sy));
        d0 = std::max(20.0f, d0);
        float scale = d1 / d0;

        float new_r = std::clamp(g_scale_initial_r * scale, 0.10f, 6.0f);
        g_scene_objects[g_selected_object_idx].radius = new_r;

        UpdateObjectFieldsFromSelection(g_selected_object_idx);
        g_viewport_dirty.store(true);
        InvalidateRect(g_hCanvas, NULL, FALSE);
    }
}

void ResetCamera() {
    g_has_rendered_image.store(false);
    if (g_current_scene == 0) {
        g_cam_target   = Vec3(-0.3f, 0.5f, 0.0f);
        g_cam_distance = 4.6f;
        g_cam_yaw      = 0.05f;
        g_cam_pitch    = 0.45f;
        g_cam_fov      = 38.0f;
    } else {
        g_cam_target   = Vec3(0.0f, 0.85f, 0.6f);
        g_cam_distance = 1.9f;
        g_cam_yaw      = 3.14159f;
        g_cam_pitch    = -0.05f;
        g_cam_fov      = 52.0f;
    }
    g_viewport_pt_spp = 0;
    if (g_live_viewport_pt.load() && g_cuda_available) {
        RenderLiveViewportFrame(false);
    } else {
        RenderFastPreview();
    }
    InvalidateRect(g_hCanvas, NULL, FALSE);
    SetWindowTextW(g_hStatusLabel, L"Zresetowano widok kamery 3D do kąta początkowego sceny.");
}

void ApplyPositionFromFields() {
    int sel = static_cast<int>(SendMessage(g_hComboObjects, CB_GETCURSEL, 0, 0));
    if (sel < 0 || sel >= static_cast<int>(g_scene_objects.size())) return;

    wchar_t buf[64];
    GetWindowTextW(g_hEditPosX, buf, 64);
    float x = wcstof(buf, nullptr);

    GetWindowTextW(g_hEditPosY, buf, 64);
    float y = wcstof(buf, nullptr);

    GetWindowTextW(g_hEditPosZ, buf, 64);
    float z = wcstof(buf, nullptr);

    GetWindowTextW(g_hEditRadius, buf, 64);
    float r = wcstof(buf, nullptr);
    if (r <= 0.05f) r = 0.05f;

    g_scene_objects[sel].center = Vec3(x, y, z);
    g_scene_objects[sel].radius = r;
    g_selected_object_idx = sel;

    wchar_t msg[128];
    swprintf(msg, 128, L"Zaktualizowano pozycję: [%s] (%.2f, %.2f, %.2f), R = %.2f",
             g_scene_objects[sel].name.c_str(), x, y, z, r);
    SetWindowTextW(g_hStatusLabel, msg);

    g_viewport_pt_spp = 0;
    if (g_live_viewport_pt.load() && g_cuda_available && !g_is_rendering.load() && !g_has_rendered_image.load()) {
        RenderLiveViewportFrame(false);
    } else {
        RenderFastPreview();
    }
    InvalidateRect(g_hCanvas, NULL, FALSE);
}

void OnResolutionComboChanged() {
    int sel = static_cast<int>(SendMessage(g_hComboRes, CB_GETCURSEL, 0, 0));
    int w = 800, h = 450;
    switch (sel) {
        case 0: w = 640;  h = 360;  break;
        case 1: w = 800;  h = 450;  break;
        case 2: w = 960;  h = 540;  break;
        case 3: w = 1280; h = 720;  break;
        case 4: w = 1920; h = 1080; break;
        default: return;
    }

    wchar_t bufW[32], bufH[32];
    swprintf(bufW, 32, L"%d", w);
    swprintf(bufH, 32, L"%d", h);
    SetWindowTextW(g_hEditResW, bufW);
    SetWindowTextW(g_hEditResH, bufH);
}

// ============================================================================
// 6. STOCHASTYCZNE KAFELKOWANIE ORAZ RENDEROWANIE
// ============================================================================

void StartRendering() {
    if (g_is_rendering.load()) return;
    g_has_rendered_image.store(false);
    if (g_is_grabbing) ConfirmGrabMode();
    if (g_is_scaling)  ConfirmScaleMode();

    wchar_t bufW[32], bufH[32];
    GetWindowTextW(g_hEditResW, bufW, 32);
    GetWindowTextW(g_hEditResH, bufH, 32);
    int reqW = wcstol(bufW, nullptr, 10);
    int reqH = wcstol(bufH, nullptr, 10);
    if (reqW < 64)   reqW = 64;
    if (reqW > 3840) reqW = 3840;
    if (reqH < 64)   reqH = 64;
    if (reqH > 2160) reqH = 2160;
    g_render_width  = reqW;
    g_render_height = reqH;

    int tile_sel = static_cast<int>(SendMessage(g_hComboTile, CB_GETCURSEL, 0, 0));
    switch (tile_sel) {
        case 0:  g_tile_size = 16; break;
        case 1:  g_tile_size = 32; break;
        case 2:  g_tile_size = 64; break;
        default: g_tile_size = 32; break;
    }

    g_render_algo = static_cast<int>(SendMessage(g_hComboAlgo, CB_GETCURSEL, 0, 0));

    wchar_t spp_buf[32], depth_buf[32], phot_buf[32];
    GetWindowTextW(g_hEditSPP, spp_buf, 32);
    GetWindowTextW(g_hEditDepth, depth_buf, 32);
    GetWindowTextW(g_hEditPhotons, phot_buf, 32);

    int spp = wcstol(spp_buf, nullptr, 10);
    int max_depth = wcstol(depth_buf, nullptr, 10);
    int photons = wcstol(phot_buf, nullptr, 10);

    if (spp < 1) spp = 1;
    if (spp > 5000) spp = 5000;
    if (max_depth < 1) max_depth = 1;
    if (max_depth > 128) max_depth = 128;
    if (photons < 1000) photons = 1000;
    if (photons > 1000000) photons = 1000000;
    g_photon_count = photons;

    // Ustawienie dokładnych wymiarów bufora w pamięci dla renderera
    g_cur_buf_w.store(g_render_width);
    g_cur_buf_h.store(g_render_height);
    g_display_buffer.assign(g_render_width * g_render_height, 0xFF18181B);
    g_raw_image.assign(g_render_width * g_render_height, Vec3(0, 0, 0));
    g_sample_counts.assign(g_render_width * g_render_height, 0);

    std::vector<Tile> tiles;
    for (int y = 0; y < g_render_height; y += g_tile_size) {
        int y_end = std::min(y + g_tile_size, g_render_height);
        for (int x = 0; x < g_render_width; x += g_tile_size) {
            int x_end = std::min(x + g_tile_size, g_render_width);
            tiles.push_back({x, y, x_end, y_end});
        }
    }
    g_total_tiles = tiles.size();

    float center_x = g_render_width * 0.5f;
    float center_y = g_render_height * 0.5f;
    std::sort(tiles.begin(), tiles.end(), [center_x, center_y](const Tile& a, const Tile& b) {
        float ax = (a.x0 + a.x1) * 0.5f - center_x;
        float ay = (a.y0 + a.y1) * 0.5f - center_y;
        float bx = (b.x0 + b.x1) * 0.5f - center_x;
        float by = (b.y0 + b.y1) * 0.5f - center_y;
        return (ax * ax + ay * ay) < (bx * bx + by * by);
    });

    EnableWindow(g_hBtnRender, FALSE);
    EnableWindow(g_hBtnStop, TRUE);
    EnableWindow(g_hBtnApplyPos, FALSE);
    EnableWindow(g_hBtnGrab, FALSE);

    const wchar_t* algoName = L"Standardowy PT (CPU)";
    if (g_render_algo == 1) algoName = L"Standardowy PT (GPU / CUDA)";
    else if (g_render_algo == 2) algoName = L"BDPT + globalna mapa fotonów";
    else if (g_render_algo == 3) algoName = L"Metropolis Light Transport (MLT)";
    else if (g_render_algo == 4) algoName = L"PT + Manifold NEE (kaustyka przez rozmaitość)";

    wchar_t statusInit[160];
    if (g_render_algo == 1) {
        g_cuda_done_spp.store(0);
        g_cuda_total_spp.store(spp);
        swprintf(statusInit, 160, L"Rozpoczynanie [GPU CUDA]: %d SPP (%s)...",
                 spp, g_cuda_device_name.empty() ? L"NVIDIA GPU" : std::wstring(g_cuda_device_name.begin(), g_cuda_device_name.end()).c_str());
        SendMessage(g_hProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, spp));
        SendMessage(g_hProgressBar, PBM_SETPOS, 0, 0);
    } else if (g_render_algo == 3) {
        size_t total_mutations = static_cast<size_t>(g_render_width) * g_render_height * spp;
        const wchar_t* tag = (g_render_algo == 4) ? L"MEMLT" : L"MLT";
        swprintf(statusInit, 160, L"Rozpoczynanie [%s]: %zu mutacji (%d/px, %d fotonów)...",
                 tag, total_mutations, spp, g_photon_count);
        SendMessage(g_hProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, 1000));
        SendMessage(g_hProgressBar, PBM_SETPOS, 0, 0);
    } else {
        swprintf(statusInit, 160, L"Rozpoczynanie [%s]: %zu kafelków %dx%d px (%d fotonów)...",
                 algoName, g_total_tiles, g_tile_size, g_tile_size, g_photon_count);
        SendMessage(g_hProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, g_total_tiles));
        SendMessage(g_hProgressBar, PBM_SETPOS, 0, 0);
    }
    SetWindowTextW(g_hStatusLabel, statusInit);

    g_is_rendering.store(true);
    g_cancel_render.store(false);
    g_completed_tiles.store(0);
    g_render_start_time = std::chrono::high_resolution_clock::now();

    InvalidateRect(g_hCanvas, NULL, FALSE);

    int algo = g_render_algo;

    std::thread([tiles, spp, max_depth, algo, photons]() {
        size_t total_pixels = static_cast<size_t>(g_render_width) * g_render_height;
        HittableList world;
        std::vector<Sphere> light_sources;
        std::shared_ptr<ConvexPolyhedron> glass_crystal = nullptr;
        float crystal_radius = 0.65f;
        float crystal_dispersion = 0.02f;
        float crystal_ior = 1.55f;

        if (g_current_scene == 0) {
            world.add(std::make_shared<Sphere>(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f,
                                               std::make_shared<Lambertian>(Vec3(0.85f, 0.85f, 0.88f))));
        }

        for (const auto& obj : g_scene_objects) {
            world.add(CreateHittableFromObject(obj));
            if (obj.is_crystal) {
                float h = obj.secondary_param > 0.1f ? obj.secondary_param : 1.40f;
                crystal_radius = obj.radius;
                crystal_dispersion = obj.mat_dispersion;
                crystal_ior = (obj.mat_param > 0.1f) ? obj.mat_param : 1.55f;
                glass_crystal = CreateFacetedCrystal(obj.center, obj.radius, h, obj.material);
            }
            if (obj.material && !obj.material->emitted(obj.center).near_zero()) {
                light_sources.push_back(Sphere(obj.center, obj.radius, obj.material));
            }
        }

        // Dokładna kamera dla wybranej rozdzielczości renderowania
        Camera cam = GetCameraForRender(g_render_width, g_render_height);

        // Tryb 4: kaustyka liczona przez rozmaitość zwierciadlaną (MNEE) zamiast mapy fotonów
        ManifoldCausticContext manifold_ctx;
        const ManifoldCausticContext* manifold_ptr = nullptr;
        if (algo == 4 && glass_crystal && glass_crystal->mat && glass_crystal->mat->is_dielectric() && !light_sources.empty()) {
            auto diel = static_cast<const Dielectric*>(glass_crystal->mat.get());
            manifold_ctx.crystal = glass_crystal.get();
            manifold_ctx.crystal_material = glass_crystal->mat.get();
            manifold_ctx.base_ior = diel->ir;
            manifold_ctx.dispersion = diel->dispersion;
            manifold_ctx.light_size = g_caustic_light_size.load();
            for (const auto& light : light_sources) {
                manifold_ctx.light_sequences.push_back(BuildManifoldSequences(manifold_ctx, light));
            }
            int seq_total = 0;
            for (const auto& ls : manifold_ctx.light_sequences) seq_total += static_cast<int>(ls.size());
            g_manifold_sequences.store(seq_total);
            manifold_ptr = &manifold_ctx;
        }

        PhotonMap photon_map;
        if ((algo == 2 || algo == 3) && glass_crystal && !light_sources.empty()) {
            // Rozmiar komórki siatki dopasowany do promienia filtru kaustyki (domyślnie 0.020f przy promieniu 0.028f)
            photon_map.init(std::clamp(g_caustic_cpu_radius.load() * 0.714f, 0.010f, 0.060f), -6.0f, 6.0f, -6.0f, 6.0f);

            const auto& light = light_sources[0];
            Vec3 crystal_center = glass_crystal->center;
            float bound_r = crystal_radius * 1.25f;

            Vec3 light_to_cryst = crystal_center - light.center;
            float dist_lc = light_to_cryst.length();
            Vec3 dir_lc = light_to_cryst / (dist_lc > 1e-4f ? dist_lc : 1.0f);

            Vec3 up(0.0f, 1.0f, 0.0f);
            if (std::fabs(dir_lc.y) > 0.9f) up = Vec3(1.0f, 0.0f, 0.0f);
            Vec3 right = unit_vector(cross(dir_lc, up));
            Vec3 v_up = cross(right, dir_lc);

            float r_light = light.radius;
            Vec3 emit_power = light.mat->emitted(light.center);

            // Fizyczne obliczenie strumienia energii w stożku emisji:
            Vec3 photon_flux_base;
            if (g_current_scene == 0) {
                // Scena 1 (Otwarta): emisja bezpośrednio w dysk otaczający kryształ
                float solid_angle = (3.14159265f * bound_r * bound_r) / (dist_lc * dist_lc + 0.01f);
                float cone_fraction = solid_angle / (4.0f * 3.14159265f);
                Vec3 total_cone_flux = emit_power * (4.0f * 3.14159265f * 3.14159265f * r_light * r_light * cone_fraction);
                photon_flux_base = total_cone_flux / static_cast<float>(photons);
            } else {
                // Scena 2 (Zamknięty pokój): światło -> lustro zewnętrzne (Z=3.2) -> szczelina okienna (Z=1.4) -> kryształ
                float mirror_w = 1.1f;
                float mirror_h = 0.9f;
                float mirror_area = mirror_w * mirror_h;
                Vec3 mirror_target(1.35f, 1.65f, 3.2f);
                float dist_lm = (mirror_target - light.center).length();
                Vec3 dir_lm = unit_vector(mirror_target - light.center);
                float cos_mir = std::fabs(dir_lm.z);
                float solid_angle_mir = (mirror_area * cos_mir) / (dist_lm * dist_lm + 0.01f);
                float cone_fraction = solid_angle_mir / (4.0f * 3.14159265f);
                Vec3 total_cone_flux = emit_power * (4.0f * 3.14159265f * 3.14159265f * r_light * r_light * cone_fraction);
                photon_flux_base = total_cone_flux / static_cast<float>(photons);
            }

            unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
            int photons_per_thread = photons / num_threads;

            std::vector<std::vector<Photon>> thread_photons(num_threads);
            std::vector<std::thread> photon_workers;

            for (unsigned int th = 0; th < num_threads; ++th) {
                photon_workers.emplace_back([&, th]() {
                    std::mt19937 rng(54321 + th * 2039);
                    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
                    std::uniform_real_distribution<float> d_theta(0.0f, 6.2831853f);

                    auto& local_photons = thread_photons[th];
                    local_photons.reserve(photons_per_thread / 2);

                    for (int i = 0; i < photons_per_thread && !g_cancel_render.load(); ++i) {
                        // Płynna dyspersja spektralna Cauchy'ego (bez artefaktów pojedynczych kanałów RGB):
                        float ior = crystal_ior;
                        Vec3 ch_tint(1.0f, 1.0f, 1.0f);

                        if (crystal_dispersion > 1e-4f) {
                            float lambda = dist01(rng);
                            ior = crystal_ior + (2.0f * lambda - 1.0f) * crystal_dispersion;
                            float wr = std::exp(-((lambda - 0.15f) * (lambda - 0.15f)) / 0.14f);
                            float wg = std::exp(-((lambda - 0.50f) * (lambda - 0.50f)) / 0.14f);
                            float wb = std::exp(-((lambda - 0.85f) * (lambda - 0.85f)) / 0.14f);
                            Vec3 spec(wr, wg, wb);
                            float avg = (spec.x + spec.y + spec.z) / 3.0f;
                            if (avg > 1e-4f) spec = spec / avg;
                            float blend = std::clamp(crystal_dispersion * 12.0f, 0.0f, 1.0f);
                            ch_tint = (1.0f - blend) * Vec3(1.0f, 1.0f, 1.0f) + blend * spec;
                        }

                        Vec3 cur_flux = photon_flux_base * ch_tint;

                        Ray cur_ray;

                        if (g_current_scene == 0) {
                            Vec3 p_light = light.center + light.radius * random_unit_vector(rng);
                            float r_disk = bound_r * std::sqrt(dist01(rng));
                            float th_angle = d_theta(rng);
                            Vec3 target = crystal_center + right * (r_disk * std::cos(th_angle)) + v_up * (r_disk * std::sin(th_angle));
                            Vec3 dir = unit_vector(target - p_light);
                            cur_ray = Ray(p_light, dir);
                        } else {
                            // Emisja na zewnętrzne lustro przy Z=3.2
                            Vec3 p_light = light.center + light.radius * random_unit_vector(rng);
                            float mx = 0.8f + dist01(rng) * 1.1f;
                            float my = 1.2f + dist01(rng) * 0.9f;
                            Vec3 p_mir(mx, my, 3.2f);
                            Vec3 dir_to_mir = unit_vector(p_mir - p_light);

                            // Odbicie lustrzane od płaszczyzny Z=3.2 z normalną (0, 0, -1)
                            Vec3 dir_refl = Vec3(dir_to_mir.x, dir_to_mir.y, -std::fabs(dir_to_mir.z));
                            cur_ray = Ray(p_mir + dir_refl * 0.002f, dir_refl);

                            // Propagacja promienia z lustra w stronę szczeliny okna przy Z=1.4
                            HitRecord hit_room;
                            if (!world.hit(cur_ray, 0.001f, 1e9f, hit_room)) continue;

                            // Jeśli promień trafił prosto w kryształ szklany:
                            if (hit_room.mat && hit_room.mat->is_dielectric()) {
                                cur_ray = Ray(hit_room.p - cur_ray.direction * 0.005f, cur_ray.direction);
                            } else {
                                // Trafienie w podłogę pokoju lub ramę okna bezpośrednim światłem
                                if (!hit_room.mat->is_specular() && hit_room.mat->emitted(hit_room.p).near_zero()) {
                                    local_photons.push_back({hit_room.p, cur_flux * 0.20f});
                                }
                                continue;
                            }
                        }

                        // Załamanie światła wewnątrz kryształu z dyspersją chromatyczną
                        for (int bounce = 0; bounce < 8; ++bounce) {
                            HitRecord hit;
                            if (!glass_crystal->hit(cur_ray, 0.001f, 1e9f, hit)) break;

                            float ref_idx = hit.front_face ? (1.0f / ior) : ior;
                            Vec3 unit_dir = unit_vector(cur_ray.direction);
                            float cos_theta = std::fmin(dot(-unit_dir, hit.normal), 1.0f);
                            float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));

                            bool cannot_refract = ref_idx * sin_theta > 1.0f;
                            float refl_prob = schlick_reflectance(cos_theta, ref_idx);

                            Vec3 next_dir;
                            if (cannot_refract || dist01(rng) < refl_prob) {
                                next_dir = reflect(unit_dir, hit.normal);
                            } else {
                                next_dir = refract(unit_dir, hit.normal, ref_idx);
                            }

                            cur_ray = Ray(hit.p + next_dir * 0.0015f, unit_vector(next_dir));
                        }

                        // Promień opuścił kryształ - odłożenie fotonu kaustyki na podłożu
                        HitRecord floor_hit;
                        if (world.hit(cur_ray, 0.001f, 1e9f, floor_hit)) {
                            if (!floor_hit.mat->is_specular() && floor_hit.mat->emitted(floor_hit.p).near_zero()) {
                                local_photons.push_back({floor_hit.p, cur_flux});
                            }
                        }
                    }
                });
            }

            for (auto& pw : photon_workers) {
                if (pw.joinable()) pw.join();
            }

            for (const auto& lph : thread_photons) {
                for (const auto& ph : lph) {
                    photon_map.add_photon(ph.pos, ph.flux);
                }
            }
        }

        if (algo == 1) {
            // ================================================================
            // RENDERING GPU: NVIDIA CUDA PATH TRACER (RTX / CUDA Cores)
            // Pełna obsługa geometrii: Kule, Czworokąty, Kryształy 20-fasetowe, Meshe OBJ
            // ================================================================
            CudaSceneData gpu_scene = BuildCudaScene(g_global_light_mult.load());

            Camera cur_cam = GetCameraForRender(g_render_width, g_render_height);
            CudaVec3 c_orig(cur_cam.origin.x, cur_cam.origin.y, cur_cam.origin.z);
            CudaVec3 c_look(cur_cam.lookat.x, cur_cam.lookat.y, cur_cam.lookat.z);
            CudaVec3 c_vup(cur_cam.vup.x, cur_cam.vup.y, cur_cam.vup.z);
            float c_fov = g_cam_fov;

            CudaRenderOptions render_opts;
            render_opts.exposure_ev = g_exposure_ev.load();
            render_opts.tonemap_mode = g_tonemap_mode.load();
            render_opts.clamp_indirect = 12.0f; // Cycles firefly suppression
            render_opts.enable_caustics = g_cuda_caustics.load() ? 1 : 0;
            render_opts.caustic_hq = g_cuda_caustics_hq.load() ? 1 : 0;
            render_opts.caustic_grid_res = g_caustic_grid_res.load();
            render_opts.caustic_blur_cm = g_caustic_blur_cm.load();
            render_opts.caustic_light_size = g_caustic_light_size.load();
            render_opts.caustic_gain = g_caustic_gain.load();
            render_opts.caustic_photons = (g_photon_count > 50000) ? g_photon_count : (render_opts.caustic_hq ? 1048576 : 524288);
            render_opts.scene_mode = g_current_scene;

            std::vector<CudaVec3> cuda_film;
            bool success = CudaPT_Render(
                g_render_width, g_render_height, spp, max_depth,
                c_orig, c_look, c_vup, c_fov,
                gpu_scene, cuda_film,
                render_opts,
                &g_cancel_render,
                [&](int done, int total, const CudaVec3* /*accum_buf*/, const uint32_t* display_bgra) {
                    g_cuda_done_spp.store(done);
                    g_cuda_total_spp.store(total);

                    if (display_bgra && done > 0) {
                        // Błyskawiczny transfer zmapowanego bufora BGRA z GPU (0.5 ms zamiast 65 ms CPU!)
                        std::memcpy(g_display_buffer.data(), display_bgra, total_pixels * sizeof(uint32_t));
                        g_has_rendered_image.store(true);
                    }
                }
            );

            if (success && !cuda_film.empty()) {
                for (size_t i = 0; i < total_pixels; ++i) {
                    g_raw_image[i] = Vec3(cuda_film[i].x, cuda_film[i].y, cuda_film[i].z);
                    g_sample_counts[i] = 1;
                }
            }
        }
        else if (algo == 3) {
            // ================================================================
            // MCMC: MLT (Cline & Egbert 2005) / MEMLT (Jakob & Marschner 2012)
            // ================================================================
            {
                std::lock_guard<std::mutex> lock(g_mlt_mutex);
                g_mlt_accum.assign(total_pixels, Vec3(0, 0, 0));
            }
            size_t total_mutations = total_pixels * spp;
            g_mlt_total_mutations.store(total_mutations);
            g_mlt_mutations_done.store(0);

            unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
            size_t bootstrap_count = 25000;
            size_t boot_per_thread = bootstrap_count / num_threads;

            struct SeedPath {
                std::vector<float> u;
                Vec3 col;
                float lum = 0.0f;
            };

            std::vector<std::vector<SeedPath>> thread_seeds(num_threads);
            std::vector<double> thread_lum_sum(num_threads, 0.0);
            std::vector<std::thread> boot_workers;

            // 1. Faza Bootstrap (Sekcja 3.5 z pracy): oszacowanie fave oraz znalezienie jasnych ścieżek
            for (unsigned int th = 0; th < num_threads; ++th) {
                boot_workers.emplace_back([&, th]() {
                    std::mt19937 rng(4321 + th * 7823);
                    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
                    double local_lum = 0.0;
                    auto& s_list = thread_seeds[th];

                    for (size_t b = 0; b < boot_per_thread && !g_cancel_render.load(); ++b) {
                        std::vector<float> u(MLT_DIM);
                        for (size_t k = 0; k < MLT_DIM; ++k) u[k] = dist01(rng);

                        PrimarySampler sampler{u, 2, &rng};
                        Ray r = cam.get_ray(u[0], 1.0f - u[1]);

                        Vec3 col = evaluate_path_mlt(r, world, light_sources, photon_map, max_depth, sampler);
                        float lum = color_luminance(col);
                        local_lum += lum;

                        if (lum > 1e-4f) {
                            s_list.push_back({u, col, lum});
                        }
                    }
                    thread_lum_sum[th] = local_lum;
                });
            }

            for (auto& bw : boot_workers) {
                if (bw.joinable()) bw.join();
            }

            if (g_cancel_render.load()) {
                g_is_rendering.store(false);
                return;
            }

            double total_lum = 0.0;
            std::vector<SeedPath> all_seeds;
            for (unsigned int th = 0; th < num_threads; ++th) {
                total_lum += thread_lum_sum[th];
                for (const auto& sp : thread_seeds[th]) {
                    all_seeds.push_back(sp);
                }
            }

            float f_ave = static_cast<float>(total_lum / static_cast<double>(bootstrap_count));
            if (f_ave < 1e-6f) f_ave = 1.0f;
            g_mlt_fave = f_ave;

            // 2. Równoległe łańcuchy Markowa MCMC (Jeden łańcuch na wątek CPU)
            size_t mutations_per_thread = total_mutations / num_threads;
            std::vector<std::thread> mlt_workers;

            for (unsigned int th = 0; th < num_threads; ++th) {
                mlt_workers.emplace_back([&, th]() {
                    std::mt19937 rng(99991 + th * 6271);
                    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

                    SeedPath cur_x;
                    if (!all_seeds.empty()) {
                        cur_x = all_seeds[(th * 17) % all_seeds.size()];
                    } else {
                        cur_x.u.resize(MLT_DIM);
                        for (size_t k = 0; k < MLT_DIM; ++k) cur_x.u[k] = dist01(rng);
                        PrimarySampler sampler{cur_x.u, 2, &rng};
                        Ray r = cam.get_ray(cur_x.u[0], 1.0f - cur_x.u[1]);
                        cur_x.col = evaluate_path_mlt(r, world, light_sources, photon_map, max_depth, sampler);
                        cur_x.lum = color_luminance(cur_x.col);
                    }

                    size_t chunk_size = 20000;
                    size_t steps_in_chunk = 0;

                    std::vector<Vec3> local_film(total_pixels, Vec3(0, 0, 0));
                    std::vector<uint32_t> touched_pixels;
                    touched_pixels.reserve(chunk_size * 2);
                    std::vector<uint8_t> pixel_touched(total_pixels, 0);
                    std::vector<float> u_prop(MLT_DIM);

                    int width = g_render_width;
                    int height = g_render_height;
                    float r1 = 0.1f;
                    float r2 = 0.08f * static_cast<float>(std::min(width, height));
                    float log_r2_r1 = std::log(r2 / r1);
                    float theta1 = 0.0001f;
                    float theta2 = 0.1f;
                    float log_theta = std::log(theta2 / theta1);

                    for (size_t m = 0; m < mutations_per_thread && !g_cancel_render.load(); ++m) {
                        bool is_large = (dist01(rng) < 0.20f);

                        if (is_large) {
                            // Nowa ścieżka (Sekcja 3.2 z pracy) - gwarantuje ergodyczność
                            for (size_t k = 0; k < MLT_DIM; ++k) u_prop[k] = dist01(rng);
                        } else {
                            {
                                u_prop = cur_x.u;
                                // Perturbacja soczewki kamery (Appendix B)
                                float px_f = cur_x.u[0] * static_cast<float>(width);
                                float py_f = cur_x.u[1] * static_cast<float>(height);
                                float phi = dist01(rng) * 6.2831853f;
                                float r = r2 * std::exp(-log_r2_r1 * dist01(rng));
                                float nx = px_f + r * std::cos(phi);
                                float ny = py_f + r * std::sin(phi);
                                nx = std::fmod(std::fmod(nx, static_cast<float>(width)) + static_cast<float>(width), static_cast<float>(width));
                                ny = std::fmod(std::fmod(ny, static_cast<float>(height)) + static_cast<float>(height), static_cast<float>(height));
                                u_prop[0] = nx / static_cast<float>(width);
                                u_prop[1] = ny / static_cast<float>(height);

                                // Perturbacja kątowa odbić promieni (Appendix C)
                                for (size_t k = 2; k < MLT_DIM; ++k) {
                                    float r_ang = theta2 * std::exp(-log_theta * dist01(rng));
                                    float sign = (dist01(rng) < 0.5f) ? -1.0f : 1.0f;
                                    float val = cur_x.u[k] + sign * r_ang * 0.25f;
                                    if (val < 0.0f) val += 1.0f;
                                    if (val >= 1.0f) val -= 1.0f;
                                    u_prop[k] = val;
                                }
                            }
                        }

                        PrimarySampler prop_sampler{u_prop, 2, &rng};
                        Ray prop_ray = cam.get_ray(u_prop[0], 1.0f - u_prop[1]);
                        Vec3 col_prop = evaluate_path_mlt(prop_ray, world, light_sources, photon_map, max_depth, prop_sampler);
                        float lum_prop = color_luminance(col_prop);

                        float A = (cur_x.lum <= 1e-7f) ? 1.0f : std::min(1.0f, lum_prop / cur_x.lum);

                        // Akumulacja wartości oczekiwanej do lokalnego filmu (Sekcja 5.1 & Rysunek 4)
                        if (lum_prop > 1e-7f && A > 1e-7f) {
                            int px_y = std::clamp(static_cast<int>(u_prop[0] * width), 0, width - 1);
                            int py_y = std::clamp(static_cast<int>(u_prop[1] * height), 0, height - 1);
                            int idx_y = py_y * width + px_y;
                            local_film[idx_y] = local_film[idx_y] + (col_prop / lum_prop) * A;
                            if (!pixel_touched[idx_y]) {
                                pixel_touched[idx_y] = 1;
                                touched_pixels.push_back(static_cast<uint32_t>(idx_y));
                            }
                        }

                        if (cur_x.lum > 1e-7f && (1.0f - A) > 1e-7f) {
                            int px_x = std::clamp(static_cast<int>(cur_x.u[0] * width), 0, width - 1);
                            int py_x = std::clamp(static_cast<int>(cur_x.u[1] * height), 0, height - 1);
                            int idx_x = py_x * width + px_x;
                            local_film[idx_x] = local_film[idx_x] + (cur_x.col / cur_x.lum) * (1.0f - A);
                            if (!pixel_touched[idx_x]) {
                                pixel_touched[idx_x] = 1;
                                touched_pixels.push_back(static_cast<uint32_t>(idx_x));
                            }
                        }

                        // Warunek akceptacji wg Detailed Balance (Równanie 2)
                        if (dist01(rng) < A) {
                            cur_x.u = u_prop;
                            cur_x.col = col_prop;
                            cur_x.lum = lum_prop;
                        }

                        steps_in_chunk++;
                        if (steps_in_chunk >= chunk_size) {
                            {
                                std::lock_guard<std::mutex> lock(g_mlt_mutex);
                                for (uint32_t p : touched_pixels) {
                                    g_mlt_accum[p] = g_mlt_accum[p] + local_film[p];
                                    local_film[p] = Vec3(0, 0, 0);
                                    pixel_touched[p] = 0;
                                }
                                g_mlt_mutations_done.fetch_add(steps_in_chunk);
                            }
                            touched_pixels.clear();
                            steps_in_chunk = 0;
                        }
                    }

                    if (steps_in_chunk > 0) {
                        std::lock_guard<std::mutex> lock(g_mlt_mutex);
                        for (uint32_t p : touched_pixels) {
                            g_mlt_accum[p] = g_mlt_accum[p] + local_film[p];
                            local_film[p] = Vec3(0, 0, 0);
                            pixel_touched[p] = 0;
                        }
                        g_mlt_mutations_done.fetch_add(steps_in_chunk);
                        touched_pixels.clear();
                    }
                });
            }

            for (auto& mw : mlt_workers) {
                if (mw.joinable()) mw.join();
            }

            // Ostateczne skalowanie histogramu metodą Metropolis (Równanie 1: s = fave / have)
            size_t m_done = g_mlt_mutations_done.load();
            if (m_done > 0) {
                float scale = (g_mlt_fave * total_pixels) / static_cast<float>(m_done);
                std::lock_guard<std::mutex> lock(g_mlt_mutex);
                for (size_t i = 0; i < total_pixels; ++i) {
                    Vec3 rad = g_mlt_accum[i] * scale;
                    g_raw_image[i] = rad;
                    g_sample_counts[i] = 1;
                }
            }
        } else {
            // ================================================================
            // KAFELKOWY STANDARDOWY PT LUB BDPT
            // ================================================================
            unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
            std::atomic<size_t> tile_index(0);
            std::vector<std::thread> workers;

            for (unsigned int t = 0; t < num_threads; ++t) {
                workers.emplace_back([&, t]() {
                    std::mt19937 rng(1337 + t * 997);
                    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

                    while (!g_cancel_render.load()) {
                        size_t idx = tile_index.fetch_add(1);
                        if (idx >= tiles.size()) break;

                        const Tile& tile = tiles[idx];
                        int tw = tile.x1 - tile.x0;
                        int th = tile.y1 - tile.y0;

                        std::vector<std::pair<int, int>> offsets;
                        offsets.reserve(tw * th);
                        for (int dy = 0; dy < th; ++dy) {
                            for (int dx = 0; dx < tw; ++dx) {
                                offsets.emplace_back(dx, dy);
                            }
                        }

                        for (int s = 0; s < spp && !g_cancel_render.load(); ++s) {
                            std::shuffle(offsets.begin(), offsets.end(), rng);

                            for (const auto& off : offsets) {
                                if (g_cancel_render.load()) break;

                                int px = tile.x0 + off.first;
                                int py = tile.y0 + off.second;

                                // Precyzyjny TOP-DOWN raytracing zgodny z buforem
                                float u = (static_cast<float>(px) + dist01(rng)) / static_cast<float>(g_render_width);
                                float v = (static_cast<float>(g_render_height - 1 - py) + dist01(rng)) / static_cast<float>(g_render_height);

                                Ray r = cam.get_ray(u, v);
                                Vec3 col;

                                if (algo == 2) {
                                    col = ray_color_bdpt(r, world, light_sources, photon_map, max_depth, rng);
                                } else if (algo == 4) {
                                    col = ray_color_manifold(r, world, light_sources, manifold_ptr, max_depth, rng);
                                } else {
                                    col = ray_color_standard(r, world, max_depth, rng);
                                }

                                int p_idx = py * g_render_width + px;
                                g_raw_image[p_idx] = g_raw_image[p_idx] + col;
                                g_sample_counts[p_idx] += 1;

                                int count = g_sample_counts[p_idx];
                                Vec3 avg = g_raw_image[p_idx] / static_cast<float>(count);

                                float ev = g_exposure_ev.load();
                                int tm_mode = g_tonemap_mode.load();
                                Vec3 srgb = ApplyExposureAndToneMap(avg, ev, tm_mode);

                                uint8_t ir = static_cast<uint8_t>(std::clamp(srgb.x * 255.0f, 0.0f, 255.0f));
                                uint8_t ig = static_cast<uint8_t>(std::clamp(srgb.y * 255.0f, 0.0f, 255.0f));
                                uint8_t ib = static_cast<uint8_t>(std::clamp(srgb.z * 255.0f, 0.0f, 255.0f));

                                g_display_buffer[p_idx] = (ib) | (ig << 8) | (ir << 16) | (0xFF << 24);
                            }
                        }

                        g_completed_tiles.fetch_add(1);
                    }
                });
            }

            for (auto& worker : workers) {
                if (worker.joinable()) worker.join();
            }

            // Normalizacja sumy próbek do uśrednionej radiancji
            for (size_t i = 0; i < total_pixels; ++i) {
                if (g_sample_counts[i] > 0) {
                    g_raw_image[i] = g_raw_image[i] / static_cast<float>(g_sample_counts[i]);
                    g_sample_counts[i] = 1;
                }
            }
        }

        // ================================================================
        // POST-PROCESSING: INTEL OPEN IMAGE DENOISE (OIDN) & DISPLAY BUFFER
        // ================================================================
        if (!g_cancel_render.load()) {
            g_unfiltered_raw_image = g_raw_image;
            g_is_currently_denoised = false;

            if (g_enable_oidn.load()) {
                SetWindowTextW(g_hStatusLabel, L"Odszumianie obrazu przez sztuczną inteligencję (Intel OIDN)...");
                std::vector<Vec3> denoised = g_raw_image;
                if (RunIntelOIDN(denoised, g_render_width, g_render_height)) {
                    g_raw_image = denoised;
                    g_is_currently_denoised = true;
                }
            }

            RefreshDisplayBufferFromRaw();
            g_has_rendered_image.store(true);
        }

        g_is_rendering.store(false);
        InvalidateRect(g_hCanvas, NULL, FALSE);
    }).detach();
}

void StopRendering() {
    if (g_is_rendering.load()) {
        g_cancel_render.store(true);
        if (g_render_algo == 1) {
            SetWindowTextW(g_hStatusLabel, L"Zatrzymywanie kernela CUDA...");
        } else if (g_render_algo == 4) {
            SetWindowTextW(g_hStatusLabel, L"Zatrzymywanie renderowania Manifold NEE...");
        } else if (g_render_algo == 3) {
            SetWindowTextW(g_hStatusLabel, L"Zatrzymywanie łańcuchów Markowa (MLT)...");
        } else {
            SetWindowTextW(g_hStatusLabel, L"Zatrzymywanie wątków renderowania...");
        }
    }
}

void SaveToPNG() {
    wchar_t filenameW[MAX_PATH] = L"render_kaustyka.png";
    if (g_hEditFilename) {
        wchar_t buf[MAX_PATH];
        GetWindowTextW(g_hEditFilename, buf, MAX_PATH);
        std::wstring ws(buf);
        while (!ws.empty() && iswspace(ws.front())) ws.erase(ws.begin());
        while (!ws.empty() && iswspace(ws.back())) ws.pop_back();
        if (!ws.empty()) {
            if (ws.length() < 4 || (ws.substr(ws.length() - 4) != L".png" && ws.substr(ws.length() - 4) != L".PNG")) {
                ws += L".png";
            }
            wcsncpy(filenameW, ws.c_str(), MAX_PATH - 1);
            filenameW[MAX_PATH - 1] = L'\0';
        }
    }

    char filenameA[MAX_PATH];
    WideCharToMultiByte(CP_UTF8, 0, filenameW, -1, filenameA, MAX_PATH, nullptr, nullptr);

    std::vector<uint8_t> pixels(g_render_width * g_render_height * 3);

    float ev = g_exposure_ev.load();
    int tm_mode = g_tonemap_mode.load();

    for (int j = 0; j < g_render_height; ++j) {
        for (int i = 0; i < g_render_width; ++i) {
            int p_idx = j * g_render_width + i;
            int count = std::max(1, g_sample_counts[p_idx]);
            Vec3 avg = g_raw_image[p_idx] / static_cast<float>(count);

            Vec3 srgb = ApplyExposureAndToneMap(avg, ev, tm_mode);

            int dst_idx = (j * g_render_width + i) * 3;
            pixels[dst_idx + 0] = static_cast<uint8_t>(std::clamp(srgb.x * 255.0f, 0.0f, 255.0f));
            pixels[dst_idx + 1] = static_cast<uint8_t>(std::clamp(srgb.y * 255.0f, 0.0f, 255.0f));
            pixels[dst_idx + 2] = static_cast<uint8_t>(std::clamp(srgb.z * 255.0f, 0.0f, 255.0f));
        }
    }

    if (stbi_write_png(filenameA, g_render_width, g_render_height, 3, pixels.data(), g_render_width * 3) != 0) {
        // Potwierdzenie bez okna dialogowego: komunikat obok przycisku zapisu oraz w pasku statusu
        wchar_t msg[MAX_PATH + 80];
        swprintf(msg, MAX_PATH + 80, L"Zapisano %dx%d: %s", g_render_width, g_render_height, filenameW);
        if (g_hLblSaveStatus) SetWindowTextW(g_hLblSaveStatus, msg);
        SetWindowTextW(g_hStatusLabel, msg);
    } else {
        wchar_t msg[MAX_PATH + 80];
        swprintf(msg, MAX_PATH + 80, L"Błąd zapisu: %s", filenameW);
        if (g_hLblSaveStatus) SetWindowTextW(g_hLblSaveStatus, msg);
        SetWindowTextW(g_hStatusLabel, msg);
    }
}

// ============================================================================
// 7. PROCEDURA OBSŁUGI OKNA KANWY (PODGLĄD 3D & BLENDER GRAB/SCALE/ORBIT)
// ============================================================================

LRESULT CALLBACK CanvasWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        // Schemat myszy: LPM = obrót kamery, ŚPM = wybór i przesuwanie obiektu, PPM = przesuwanie widoku (pan)
        case WM_LBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_RBUTTONDOWN: {
            int mx = static_cast<short>(LOWORD(lParam));
            int my = static_cast<short>(HIWORD(lParam));
            SetFocus(hWnd);

            // Tryby klawiszowe G / S: LPM zatwierdza, PPM anuluje (jak w Blenderze)
            if (g_is_grabbing || g_is_scaling) {
                if (msg == WM_LBUTTONDOWN) {
                    if (g_is_grabbing) ConfirmGrabMode(); else ConfirmScaleMode();
                } else if (msg == WM_RBUTTONDOWN) {
                    if (g_is_grabbing) CancelGrabMode(); else CancelScaleMode();
                }
                if (msg != WM_MBUTTONDOWN) g_grab_by_drag = false;
                return 0;
            }
            if (g_is_orbiting || g_is_panning) return 0;

            if (g_has_rendered_image.load()) {
                g_has_rendered_image.store(false);
                g_viewport_pt_spp = 0;
                if (g_live_viewport_pt.load() && g_cuda_available) {
                    RenderLiveViewportFrame(false);
                } else {
                    RenderFastPreview();
                }
                InvalidateRect(hWnd, NULL, FALSE);
            }

            if (msg == WM_LBUTTONDOWN) {
                g_is_orbiting = true;
                g_nav_last_mouse = {mx, my};
                SetCapture(hWnd);
            } else if (msg == WM_RBUTTONDOWN) {
                g_is_panning = true;
                g_pan_depth = RayPickDepth(mx, my);
                g_nav_last_mouse = {mx, my};
                SetCapture(hWnd);
            } else {
                // ŚPM: wybór obiektu pod kursorem i natychmiastowe złapanie go do przesuwania (puść przycisk, aby zatwierdzić)
                int hit_idx = RayPickObject(mx, my);
                if (hit_idx >= 0) {
                    g_selected_object_idx = hit_idx;
                    SendMessage(g_hComboObjects, CB_SETCURSEL, hit_idx, 0);
                    UpdateObjectFieldsFromSelection(hit_idx);

                    StartGrabMode(mx, my);
                    g_grab_by_drag = g_is_grabbing;
                    if (!g_is_grabbing) {
                        RenderFastPreview();
                        InvalidateRect(hWnd, NULL, FALSE);
                    }
                }
            }
            return 0;
        }

        case WM_LBUTTONUP:
        case WM_RBUTTONUP: {
            bool was_nav = (msg == WM_LBUTTONUP) ? g_is_orbiting : g_is_panning;
            if (was_nav) {
                g_is_orbiting = false;
                g_is_panning  = false;
                ReleaseCapture();
                g_viewport_pt_spp = 0;
                if (g_live_viewport_pt.load() && g_cuda_available) {
                    RenderLiveViewportFrame(false);
                } else {
                    RenderFastPreview();
                }
                InvalidateRect(hWnd, NULL, FALSE);
            }
            return 0;
        }

        case WM_MBUTTONUP: {
            if (g_grab_by_drag) {
                g_grab_by_drag = false;
                if (g_is_grabbing) ConfirmGrabMode();
            }
            return 0;
        }

        case WM_MOUSEWHEEL: {
            if (g_has_rendered_image.load()) {
                g_has_rendered_image.store(false);
            }
            short zDelta = GET_WHEEL_DELTA_WPARAM(wParam);
            float factor = (zDelta > 0) ? 0.90f : 1.10f;
            g_cam_distance = std::clamp(g_cam_distance * factor, 0.8f, 35.0f);
            g_viewport_pt_spp = 0;
            if (g_live_viewport_pt.load() && g_cuda_available) {
                RenderLiveViewportFrame(false);
            } else {
                RenderFastPreview();
            }
            InvalidateRect(hWnd, NULL, FALSE);
            return 0;
        }

        case WM_MOUSEMOVE: {
            // Rzutowanie na short: przy przechwyconej myszy współrzędne poza kanwą są ujemne
            int mx = static_cast<short>(LOWORD(lParam));
            int my = static_cast<short>(HIWORD(lParam));

            if (g_is_orbiting) {
                int dx = mx - g_nav_last_mouse.x;
                int dy = my - g_nav_last_mouse.y;
                // Orbita typu turntable (Blender / CAD): scena podąża za kursorem.
                // Ruch myszy w prawo obraca scenę w prawo (kamera okrąża cel w lewo), ruch w dół pokazuje scenę bardziej z góry.
                g_cam_yaw   -= dx * 0.007f;
                g_cam_pitch = std::clamp(g_cam_pitch + dy * 0.007f, -1.45f, 1.45f);
                g_nav_last_mouse = {mx, my};
                g_viewport_dirty.store(true);
                g_viewport_pt_spp = 0;
                InvalidateRect(hWnd, NULL, FALSE);
                return 0;
            }
            else if (g_is_panning) {
                int dx = mx - g_nav_last_mouse.x;
                int dy = my - g_nav_last_mouse.y;
                Camera cam = GetCurrentCamera();
                Vec3 w = unit_vector(cam.origin - cam.lookat);
                Vec3 u = unit_vector(cross(cam.vup, w));
                Vec3 v = cross(w, u);
                // Transformacja frustum: rozmiar piksela w przestrzeni świata na głębokości złapanego punktu.
                // Wysokość frustum na głębokości d wynosi 2*d*tan(fov/2), więc punkt pod kursorem podąża za nim 1:1.
                float world_per_px = 2.0f * g_pan_depth * std::tan(g_cam_fov * 3.14159265f / 360.0f)
                                     / static_cast<float>(std::max(1, g_canvas_view_height));
                g_cam_target = g_cam_target - u * (dx * world_per_px) + v * (dy * world_per_px);
                g_nav_last_mouse = {mx, my};
                g_viewport_dirty.store(true);
                g_viewport_pt_spp = 0;
                InvalidateRect(hWnd, NULL, FALSE);
                return 0;
            }
            else if (g_is_grabbing) {
                ProcessGrabMove(mx, my);
            }
            else if (g_is_scaling) {
                ProcessScaleMove(mx, my);
            }
            else {
                int hit_idx = RayPickObject(mx, my);
                if (hit_idx >= 0) {
                    SetCursor(LoadCursor(NULL, IDC_HAND));
                } else {
                    SetCursor(LoadCursor(NULL, IDC_ARROW));
                }
            }
            return 0;
        }

        case WM_SETCURSOR: {
            if (g_is_grabbing) {
                SetCursor(LoadCursor(NULL, IDC_SIZEALL));
                return TRUE;
            } else if (g_is_scaling) {
                SetCursor(LoadCursor(NULL, IDC_SIZENWSE));
                return TRUE;
            } else if (g_is_orbiting || g_is_panning) {
                SetCursor(LoadCursor(NULL, IDC_SIZEALL));
                return TRUE;
            }
            break;
        }

        case WM_KEYDOWN: {
            if (g_has_rendered_image.load()) {
                g_has_rendered_image.store(false);
                g_viewport_pt_spp = 0;
                if (g_live_viewport_pt.load() && g_cuda_available) {
                    RenderLiveViewportFrame(false);
                } else {
                    RenderFastPreview();
                }
                InvalidateRect(hWnd, NULL, FALSE);
            }

            if (!g_is_grabbing && (wParam == 'Z' || wParam == 'z')) {
                ToggleLiveViewportPT();
                return 0;
            }

            if (wParam == 'G' || wParam == 'g') {
                if (!g_is_grabbing) StartGrabMode();
                else ConfirmGrabMode();
                return 0;
            }
            else if (wParam == 'S' || wParam == 's') {
                if (!g_is_scaling) StartScaleMode();
                else ConfirmScaleMode();
                return 0;
            }
            else if (wParam == 'F' || wParam == 'f') {
                g_show_floor_grid = !g_show_floor_grid;
                g_viewport_pt_spp = 0;
                if (g_live_viewport_pt.load() && g_cuda_available) {
                    RenderLiveViewportFrame(false);
                } else {
                    RenderFastPreview();
                }
                InvalidateRect(hWnd, NULL, FALSE);
                return 0;
            }
            else if (g_is_grabbing) {
                if (wParam == 'X' || wParam == 'x') {
                    g_axis_lock = (g_axis_lock == AXIS_X) ? AXIS_NONE : AXIS_X;
                    POINT pt; GetCursorPos(&pt); ScreenToClient(hWnd, &pt);
                    ProcessGrabMove(pt.x, pt.y);
                    return 0;
                }
                else if (wParam == 'Y' || wParam == 'y') {
                    g_axis_lock = (g_axis_lock == AXIS_Y) ? AXIS_NONE : AXIS_Y;
                    POINT pt; GetCursorPos(&pt); ScreenToClient(hWnd, &pt);
                    ProcessGrabMove(pt.x, pt.y);
                    return 0;
                }
                else if (wParam == 'Z' || wParam == 'z') {
                    g_axis_lock = (g_axis_lock == AXIS_Z) ? AXIS_NONE : AXIS_Z;
                    POINT pt; GetCursorPos(&pt); ScreenToClient(hWnd, &pt);
                    ProcessGrabMove(pt.x, pt.y);
                    return 0;
                }
                else if (wParam == VK_RETURN) {
                    ConfirmGrabMode();
                    return 0;
                }
                else if (wParam == VK_ESCAPE) {
                    CancelGrabMode();
                    return 0;
                }
            }
            else if (g_is_scaling) {
                if (wParam == VK_RETURN) {
                    ConfirmScaleMode();
                    return 0;
                }
                else if (wParam == VK_ESCAPE) {
                    CancelScaleMode();
                    return 0;
                }
            }
            break;
        }

        case WM_PAINT: {
            if (g_viewport_dirty.load() && !g_is_rendering.load() && !g_has_rendered_image.load()) {
                if (g_live_viewport_pt.load() && g_cuda_available) {
                    RenderLiveViewportFrame(true);
                } else {
                    RenderFastPreview();
                }
                g_viewport_dirty.store(false);
            }

            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);

            int bufW = g_cur_buf_w.load();
            int bufH = g_cur_buf_h.load();

            if (bufW > 0 && bufH > 0 && g_display_buffer.size() >= static_cast<size_t>(bufW * bufH)) {
                int destX = 0, destY = 0;
                int destW = g_canvas_view_width;
                int destH = g_canvas_view_height;

                // Centrowanie i zachowanie proporcji kadru dla aktywnego oraz ukoczonego renderu
                if (g_is_rendering.load() || g_has_rendered_image.load()) {
                    float canvasAspect = static_cast<float>(g_canvas_view_width) / static_cast<float>(std::max(1, g_canvas_view_height));
                    float bufAspect = static_cast<float>(bufW) / static_cast<float>(bufH);

                    if (canvasAspect > bufAspect) {
                        destH = g_canvas_view_height;
                        destW = static_cast<int>(destH * bufAspect);
                        destX = (g_canvas_view_width - destW) / 2;
                        destY = 0;
                    } else {
                        destW = g_canvas_view_width;
                        destH = static_cast<int>(destW / bufAspect);
                        destX = 0;
                        destY = (g_canvas_view_height - destH) / 2;
                    }

                    if (destX > 0 || destY > 0) {
                        HBRUSH hBg = CreateSolidBrush(RGB(18, 18, 22));
                        if (destX > 0) {
                            RECT rLeft = {0, 0, destX, g_canvas_view_height};
                            RECT rRight = {destX + destW, 0, g_canvas_view_width, g_canvas_view_height};
                            FillRect(hdc, &rLeft, hBg);
                            FillRect(hdc, &rRight, hBg);
                        }
                        if (destY > 0) {
                            RECT rTop = {0, 0, g_canvas_view_width, destY};
                            RECT rBottom = {0, destY + destH, g_canvas_view_width, g_canvas_view_height};
                            FillRect(hdc, &rTop, hBg);
                            FillRect(hdc, &rBottom, hBg);
                        }
                        DeleteObject(hBg);
                    }
                }

                BITMAPINFO bmi = {};
                bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                bmi.bmiHeader.biWidth = bufW;
                bmi.bmiHeader.biHeight = -bufH; // Top-Down DIB zgodny ze skanliniami bufora
                bmi.bmiHeader.biPlanes = 1;
                bmi.bmiHeader.biBitCount = 32;
                bmi.bmiHeader.biCompression = BI_RGB;

                SetStretchBltMode(hdc, COLORONCOLOR);
                SetBrushOrgEx(hdc, 0, 0, NULL);

                StretchDIBits(
                    hdc,
                    destX, destY, destW, destH,
                    0, 0, bufW, bufH,
                    g_display_buffer.data(),
                    &bmi,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
            }

            // 2. Nakładki 3D (Wskaźnik zaznaczenia, osie X/Y/Z i kropkowane linie)
            if (!g_is_rendering.load() && !g_has_rendered_image.load() && g_selected_object_idx >= 0 && g_selected_object_idx < static_cast<int>(g_scene_objects.size())) {
                Camera cam = GetCurrentCamera();
                const auto& sel_obj = g_scene_objects[g_selected_object_idx];

                int sx = 0, sy = 0;
                float v_w = 2.0f * std::tan(g_cam_fov * 3.14159f / 360.0f) * (static_cast<float>(g_canvas_view_width) / g_canvas_view_height);
                float v_h = 2.0f * std::tan(g_cam_fov * 3.14159f / 360.0f);

                if (ProjectToScreen(sel_obj.center, cam, v_w, v_h, g_canvas_view_width, g_canvas_view_height, sx, sy)) {
                    if (g_is_grabbing) {
                        HPEN hPenAxis = nullptr;
                        Vec3 pA, pB;
                        if (g_axis_lock == AXIS_X) {
                            hPenAxis = CreatePen(PS_SOLID, 2, RGB(255, 60, 60));
                            pA = sel_obj.center + Vec3(-20.0f, 0.0f, 0.0f);
                            pB = sel_obj.center + Vec3( 20.0f, 0.0f, 0.0f);
                        } else if (g_axis_lock == AXIS_Y) {
                            hPenAxis = CreatePen(PS_SOLID, 2, RGB(60, 255, 60));
                            pA = sel_obj.center + Vec3(0.0f, -20.0f, 0.0f);
                            pB = sel_obj.center + Vec3(0.0f,  20.0f, 0.0f);
                        } else if (g_axis_lock == AXIS_Z) {
                            hPenAxis = CreatePen(PS_SOLID, 2, RGB(60, 140, 255));
                            pA = sel_obj.center + Vec3(0.0f, 0.0f, -20.0f);
                            pB = sel_obj.center + Vec3(0.0f, 0.0f,  20.0f);
                        }

                        if (hPenAxis) {
                            HPEN hOldPen = (HPEN)SelectObject(hdc, hPenAxis);
                            int ax1 = 0, ay1 = 0, ax2 = 0, ay2 = 0;
                            if (ProjectToScreen(pA, cam, v_w, v_h, g_canvas_view_width, g_canvas_view_height, ax1, ay1) &&
                                ProjectToScreen(pB, cam, v_w, v_h, g_canvas_view_width, g_canvas_view_height, ax2, ay2)) {
                                MoveToEx(hdc, ax1, ay1, NULL);
                                LineTo(hdc, ax2, ay2);
                            }
                            SelectObject(hdc, hOldPen);
                            DeleteObject(hPenAxis);
                        }

                        int orig_sx = 0, orig_sy = 0;
                        if (ProjectToScreen(g_grab_initial_pos, cam, v_w, v_h, g_canvas_view_width, g_canvas_view_height, orig_sx, orig_sy)) {
                            HPEN hPenDash = CreatePen(PS_DOT, 1, RGB(230, 230, 230));
                            HPEN hOldPen = (HPEN)SelectObject(hdc, hPenDash);
                            MoveToEx(hdc, orig_sx, orig_sy, NULL);
                            LineTo(hdc, sx, sy);
                            SelectObject(hdc, hOldPen);
                            DeleteObject(hPenDash);
                        }
                    }

                    COLORREF ringColor = RGB(255, 140, 0);
                    if (g_is_grabbing) ringColor = RGB(255, 230, 20);
                    else if (g_is_scaling) ringColor = RGB(80, 220, 255);

                    HPEN hPenOrange = CreatePen(PS_SOLID, 2, ringColor);
                    HBRUSH hNullBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
                    HPEN hOldPen = (HPEN)SelectObject(hdc, hPenOrange);
                    HBRUSH hOldBrush = (HBRUSH)SelectObject(hdc, hNullBrush);

                    int ring_r = 18;
                    Ellipse(hdc, sx - ring_r, sy - ring_r, sx + ring_r, sy + ring_r);
                    MoveToEx(hdc, sx - 6, sy, NULL); LineTo(hdc, sx + 7, sy);
                    MoveToEx(hdc, sx, sy - 6, NULL); LineTo(hdc, sx, sy + 7);

                    SelectObject(hdc, hOldPen);
                    SelectObject(hdc, hOldBrush);
                    DeleteObject(hPenOrange);
                }
            }

            // 3. Pasek pomocniczy HUD na dole kanwy
            if (!g_is_rendering.load() && !g_has_rendered_image.load()) {
                SetBkMode(hdc, TRANSPARENT);
                HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
                HFONT hOldFont = (HFONT)SelectObject(hdc, hFont);

                RECT hudRect = { 10, g_canvas_view_height - 30, g_canvas_view_width - 10, g_canvas_view_height - 6 };
                HBRUSH hBgBrush = CreateSolidBrush(RGB(18, 22, 28));
                FillRect(hdc, &hudRect, hBgBrush);
                DeleteObject(hBgBrush);

                wchar_t hudText[240];
                if (g_is_grabbing) {
                    const wchar_t* axisName = L"swobodny";
                    if (g_axis_lock == AXIS_X) axisName = L"oś X (czerwona)";
                    else if (g_axis_lock == AXIS_Y) axisName = L"oś Y (zielona)";
                    else if (g_axis_lock == AXIS_Z) axisName = L"oś Z (niebieska)";

                    swprintf(hudText, 240, L" ✋ [BLENDER GRAB] %s | Oś: %s | [X/Y/Z]: zmiana osi | [LPM/Enter]: zatwierdź | [PPM/Esc]: anuluj",
                             g_scene_objects[g_selected_object_idx].name.c_str(), axisName);
                    SetTextColor(hdc, RGB(255, 215, 0));
                } else if (g_is_scaling) {
                    swprintf(hudText, 240, L" 📐 [BLENDER SCALE] %s | R = %.2f | ruch myszą | [LPM/Enter]: zatwierdź | [PPM/Esc]: anuluj",
                             g_scene_objects[g_selected_object_idx].name.c_str(), g_scene_objects[g_selected_object_idx].radius);
                    SetTextColor(hdc, RGB(80, 220, 255));
                } else {
                    swprintf(hudText, 240, L" 🎮 [BLENDER 3D] LPM: obrót | SPM: wybór i przesuwanie obiektu | PPM: przesuwanie widoku | Scroll: zoom | G: przesuń | S: skaluj | F: siatka",
                             (g_selected_object_idx >= 0 ? g_scene_objects[g_selected_object_idx].name.c_str() : L"brak"));
                    SetTextColor(hdc, RGB(210, 225, 245));
                }

                DrawTextW(hdc, hudText, -1, &hudRect, DT_SINGLELINE | DT_VCENTER | DT_LEFT);

                RECT topRect = { 10, 8, 380, 28 };
                HBRUSH hTopBg = CreateSolidBrush(RGB(20, 24, 32));
                FillRect(hdc, &topRect, hTopBg);
                DeleteObject(hTopBg);
                SetTextColor(hdc, RGB(180, 200, 230));
                wchar_t topText[120];
                swprintf(topText, 120, L"  Widok 3D: płynny cieniowany | Siatka podłoża: [%s]  ", g_show_floor_grid ? L"WŁ" : L"WYŁ");
                DrawTextW(hdc, topText, -1, &topRect, DT_SINGLELINE | DT_VCENTER | DT_LEFT);

                SelectObject(hdc, hOldFont);
            }

            EndPaint(hWnd, &ps);
            return 0;
        }

        case WM_ERASEBKGND:
            return 1;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

// ============================================================================
// 8. RESPONSYWNE SKALOWANIE OKNA (PEŁNE WSPARCIE MAKSYMALIZACJI)
// ============================================================================


void ResizeLayout(int clientW, int clientH) {
    const int panelW = 390;

    // Kanwa widoku 3D wypełnia całą przestrzeń na lewo od panelu sterowania
    g_canvas_view_width  = std::max(320, clientW - panelW - 50);
    g_canvas_view_height = std::max(180, clientH - 100);
    if (g_hCanvas) MoveWindow(g_hCanvas, 15, 15, g_canvas_view_width, g_canvas_view_height, TRUE);

    // Pasek narzędzi pod kanwą, pasek postępu i etykieta statusu
    int rowY = 15 + g_canvas_view_height + 5;
    if (g_hChkLiveViewportPT) MoveWindow(g_hChkLiveViewportPT, 15, rowY, 260, 22, TRUE);
    if (g_hLblLightPower) MoveWindow(g_hLblLightPower, 280, rowY + 2, 75, 18, TRUE);
    if (g_hEditLightPower) MoveWindow(g_hEditLightPower, 360, rowY, 50, 22, TRUE);
    if (g_hBtnLightDown) MoveWindow(g_hBtnLightDown, 415, rowY, 28, 22, TRUE);
    if (g_hBtnLightUp) MoveWindow(g_hBtnLightUp, 447, rowY, 28, 22, TRUE);
    if (g_hBtnLightHalf) MoveWindow(g_hBtnLightHalf, 480, rowY, 42, 22, TRUE);
    if (g_hBtnLightDouble) MoveWindow(g_hBtnLightDouble, 526, rowY, 42, 22, TRUE);
    if (g_hBtnLightReset) MoveWindow(g_hBtnLightReset, 572, rowY, 42, 22, TRUE);
    if (g_hLblVpSpp) MoveWindow(g_hLblVpSpp, 620, rowY + 2, 160, 18, TRUE);
    if (g_hProgressBar) MoveWindow(g_hProgressBar, 15, rowY + 26, g_canvas_view_width, 20, TRUE);
    if (g_hStatusLabel) MoveWindow(g_hStatusLabel, 15, rowY + 50, g_canvas_view_width + panelW, 24, TRUE);

    // Prawy panel sterowania zadokowany przy prawej krawędzi
    int sideX = 15 + g_canvas_view_width + 20;
    int curY = 15;

    // 1. Grupa: Scena 3D - import geometrii
    if (g_hGroupScene) MoveWindow(g_hGroupScene, sideX, curY, panelW, 82, TRUE);
    curY += 20;
    if (g_hLblScene) MoveWindow(g_hLblScene, sideX + 15, curY, 80, 20, TRUE);
    if (g_hComboScene) MoveWindow(g_hComboScene, sideX + 100, curY - 2, 275, 150, TRUE);

    curY += 26;
    if (g_hBtnImportObj) MoveWindow(g_hBtnImportObj, sideX + 15, curY, 175, 28, TRUE);
    if (g_hBtnExportObj) MoveWindow(g_hBtnExportObj, sideX + 200, curY, 175, 28, TRUE);

    curY += 44;
    // 2. Grupa: Parametry renderowania & Algorytm
    if (g_hGroupRender) MoveWindow(g_hGroupRender, sideX, curY, panelW, 330, TRUE);
    curY += 20;
    if (g_hLblAlgo) MoveWindow(g_hLblAlgo, sideX + 15, curY, 95, 20, TRUE);
    if (g_hComboAlgo) MoveWindow(g_hComboAlgo, sideX + 115, curY - 2, 260, 150, TRUE);

    curY += 26;
    if (g_hChkCudaCaustics) MoveWindow(g_hChkCudaCaustics, sideX + 15, curY, 175, 22, TRUE);
    if (g_hChkCausticsHQ) MoveWindow(g_hChkCausticsHQ, sideX + 195, curY, 180, 22, TRUE);

    curY += 26;
    if (g_hLblPhotons) MoveWindow(g_hLblPhotons, sideX + 15, curY, 100, 20, TRUE);
    if (g_hEditPhotons) MoveWindow(g_hEditPhotons, sideX + 115, curY - 2, 85, 22, TRUE);
    if (g_hLblPhotHint) MoveWindow(g_hLblPhotHint, sideX + 210, curY, 165, 20, TRUE);

    curY += 26;
    if (g_hLblCausticBlur) MoveWindow(g_hLblCausticBlur, sideX + 15, curY, 82, 20, TRUE);
    if (g_hEditCausticBlur) MoveWindow(g_hEditCausticBlur, sideX + 100, curY - 2, 45, 22, TRUE);
    if (g_hLblCausticLight) MoveWindow(g_hLblCausticLight, sideX + 153, curY, 42, 20, TRUE);
    if (g_hEditCausticLight) MoveWindow(g_hEditCausticLight, sideX + 197, curY - 2, 45, 22, TRUE);
    if (g_hLblCausticGain) MoveWindow(g_hLblCausticGain, sideX + 252, curY, 48, 20, TRUE);
    if (g_hEditCausticGain) MoveWindow(g_hEditCausticGain, sideX + 302, curY - 2, 45, 22, TRUE);

    curY += 26;
    if (g_hLblCausticGrid) MoveWindow(g_hLblCausticGrid, sideX + 15, curY, 65, 20, TRUE);
    if (g_hComboCausticGrid) MoveWindow(g_hComboCausticGrid, sideX + 82, curY - 2, 88, 150, TRUE);
    if (g_hLblCausticRadius) MoveWindow(g_hLblCausticRadius, sideX + 178, curY, 72, 20, TRUE);
    if (g_hEditCausticRadius) MoveWindow(g_hEditCausticRadius, sideX + 252, curY - 2, 50, 22, TRUE);
    if (g_hBtnCausticApply) MoveWindow(g_hBtnCausticApply, sideX + 310, curY - 2, 65, 24, TRUE);

    curY += 26;
    if (g_hLblRes) MoveWindow(g_hLblRes, sideX + 15, curY, 95, 20, TRUE);
    if (g_hComboRes) MoveWindow(g_hComboRes, sideX + 115, curY - 2, 260, 200, TRUE);

    curY += 26;
    if (g_hLblW) MoveWindow(g_hLblW, sideX + 15, curY, 80, 20, TRUE);
    if (g_hEditResW) MoveWindow(g_hEditResW, sideX + 100, curY - 2, 65, 22, TRUE);
    if (g_hLblH) MoveWindow(g_hLblH, sideX + 190, curY, 75, 20, TRUE);
    if (g_hEditResH) MoveWindow(g_hEditResH, sideX + 270, curY - 2, 65, 22, TRUE);

    curY += 26;
    if (g_hLblTile) MoveWindow(g_hLblTile, sideX + 15, curY, 95, 20, TRUE);
    if (g_hComboTile) MoveWindow(g_hComboTile, sideX + 115, curY - 2, 260, 150, TRUE);

    curY += 26;
    if (g_hLblSPP) MoveWindow(g_hLblSPP, sideX + 15, curY, 80, 20, TRUE);
    if (g_hEditSPP) MoveWindow(g_hEditSPP, sideX + 100, curY - 2, 65, 22, TRUE);
    if (g_hLblDepth) MoveWindow(g_hLblDepth, sideX + 190, curY, 75, 20, TRUE);
    if (g_hEditDepth) MoveWindow(g_hEditDepth, sideX + 270, curY - 2, 65, 22, TRUE);

    curY += 26;
    if (g_hChkDenoise) MoveWindow(g_hChkDenoise, sideX + 15, curY, 230, 22, TRUE);
    if (g_hBtnDenoiseNow) MoveWindow(g_hBtnDenoiseNow, sideX + 250, curY - 2, 125, 24, TRUE);

    curY += 30;
    if (g_hBtnRender) MoveWindow(g_hBtnRender, sideX + 15, curY, 200, 32, TRUE);
    if (g_hBtnStop) MoveWindow(g_hBtnStop, sideX + 225, curY, 150, 32, TRUE);

    curY += 46;
    // Grupa: Ekspozycja (F-stopy) & Tone Mapping (HDR)
    if (g_hGroupExposure) MoveWindow(g_hGroupExposure, sideX, curY, panelW, 96, TRUE);
    curY += 20;
    if (g_hLblExposure) MoveWindow(g_hLblExposure, sideX + 15, curY, 100, 20, TRUE);
    if (g_hEditExposure) MoveWindow(g_hEditExposure, sideX + 120, curY - 2, 50, 22, TRUE);
    if (g_hBtnEvDown) MoveWindow(g_hBtnEvDown, sideX + 175, curY - 2, 45, 22, TRUE);
    if (g_hBtnEvUp) MoveWindow(g_hBtnEvUp, sideX + 225, curY - 2, 45, 22, TRUE);
    if (g_hBtnEvReset) MoveWindow(g_hBtnEvReset, sideX + 275, curY - 2, 45, 22, TRUE);
    if (g_hBtnEvApply) MoveWindow(g_hBtnEvApply, sideX + 325, curY - 2, 50, 22, TRUE);
    curY += 24;
    if (g_hLblFStopHint) MoveWindow(g_hLblFStopHint, sideX + 15, curY, 360, 18, TRUE);
    curY += 22;
    if (g_hLblToneMap) MoveWindow(g_hLblToneMap, sideX + 15, curY, 95, 20, TRUE);
    if (g_hComboToneMap) MoveWindow(g_hComboToneMap, sideX + 115, curY - 2, 260, 150, TRUE);

    curY += 36;
    // 3. Grupa: Obiekt, Pozycja & Material
    if (g_hGroupObj) MoveWindow(g_hGroupObj, sideX, curY, panelW, 280, TRUE);
    curY += 18;
    if (g_hLblObj) MoveWindow(g_hLblObj, sideX + 15, curY, 90, 20, TRUE);
    if (g_hComboObjects) MoveWindow(g_hComboObjects, sideX + 110, curY - 2, 265, 200, TRUE);

    curY += 25;
    if (g_hLblPosX) MoveWindow(g_hLblPosX, sideX + 15, curY, 75, 20, TRUE);
    if (g_hEditPosX) MoveWindow(g_hEditPosX, sideX + 90, curY - 2, 75, 22, TRUE);
    if (g_hLblRadius) MoveWindow(g_hLblRadius, sideX + 185, curY, 70, 20, TRUE);
    if (g_hEditRadius) MoveWindow(g_hEditRadius, sideX + 260, curY - 2, 80, 22, TRUE);

    curY += 24;
    if (g_hLblPosY) MoveWindow(g_hLblPosY, sideX + 15, curY, 75, 20, TRUE);
    if (g_hEditPosY) MoveWindow(g_hEditPosY, sideX + 90, curY - 2, 75, 22, TRUE);
    if (g_hBtnApplyPos) MoveWindow(g_hBtnApplyPos, sideX + 185, curY - 2, 185, 24, TRUE);

    curY += 24;
    if (g_hLblPosZ) MoveWindow(g_hLblPosZ, sideX + 15, curY, 75, 20, TRUE);
    if (g_hEditPosZ) MoveWindow(g_hEditPosZ, sideX + 90, curY - 2, 75, 22, TRUE);
    if (g_hBtnGrab) MoveWindow(g_hBtnGrab, sideX + 185, curY - 2, 185, 24, TRUE);

    curY += 25;
    if (g_hBtnResetCam) MoveWindow(g_hBtnResetCam, sideX + 185, curY - 2, 185, 24, TRUE);

    // Sekcja Materialu wewnatrz grupy obiektow:
    curY += 28;
    if (g_hLblMatType) MoveWindow(g_hLblMatType, sideX + 15, curY, 90, 20, TRUE);
    if (g_hComboMatType) MoveWindow(g_hComboMatType, sideX + 110, curY - 2, 265, 150, TRUE);

    curY += 25;
    if (g_hLblMatCol) MoveWindow(g_hLblMatCol, sideX + 15, curY, 75, 20, TRUE);
    if (g_hEditMatR) MoveWindow(g_hEditMatR, sideX + 90, curY - 2, 50, 22, TRUE);
    if (g_hEditMatG) MoveWindow(g_hEditMatG, sideX + 145, curY - 2, 50, 22, TRUE);
    if (g_hEditMatB) MoveWindow(g_hEditMatB, sideX + 200, curY - 2, 50, 22, TRUE);
    if (g_hEditMatParam) MoveWindow(g_hEditMatParam, sideX + 260, curY - 2, 80, 22, TRUE);

    curY += 25;
    if (g_hLblMatDisp) MoveWindow(g_hLblMatDisp, sideX + 15, curY, 75, 20, TRUE);
    if (g_hEditMatDisp) MoveWindow(g_hEditMatDisp, sideX + 90, curY - 2, 75, 22, TRUE);
    if (g_hLblMatDispHint) MoveWindow(g_hLblMatDispHint, sideX + 175, curY, 195, 20, TRUE);

    curY += 26;
    if (g_hBtnApplyMat) MoveWindow(g_hBtnApplyMat, sideX + 15, curY, 360, 26, TRUE);

    curY += 36;
    // 4. Grupa: Zapis PNG
    if (g_hGroupSave) MoveWindow(g_hGroupSave, sideX, curY, panelW, 78, TRUE);
    curY += 18;
    if (g_hLblFile) MoveWindow(g_hLblFile, sideX + 15, curY, 80, 20, TRUE);
    if (g_hEditFilename) MoveWindow(g_hEditFilename, sideX + 100, curY - 2, 275, 22, TRUE);

    curY += 24;
    if (g_hBtnSave) MoveWindow(g_hBtnSave, sideX + 15, curY, 150, 26, TRUE);
    if (g_hLblSaveStatus) MoveWindow(g_hLblSaveStatus, sideX + 172, curY + 5, 205, 18, TRUE);

    // Renderujemy szybki podgląd tylko jeśli renderowanie PT nie trwa w tle i nie ma wyrenderowanego obrazu!
    if (!g_is_rendering.load() && !g_has_rendered_image.load()) {
        RenderFastPreview();
    }
    if (g_hCanvas) InvalidateRect(g_hCanvas, NULL, FALSE);
}

// ============================================================================
// 9. GŁÓWNA PROCEDURA OKNA
// ============================================================================

LRESULT CALLBACK MainWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            WNDCLASSW wcCanvas = {};
            wcCanvas.lpfnWndProc = CanvasWndProc;
            wcCanvas.hInstance = GetModuleHandle(NULL);
            wcCanvas.lpszClassName = L"PathTracerCanvas";
            wcCanvas.hCursor = LoadCursor(NULL, IDC_ARROW);
            RegisterClassW(&wcCanvas);

            g_hCanvas = CreateWindowExW(
                WS_EX_CLIENTEDGE, L"PathTracerCanvas", nullptr,
                WS_CHILD | WS_VISIBLE,
                15, 15, g_canvas_view_width, g_canvas_view_height,
                hWnd, nullptr, GetModuleHandle(NULL), nullptr
            );

            HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
            int sideX = 835;
            int curY = 15;

            // 1. Grupa: Scena 3D - import geometrii
            g_hGroupScene = CreateWindowW(L"BUTTON", L" Wybór sceny 3D - import geometrii ",
                WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
                sideX, curY, 390, 82, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hGroupScene, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 20;
            g_hLblScene = CreateWindowW(L"STATIC", L"Scena 3D:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 80, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblScene, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hComboScene = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                sideX + 100, curY - 2, 275, 150, hWnd, (HMENU)IDC_COMBO_SCENE, nullptr, nullptr);
            SendMessage(g_hComboScene, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hComboScene, CB_ADDSTRING, 0, (LPARAM)L"1. Kryształ szklany i kule (kaustyka)");
            SendMessage(g_hComboScene, CB_ADDSTRING, 0, (LPARAM)L"2. Zamknięty pokój ze szczeliną (Manifold)");
            SendMessage(g_hComboScene, CB_SETCURSEL, 0, 0);

            curY += 26;
            g_hBtnImportObj = CreateWindowW(L"BUTTON", L"📂 Importuj (.OBJ)", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 15, curY, 175, 28, hWnd, (HMENU)IDC_BTN_IMPORT_OBJ, nullptr, nullptr);
            SendMessage(g_hBtnImportObj, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnExportObj = CreateWindowW(L"BUTTON", L"💾 Eksportuj (.OBJ)", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 200, curY, 175, 28, hWnd, (HMENU)IDC_BTN_EXPORT_OBJ, nullptr, nullptr);
            SendMessage(g_hBtnExportObj, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 44;
            // 2. Grupa: Parametry renderowania & Algorytm
            g_hGroupRender = CreateWindowW(L"BUTTON", L" Parametry renderowania i algorytm ",
                WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
                sideX, curY, 390, 330, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hGroupRender, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 20;
            g_hLblAlgo = CreateWindowW(L"STATIC", L"Algorytm PT:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 95, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblAlgo, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hComboAlgo = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                sideX + 115, curY - 2, 260, 150, hWnd, (HMENU)IDC_COMBO_ALGO, nullptr, nullptr);
            SendMessage(g_hComboAlgo, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hComboAlgo, CB_ADDSTRING, 0, (LPARAM)L"Standardowy Path Tracing (CPU)");
            SendMessage(g_hComboAlgo, CB_ADDSTRING, 0, (LPARAM)L"Standardowy Path Tracing (GPU / CUDA)");
            SendMessage(g_hComboAlgo, CB_ADDSTRING, 0, (LPARAM)L"BDPT (globalna mapa fotonów – kryształ)");
            SendMessage(g_hComboAlgo, CB_ADDSTRING, 0, (LPARAM)L"Metropolis Light Transport (MLT)");
            SendMessage(g_hComboAlgo, CB_ADDSTRING, 0, (LPARAM)L"Manifold NEE (MNEE – kaustyka przez rozmaitość)");

            if (CudaPT_IsAvailable(g_cuda_device_name)) {
                g_cuda_available = true;
                SendMessage(g_hComboAlgo, CB_SETCURSEL, 1, 0); // Domyślnie GPU CUDA!
                g_render_algo = 1;
            } else {
                g_cuda_available = false;
                SendMessage(g_hComboAlgo, CB_SETCURSEL, 0, 0); // CPU PT
                g_render_algo = 0;
            }

            curY += 26;
            g_hChkCudaCaustics = CreateWindowW(L"BUTTON", L"Kaustyka GPU (CUDA)", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                sideX + 15, curY, 175, 22, hWnd, (HMENU)IDC_CHK_CUDA_CAUSTICS, nullptr, nullptr);
            SendMessage(g_hChkCudaCaustics, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hChkCudaCaustics, BM_SETCHECK, BST_CHECKED, 0);

            g_hChkCausticsHQ = CreateWindowW(L"BUTTON", L"Precyzyjny render HD", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                sideX + 195, curY, 180, 22, hWnd, (HMENU)IDC_CHK_CAUSTICS_HQ, nullptr, nullptr);
            SendMessage(g_hChkCausticsHQ, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hChkCausticsHQ, BM_SETCHECK, BST_CHECKED, 0);

            curY += 26;
            g_hLblPhotons = CreateWindowW(L"STATIC", L"Fotony kaustyki:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 100, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblPhotons, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hEditPhotons = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"50000", WS_CHILD | WS_VISIBLE | ES_NUMBER,
                sideX + 115, curY - 2, 85, 22, hWnd, (HMENU)IDC_EDIT_PHOTONS, nullptr, nullptr);
            SendMessage(g_hEditPhotons, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hLblPhotHint = CreateWindowW(L"STATIC", L"(np. 20 000 – 150 000)", WS_CHILD | WS_VISIBLE, sideX + 210, curY, 165, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblPhotHint, WM_SETFONT, (WPARAM)hFont, TRUE);

            // Parametry ostrości kaustyki: rozmycie, rozmiar źródła, jasność, siatka GPU, promień CPU
            curY += 26;
            g_hLblCausticBlur = CreateWindowW(L"STATIC", L"Rozmycie [cm]:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 82, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblCausticBlur, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditCausticBlur = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.25", WS_CHILD | WS_VISIBLE, sideX + 100, curY - 2, 45, 22, hWnd, (HMENU)IDC_EDIT_CAUSTIC_BLUR, nullptr, nullptr);
            SendMessage(g_hEditCausticBlur, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hLblCausticLight = CreateWindowW(L"STATIC", L"Źródło:", WS_CHILD | WS_VISIBLE, sideX + 153, curY, 42, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblCausticLight, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditCausticLight = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.12", WS_CHILD | WS_VISIBLE, sideX + 197, curY - 2, 45, 22, hWnd, (HMENU)IDC_EDIT_CAUSTIC_LIGHT, nullptr, nullptr);
            SendMessage(g_hEditCausticLight, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hLblCausticGain = CreateWindowW(L"STATIC", L"Jasność:", WS_CHILD | WS_VISIBLE, sideX + 252, curY, 48, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblCausticGain, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditCausticGain = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"1.00", WS_CHILD | WS_VISIBLE, sideX + 302, curY - 2, 45, 22, hWnd, (HMENU)IDC_EDIT_CAUSTIC_GAIN, nullptr, nullptr);
            SendMessage(g_hEditCausticGain, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 26;
            g_hLblCausticGrid = CreateWindowW(L"STATIC", L"Siatka GPU:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 65, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblCausticGrid, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hComboCausticGrid = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                sideX + 82, curY - 2, 88, 150, hWnd, (HMENU)IDC_COMBO_CAUSTIC_GRID, nullptr, nullptr);
            SendMessage(g_hComboCausticGrid, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hComboCausticGrid, CB_ADDSTRING, 0, (LPARAM)L"1024 (szybka)");
            SendMessage(g_hComboCausticGrid, CB_ADDSTRING, 0, (LPARAM)L"2048 (ostra)");
            SendMessage(g_hComboCausticGrid, CB_ADDSTRING, 0, (LPARAM)L"4096 (b. ostra)");
            SendMessage(g_hComboCausticGrid, CB_SETCURSEL, 1, 0);
            g_hLblCausticRadius = CreateWindowW(L"STATIC", L"Promień CPU:", WS_CHILD | WS_VISIBLE, sideX + 178, curY, 72, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblCausticRadius, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditCausticRadius = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.028", WS_CHILD | WS_VISIBLE, sideX + 252, curY - 2, 50, 22, hWnd, (HMENU)IDC_EDIT_CAUSTIC_RADIUS, nullptr, nullptr);
            SendMessage(g_hEditCausticRadius, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hBtnCausticApply = CreateWindowW(L"BUTTON", L"Zastosuj", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 310, curY - 2, 65, 24, hWnd, (HMENU)IDC_BTN_CAUSTIC_APPLY, nullptr, nullptr);
            SendMessage(g_hBtnCausticApply, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 26;
            g_hLblRes = CreateWindowW(L"STATIC", L"Rozdzielczość:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 95, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblRes, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hComboRes = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
                sideX + 115, curY - 2, 260, 200, hWnd, (HMENU)IDC_COMBO_RES, nullptr, nullptr);
            SendMessage(g_hComboRes, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hComboRes, CB_ADDSTRING, 0, (LPARAM)L"640 x 360  (bardzo szybki podgląd)");
            SendMessage(g_hComboRes, CB_ADDSTRING, 0, (LPARAM)L"800 x 450  (standardowy podgląd)");
            SendMessage(g_hComboRes, CB_ADDSTRING, 0, (LPARAM)L"960 x 540  (qHD)");
            SendMessage(g_hComboRes, CB_ADDSTRING, 0, (LPARAM)L"1280 x 720 (HD 720p)");
            SendMessage(g_hComboRes, CB_ADDSTRING, 0, (LPARAM)L"1920 x 1080 (Full HD 1080p)");
            SendMessage(g_hComboRes, CB_ADDSTRING, 0, (LPARAM)L"Niestandardowa (wpisz poniżej)");
            SendMessage(g_hComboRes, CB_SETCURSEL, 1, 0);

            curY += 26;
            g_hLblW = CreateWindowW(L"STATIC", L"Szerokość (W):", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 80, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblW, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditResW = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"800", WS_CHILD | WS_VISIBLE | ES_NUMBER, sideX + 100, curY - 2, 65, 22, hWnd, (HMENU)IDC_EDIT_RES_W, nullptr, nullptr);
            SendMessage(g_hEditResW, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hLblH = CreateWindowW(L"STATIC", L"Wysokość (H):", WS_CHILD | WS_VISIBLE, sideX + 190, curY, 75, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblH, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditResH = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"450", WS_CHILD | WS_VISIBLE | ES_NUMBER, sideX + 270, curY - 2, 65, 22, hWnd, (HMENU)IDC_EDIT_RES_H, nullptr, nullptr);
            SendMessage(g_hEditResH, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 26;
            g_hLblTile = CreateWindowW(L"STATIC", L"Rozmiar kafelka:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 95, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblTile, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hComboTile = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                sideX + 115, curY - 2, 260, 150, hWnd, (HMENU)IDC_COMBO_TILE, nullptr, nullptr);
            SendMessage(g_hComboTile, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hComboTile, CB_ADDSTRING, 0, (LPARAM)L"16 x 16 px (drobny – bez szwów)");
            SendMessage(g_hComboTile, CB_ADDSTRING, 0, (LPARAM)L"32 x 32 px (standardowy – optymalny)");
            SendMessage(g_hComboTile, CB_ADDSTRING, 0, (LPARAM)L"64 x 64 px (duże kafelki)");
            SendMessage(g_hComboTile, CB_SETCURSEL, 0, 0);

            curY += 26;
            g_hLblSPP = CreateWindowW(L"STATIC", L"Próbki (SPP):", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 80, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblSPP, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditSPP = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"64", WS_CHILD | WS_VISIBLE | ES_NUMBER, sideX + 100, curY - 2, 65, 22, hWnd, (HMENU)IDC_EDIT_SPP, nullptr, nullptr);
            SendMessage(g_hEditSPP, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hLblDepth = CreateWindowW(L"STATIC", L"Maks. odbić:", WS_CHILD | WS_VISIBLE, sideX + 190, curY, 75, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblDepth, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditDepth = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"16", WS_CHILD | WS_VISIBLE | ES_NUMBER, sideX + 270, curY - 2, 65, 22, hWnd, (HMENU)IDC_EDIT_DEPTH, nullptr, nullptr);
            SendMessage(g_hEditDepth, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 26;
            g_hChkDenoise = CreateWindowW(L"BUTTON", L"Odszumianie AI (Intel OIDN)", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                sideX + 15, curY, 230, 22, hWnd, (HMENU)IDC_CHK_DENOISE, nullptr, nullptr);
            SendMessage(g_hChkDenoise, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hChkDenoise, BM_SETCHECK, BST_CHECKED, 0);

            g_hBtnDenoiseNow = CreateWindowW(L"BUTTON", L"⚡ Odszum teraz", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 250, curY - 2, 125, 24, hWnd, (HMENU)IDC_BTN_DENOISE_NOW, nullptr, nullptr);
            SendMessage(g_hBtnDenoiseNow, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 28;
            g_hBtnRender = CreateWindowW(L"BUTTON", L"▶ Rozpocznij renderowanie", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 15, curY, 200, 32, hWnd, (HMENU)IDC_BTN_RENDER, nullptr, nullptr);
            SendMessage(g_hBtnRender, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnStop = CreateWindowW(L"BUTTON", L"⏹ Przerwij", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | WS_DISABLED,
                sideX + 225, curY, 150, 32, hWnd, (HMENU)IDC_BTN_STOP, nullptr, nullptr);
            SendMessage(g_hBtnStop, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 46;

            // Grupa: Ekspozycja (F-stopy) & Tone Mapping (HDR)
            g_hGroupExposure = CreateWindowW(L"BUTTON", L" Ekspozycja (F-stopy) i Tone Mapping (HDR) ",
                WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
                sideX, curY, 390, 96, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hGroupExposure, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 20;
            g_hLblExposure = CreateWindowW(L"STATIC", L"Ekspozycja (EV):", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 100, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblExposure, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hEditExposure = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"-2.00", WS_CHILD | WS_VISIBLE,
                sideX + 120, curY - 2, 50, 22, hWnd, (HMENU)IDC_EDIT_EXPOSURE, nullptr, nullptr);
            SendMessage(g_hEditExposure, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnEvDown = CreateWindowW(L"BUTTON", L"-0.5", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 175, curY - 2, 45, 22, hWnd, (HMENU)IDC_BTN_EV_DOWN, nullptr, nullptr);
            SendMessage(g_hBtnEvDown, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnEvUp = CreateWindowW(L"BUTTON", L"+0.5", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 225, curY - 2, 45, 22, hWnd, (HMENU)IDC_BTN_EV_UP, nullptr, nullptr);
            SendMessage(g_hBtnEvUp, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnEvReset = CreateWindowW(L"BUTTON", L"0 EV", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 275, curY - 2, 45, 22, hWnd, (HMENU)IDC_BTN_EV_RESET, nullptr, nullptr);
            SendMessage(g_hBtnEvReset, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnEvApply = CreateWindowW(L"BUTTON", L"OK", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 325, curY - 2, 50, 22, hWnd, (HMENU)IDC_BTN_EV_APPLY, nullptr, nullptr);
            SendMessage(g_hBtnEvApply, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 24;
            g_hLblFStopHint = CreateWindowW(L"STATIC", L"Przysłona: f/5.6 (-2.0 stopów | 25% światła)", WS_CHILD | WS_VISIBLE,
                sideX + 15, curY, 360, 18, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblFStopHint, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 22;
            g_hLblToneMap = CreateWindowW(L"STATIC", L"Tone Mapping:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 95, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblToneMap, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hComboToneMap = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                sideX + 115, curY - 2, 260, 150, hWnd, (HMENU)IDC_COMBO_TONEMAP, nullptr, nullptr);
            SendMessage(g_hComboToneMap, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hComboToneMap, CB_ADDSTRING, 0, (LPARAM)L"ACES Filmic (kinowy – 16+ F-stopów)");
            SendMessage(g_hComboToneMap, CB_ADDSTRING, 0, (LPARAM)L"Reinhard (kompresja jasnych świateł)");
            SendMessage(g_hComboToneMap, CB_ADDSTRING, 0, (LPARAM)L"Filmic Soft / AgX (miękki roll-off)");
            SendMessage(g_hComboToneMap, CB_ADDSTRING, 0, (LPARAM)L"Liniowy (standard z obcięciem)");
            SendMessage(g_hComboToneMap, CB_SETCURSEL, 0, 0);

            curY += 36;

            // 3. Grupa: Obiekt, Pozycja & Materiał
            g_hGroupObj = CreateWindowW(L"BUTTON", L" Edycja obiektu, pozycji (Blender 3D) i materiału ",
                WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
                sideX, curY, 390, 280, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hGroupObj, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 18;
            g_hLblObj = CreateWindowW(L"STATIC", L"Wybierz obiekt:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 90, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblObj, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hComboObjects = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
                sideX + 110, curY - 2, 265, 200, hWnd, (HMENU)IDC_COMBO_OBJECTS, nullptr, nullptr);
            SendMessage(g_hComboObjects, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 25;
            g_hLblPosX = CreateWindowW(L"STATIC", L"Pozycja X:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 75, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblPosX, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditPosX = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.00", WS_CHILD | WS_VISIBLE, sideX + 90, curY - 2, 75, 22, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hEditPosX, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hLblRadius = CreateWindowW(L"STATIC", L"Promień (R):", WS_CHILD | WS_VISIBLE, sideX + 185, curY, 70, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblRadius, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditRadius = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.65", WS_CHILD | WS_VISIBLE, sideX + 260, curY - 2, 80, 22, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hEditRadius, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 24;
            g_hLblPosY = CreateWindowW(L"STATIC", L"Pozycja Y:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 75, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblPosY, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditPosY = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.00", WS_CHILD | WS_VISIBLE, sideX + 90, curY - 2, 75, 22, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hEditPosY, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnApplyPos = CreateWindowW(L"BUTTON", L"Zastosuj pozycję", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 185, curY - 2, 185, 24, hWnd, (HMENU)IDC_BTN_APPLY_POS, nullptr, nullptr);
            SendMessage(g_hBtnApplyPos, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 24;
            g_hLblPosZ = CreateWindowW(L"STATIC", L"Pozycja Z:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 75, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblPosZ, WM_SETFONT, (WPARAM)hFont, TRUE);
            g_hEditPosZ = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.00", WS_CHILD | WS_VISIBLE, sideX + 90, curY - 2, 75, 22, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hEditPosZ, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnGrab = CreateWindowW(L"BUTTON", L"✋ Przesuń w 3D (G)", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 185, curY - 2, 185, 24, hWnd, (HMENU)IDC_BTN_GRAB, nullptr, nullptr);
            SendMessage(g_hBtnGrab, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 25;
            g_hBtnResetCam = CreateWindowW(L"BUTTON", L"🎥 Reset kamery 3D", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 185, curY - 2, 185, 24, hWnd, (HMENU)IDC_BTN_RESET_CAM, nullptr, nullptr);
            SendMessage(g_hBtnResetCam, WM_SETFONT, (WPARAM)hFont, TRUE);

            // Material Editor wewnątrz grupy obiektów:
            curY += 28;
            g_hLblMatType = CreateWindowW(L"STATIC", L"Materiał:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 90, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblMatType, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hComboMatType = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                sideX + 110, curY - 2, 265, 150, hWnd, (HMENU)IDC_COMBO_MAT_TYPE, nullptr, nullptr);
            SendMessage(g_hComboMatType, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hComboMatType, CB_ADDSTRING, 0, (LPARAM)L"Lambertian (matowy / rozproszony)");
            SendMessage(g_hComboMatType, CB_ADDSTRING, 0, (LPARAM)L"Metal (zwierciadlany / lustro)");
            SendMessage(g_hComboMatType, CB_ADDSTRING, 0, (LPARAM)L"Dielectric (szkło / kryształ)");
            SendMessage(g_hComboMatType, CB_ADDSTRING, 0, (LPARAM)L"DiffuseLight (emisja światła)");
            SendMessage(g_hComboMatType, CB_SETCURSEL, 0, 0);

            curY += 25;
            g_hLblMatCol = CreateWindowW(L"STATIC", L"RGB / parametr:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 75, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblMatCol, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hEditMatR = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.85", WS_CHILD | WS_VISIBLE, sideX + 90, curY - 2, 50, 22, hWnd, (HMENU)IDC_EDIT_MAT_R, nullptr, nullptr);
            SendMessage(g_hEditMatR, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hEditMatG = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.85", WS_CHILD | WS_VISIBLE, sideX + 145, curY - 2, 50, 22, hWnd, (HMENU)IDC_EDIT_MAT_G, nullptr, nullptr);
            SendMessage(g_hEditMatG, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hEditMatB = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.85", WS_CHILD | WS_VISIBLE, sideX + 200, curY - 2, 50, 22, hWnd, (HMENU)IDC_EDIT_MAT_B, nullptr, nullptr);
            SendMessage(g_hEditMatB, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hEditMatParam = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.05", WS_CHILD | WS_VISIBLE, sideX + 260, curY - 2, 80, 22, hWnd, (HMENU)IDC_EDIT_MAT_PARAM, nullptr, nullptr);
            SendMessage(g_hEditMatParam, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 25;
            g_hLblMatDisp = CreateWindowW(L"STATIC", L"Dyspersja:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 75, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblMatDisp, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hEditMatDisp = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"0.020", WS_CHILD | WS_VISIBLE, sideX + 90, curY - 2, 75, 22, hWnd, (HMENU)IDC_EDIT_MAT_DISP, nullptr, nullptr);
            SendMessage(g_hEditMatDisp, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hLblMatDispHint = CreateWindowW(L"STATIC", L"(0.00 = brak, 0.05 = pryzmat)", WS_CHILD | WS_VISIBLE, sideX + 175, curY, 195, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblMatDispHint, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 26;
            g_hBtnApplyMat = CreateWindowW(L"BUTTON", L"🎨 Zastosuj materiał do wybranego obiektu", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 15, curY, 360, 26, hWnd, (HMENU)IDC_BTN_APPLY_MAT, nullptr, nullptr);
            SendMessage(g_hBtnApplyMat, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 36;

            // 4. Grupa: Zapis PNG
            g_hGroupSave = CreateWindowW(L"BUTTON", L" Zapis obrazu do pliku PNG ",
                WS_CHILD | WS_VISIBLE | BS_GROUPBOX,
                sideX, curY, 390, 78, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hGroupSave, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 18;
            g_hLblFile = CreateWindowW(L"STATIC", L"Nazwa pliku:", WS_CHILD | WS_VISIBLE, sideX + 15, curY, 80, 20, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblFile, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hEditFilename = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"render_kaustyka.png", WS_CHILD | WS_VISIBLE,
                sideX + 100, curY - 2, 275, 22, hWnd, (HMENU)IDC_EDIT_FILENAME, nullptr, nullptr);
            SendMessage(g_hEditFilename, WM_SETFONT, (WPARAM)hFont, TRUE);

            curY += 24;
            g_hBtnSave = CreateWindowW(L"BUTTON", L"💾 Zapisz obraz PNG",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                sideX + 15, curY, 150, 26, hWnd, (HMENU)IDC_BTN_SAVE_PNG, nullptr, nullptr);
            SendMessage(g_hBtnSave, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hLblSaveStatus = CreateWindowW(L"STATIC", L"", WS_CHILD | WS_VISIBLE | SS_ENDELLIPSIS,
                sideX + 172, curY + 5, 205, 18, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblSaveStatus, WM_SETFONT, (WPARAM)hFont, TRUE);

            // Pasek narzędzi podglądu na żywo i mocy świateł
            g_hChkLiveViewportPT = CreateWindowW(L"BUTTON", L"⚡ Podgląd Path Tracing (GPU RTX) [Z]",
                WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                15, 470, 260, 22, hWnd, (HMENU)IDC_CHK_LIVE_VIEWPORT_PT, nullptr, nullptr);
            SendMessage(g_hChkLiveViewportPT, WM_SETFONT, (WPARAM)hFont, TRUE);
            SendMessage(g_hChkLiveViewportPT, BM_SETCHECK, g_cuda_available ? BST_CHECKED : BST_UNCHECKED, 0);

            g_hLblLightPower = CreateWindowW(L"STATIC", L"Moc świateł:", WS_CHILD | WS_VISIBLE,
                280, 472, 75, 18, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblLightPower, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hEditLightPower = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"1.00x", WS_CHILD | WS_VISIBLE,
                360, 470, 50, 22, hWnd, (HMENU)IDC_EDIT_LIGHT_POWER, nullptr, nullptr);
            SendMessage(g_hEditLightPower, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnLightDown = CreateWindowW(L"BUTTON", L"-", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                415, 470, 28, 22, hWnd, (HMENU)IDC_BTN_LIGHT_DOWN, nullptr, nullptr);
            SendMessage(g_hBtnLightDown, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnLightUp = CreateWindowW(L"BUTTON", L"+", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                447, 470, 28, 22, hWnd, (HMENU)IDC_BTN_LIGHT_UP, nullptr, nullptr);
            SendMessage(g_hBtnLightUp, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnLightHalf = CreateWindowW(L"BUTTON", L"0.5x", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                480, 470, 42, 22, hWnd, (HMENU)IDC_BTN_LIGHT_HALF, nullptr, nullptr);
            SendMessage(g_hBtnLightHalf, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnLightDouble = CreateWindowW(L"BUTTON", L"2.0x", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                526, 470, 42, 22, hWnd, (HMENU)IDC_BTN_LIGHT_DOUBLE, nullptr, nullptr);
            SendMessage(g_hBtnLightDouble, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hBtnLightReset = CreateWindowW(L"BUTTON", L"1.0x", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                572, 470, 42, 22, hWnd, (HMENU)IDC_BTN_LIGHT_RESET, nullptr, nullptr);
            SendMessage(g_hBtnLightReset, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hLblVpSpp = CreateWindowW(L"STATIC", L"VP: 0 / 256 SPP", WS_CHILD | WS_VISIBLE,
                620, 472, 160, 18, hWnd, nullptr, nullptr, nullptr);
            SendMessage(g_hLblVpSpp, WM_SETFONT, (WPARAM)hFont, TRUE);

            g_hProgressBar = CreateWindowExW(0, PROGRESS_CLASSW, nullptr,
                WS_CHILD | WS_VISIBLE | PBS_SMOOTH,
                15, 496, g_canvas_view_width, 20, hWnd, (HMENU)IDC_PROGRESS_BAR, nullptr, nullptr);

            g_hStatusLabel = CreateWindowW(L"STATIC", L"Gotowy. LPM = obrót kamery, SPM = wybór i przesuwanie obiektu, PPM = przesuwanie widoku, Scroll = zoom, G / S = przesuń / skaluj.",
                WS_CHILD | WS_VISIBLE,
                15, 520, g_canvas_view_width + 390, 24, hWnd, (HMENU)IDC_STATUS_LABEL, nullptr, nullptr);
            SendMessage(g_hStatusLabel, WM_SETFONT, (WPARAM)hFont, TRUE);

            SwitchScene(0);

            SetTimer(hWnd, TIMER_UPDATE_ID, 33, nullptr);
            break;
        }

        case WM_SIZE: {
            if (wParam != SIZE_MINIMIZED) {
                int clientW = LOWORD(lParam);
                int clientH = HIWORD(lParam);
                ResizeLayout(clientW, clientH);
            }
            break;
        }

        case WM_TIMER: {
            if (wParam == TIMER_UPDATE_ID) {
                if (g_is_rendering.load()) {
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = now - g_render_start_time;

                    if (g_render_algo == 1) {
                        int done = g_cuda_done_spp.load();
                        int total = g_cuda_total_spp.load();
                        SendMessage(g_hProgressBar, PBM_SETPOS, done, 0);

                        int percent = (total > 0) ? (done * 100 / total) : 0;
                        wchar_t status[220];
                        swprintf(status, 220, L"Renderowanie [GPU CUDA: %s]... %d%% (%d/%d SPP) [czas: %.2f s]",
                                 g_cuda_device_name.empty() ? L"NVIDIA GPU" : std::wstring(g_cuda_device_name.begin(), g_cuda_device_name.end()).c_str(),
                                 percent, done, total, elapsed.count());
                        SetWindowTextW(g_hStatusLabel, status);
                    } else if (g_render_algo == 3) {
                        size_t done = g_mlt_mutations_done.load();
                        size_t total = g_mlt_total_mutations.load();
                        int pos = (total > 0) ? static_cast<int>(done * 1000 / total) : 0;
                        int percent = (total > 0) ? static_cast<int>(done * 100 / total) : 0;
                        SendMessage(g_hProgressBar, PBM_SETPOS, pos, 0);

                        size_t total_pixels = static_cast<size_t>(g_render_width) * g_render_height;
                        if (done > 0 && g_mlt_accum.size() >= total_pixels) {
                            float scale = (g_mlt_fave * total_pixels) / static_cast<float>(done);
                            std::lock_guard<std::mutex> lock(g_mlt_mutex);
                            for (size_t i = 0; i < total_pixels; ++i) {
                                Vec3 rad = g_mlt_accum[i] * scale;
                                g_raw_image[i] = rad;
                                g_sample_counts[i] = 1;

                                float ev = g_exposure_ev.load();
                                int tm_mode = g_tonemap_mode.load();
                                Vec3 srgb = ApplyExposureAndToneMap(rad, ev, tm_mode);

                                uint8_t ir = static_cast<uint8_t>(std::clamp(srgb.x * 255.0f, 0.0f, 255.0f));
                                uint8_t ig = static_cast<uint8_t>(std::clamp(srgb.y * 255.0f, 0.0f, 255.0f));
                                uint8_t ib = static_cast<uint8_t>(std::clamp(srgb.z * 255.0f, 0.0f, 255.0f));

                                g_display_buffer[i] = (ib) | (ig << 8) | (ir << 16) | (0xFF << 24);
                            }
                        }

                        const wchar_t* algoLabel = (g_render_algo == 4) ? L"Manifold Exploration MEMLT" : L"Metropolis Light Transport";
                        wchar_t status[200];
                        swprintf(status, 200, L"Renderowanie [%s]... %d%% (%zu/%zu mutacji) [czas: %.1f s]",
                                 algoLabel, percent, done, total, elapsed.count());
                        SetWindowTextW(g_hStatusLabel, status);
                    } else {
                        size_t done = g_completed_tiles.load();
                        SendMessage(g_hProgressBar, PBM_SETPOS, done, 0);

                        int percent = (g_total_tiles > 0) ? static_cast<int>(done * 100 / g_total_tiles) : 0;
                        const wchar_t* modeTag = (g_render_algo == 2) ? L"BDPT + mapa fotonów" : ((g_render_algo == 4) ? L"PT + Manifold NEE" : L"PT (CPU)");

                        wchar_t status[180];
                        if (g_render_algo == 4) {
                            swprintf(status, 180, L"Renderowanie [%s]... %d%% (%zu/%zu kafelków) [czas: %.1f s, sekwencje faset: %d]",
                                     modeTag, percent, done, g_total_tiles, elapsed.count(), g_manifold_sequences.load());
                        } else {
                            swprintf(status, 180, L"Renderowanie [%s]... %d%% (%zu/%zu kafelków) [czas: %.1f s, fotony: %d]",
                                     modeTag, percent, done, g_total_tiles, elapsed.count(), g_photon_count);
                        }
                        SetWindowTextW(g_hStatusLabel, status);
                    }

                    InvalidateRect(g_hCanvas, NULL, FALSE);
                } else {
                    if (g_live_viewport_pt.load() && g_cuda_available && !g_has_rendered_image.load() && !g_viewport_dirty.load()) {
                        if (g_viewport_pt_spp < 256) {
                            RenderLiveViewportFrame(false);
                            InvalidateRect(g_hCanvas, NULL, FALSE);
                        }
                    }
                    if (!IsWindowEnabled(g_hBtnRender)) {
                        EnableWindow(g_hBtnRender, TRUE);
                        EnableWindow(g_hBtnStop, FALSE);
                        EnableWindow(g_hBtnApplyPos, TRUE);
                        EnableWindow(g_hBtnGrab, TRUE);

                        auto now = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> elapsed = now - g_render_start_time;

                        const wchar_t* modeTag = (g_render_algo == 4) ? L"MNEE" : ((g_render_algo == 3) ? L"MLT" : ((g_render_algo == 2) ? L"BDPT" : ((g_render_algo == 1) ? L"GPU CUDA" : L"CPU PT")));
                        const wchar_t* oidnTag = g_is_currently_denoised ? L" [Intel OIDN: włączony]" : L" [OIDN: wyłączony]";
                        wchar_t status[220];

                        if (g_render_algo == 1) {
                            int done = g_cuda_done_spp.load();
                            int total = g_cuda_total_spp.load();
                            SendMessage(g_hProgressBar, PBM_SETPOS, total, 0);
                            if (g_cancel_render.load()) {
                                swprintf(status, 220, L"[GPU CUDA] Renderowanie przerwane na %d/%d SPP.", done, total);
                            } else {
                                swprintf(status, 220, L"[GPU CUDA] Renderowanie ukończone w %.2f s! Rozdzielczość: %dx%d px (%d SPP)%s.",
                                         elapsed.count(), g_render_width, g_render_height, total, oidnTag);
                            }
                        } else if (g_render_algo == 3) {
                            size_t done = g_mlt_mutations_done.load();
                            size_t total = g_mlt_total_mutations.load();
                            SendMessage(g_hProgressBar, PBM_SETPOS, 1000, 0);
                            if (g_cancel_render.load()) {
                                swprintf(status, 220, L"[%s] Renderowanie przerwane na %zu/%zu mutacjach.", modeTag, done, total);
                            } else {
                                swprintf(status, 220, L"[%s] Renderowanie ukończone w %.2f s! (%zu mutacji, %dx%d px)%s.",
                                         modeTag, elapsed.count(), done, g_render_width, g_render_height, oidnTag);
                            }
                        } else {
                            size_t done = g_completed_tiles.load();
                            SendMessage(g_hProgressBar, PBM_SETPOS, done, 0);
                            if (g_cancel_render.load()) {
                                swprintf(status, 220, L"[%s] Renderowanie przerwane na kafelku %zu/%zu.", modeTag, done, g_total_tiles);
                            } else {
                                swprintf(status, 220, L"[%s] Renderowanie ukończone w %.2f s! Rozdzielczość: %dx%d px%s.",
                                         modeTag, elapsed.count(), g_render_width, g_render_height, oidnTag);
                            }
                        }

                        SetWindowTextW(g_hStatusLabel, status);
                        InvalidateRect(g_hCanvas, NULL, FALSE);
                    }
                }
            }
            break;
        }

        case WM_COMMAND: {
            int wmId = LOWORD(wParam);
            int wmEvent = HIWORD(wParam);

            if (wmId == IDC_COMBO_SCENE && wmEvent == CBN_SELCHANGE) {
                int sel = static_cast<int>(SendMessage(g_hComboScene, CB_GETCURSEL, 0, 0));
                SwitchScene(sel);
            }
            else if (wmId == IDC_BTN_IMPORT_OBJ) {
                ImportOBJModel();
            }
            else if (wmId == IDC_BTN_EXPORT_OBJ) {
                ExportSceneToOBJ();
            }
            else if (wmId == IDC_BTN_APPLY_MAT) {
                ApplyMaterialFromFields();
            }
            else if (wmId == IDC_COMBO_ALGO && wmEvent == CBN_SELCHANGE) {
                int sel = static_cast<int>(SendMessage(g_hComboAlgo, CB_GETCURSEL, 0, 0));
                g_render_algo = sel;
                if (sel == 3) {
                    SetWindowTextW(g_hLblSPP, L"Mutacje na piksel:");
                    EnableWindow(g_hComboTile, FALSE);
                } else if (sel == 1) {
                    SetWindowTextW(g_hLblSPP, L"Próbki (SPP):");
                    EnableWindow(g_hComboTile, FALSE);
                } else {
                    SetWindowTextW(g_hLblSPP, L"Próbki (SPP):");
                    EnableWindow(g_hComboTile, TRUE);
                }
            }
            else if (wmId == IDC_COMBO_RES && wmEvent == CBN_SELCHANGE) {
                OnResolutionComboChanged();
            }
            else if (wmId == IDC_COMBO_OBJECTS && wmEvent == CBN_SELCHANGE) {
                int sel = static_cast<int>(SendMessage(g_hComboObjects, CB_GETCURSEL, 0, 0));
                g_selected_object_idx = sel;
                UpdateObjectFieldsFromSelection(sel);
                RenderFastPreview();
                InvalidateRect(g_hCanvas, NULL, FALSE);
            }
            else if (wmId == IDC_BTN_APPLY_POS) {
                ApplyPositionFromFields();
            }
            else if (wmId == IDC_BTN_GRAB) {
                if (!g_is_grabbing) StartGrabMode();
                else ConfirmGrabMode();
            }
            else if (wmId == IDC_BTN_RESET_CAM) {
                ResetCamera();
            }
            else if (wmId == IDC_BTN_EV_DOWN) {
                float ev = g_exposure_ev.load() - 0.5f;
                g_exposure_ev.store(ev);
                wchar_t buf[32];
                swprintf(buf, 32, L"%.2f", ev);
                SetWindowTextW(g_hEditExposure, buf);
                UpdateFStopLabel();
                RefreshDisplayBufferFromRaw();
            }
            else if (wmId == IDC_BTN_EV_UP) {
                float ev = g_exposure_ev.load() + 0.5f;
                g_exposure_ev.store(ev);
                wchar_t buf[32];
                swprintf(buf, 32, L"%.2f", ev);
                SetWindowTextW(g_hEditExposure, buf);
                UpdateFStopLabel();
                RefreshDisplayBufferFromRaw();
            }
            else if (wmId == IDC_BTN_EV_RESET) {
                g_exposure_ev.store(0.0f);
                SetWindowTextW(g_hEditExposure, L"0.00");
                UpdateFStopLabel();
                RefreshDisplayBufferFromRaw();
            }
            else if (wmId == IDC_BTN_EV_APPLY || (wmId == IDC_EDIT_EXPOSURE && wmEvent == EN_KILLFOCUS)) {
                wchar_t buf[64];
                GetWindowTextW(g_hEditExposure, buf, 64);
                try {
                    float val = std::stof(buf);
                    g_exposure_ev.store(val);
                    UpdateFStopLabel();
                    RefreshDisplayBufferFromRaw();
                } catch (...) {}
            }
            else if (wmId == IDC_COMBO_TONEMAP && wmEvent == CBN_SELCHANGE) {
                int sel = static_cast<int>(SendMessage(g_hComboToneMap, CB_GETCURSEL, 0, 0));
                g_tonemap_mode.store(sel);
                RefreshDisplayBufferFromRaw();
            }
            else if (wmId == IDC_CHK_DENOISE) {
                LRESULT checked = SendMessage(g_hChkDenoise, BM_GETCHECK, 0, 0);
                g_enable_oidn.store(checked == BST_CHECKED);

                if (!g_unfiltered_raw_image.empty() && !g_is_rendering.load()) {
                    if (g_enable_oidn.load()) {
                        std::vector<Vec3> denoised = g_unfiltered_raw_image;
                        if (RunIntelOIDN(denoised, g_render_width, g_render_height)) {
                            g_raw_image = denoised;
                            g_is_currently_denoised = true;
                        }
                    } else {
                        g_raw_image = g_unfiltered_raw_image;
                        g_is_currently_denoised = false;
                    }
                    RefreshDisplayBufferFromRaw();
                    wchar_t statusBuf[180];
                    swprintf(statusBuf, 180, L"Podgląd przełączony: %s", g_is_currently_denoised ? L"Intel OIDN aktywny" : L"surowy (bez odszumiania)");
                    SetWindowTextW(g_hStatusLabel, statusBuf);
                }
            }
            else if (wmId == IDC_BTN_DENOISE_NOW) {
                if (!g_raw_image.empty() && !g_is_rendering.load()) {
                    if (g_unfiltered_raw_image.empty()) {
                        g_unfiltered_raw_image = g_raw_image;
                    }
                    SetWindowTextW(g_hStatusLabel, L"Odszumianie obrazu przez Intel Open Image Denoise (OIDN)...");
                    UpdateWindow(g_hStatusLabel);
                    std::vector<Vec3> denoised = g_unfiltered_raw_image;
                    if (RunIntelOIDN(denoised, g_render_width, g_render_height)) {
                        g_raw_image = denoised;
                        g_is_currently_denoised = true;
                        SendMessage(g_hChkDenoise, BM_SETCHECK, BST_CHECKED, 0);
                        g_enable_oidn.store(true);
                        RefreshDisplayBufferFromRaw();
                        SetWindowTextW(g_hStatusLabel, L"Odszumianie ukończone pomyślnie! (Intel OIDN)");
                    } else {
                        SetWindowTextW(g_hStatusLabel, L"Błąd podczas odszumiania OIDN.");
                    }
                }
            }
            else if (wmId == IDC_CHK_CUDA_CAUSTICS) {
                LRESULT checked = SendMessage(g_hChkCudaCaustics, BM_GETCHECK, 0, 0);
                bool enable = (checked == BST_CHECKED);
                g_cuda_caustics.store(enable);
                g_viewport_pt_spp = 0;
                CudaPT_ResetViewportAccum();
                if (g_live_viewport_pt.load()) {
                    RenderLiveViewportFrame(false);
                    InvalidateRect(g_hCanvas, NULL, FALSE);
                }
                if (enable) {
                    SetWindowTextW(g_hStatusLabel, L"Kaustyka GPU CUDA: Włączona (ostre wstęgi skupień i dyspersja tęczowa).");
                } else {
                    SetWindowTextW(g_hStatusLabel, L"Kaustyka GPU CUDA: Wyłączona (transparentne cienie szkła – brak czarnej plamy).");
                }
            }
            else if (wmId == IDC_BTN_CAUSTIC_APPLY || (wmId == IDC_COMBO_CAUSTIC_GRID && wmEvent == CBN_SELCHANGE)) {
                auto readFloat = [](HWND hEdit, float def, float lo, float hi) {
                    wchar_t buf[64];
                    GetWindowTextW(hEdit, buf, 64);
                    for (wchar_t* c = buf; *c; ++c) if (*c == L',') *c = L'.';
                    float val = def;
                    try { val = std::stof(buf); } catch (...) {}
                    val = std::clamp(val, lo, hi);
                    wchar_t out[32];
                    swprintf(out, 32, hi <= 0.5f ? L"%.3f" : L"%.2f", val);
                    SetWindowTextW(hEdit, out);
                    return val;
                };
                g_caustic_blur_cm.store(readFloat(g_hEditCausticBlur, 0.25f, 0.0f, 5.0f));
                g_caustic_light_size.store(readFloat(g_hEditCausticLight, 0.12f, 0.0f, 1.0f));
                g_caustic_gain.store(readFloat(g_hEditCausticGain, 1.0f, 0.0f, 20.0f));
                g_caustic_cpu_radius.store(readFloat(g_hEditCausticRadius, 0.028f, 0.004f, 0.3f));
                static const int grid_sizes[3] = {1024, 2048, 4096};
                int gsel = std::clamp(static_cast<int>(SendMessage(g_hComboCausticGrid, CB_GETCURSEL, 0, 0)), 0, 2);
                g_caustic_grid_res.store(grid_sizes[gsel]);

                wchar_t pbuf[32];
                GetWindowTextW(g_hEditPhotons, pbuf, 32);
                int photons = wcstol(pbuf, nullptr, 10);
                if (photons >= 1000) g_photon_count = std::min(photons, 8000000);

                g_viewport_pt_spp = 0;
                CudaPT_ResetViewportAccum();
                if (g_live_viewport_pt.load()) {
                    RenderLiveViewportFrame(false);
                    InvalidateRect(g_hCanvas, NULL, FALSE);
                }
                wchar_t st[220];
                swprintf(st, 220, L"Kaustyka: rozmycie %.2f cm | źródło %.2f | jasność %.2fx | siatka GPU %d | promień CPU %.3f m (CPU: widoczne po ponownym renderze).",
                         g_caustic_blur_cm.load(), g_caustic_light_size.load(), g_caustic_gain.load(), g_caustic_grid_res.load(), g_caustic_cpu_radius.load());
                SetWindowTextW(g_hStatusLabel, st);
            }
            else if (wmId == IDC_CHK_CAUSTICS_HQ) {
                LRESULT checked = SendMessage(g_hChkCausticsHQ, BM_GETCHECK, 0, 0);
                bool enable = (checked == BST_CHECKED);
                g_cuda_caustics_hq.store(enable);
                g_caustic_grid_res.store(enable ? 2048 : 1024);
                SendMessage(g_hComboCausticGrid, CB_SETCURSEL, enable ? 1 : 0, 0);
                g_viewport_pt_spp = 0;
                CudaPT_ResetViewportAccum();
                if (enable) {
                    SetWindowTextW(g_hStatusLabel, L"Tryb kaustyki dla renderu: Precyzyjny HD (siatka 2048x2048, 1M fotonów).");
                } else {
                    SetWindowTextW(g_hStatusLabel, L"Tryb kaustyki dla renderu: Szybki standardowy (siatka 1024x1024).");
                }
            }
            else if (wmId == IDC_CHK_LIVE_VIEWPORT_PT) {
                LRESULT checked = SendMessage(g_hChkLiveViewportPT, BM_GETCHECK, 0, 0);
                bool enable = (checked == BST_CHECKED);
                g_live_viewport_pt.store(enable);
                g_viewport_pt_spp = 0;
                if (enable && g_cuda_available) {
                    RenderLiveViewportFrame(true);
                    SetWindowTextW(g_hStatusLabel, L"Włączono podgląd Path Tracing na żywo (GPU RTX). [Z] przełącza tryb.");
                } else {
                    CudaPT_ResetViewportAccum();
                    RenderFastPreview();
                    SetWindowTextW(g_hStatusLabel, L"Przełączono na podgląd bryłowy (Workbench).");
                    if (g_hLblVpSpp) SetWindowTextW(g_hLblVpSpp, L"VP: Wyłączony");
                }
                InvalidateRect(g_hCanvas, NULL, FALSE);
            }
            else if (wmId == IDC_BTN_LIGHT_DOWN) {
                UpdateGlobalLightMultiplier(g_global_light_mult.load() - 0.25f);
            }
            else if (wmId == IDC_BTN_LIGHT_UP) {
                UpdateGlobalLightMultiplier(g_global_light_mult.load() + 0.25f);
            }
            else if (wmId == IDC_BTN_LIGHT_HALF) {
                UpdateGlobalLightMultiplier(g_global_light_mult.load() * 0.5f);
            }
            else if (wmId == IDC_BTN_LIGHT_DOUBLE) {
                UpdateGlobalLightMultiplier(g_global_light_mult.load() * 2.0f);
            }
            else if (wmId == IDC_BTN_LIGHT_RESET) {
                UpdateGlobalLightMultiplier(1.0f);
            }
            else if (wmId == IDC_BTN_LIGHT_APPLY || (wmId == IDC_EDIT_LIGHT_POWER && wmEvent == EN_KILLFOCUS)) {
                wchar_t buf[64];
                GetWindowTextW(g_hEditLightPower, buf, 64);
                try {
                    float val = std::stof(buf);
                    UpdateGlobalLightMultiplier(val);
                } catch (...) {}
            }
            else if (wmId == IDC_BTN_RENDER) {
                StartRendering();
            }
            else if (wmId == IDC_BTN_STOP) {
                StopRendering();
                RenderFastPreview();
                InvalidateRect(g_hCanvas, NULL, FALSE);
            }
            else if (wmId == IDC_BTN_SAVE_PNG) {
                SaveToPNG();
            }
            break;
        }

        case WM_DESTROY:
            KillTimer(hWnd, TIMER_UPDATE_ID);
            StopRendering();
            PostQuitMessage(0);
            break;

        default:
            return DefWindowProc(hWnd, msg, wParam, lParam);
    }
    return 0;
}

// ============================================================================
// 10. PUNKT WEJŚCIA PROGRAMU (WinMain)
// ============================================================================

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    INITCOMMONCONTROLSEX icex;
    icex.dwSize = sizeof(INITCOMMONCONTROLSEX);
    icex.dwICC = ICC_STANDARD_CLASSES | ICC_PROGRESS_CLASS;
    InitCommonControlsEx(&icex);

    WNDCLASSW wc = {};
    wc.lpfnWndProc = MainWndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"PathTracerMainWindow";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);

    RegisterClassW(&wc);

    int winWidth  = 1320;
    int winHeight = 720;

    g_hWnd = CreateWindowW(
        L"PathTracerMainWindow",
        L"Path Tracer 3D - Blender Viewport, Manifold Exploration Room & OBJ Importer",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        winWidth, winHeight,
        nullptr, nullptr, hInstance, nullptr
    );

    if (!g_hWnd) return 0;

    ShowWindow(g_hWnd, nCmdShow);
    UpdateWindow(g_hWnd);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        if (msg.message == WM_KEYDOWN) {
            HWND hFocus = GetFocus();
            bool isEditing = (hFocus == g_hEditPosX || hFocus == g_hEditPosY || hFocus == g_hEditPosZ ||
                              hFocus == g_hEditRadius || hFocus == g_hEditFilename || hFocus == g_hEditPhotons ||
                              hFocus == g_hEditSPP || hFocus == g_hEditDepth || hFocus == g_hEditResW || hFocus == g_hEditResH ||
                              hFocus == g_hEditMatR || hFocus == g_hEditMatG || hFocus == g_hEditMatB || hFocus == g_hEditMatParam ||
                              hFocus == g_hEditExposure || hFocus == g_hEditLightPower ||
                              hFocus == g_hEditCausticBlur || hFocus == g_hEditCausticLight || hFocus == g_hEditCausticGain ||
                              hFocus == g_hEditCausticRadius);

            if (!isEditing && !g_is_grabbing && (msg.wParam == 'Z' || msg.wParam == 'z')) {
                ToggleLiveViewportPT();
                continue;
            }

            if (msg.wParam == 'G' || msg.wParam == 'g' ||
                msg.wParam == 'S' || msg.wParam == 's' ||
                msg.wParam == 'F' || msg.wParam == 'f' ||
                (g_is_grabbing && (msg.wParam == 'X' || msg.wParam == 'x' ||
                                   msg.wParam == 'Y' || msg.wParam == 'y' ||
                                   msg.wParam == 'Z' || msg.wParam == 'z' ||
                                   msg.wParam == VK_ESCAPE || msg.wParam == VK_RETURN)) ||
                (g_is_scaling  && (msg.wParam == VK_ESCAPE || msg.wParam == VK_RETURN))) {
                if (!isEditing || g_is_grabbing || g_is_scaling) {
                    SendMessage(g_hCanvas, WM_KEYDOWN, msg.wParam, msg.lParam);
                    continue;
                }
            }
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return (int)msg.wParam;
}
