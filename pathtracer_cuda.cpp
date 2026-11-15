#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "cuda_pt.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Pomocnicze wektory i funkcje geometrii
struct LocalVec3 {
    float x, y, z;
    LocalVec3() : x(0), y(0), z(0) {}
    LocalVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    LocalVec3 operator-(const LocalVec3& v) const { return LocalVec3(x - v.x, y - v.y, z - v.z); }
    LocalVec3 operator+(const LocalVec3& v) const { return LocalVec3(x + v.x, y + v.y, z + v.z); }
    LocalVec3 operator*(float t) const { return LocalVec3(x * t, y * t, z * t); }
    LocalVec3 operator/(float t) const { return LocalVec3(x / t, y / t, z / t); }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
};

static inline float dot_v(const LocalVec3& u, const LocalVec3& v) { return u.x * v.x + u.y * v.y + u.z * v.z; }
static inline LocalVec3 cross_v(const LocalVec3& u, const LocalVec3& v) {
    return LocalVec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}
static inline LocalVec3 unit_v(const LocalVec3& v) {
    float l = v.length();
    return l > 1e-8f ? v / l : LocalVec3(0, 0, 0);
}

// Scena 0: Szlifowany kryształ 20-fasetowy, kula złota, rubinowa, chromowana i światło
static CudaSceneData BuildSceneCrystalSpheres() {
    CudaSceneData scene;
    // Podłoże
    scene.spheres.push_back({
        CudaVec3(0.0f, -1000.0f, 0.0f), 1000.0f,
        CUDA_MAT_LAMBERTIAN, CudaVec3(0.85f, 0.85f, 0.88f), 0.0f
    });

    // 1. Kryształ szklany (ConvexPolyhedron z 20 fasetami)
    float radius = 0.65f;
    float height = 1.40f;
    float half_h = height * 0.5f;
    float b_rad = std::sqrt(radius * radius + half_h * half_h) * 1.18f;

    CudaCrystal cc;
    cc.center = CudaVec3(-0.5f, 0.70f, 0.1f);
    cc.bounding_radius = b_rad;
    cc.plane_start = 0;
    cc.plane_count = 20;
    cc.mat_type = CUDA_MAT_DIELECTRIC;
    cc.color = CudaVec3(1.0f, 1.0f, 1.0f);
    cc.param = 1.55f;
    cc.dispersion = 0.02f;

    // Dno i szczyt
    scene.crystal_planes.push_back({ CudaVec3(0.0f, -1.0f, 0.0f), half_h });
    scene.crystal_planes.push_back({ CudaVec3(0.0f, 1.0f, 0.0f), half_h });

    // 6 pasów pionowych
    for (int i = 0; i < 6; ++i) {
        float angle = i * (3.14159265f / 3.0f);
        scene.crystal_planes.push_back({ CudaVec3(std::cos(angle), 0.0f, std::sin(angle)), radius });
    }

    // 6 faset korony
    for (int i = 0; i < 6; ++i) {
        float angle = (i + 0.5f) * (3.14159265f / 3.0f);
        LocalVec3 slanted = unit_v(LocalVec3(std::cos(angle) * 0.85f, 1.15f, std::sin(angle) * 0.85f));
        scene.crystal_planes.push_back({ CudaVec3(slanted.x, slanted.y, slanted.z), radius * 1.08f });
    }

    // 6 faset podstawy
    for (int i = 0; i < 6; ++i) {
        float angle = i * (3.14159265f / 3.0f);
        LocalVec3 slanted = unit_v(LocalVec3(std::cos(angle) * 0.80f, -1.25f, std::sin(angle) * 0.80f));
        scene.crystal_planes.push_back({ CudaVec3(slanted.x, slanted.y, slanted.z), radius * 1.18f });
    }

    scene.crystals.push_back(cc);

    // Kule towarzyszące
    scene.spheres.push_back({ CudaVec3(3.2f, 2.4f, -1.2f),   0.85f, CUDA_MAT_LIGHT,      CudaVec3(35.0f, 32.0f, 28.0f), 1.0f });
    scene.spheres.push_back({ CudaVec3(1.7f, 0.80f, 0.6f),   0.80f, CUDA_MAT_METAL,      CudaVec3(0.95f, 0.78f, 0.35f), 0.02f });
    scene.spheres.push_back({ CudaVec3(-2.0f, 0.65f, -0.6f), 0.65f, CUDA_MAT_LAMBERTIAN, CudaVec3(0.85f, 0.15f, 0.15f), 0.0f });
    scene.spheres.push_back({ CudaVec3(0.7f, 0.35f, 1.7f),   0.35f, CUDA_MAT_METAL,      CudaVec3(0.92f, 0.95f, 0.98f), 0.05f });

    return scene;
}

// Scena 1: Zamknięty pokój z panelami ściennymi Quad, oknem, lustrem zewnętrznym i kryształem
static CudaSceneData BuildSceneRoomQuads() {
    CudaSceneData scene;

    // 1. Kryształ szklany
    float radius = 0.50f;
    float height = 1.10f;
    float half_h = height * 0.5f;
    float b_rad = std::sqrt(radius * radius + half_h * half_h) * 1.18f;

    CudaCrystal cc;
    cc.center = CudaVec3(0.0f, 0.55f, 0.2f);
    cc.bounding_radius = b_rad;
    cc.plane_start = 0;
    cc.plane_count = 20;
    cc.mat_type = CUDA_MAT_DIELECTRIC;
    cc.color = CudaVec3(1.0f, 1.0f, 1.0f);
    cc.param = 1.55f;
    cc.dispersion = 0.02f;

    scene.crystal_planes.push_back({ CudaVec3(0.0f, -1.0f, 0.0f), half_h });
    scene.crystal_planes.push_back({ CudaVec3(0.0f, 1.0f, 0.0f), half_h });
    for (int i = 0; i < 6; ++i) {
        float angle = i * (3.14159265f / 3.0f);
        scene.crystal_planes.push_back({ CudaVec3(std::cos(angle), 0.0f, std::sin(angle)), radius });
    }
    for (int i = 0; i < 6; ++i) {
        float angle = (i + 0.5f) * (3.14159265f / 3.0f);
        LocalVec3 slanted = unit_v(LocalVec3(std::cos(angle) * 0.85f, 1.15f, std::sin(angle) * 0.85f));
        scene.crystal_planes.push_back({ CudaVec3(slanted.x, slanted.y, slanted.z), radius * 1.08f });
    }
    for (int i = 0; i < 6; ++i) {
        float angle = i * (3.14159265f / 3.0f);
        LocalVec3 slanted = unit_v(LocalVec3(std::cos(angle) * 0.80f, -1.25f, std::sin(angle) * 0.80f));
        scene.crystal_planes.push_back({ CudaVec3(slanted.x, slanted.y, slanted.z), radius * 1.18f });
    }
    scene.crystals.push_back(cc);

    // 2. Kula złota
    scene.spheres.push_back({ CudaVec3(0.75f, 0.35f, 0.4f), 0.35f, CUDA_MAT_METAL, CudaVec3(0.95f, 0.78f, 0.35f), 0.03f });
    // 3. Zewnętrzne źródło światła
    scene.spheres.push_back({ CudaVec3(2.2f, 1.15f, 2.1f), 0.40f, CUDA_MAT_LIGHT, CudaVec3(70.0f, 65.0f, 55.0f), 1.0f });

    auto add_quad = [&](LocalVec3 corner, LocalVec3 u, LocalVec3 v, int mat, CudaVec3 col, float param) {
        LocalVec3 n = cross_v(u, v);
        LocalVec3 norm = unit_v(n);
        float d = dot_v(norm, corner);
        float len_sq = dot_v(n, n);
        LocalVec3 w = len_sq > 1e-8f ? (n / len_sq) : LocalVec3(0, 0, 0);

        CudaQuad q;
        q.q = CudaVec3(corner.x, corner.y, corner.z);
        q.u = CudaVec3(u.x, u.y, u.z);
        q.v = CudaVec3(v.x, v.y, v.z);
        q.normal = CudaVec3(norm.x, norm.y, norm.z);
        q.d = d;
        q.w = CudaVec3(w.x, w.y, w.z);
        q.mat_type = mat;
        q.color = col;
        q.param = param;
        scene.quads.push_back(q);
    };

    // Zewnętrzne lustro odbijające
    add_quad(LocalVec3(-2.0f, 0.0f, 3.2f), LocalVec3(4.0f, 0.0f, 0.0f), LocalVec3(0.0f, 2.4f, 0.0f), CUDA_MAT_METAL, CudaVec3(0.98f, 0.98f, 0.98f), 0.01f);
    // Baffle
    add_quad(LocalVec3(1.35f, 0.0f, 1.4f), LocalVec3(0.0f, 0.0f, 1.4f), LocalVec3(0.0f, 2.4f, 0.0f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.2f, 0.2f, 0.25f), 0.0f);
    // Ściana z oknem (dół, góra, lewo, prawo)
    add_quad(LocalVec3(-2.0f, 0.0f, 1.4f), LocalVec3(4.0f, 0.0f, 0.0f), LocalVec3(0.0f, 0.95f, 0.0f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.72f, 0.70f, 0.68f), 0.0f);
    add_quad(LocalVec3(-2.0f, 1.35f, 1.4f), LocalVec3(4.0f, 0.0f, 0.0f), LocalVec3(0.0f, 1.05f, 0.0f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.72f, 0.70f, 0.68f), 0.0f);
    add_quad(LocalVec3(-2.0f, 0.95f, 1.4f), LocalVec3(0.8f, 0.0f, 0.0f), LocalVec3(0.0f, 0.40f, 0.0f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.72f, 0.70f, 0.68f), 0.0f);
    add_quad(LocalVec3(1.2f, 0.95f, 1.4f), LocalVec3(0.8f, 0.0f, 0.0f), LocalVec3(0.0f, 0.40f, 0.0f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.72f, 0.70f, 0.68f), 0.0f);
    // Pokój (podłoga, sufit, ściany)
    add_quad(LocalVec3(-2.0f, 0.0f, -2.0f), LocalVec3(4.0f, 0.0f, 0.0f), LocalVec3(0.0f, 0.0f, 3.4f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.75f, 0.72f, 0.68f), 0.0f);
    add_quad(LocalVec3(-2.0f, 2.4f, -2.0f), LocalVec3(4.0f, 0.0f, 0.0f), LocalVec3(0.0f, 0.0f, 3.4f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.82f, 0.82f, 0.82f), 0.0f);
    add_quad(LocalVec3(-2.0f, 0.0f, -2.0f), LocalVec3(0.0f, 0.0f, 3.4f), LocalVec3(0.0f, 2.4f, 0.0f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.75f, 0.35f, 0.32f), 0.0f);
    add_quad(LocalVec3(2.0f, 0.0f, 1.4f), LocalVec3(0.0f, 0.0f, -3.4f), LocalVec3(0.0f, 2.4f, 0.0f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.32f, 0.65f, 0.42f), 0.0f);
    add_quad(LocalVec3(2.0f, 0.0f, -2.0f), LocalVec3(-4.0f, 0.0f, 0.0f), LocalVec3(0.0f, 2.4f, 0.0f), CUDA_MAT_LAMBERTIAN, CudaVec3(0.68f, 0.68f, 0.72f), 0.0f);

    return scene;
}

int main(int argc, char* argv[]) {
    // 1. Sprawdzenie aktywnej karty graficznej NVIDIA CUDA
    std::string device_name;
    if (!CudaPT_IsAvailable(device_name)) {
        std::cerr << "Blad: Nie znaleziono kompatybilnego urzadzenia NVIDIA CUDA!\n";
        return 1;
    }

    std::cout << "==========================================\n";
    std::cout << "      NVIDIA CUDA GPU PATH TRACER         \n";
    std::cout << "==========================================\n";
    std::cout << "Aktywna karta graficzna: " << device_name << "\n";

    // 2. Parsowanie argumentow wiersza polecen
    int width = 1280;
    int height = 720;
    int spp = 256;
    int max_depth = 32;
    int scene_idx = 0; // 0 = Crystal+Spheres, 1 = Room+Quads
    float exposure_ev = -2.0f;
    int enable_caustics = 1;
    int caustic_hq = 1;
    int caustic_photons = 524288;
    std::string out_filename = "output_cuda.png";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc)         width = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc)   height = std::stoi(argv[++i]);
        else if (arg == "--spp" && i + 1 < argc)      spp = std::stoi(argv[++i]);
        else if (arg == "--depth" && i + 1 < argc)    max_depth = std::stoi(argv[++i]);
        else if (arg == "--scene" && i + 1 < argc)    scene_idx = std::stoi(argv[++i]);
        else if (arg == "--exposure" && i + 1 < argc) exposure_ev = std::stof(argv[++i]);
        else if (arg == "--caustics" && i + 1 < argc) enable_caustics = std::stoi(argv[++i]);
        else if (arg == "--caustic-hq" && i + 1 < argc) caustic_hq = std::stoi(argv[++i]);
        else if (arg == "--photons" && i + 1 < argc)  caustic_photons = std::stoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc)      out_filename = argv[++i];
    }

    CudaSceneData scene;
    CudaVec3 lookfrom, lookat, vup(0.0f, 1.0f, 0.0f);
    float vfov = 38.0f;

    if (scene_idx == 0) {
        std::cout << "Wybrana scena: 1. Krysztal szklany i kule (Faceted Crystal + Spheres)\n";
        scene = BuildSceneCrystalSpheres();
        // Kamera dla sceny 1
        lookfrom = CudaVec3(-0.3f, 2.5f, 4.2f);
        lookat   = CudaVec3(-0.3f, 0.5f, 0.0f);
        vfov = 38.0f;
    } else {
        std::cout << "Wybrana scena: 2. Zamkniety pokoj z oknem, lustrem i krysztalem (Quads + Crystal)\n";
        scene = BuildSceneRoomQuads();
        // Kamera dla sceny 2 (wewnątrz pokoju skierowana na okno i kryształ)
        lookfrom = CudaVec3(0.0f, 0.85f, -1.3f);
        lookat   = CudaVec3(0.0f, 0.85f, 0.6f);
        vfov = 45.0f;
    }

    std::cout << "Geometria sceny: " 
              << scene.spheres.size() << " kul, "
              << scene.quads.size() << " czworokatow (quads), "
              << scene.crystals.size() << " krysztalow (" << scene.crystal_planes.size() << " faset), "
              << scene.meshes.size() << " siatek OBJ\n";
    std::cout << "Rozdzielczosc: " << width << "x" << height << "\n";
    std::cout << "Probki na piksel (SPP): " << spp << "\n";
    std::cout << "Maksymalna liczba odbic: " << max_depth << "\n";
    std::cout << "Ekspozycja (EV / F-stopy): " << exposure_ev << " EV\n";
    std::cout << "Tone Mapping: ACES Filmic HDR (16+ F-stopow)\n";
    std::cout << "Plik wyjsciowy: " << out_filename << "\n\n";

    std::vector<CudaVec3> pixels;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 5. Uruchomienie renderingu na GPU
    std::cout << "Uruchamianie kernela CUDA na karcie " << device_name << "...\n";
    CudaRenderOptions opts;
    opts.exposure_ev = exposure_ev;
    opts.tonemap_mode = 0; // ACES Filmic HDR
    opts.clamp_indirect = 12.0f; // Cycles firefly suppression
    opts.enable_caustics = enable_caustics;
    opts.caustic_hq = caustic_hq;
    opts.caustic_photons = caustic_photons;
    opts.scene_mode = scene_idx;

    bool success = CudaPT_Render(
        width, height, spp, max_depth,
        lookfrom, lookat, vup, vfov,
        scene, pixels,
        opts,
        nullptr,
        [&](int done_spp, int total_spp, const CudaVec3* /*accum_pixels*/, const uint32_t* /*display_bgra*/) {
            int pct = (done_spp * 100) / total_spp;
            std::cout << "\rPostep GPU: [" << pct << "%] (" << done_spp << "/" << total_spp << " SPP)" << std::flush;
        }
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nRenderowanie na GPU zakonczone w " << elapsed.count() << " s!\n";

    if (!success || pixels.empty()) {
        std::cerr << "Blad podczas renderingu GPU!\n";
        return 1;
    }

    // 6. Zapis obrazu do pliku PNG z ACES Filmic Tone Mapping i korekcją F-stopów
    auto aces_film = [](float c) {
        float a = 2.51f, b = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
        return std::clamp((c * (a * c + b)) / (c * (cc * c + d) + e), 0.0f, 1.0f);
    };

    float scale = std::pow(2.0f, exposure_ev);
    constexpr float inv_gamma = 1.0f / 2.2f;

    std::vector<uint8_t> out_bytes(width * height * 3);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            const CudaVec3& col = pixels[j * width + i];

            float r = std::pow(aces_film(std::max(0.0f, col.x * scale)), inv_gamma);
            float g = std::pow(aces_film(std::max(0.0f, col.y * scale)), inv_gamma);
            float b = std::pow(aces_film(std::max(0.0f, col.z * scale)), inv_gamma);

            int dst_idx = (j * width + i) * 3;
            out_bytes[dst_idx + 0] = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
            out_bytes[dst_idx + 1] = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
            out_bytes[dst_idx + 2] = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));
        }
    }

    std::cout << "Zapisywanie obrazu do '" << out_filename << "'...";
    if (stbi_write_png(out_filename.c_str(), width, height, 3, out_bytes.data(), width * 3)) {
        std::cout << " OK!\n";
    } else {
        std::cerr << " BLAD zapisu PNG!\n";
        return 1;
    }

    return 0;
}
