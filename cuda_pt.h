#pragma once
#include <string>
#include <vector>
#include <functional>
#include <atomic>

// Struktura wektora 3D dla transferu danych CPU-GPU
struct CudaVec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    CudaVec3() = default;
    CudaVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// Typ materiału
enum CudaMaterialType {
    CUDA_MAT_LAMBERTIAN = 0,
    CUDA_MAT_METAL      = 1,
    CUDA_MAT_DIELECTRIC = 2,
    CUDA_MAT_LIGHT      = 3
};

// Sfera w przestrzeni 3D dla GPU
struct CudaSphere {
    CudaVec3 center;
    float radius = 1.0f;
    int mat_type = CUDA_MAT_LAMBERTIAN;
    CudaVec3 color = CudaVec3(0.8f, 0.8f, 0.8f);
    float param = 0.05f; // Roughness (Metal), IOR (Dielectric), Intensity (Light)
};

// Płaszczyzna tnąca dla szlifowanego kryształu (Slab Method)
struct CudaPlane {
    CudaVec3 n;
    float d = 0.0f;
};

// Czworokąt / panel ścienny (Quad)
struct CudaQuad {
    CudaVec3 q;
    CudaVec3 u;
    CudaVec3 v;
    CudaVec3 normal;
    float d = 0.0f;
    CudaVec3 w;
    int mat_type = CUDA_MAT_LAMBERTIAN;
    CudaVec3 color = CudaVec3(0.8f, 0.8f, 0.8f);
    float param = 0.05f;
};

// Szlifowany kryształ szklany (ConvexPolyhedron z 20 fasetami)
struct CudaCrystal {
    CudaVec3 center;
    float bounding_radius = 1.0f;
    int plane_start = 0;
    int plane_count = 0;
    int mat_type = CUDA_MAT_DIELECTRIC;
    CudaVec3 color = CudaVec3(1.0f, 1.0f, 1.0f);
    float param = 1.55f; // IOR
    float dispersion = 0.02f; // Cauchy dispersion (rozszczepienie tęczowe światła)
};

// Trójkąt siatki mesh (z importu OBJ)
struct CudaTriangle {
    CudaVec3 v0;
    CudaVec3 v1;
    CudaVec3 v2;
    CudaVec3 normal;
};

// Nagłówek zaimportowanej siatki mesh (AABB culling + trójkąty)
struct CudaMesh {
    CudaVec3 bmin;
    CudaVec3 bmax;
    int tri_start = 0;
    int tri_count = 0;
    int mat_type = CUDA_MAT_LAMBERTIAN;
    CudaVec3 color = CudaVec3(0.8f, 0.8f, 0.8f);
    float param = 0.05f;
};

// Kompletna struktura wielogeometrycznej sceny dla GPU CUDA
struct CudaSceneData {
    std::vector<CudaSphere>   spheres;
    std::vector<CudaQuad>     quads;
    std::vector<CudaCrystal>  crystals;
    std::vector<CudaPlane>    crystal_planes;
    std::vector<CudaTriangle> triangles;
    std::vector<CudaMesh>     meshes;
};

// Opcje renderingu ścieżkowego w stylu Cycles (F-stopy, Clamping fireflies, Tone mapping, Kaustyka GPU)
struct CudaRenderOptions {
    float exposure_ev = -2.0f;
    int tonemap_mode = 0; // 0=ACES Filmic HDR, 1=Reinhard, 2=Linear Clamp
    float clamp_indirect = 12.0f; // Cycles firefly suppression (tłumienie białych kropek)
    int enable_caustics = 1; // 1 = Włączona kaustyka GPU, 0 = Wyłączona (Transparentne cienie szkła, brak czarnej plamy)
    int caustic_photons = 262144; // Liczba fotonów kaustyki GPU z dyspersją Cauchy'ego
    int caustic_hq = 1; // 1 = Wysoka precyzja (2048x2048 HD, ostre wstęgi), 0 = Szybki podgląd (1024x1024)
    int scene_mode = 0; // 0 = Scena otwarta, 1 = Pokój ze zwierciadłem
};

// Sprawdzenie dostępności urządzenia NVIDIA CUDA
bool CudaPT_IsAvailable(std::string& out_device_name);

// Callback postępu i podglądu klatki na żywo:
// done_spp: ile próbek już zrobiono
// total_spp: ile próbek zaplanowano
// accum_pixels: surowy bufor sumy z GPU (dla końcowego zapisu HDR lub nullptr)
// display_bgra: natywny, zmapowany na GPU bufor 32-bit BGRA (zero narzutu CPU!)
using CudaProgressCallback = std::function<void(
    int done_spp, int total_spp,
    const CudaVec3* accum_pixels,
    const uint32_t* display_bgra
)>;

// Uruchomienie śledzenia ścieżek na GPU z obsługą WSZYSTKICH kształtów:
// Kule (Spheres), Czworokąty (Quads), Szlifowane kryształy (Crystals), Siatki trójkątów OBJ (Meshes)
// z algorytmem w stylu Cycles: Next Event Estimation, Firefly Clamping, Cauchy Dispersion i pełną saturacją SM
bool CudaPT_Render(
    int width, int height, int spp, int max_depth,
    const CudaVec3& cam_origin, const CudaVec3& cam_lookat, const CudaVec3& cam_vup, float cam_fov,
    const CudaSceneData& scene,
    std::vector<CudaVec3>& out_pixels,
    const CudaRenderOptions& options = CudaRenderOptions(),
    std::atomic<bool>* cancel_flag = nullptr,
    CudaProgressCallback progress_cb = nullptr
);

// Przeciążenie dla kompatybilności z samymi kulami
bool CudaPT_Render(
    int width, int height, int spp, int max_depth,
    const CudaVec3& cam_origin, const CudaVec3& cam_lookat, const CudaVec3& cam_vup, float cam_fov,
    const std::vector<CudaSphere>& spheres,
    std::vector<CudaVec3>& out_pixels,
    const CudaRenderOptions& options = CudaRenderOptions(),
    std::atomic<bool>* cancel_flag = nullptr,
    CudaProgressCallback progress_cb = nullptr
);

// ============================================================================
// Interaktywny podgląd Path Tracingu w viewporcie na żywo (GPU CUDA / RTX)
// Płynna akumulacja progresywna w czasie rzeczywistym (60+ FPS podczas ruchu)
// ============================================================================
bool CudaPT_RenderViewportStep(
    int width, int height,
    int spp_step, int& inout_accum_spp,
    const CudaVec3& cam_origin, const CudaVec3& cam_lookat, const CudaVec3& cam_vup, float cam_fov,
    const CudaSceneData& scene,
    const CudaRenderOptions& options,
    uint32_t* out_display_bgra
);

void CudaPT_ResetViewportAccum();
void CudaPT_CleanupViewport();
