#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <fstream>
#include "gpu.h"
using namespace std;


// --- Twin-Turbo 5.2L V10 ----
struct Engine {
    GLFWwindow* window;
    int WIDTH = 800, HEIGHT = 600;
    double prev = 0.0;
    float yaw = 0.5f, pitch = 0.3f, dist = 2.0f;
    double mx = 0, my = 0;

    Engine() {
        // wake up glfw
        if (!glfwInit()) {
            cerr << "glfw died lol" << endl;
            exit(EXIT_FAILURE);
        }

        //window
        window = glfwCreateWindow(WIDTH, HEIGHT, "subscribe to kavan :D!", nullptr, nullptr);
        if (!window) {
            cerr << "no window for u lol" << endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        glfwMakeContextCurrent(window);
        glewInit();

        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);
        prev = glfwGetTime();

        glfwSetWindowUserPointer(window, this);
        glfwSetScrollCallback(window, [](GLFWwindow* w, double, double dy) {
            ((Engine*)glfwGetWindowUserPointer(w))->dist *= pow(0.9, dy);
        });
        glEnable(GL_DEPTH_TEST);
    }
    float run() {
        glfwSwapBuffers(window);
        glfwPollEvents();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        double x, y; glfwGetCursorPos(window, &x, &y);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
            yaw -= (x - mx) * 0.005;
            pitch = clamp(pitch + float(y - my) * 0.005f, -1.5f, 1.5f);
        }
        mx = x; my = y;
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, (double)WIDTH / HEIGHT, 0.01, 100);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(dist*cos(pitch)*sin(yaw), dist*sin(pitch), dist*cos(pitch)*cos(yaw), 0, 0, 0, 0, 1, 0);

        double now = glfwGetTime();
        float dt = (float)(now - prev);
        prev = now;
        if (dt <= 0.0f || dt > 0.1f) dt = 1.0f / 60.0f;
        return dt;
    }
    // Q / E: spin rate in rad/s
    float spin() {
        return 1.5f * ((glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) - (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS));
    }

    // Q / E: spin rate in rad/s
    float rotateInput() {
        float speed = 1.5f;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) return  speed;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) return -speed;
        return 0.0f;
    }
    int slice(int k, int n, int up, int down) {
        if (glfwGetKey(window, up) == GLFW_PRESS) k++;
        if (glfwGetKey(window, down) == GLFW_PRESS) k--;
        return clamp(k, 0, n - 1);
    }

    // ---- GPU ----
    template <class T> T* alloc(size_t n) { T* p; cudaMallocManaged(&p, n * sizeof(T)); cudaMemset(p, 0, n * sizeof(T)); sync(); return p; }
    template <class F> void gpu(int n, F f) { kernel<<<(n + 255) / 256, 256>>>(n, f); }
    template <class T> void copy(T* to, T* from, size_t n) { cudaMemcpyAsync(to, from, n * sizeof(T), cudaMemcpyDefault); }
    void sync() { cudaDeviceSynchronize(); }
};
Engine engine;

struct Fluid {
    int Nx = 384, Ny = 192, Nz = 128;
    float dL = 0.005, dt = 1.0/60.0, maxVal = 2.2;
    float rho = 1.2, nu = 1.5e-5, Uin = 1.0;

    float *u, *v, *w, *p, *div, *f1, *f2, *lo, *hi, *res[3], *acc;
    char* solid;
    GPU int U(int i, int j, int k) const { return i + j*(Nx+1) + k*(Nx+1)*Ny; }
    GPU int V(int i, int j, int k) const { return i + j*Nx + k*Nx*(Ny+1); }
    GPU int W(int i, int j, int k) const { return i + j*Nx + k*Nx*Ny; }
    GPU int C(int i, int j, int k) const { return i + j*Nx + k*Nx*Ny; }
    GPU ivec3 N(int c) const { return ivec3(Nx + (c==0), Ny + (c==1), Nz + (c==2)); }
    GPU int size(int c) const { ivec3 n = N(c); return n.x*n.y*n.z; }
    GPU vec3 off(int c) const { return vec3(c == 0 ? 0 : 0.5f, c == 1 ? 0 : 0.5f, c == 2 ? 0 : 0.5f); }
    GPU ivec3 at(int id, ivec3 n) const { return ivec3(id % n.x, id / n.x % n.y, id / (n.x*n.y)); }
    float*& F(int c) { return c == 0 ? u : c == 1 ? v : w; }

    // --- multigrid
    struct Level { ivec3 n; float *x, *f, *r; char* solid; };
    Level lv[8]; int nl = 0;

    Fluid() {
        int n = (Nx+1)*(Ny+1)*(Nz+1);
        for (float** b : {&u, &v, &w, &p, &div, &f1, &f2, &lo, &hi, &res[0], &res[1], &res[2]}) *b = engine.alloc<float>(n);
        acc = engine.alloc<float>(4);
        solid = engine.alloc<char>(n);
        for (int i = 0; i < size(0); i++) u[i] = Uin;

        // --- multigrid ---
        ivec3 m(Nx, Ny, Nz);
        lv[nl++] = {m, p, div, engine.alloc<float>(n), solid};
        while (m.x % 2 == 0 && m.y % 2 == 0 && m.z % 2 == 0 && std::min({m.x, m.y, m.z}) >= 8) {
            m = ivec3(m.x/2, m.y/2, m.z/2);
            int c = m.x*m.y*m.z;
            lv[nl++] = {m, engine.alloc<float>(c), engine.alloc<float>(c), engine.alloc<float>(c), engine.alloc<char>(c)};
        }
    }
    // ---- drawing ----
    vec3 colorMap(float t) {
        float s1 = 0.35f, s2 = 0.55f, s3 = 0.6f, s4 = 0.8f;
        vec3 c1(0,0,0), c2(0,0,1), c3(0,1,0), c4(1,1,0), c5(1,0,0);
        if (t < s1) return mix(c1, c2, t / s1);
        if (t < s2) return mix(c2, c3, (t-s1) / (s2-s1));
        if (t < s3) return mix(c3, c4, (t-s2) / (s3-s2));
        return mix(c4, c5, std::min((t-s3) / (s4-s3), 1.0f));
    }
    void draw(int k, int jy) {
        engine.sync();
        vec3 a = -vec3(Nx, Ny, Nz) * dL / 2.0f, b = -a;
        auto cell = [&](int x, int y, int z) {
            vec3 s((u[U(x,y,z)] + u[U(x+1,y,z)]) / 2, (v[V(x,y,z)] + v[V(x,y+1,z)]) / 2, (w[W(x,y,z)] + w[W(x,y,z+1)]) / 2);
            vec3 col = solid[C(x,y,z)] ? vec3(1) : colorMap(clamp(length(s) / (maxVal * Uin), 0.0f, 1.0f));
            glColor3f(col.x, col.y, col.z);
        };
        glBegin(GL_QUADS);
        for (int y = 0; y < Ny; y++) for (int x = 0; x < Nx; x++) {
            cell(x, y, k);
            float px = a.x + x*dL, py = a.y + y*dL, z = a.z + (k + 0.5f)*dL;
            glVertex3f(px, py, z); glVertex3f(px+dL, py, z); glVertex3f(px+dL, py+dL, z); glVertex3f(px, py+dL, z);
        }
        // for (int z = 0; z < Nz; z++) for (int x = 0; x < Nx; x++) {
        //     cell(x, jy, z);
        //     float px = a.x + x*dL, pz = a.z + z*dL, y = a.y + (jy + 0.5f)*dL;
        //     glVertex3f(px, y, pz); glVertex3f(px+dL, y, pz); glVertex3f(px+dL, y, pz+dL); glVertex3f(px, y, pz+dL);
        // }
        glEnd();

        glColor3f(1, 1, 1);
        glBegin(GL_LINES);
        for (int m = 0; m < 8; m++) for (int d = 0; d < 3; d++) if (!(m >> d & 1)) {
            int n = m | 1 << d;
            glVertex3f(m&1 ? b.x : a.x, m&2 ? b.y : a.y, m&4 ? b.z : a.z);
            glVertex3f(n&1 ? b.x : a.x, n&2 ? b.y : a.y, n&4 ? b.z : a.z);
        }
        glEnd();
    }

    // ---- moving the fluid ----
    void boundaries(vec3 c = vec3(0), vec3 omega = vec3(0)) {
        engine.gpu(Ny*Nz, [=, *this] __device__ (int id) {
            int j = id % Ny, k = id / Ny;
            u[U(0,j,k)]  = Uin;
            u[U(Nx,j,k)] = u[U(Nx-1,j,k)];
        });
        engine.gpu(Nx*Nz, [=, *this] __device__ (int id) { int i = id % Nx, k = id / Nx; v[V(i,0,k)] = v[V(i,Ny,k)] = 0; });
        engine.gpu(Nx*Ny, [=, *this] __device__ (int id) { int i = id % Nx, j = id / Nx; w[W(i,j,0)] = w[W(i,j,Nz)] = 0; });
        engine.gpu(Nx*Ny*Nz, [=, *this] __device__ (int id) {
            if (!solid[id]) return;
            ivec3 q = at(id, ivec3(Nx, Ny, Nz));
            vec3 s = cross(omega, (vec3(q) + 0.5f) * dL - vec3(Nx, Ny, Nz) * dL / 2.0f - c);
            u[U(q.x,q.y,q.z)] = u[U(q.x+1,q.y,q.z)] = s.x;
            v[V(q.x,q.y,q.z)] = v[V(q.x,q.y+1,q.z)] = s.y;
            w[W(q.x,q.y,q.z)] = w[W(q.x,q.y,q.z+1)] = s.z;
        });
    }
    GPU float sample(const float* f, ivec3 n, vec3 q, float* l = nullptr, float* h = nullptr) const {
        q = clamp(q, vec3(0), vec3(n) - 1.001f);
        ivec3 a(q); vec3 t = q - vec3(a);
        float c[8];
        for (int m = 0; m < 8; m++) c[m] = f[(a.x + (m&1)) + (a.y + (m>>1&1))*n.x + (a.z + (m>>2))*n.x*n.y];
        if (l) { *l = *h = c[0]; for (int m = 1; m < 8; m++) { *l = fminf(*l, c[m]); *h = fmaxf(*h, c[m]); } }
        return mix(mix(mix(c[0], c[1], t.x), mix(c[2], c[3], t.x), t.y),
                mix(mix(c[4], c[5], t.x), mix(c[6], c[7], t.x), t.y), t.z);
    }
    GPU vec3 velAt(vec3 q) const {
        return vec3(sample(u, N(0), q - off(0)), sample(v, N(1), q - off(1)), sample(w, N(2), q - off(2)));
    }
    void trace(int c, float* f, float* out, float s, bool limits = false) {
        engine.gpu(size(c), [=, *this] __device__ (int id) {
            vec3 q = vec3(at(id, N(c))) + off(c);
            vec3 a = velAt(q), m = velAt(q - a*s/2.0f);
            out[id] = sample(f, N(c), q - m*s - off(c), limits ? lo + id : nullptr, limits ? hi + id : nullptr);
        });
    }
    void advect() {
        float s = dt / dL;
        for (int c = 0; c < 3; c++) {
            float *f = F(c), *r = res[c];
            trace(c, f, f1, s, true);
            trace(c, f1, f2, -s);
            engine.gpu(size(c), [=, *this] __device__ (int id) {
                float e = f1[id] + 0.5f*(f[id] - f2[id]);
                r[id] = (e < lo[id] || e > hi[id]) ? f1[id] : e;
            });
        }
        for (int c = 0; c < 3; c++) swap(F(c), res[c]);
    }

    // ---- pressure: multigrid ----
    GPU float lap(Level L, ivec3 q, int id, int& cnt) const {
        float sum = 0; cnt = 0;
        for (int d = 0; d < 3; d++) for (int sg = -1; sg <= 1; sg += 2) {
            ivec3 m = q; m[d] += sg;
            if (m.x == L.n.x) { cnt++; continue; }
            if (m[d] < 0 || m[d] >= L.n[d]) continue;
            int mi = m.x + m.y*L.n.x + m.z*L.n.x*L.n.y;
            if (!L.solid[mi]) { sum += L.x[mi]; cnt++; }
        }
        return sum - cnt*L.x[id];
    }
    void smooth(int l, int iters, float om) {
        Level L = lv[l];
        for (int it = 0; it < iters; it++) for (int color = 0; color < 2; color++)
            engine.gpu(L.n.x*L.n.y*L.n.z, [=, *this] __device__ (int id) {
                ivec3 q = at(id, L.n);
                if ((q.x + q.y + q.z) % 2 != color || L.solid[id]) return;
                int cnt; float r = lap(L, q, id, cnt) - L.f[id];
                if (cnt) L.x[id] += om * r / cnt;
            });
    }
    void residual(int l, bool check = false) {
        Level L = lv[l];
        engine.gpu(L.n.x*L.n.y*L.n.z, [=, *this] __device__ (int id) {
            if (L.solid[id]) { L.r[id] = 0; return; }
            int cnt; L.r[id] = L.f[id] - lap(L, at(id, L.n), id, cnt);
            if (check && cnt) atomicMax((int*)acc, __float_as_int(fabsf(L.r[id])));
        });
    }
    void vcycle(int l) {
        if (l == nl-1) { smooth(l, 30, 1.5f); return; }
        Level F = lv[l], C = lv[l+1];
        smooth(l, 4, 1.15f);
        residual(l);
        engine.gpu(C.n.x*C.n.y*C.n.z, [=, *this] __device__ (int id) {
            ivec3 q = at(id, C.n);
            float sum = 0; bool all = true;
            for (int m = 0; m < 8; m++) {
                ivec3 c = q*2 + ivec3(m&1, m>>1&1, m>>2);
                int fi = c.x + c.y*F.n.x + c.z*F.n.x*F.n.y;
                sum += F.r[fi]; all = all && F.solid[fi];
            }
            C.solid[id] = all;
            C.f[id] = all ? 0 : sum / 2;
            C.x[id] = 0;
        });
        vcycle(l+1);
        engine.gpu(F.n.x*F.n.y*F.n.z, [=, *this] __device__ (int id) {
            if (F.solid[id]) return;
            F.x[id] += sample(C.x, C.n, (vec3(at(id, F.n)) + 0.5f) / 2.0f - 0.5f);
        });
        smooth(l, 4, 1.15f);
    }
    void project() {
        ivec3 n(Nx, Ny, Nz);
        float s = rho * dL*dL / dt;
        engine.gpu(Nx*Ny*Nz, [=, *this] __device__ (int id) {
            ivec3 q = at(id, n);
            div[id] = solid[id] ? 0 : s * (u[U(q.x+1,q.y,q.z)] - u[U(q.x,q.y,q.z)] + v[V(q.x,q.y+1,q.z)] - v[V(q.x,q.y,q.z)]
                                        + w[W(q.x,q.y,q.z+1)] - w[W(q.x,q.y,q.z)]) / dL;
        });
        int it = 0;
        for (; it < 15; it++) {
            vcycle(0);
            engine.gpu(1, [=, *this] __device__ (int) { *acc = 0; });
            residual(0, true);
            engine.sync();
            if (*acc / s * dt < 1e-3f) break;
        }
        static int frame = 0;
        if (frame++ % 60 == 0) cout << "pressure cycles: " << it + 1 << "  err: " << *acc / s * dt << endl;

        float g = dt / (rho * dL);
        engine.gpu(size(0), [=, *this] __device__ (int id) {
            ivec3 q = at(id, N(0));
            if (q.x == 0) return;
            if (q.x == Nx) { if (!solid[C(Nx-1,q.y,q.z)]) u[id] -= g * (0 - p[C(Nx-1,q.y,q.z)]); return; }
            int a = C(q.x,q.y,q.z), b = C(q.x-1,q.y,q.z);
            if (!solid[a] && !solid[b]) u[id] -= g * (p[a] - p[b]);
        });
        engine.gpu(size(1), [=, *this] __device__ (int id) {
            ivec3 q = at(id, N(1));
            if (q.y == 0 || q.y == Ny) return;
            int a = C(q.x,q.y,q.z), b = C(q.x,q.y-1,q.z);
            if (!solid[a] && !solid[b]) v[id] -= g * (p[a] - p[b]);
        });
        engine.gpu(size(2), [=, *this] __device__ (int id) {
            ivec3 q = at(id, N(2));
            if (q.z == 0 || q.z == Nz) return;
            int a = C(q.x,q.y,q.z), b = C(q.x,q.y,q.z-1);
            if (!solid[a] && !solid[b]) w[id] -= g * (p[a] - p[b]);
        });
    }
        
    void diffuse() {
        float* nt = lo;
        engine.gpu(Nx*Ny*Nz, [=, *this] __device__ (int id) {
            ivec3 q = at(id, ivec3(Nx, Ny, Nz));
            nt[id] = 0;
            if (solid[id] || q.x < 1 || q.y < 1 || q.z < 1 || q.x > Nx-2 || q.y > Ny-2 || q.z > Nz-2) return;
            vec3 c = vec3(q) + 0.5f, g[3];
            for (int b = 0; b < 3; b++) { vec3 e(0); e[b] = 1; g[b] = (velAt(c + e) - velAt(c - e)) / 2.0f; }
            float S2 = 0;
            for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) { float s = (g[b][a] + g[a][b]) / 2; S2 += s*s; }
            nt[id] = 0.17f*0.17f * dL * sqrtf(2*S2);
        });
        int iters = std::min(50, 4 + (int)(10 * nu * dt / (dL*dL)));
        for (int c = 0; c < 3; c++) {
            engine.copy(f1, F(c), size(c));
            for (int it = 0; it < iters; it++) {
                float *f = F(c), *g = f2;
                engine.gpu(size(c), [=, *this] __device__ (int id) {
                    ivec3 n = N(c), q = at(id, n);
                    if (q[c] == 0 || q[c] == n[c]-1) { g[id] = f[id]; return; }
                    ivec3 b = q; b[c] -= 1;
                    float a = (nu + (nt[C(q.x,q.y,q.z)] + nt[C(b.x,b.y,b.z)]) / 2) * dt / (dL*dL);
                    float sum = 0;
                    for (int d = 0; d < 3; d++) for (int s = -1; s <= 1; s += 2) {
                        ivec3 m = q; m[d] += s;
                        sum += (m[d] < 0 || m[d] >= n[d]) ? f[id] : f[m.x + m.y*n.x + m.z*n.x*n.y];
                    }
                    g[id] = (f1[id] + a*sum) / (1 + 6*a);
                });
                swap(F(c), f2);
            }
        }
    }

    vec3 drag() {
        engine.sync();
        acc[0] = acc[1] = acc[2] = 0;
        float A = dL*dL, mu = rho*nu;
        engine.gpu(Nx*Ny*Nz, [=, *this] __device__ (int id) {
            if (solid[id]) return;
            ivec3 n(Nx, Ny, Nz), q = at(id, n);
            vec3 uc((u[U(q.x,q.y,q.z)] + u[U(q.x+1,q.y,q.z)]) / 2, (v[V(q.x,q.y,q.z)] + v[V(q.x,q.y+1,q.z)]) / 2, (w[W(q.x,q.y,q.z)] + w[W(q.x,q.y,q.z+1)]) / 2);
            for (int d = 0; d < 3; d++) for (int s = -1; s <= 1; s += 2) {
                ivec3 m = q; m[d] += s;
                if (m[d] < 0 || m[d] >= n[d] || !solid[C(m.x, m.y, m.z)]) continue;
                atomicAdd(&acc[d], s * p[id] * A);
                for (int t = 0; t < 3; t++) if (t != d) atomicAdd(&acc[t], mu * uc[t] / (dL/2) * A);
            }
        });
        engine.sync();
        return vec3(acc[0], acc[1], acc[2]);
    }
};
Fluid fluid;

struct Object {
    vec3 pos; float r; vec3 omega = vec3(0);
    vector<vec3> tri;
    GLuint list = 0;

    Object(vec3 pos, float r, const char* path) : pos(pos), r(r) {
        ifstream f(path, ios::binary);
        char h[80]; uint32_t n = 0;
        f.read(h, 80); f.read((char*)&n, 4);
        tri.resize(n*3);
        for (uint32_t t = 0; t < n; t++) {
            float d[12]; uint16_t a;
            f.read((char*)d, 48); f.read((char*)&a, 2);
            for (int m = 0; m < 3; m++) tri[t*3+m] = vec3(d[3+3*m], d[5+3*m], -d[4+3*m]);
        }
        vec3 lo(1e30f), hi(-1e30f);
        float a = 3 * M_PI/2;
        for (auto& p : tri) p = vec3(cos(a)*p.x + sin(a)*p.z, p.y, -sin(a)*p.x + cos(a)*p.z);

        for (auto& p : tri) { lo = min(lo, p); hi = max(hi, p); }
        vec3 e = hi - lo, c = (lo + hi) / 2.0f;
        float s = 2*r / std::max({e.x, e.y, e.z});
        for (auto& p : tri) p = (p - c) * s;
        cout << n << " triangles" << endl;
    }

    void block(Fluid& f) {
        ivec3 n(f.Nx, f.Ny, f.Nz);
        vec3 L = vec3(n) * f.dL;
        vector<char> votes(f.Nx*f.Ny*f.Nz, 0);
        for (int d = 0; d < 3; d++) {
            int e = (d+1) % 3, g = (d+2) % 3;
            vector<vector<float>> hit(n[e] * n[g]);
            for (size_t t = 0; t < tri.size(); t += 3) {
                vec3 a = tri[t] + pos + L/2.0f, b = tri[t+1] + pos + L/2.0f, c = tri[t+2] + pos + L/2.0f;
                int e0 = std::max(0, (int)ceil(std::min({a[e], b[e], c[e]}) / f.dL - 0.5f));
                int e1 = std::min(n[e]-1, (int)floor(std::max({a[e], b[e], c[e]}) / f.dL - 0.5f));
                int g0 = std::max(0, (int)ceil(std::min({a[g], b[g], c[g]}) / f.dL - 0.5f));
                int g1 = std::min(n[g]-1, (int)floor(std::max({a[g], b[g], c[g]}) / f.dL - 0.5f));
                float det = (b[e]-a[e])*(c[g]-a[g]) - (c[e]-a[e])*(b[g]-a[g]);
                if (abs(det) < 1e-12f) continue;
                for (int y = g0; y <= g1; y++) for (int x = e0; x <= e1; x++) {
                    float pe = (x + 0.5f)*f.dL - a[e], pg = (y + 0.5f)*f.dL - a[g];
                    float s = (pe*(c[g]-a[g]) - (c[e]-a[e])*pg) / det;
                    float w = ((b[e]-a[e])*pg - pe*(b[g]-a[g])) / det;
                    if (s < 0 || w < 0 || s + w > 1) continue;
                    hit[x + y*n[e]].push_back(a[d] + s*(b[d]-a[d]) + w*(c[d]-a[d]));
                }
            }
            #pragma omp parallel for
            for (int r = 0; r < n[e]*n[g]; r++) {
                vector<float>& h = hit[r];
                sort(h.begin(), h.end());
                for (int m = 0; m < n[d]; m++) {
                    ivec3 q; q[d] = m; q[e] = r % n[e]; q[g] = r / n[e];
                    votes[f.C(q.x, q.y, q.z)] += (lower_bound(h.begin(), h.end(), (m + 0.5f)*f.dL) - h.begin()) % 2;
                }
            }
        }
        for (size_t i = 0; i < votes.size(); i++) f.solid[i] = votes[i] >= 2;
    }
    void draw() {
        if (!list) {
            list = glGenLists(1);
            glNewList(list, GL_COMPILE);
            vec3 light = normalize(vec3(0.3, 1, 0.5));
            glBegin(GL_TRIANGLES);
            for (size_t t = 0; t < tri.size(); t += 3) {
                vec3 n = normalize(cross(tri[t+1] - tri[t], tri[t+2] - tri[t]));
                float b = 0.3f + 0.7f * std::max(0.0f, dot(n, light));
                glColor3f(b, b, b);
                for (int m = 0; m < 3; m++) { vec3 p = tri[t+m] + pos; glVertex3f(p.x, p.y, p.z); }
            }
            glEnd();
            glEndList();
        }
        glCallList(list);
    }
};
Object model(vec3(0.0, 0, 0), 0.5, "models/rb22.stl");


int main() {
    cout << "Re = " << fluid.Uin * 2*model.r / fluid.nu << endl;
    model.block(fluid);
    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();

        // --- physiques ---
        fluid.boundaries(model.pos, model.omega);
        fluid.advect();
        fluid.diffuse();
        fluid.boundaries(model.pos, model.omega);
        fluid.project();

        static double t0 = glfwGetTime(); static int fr = 0;
if (++fr % 60 == 0) { cout << "ms/frame: " << (glfwGetTime() - t0) * 1000 / 60 << endl; t0 = glfwGetTime(); }

        // static int f = 0;
        // if (f++ % 60 == 0) {
        //     vec3 F = fluid.drag();
        //     cout << "Cd = " << F.x / (0.5f * fluid.rho * fluid.Uin*fluid.Uin * M_PI * sphere.r*sphere.r) << endl;
        // }

        // --- draw ---
        static int k = fluid.Nz / 2, j = fluid.Ny / 2;
        k = engine.slice(k, fluid.Nz, GLFW_KEY_UP, GLFW_KEY_DOWN);
        j = engine.slice(j, fluid.Ny, GLFW_KEY_RIGHT, GLFW_KEY_LEFT);
        fluid.draw(k, j);
        model.draw();
    }
    glfwTerminate();
    return 0;
}



