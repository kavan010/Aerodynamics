#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
using namespace glm;
using namespace std;
using real = float;

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
            pitch = glm::clamp(pitch + float(y - my) * 0.005f, -1.5f, 1.5f);
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
        return glm::clamp(k, 0, n - 1);
    }
};
Engine engine;

struct Fluid {
    int Nx = 384, Ny = 192, Nz = 128;
    real dL = 0.005, dt = 0.25, maxVal = 2.2;
    real rho = 1.2, nu = 1.5e-5, Uin = 0.009375;

    vector<real> u, v, w, p;
    vector<char> solid;
    int U(int i, int j, int k) { return i + j*(Nx+1) + k*(Nx+1)*Ny; }
    int V(int i, int j, int k) { return i + j*Nx + k*Nx*(Ny+1); }
    int W(int i, int j, int k) { return i + j*Nx + k*Nx*Ny; }
    int C(int i, int j, int k) { return i + j*Nx + k*Nx*Ny; }
    ivec3 N(int c) { return ivec3(Nx + (c==0), Ny + (c==1), Nz + (c==2)); }
    vec3 off(int c) { return vec3(c != 0, c != 1, c != 2) * 0.5f; }
    vector<real>& F(int c) { return c == 0 ? u : c == 1 ? v : w; }

    Fluid() {
        u.resize((Nx+1)*Ny*Nz, Uin);
        v.resize(Nx*(Ny+1)*Nz, 0);
        w.resize(Nx*Ny*(Nz+1), 0);
        p.resize(Nx*Ny*Nz, 0);
        solid.resize(Nx*Ny*Nz, 0);
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
        vec3 a = -vec3(Nx, Ny, Nz) * dL / 2.0f, b = -a;
        auto cell = [&](int x, int y, int z) {
            vec3 s((u[U(x,y,z)] + u[U(x+1,y,z)]) / 2, (v[V(x,y,z)] + v[V(x,y+1,z)]) / 2, (w[W(x,y,z)] + w[W(x,y,z+1)]) / 2);
            vec3 col = solid[C(x,y,z)] ? vec3(1) : colorMap(glm::clamp(length(s) / (maxVal * Uin), 0.0f, 1.0f));
            glColor3f(col.r, col.g, col.b);
        };
        glBegin(GL_QUADS);
        for (int y = 0; y < Ny; y++) for (int x = 0; x < Nx; x++) {
            cell(x, y, k);
            float px = a.x + x*dL, py = a.y + y*dL, z = a.z + (k + 0.5f)*dL;
            glVertex3f(px, py, z); glVertex3f(px+dL, py, z); glVertex3f(px+dL, py+dL, z); glVertex3f(px, py+dL, z);
        }
        for (int z = 0; z < Nz; z++) for (int x = 0; x < Nx; x++) {
            cell(x, jy, z);
            float px = a.x + x*dL, pz = a.z + z*dL, y = a.y + (jy + 0.5f)*dL;
            glVertex3f(px, y, pz); glVertex3f(px+dL, y, pz); glVertex3f(px+dL, y, pz+dL); glVertex3f(px, y, pz+dL);
        }
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
        for (int k = 0; k < Nz; k++) for (int j = 0; j < Ny; j++) {
            u[U(0,j,k)]  = Uin;
            u[U(Nx,j,k)] = u[U(Nx-1,j,k)];
        }
        for (int k = 0; k < Nz; k++) for (int i = 0; i < Nx; i++) v[V(i,0,k)] = v[V(i,Ny,k)] = 0;
        for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++) w[W(i,j,0)] = w[W(i,j,Nz)] = 0;
        vec3 L = vec3(Nx, Ny, Nz) * dL;
        for (int k = 0; k < Nz; k++) for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++) {
            if (!solid[C(i,j,k)]) continue;
            vec3 s = cross(omega, (vec3(i,j,k) + 0.5f) * dL - L/2.0f - c);
            u[U(i,j,k)] = u[U(i+1,j,k)] = s.x;
            v[V(i,j,k)] = v[V(i,j+1,k)] = s.y;
            w[W(i,j,k)] = w[W(i,j,k+1)] = s.z;
        }
    }
    real sample(const vector<real>& f, ivec3 n, vec3 q, real* lo = nullptr, real* hi = nullptr) {
        q = glm::clamp(q, vec3(0), vec3(n) - 1.001f);
        ivec3 a(q); vec3 t = q - vec3(a);
        real c[8];
        for (int m = 0; m < 8; m++) c[m] = f[(a.x + (m&1)) + (a.y + (m>>1&1))*n.x + (a.z + (m>>2))*n.x*n.y];
        if (lo) { *lo = *std::min_element(c, c+8); *hi = *std::max_element(c, c+8); }
        return mix(mix(mix(c[0], c[1], t.x), mix(c[2], c[3], t.x), t.y),
                mix(mix(c[4], c[5], t.x), mix(c[6], c[7], t.x), t.y), t.z);
    }
    vec3 velAt(vec3 q) {
        return vec3(sample(u, N(0), q - off(0)), sample(v, N(1), q - off(1)), sample(w, N(2), q - off(2)));
    }
    void trace(int c, const vector<real>& f, vector<real>& out, real s, vector<real>* lo = nullptr, vector<real>* hi = nullptr) {
        ivec3 n = N(c);
        #pragma omp parallel for
        for (int k = 0; k < n.z; k++) for (int j = 0; j < n.y; j++) for (int i = 0; i < n.x; i++) {
            int id = i + j*n.x + k*n.x*n.y;
            vec3 q = vec3(i, j, k) + off(c);
            vec3 a = velAt(q), m = velAt(q - a*s/2.0f);
            out[id] = sample(f, n, q - m*s - off(c), lo ? &(*lo)[id] : nullptr, hi ? &(*hi)[id] : nullptr);
        }
    }
    void advect() {
        real s = dt / dL;
        vector<real> res[3];
        for (int c = 0; c < 3; c++) {
            const vector<real>& f = F(c);
            vector<real> f1 = f, f2 = f, lo = f, hi = f;
            trace(c, f, f1, s, &lo, &hi);
            trace(c, f1, f2, -s);
            for (size_t n = 0; n < f.size(); n++) {
                real e = f1[n] + 0.5f*(f[n] - f2[n]);
                f2[n] = (e < lo[n] || e > hi[n]) ? f1[n] : e;
            }
            res[c] = f2;
        }
        u = res[0]; v = res[1]; w = res[2];
    }
    void project() {
        vector<real> div(Nx*Ny*Nz, 0);
        #pragma omp parallel for
        for (int k = 0; k < Nz; k++) for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++)
            if (!solid[C(i,j,k)]) div[C(i,j,k)] = (u[U(i+1,j,k)] - u[U(i,j,k)] + v[V(i,j+1,k)] - v[V(i,j,k)]
                                                + w[W(i,j,k+1)] - w[W(i,j,k)]) / dL;
        real s = rho * dL*dL / dt, err = 0;
        int it = 0;
        for (; it < 500; it++) {
            err = 0;
            for (int color = 0; color < 2; color++) {
                #pragma omp parallel for reduction(max:err)
                for (int k = 0; k < Nz; k++) for (int j = 0; j < Ny; j++) for (int i = (j + k + color) % 2; i < Nx; i += 2) {
                    int c = C(i,j,k); if (solid[c]) continue;
                    real sum = 0; int n = 0;
                    int nb[6][3] = {{i-1,j,k}, {i+1,j,k}, {i,j-1,k}, {i,j+1,k}, {i,j,k-1}, {i,j,k+1}};
                    for (auto& q : nb) {
                        if (q[0] == Nx) { n++; continue; }
                        if (q[0] < 0 || q[1] < 0 || q[1] >= Ny || q[2] < 0 || q[2] >= Nz) continue;
                        int m = C(q[0], q[1], q[2]);
                        if (!solid[m]) { sum += p[m]; n++; }
                    }
                    if (!n) continue;
                    real r = sum - n*p[c] - s*div[c];
                    err = std::max(err, abs(r));
                    p[c] += 1.97f * r / n;
                }
            }
            if (err / s * dt < 1e-4f) break;
        }
        static int frame = 0;
        if (frame++ % 60 == 0) cout << "pressure iters: " << it << "  err: " << err / s * dt << endl;

        real g = dt / (rho * dL);
        #pragma omp parallel for
        for (int k = 0; k < Nz; k++) for (int j = 0; j < Ny; j++) {
            for (int i = 1; i < Nx; i++)
                if (!solid[C(i,j,k)] && !solid[C(i-1,j,k)]) u[U(i,j,k)] -= g * (p[C(i,j,k)] - p[C(i-1,j,k)]);
            if (!solid[C(Nx-1,j,k)]) u[U(Nx,j,k)] -= g * (0 - p[C(Nx-1,j,k)]);
            for (int i = 0; i < Nx; i++) {
                if (j > 0 && !solid[C(i,j,k)] && !solid[C(i,j-1,k)]) v[V(i,j,k)] -= g * (p[C(i,j,k)] - p[C(i,j-1,k)]);
                if (k > 0 && !solid[C(i,j,k)] && !solid[C(i,j,k-1)]) w[W(i,j,k)] -= g * (p[C(i,j,k)] - p[C(i,j,k-1)]);
            }
        }
    }
    void diffuse() {
        real a = nu * dt / (dL*dL);
        int iters = std::min(50, 2 + (int)(10*a));
        for (int c = 0; c < 3; c++) {
            vector<real>& f = F(c); vector<real> f0 = f; ivec3 n = N(c);
            for (int it = 0; it < iters; it++)
                for (int k = 0; k < n.z; k++) for (int j = 0; j < n.y; j++) for (int i = 0; i < n.x; i++) {
                    ivec3 q(i, j, k);
                    if (q[c] == 0 || q[c] == n[c]-1) continue;
                    int id = i + j*n.x + k*n.x*n.y; real sum = 0;
                    for (int d = 0; d < 3; d++) for (int s = -1; s <= 1; s += 2) {
                        ivec3 m = q; m[d] += s;
                        sum += (m[d] < 0 || m[d] >= n[d]) ? f[id] : f[m.x + m.y*n.x + m.z*n.x*n.y];
                    }
                    f[id] = (f0[id] + a*sum) / (1 + 6*a);
                }
        }
    }

    vec3 drag() {
        vec3 F(0); ivec3 n(Nx, Ny, Nz);
        real A = dL*dL, mu = rho*nu;
        for (int k = 0; k < Nz; k++) for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++) {
            if (solid[C(i,j,k)]) continue;
            vec3 uc((u[U(i,j,k)] + u[U(i+1,j,k)]) / 2, (v[V(i,j,k)] + v[V(i,j+1,k)]) / 2, (w[W(i,j,k)] + w[W(i,j,k+1)]) / 2);
            for (int d = 0; d < 3; d++) for (int s = -1; s <= 1; s += 2) {
                ivec3 m(i, j, k); m[d] += s;
                if (m[d] < 0 || m[d] >= n[d] || !solid[C(m.x, m.y, m.z)]) continue;
                F[d] += s * p[C(i,j,k)] * A;
                for (int t = 0; t < 3; t++) if (t != d) F[t] += mu * uc[t] / (dL/2) * A;
            }
        }
        return F;
    }
};
Fluid fluid;

struct Object {
    vec3 pos; real r; vec3 omega = vec3(0);
    bool inside(vec3 q) { return length(q - pos) < r; }
    void block(Fluid& f) {
        vec3 L = vec3(f.Nx, f.Ny, f.Nz) * f.dL;
        for (int k = 0; k < f.Nz; k++) for (int j = 0; j < f.Ny; j++) for (int i = 0; i < f.Nx; i++)
            f.solid[f.C(i,j,k)] = inside((vec3(i,j,k) + 0.5f) * f.dL - L/2.0f);
    }
    void draw() {
        glPushMatrix();
        glTranslatef(pos.x, pos.y, pos.z);
        glColor3f(1, 1, 1);
        gluSphere(gluNewQuadric(), r, 32, 16);
        glPopMatrix();
    }
};
Object sphere{vec3(-0.4, 0, 0), 0.08};

int main() {
    cout << "Re = " << fluid.Uin * 2*sphere.r / fluid.nu << endl;
    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();

        // --- physiques ---
        sphere.block(fluid);
        fluid.boundaries(sphere.pos, sphere.omega);
        fluid.advect();
        fluid.diffuse();
        fluid.boundaries(sphere.pos, sphere.omega);
        fluid.project();

        static int f = 0;
        if (f++ % 60 == 0) {
            vec3 F = fluid.drag();
            cout << "Cd = " << F.x / (0.5f * fluid.rho * fluid.Uin*fluid.Uin * M_PI * sphere.r*sphere.r) << endl;
        }

        // --- draw ---
        static int k = fluid.Nz / 2, j = fluid.Ny / 2;
        k = engine.slice(k, fluid.Nz, GLFW_KEY_UP, GLFW_KEY_DOWN);
        j = engine.slice(j, fluid.Ny, GLFW_KEY_RIGHT, GLFW_KEY_LEFT);
        fluid.draw(k, j);
        sphere.draw();
    }
    glfwTerminate();
    return 0;
}



