#include <GL/glew.h>
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

// --- Twin-Turbo 5.2L V10 ----
struct Engine {
    GLFWwindow* window;
    int WIDTH = 800, HEIGHT = 600;
    double prev = 0.0;

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
    }
    float run() {
        glfwSwapBuffers(window);
        glfwPollEvents();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-WIDTH / 2.0, WIDTH / 2.0, -HEIGHT / 2.0, HEIGHT / 2.0, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
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

};
Engine engine;

struct Fluid {
    int Nx = 266, Ny=200;
    float dL = 0.005f, dt = 1.f/60.f;
    float maxVal = 2.2f;
    //     air density, viscosity, inlet wind speed
    float rho = 1.2f, nu = 1.5e-5f, Uin = 0.33f; 

    vector<float> u, v, p;
    vector<bool> solid;
    int U(int i, int j) { return i + j*(Nx+1); }
    int V(int i, int j) { return i + j*Nx; }

    Fluid (){
        u.resize((Nx+1)*Ny, Uin);
        v.resize(Nx*(Ny+1), 0.0f);
        p.resize(Nx*Ny, 0.0f);
        solid.resize(Nx*Ny, false);
        for (float& x : u) x = 0.5f;
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
    void draw () {
        float cellW = (float)engine.WIDTH / Nx;
        float cellH = (float)engine.HEIGHT / Ny;

        for (int y = 0; y < Ny; y++) {
            for (int x = 0; x < Nx; x++) {
                float val = length(vec2(u[U(x,y)]+u[U(x+1,y)], v[V(x,y)]+v[V(x,y+1)]) * 0.5f);
                vec3 col = colorMap(glm::clamp(val / (maxVal * Uin), 0.0f, 1.0f));
                if (solid[x + y*Nx]) col = vec3(1);
                glColor3f(col.r, col.g, col.b);

                float px = x*cellW - engine.WIDTH / 2.0f;
                float py = y*cellH - engine.HEIGHT / 2.0f;

                glBegin(GL_QUADS);
                glVertex2f(px, py);
                glVertex2f(px+cellW, py);
                glVertex2f(px+cellW, py+cellH);
                glVertex2f(px, py+cellH);
                glEnd();
            }
        }
    }
    
    // ---- moving the fluid ----
    void boundaries(vec2 c = vec2(0), float omega = 0.0f) {
        float x0 = Nx*dL/2, y0 = Ny*dL/2;
        for (int j = 0; j < Ny; j++) {
            u[U(0,j)]  = Uin;          
            u[U(Nx,j)] = u[U(Nx-1,j)];
        }
        for (int i = 0; i < Nx; i++) v[V(i,0)] = v[V(i,Ny)] = 0.0f;
        for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++) {
            if (!solid[i + j*Nx]) continue;
            float us = -omega * ((j + 0.5f)*dL - y0 - c.y);
            float vs =  omega * ((i + 0.5f)*dL - x0 - c.x);
            u[U(i,j)] = u[U(i+1,j)] = us;
            v[V(i,j)] = v[V(i,j+1)] = vs;
        }
    }

    float sample(const vector<float>& f, int w, int h, float x, float y, float* lo = nullptr, float* hi = nullptr) {
        x = glm::clamp(x, 0.0f, w - 1.001f);
        y = glm::clamp(y, 0.0f, h - 1.001f);
        int i = (int)x, j = (int)y;
        float fx = x - i, fy = y - j;
        float a = f[i + j*w], b = f[(i+1) + j*w], c = f[i + (j+1)*w], d = f[(i+1) + (j+1)*w];
        if (lo) { *lo = std::min({a, b, c, d}); *hi = std::max({a, b, c, d}); }
        return mix(mix(a, b, fx), mix(c, d, fx), fy);
    }
    void trace(const vector<float>& f, vector<float>& out, bool isU, float k, vector<float>* lo = nullptr, vector<float>* hi = nullptr) {
        int w = isU ? Nx+1 : Nx, h = isU ? Ny : Ny+1;
        for (int j = 0; j < h; j++) for (int i = 0; i < w; i++) {
            int n = i + j*w;                                                   // same as U(i,j) / V(i,j)
            float vx = isU ? u[n] : sample(u, Nx+1, Ny, i + 0.5f, j - 0.5f);
            float vy = isU ? sample(v, Nx, Ny+1, i - 0.5f, j + 0.5f) : v[n];
            out[n] = sample(f, w, h, i - vx*k, j - vy*k, lo ? &(*lo)[n] : nullptr, hi ? &(*hi)[n] : nullptr);
        }
    }
    void advect() {
        float k = dt / dL;
        vector<float> res[2];
        for (int c = 0; c < 2; c++) {
            const vector<float>& f = c ? v : u;
            vector<float> f1 = f, f2 = f, lo = f, hi = f;
            trace(f, f1, !c, k, &lo, &hi);             
            trace(f1, f2, !c, -k);                     
            for (size_t n = 0; n < f.size(); n++)       
                f2[n] = glm::clamp(f1[n] + 0.5f*(f[n] - f2[n]), lo[n], hi[n]);
            res[c] = f2;
        }
        u = res[0]; v = res[1];                        
    }
    void project() {
        vector<float> div(Nx*Ny, 0.0f);
        for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++)
            if (!solid[i+j*Nx]) div[i+j*Nx] = (u[U(i+1,j)] - u[U(i,j)] + v[V(i,j+1)] - v[V(i,j)]) / dL;
        float s = rho * dL*dL / dt, err = 0.0f;
        int it = 0;
        for (; it < 500; it++) {
            err = 0.0f;
            for (int color = 0; color < 2; color++) {       
                #pragma omp parallel for reduction(max:err)
                for (int j = 0; j < Ny; j++) for (int i = (j + color) % 2; i < Nx; i += 2) {
                    if (solid[i+j*Nx]) continue;
                    float sum = 0.0f; int n = 0;
                    if (i > 0    && !solid[(i-1)+j*Nx]) { sum += p[(i-1)+j*Nx]; n++; }
                    if (i < Nx-1) { if (!solid[(i+1)+j*Nx]) { sum += p[(i+1)+j*Nx]; n++; } }
                    else n++;                       
                    if (j > 0    && !solid[i+(j-1)*Nx]) { sum += p[i+(j-1)*Nx]; n++; }
                    if (j < Ny-1 && !solid[i+(j+1)*Nx]) { sum += p[i+(j+1)*Nx]; n++; }
                    if (!n) continue;
                    float r = sum - n*p[i+j*Nx] - s*div[i+j*Nx];
                    err = std::max(err, abs(r));
                    p[i+j*Nx] += 1.97f * r / n;     
                }
            }
            if (err / s * dt < 1e-4f) break;
        }
        // static int frame = 0;
        // if (frame++ % 60 == 0) cout << "pressure iters: " << it << "  err: " << err / s * dt << endl;


        float k = dt / (rho * dL);
        for (int j = 0; j < Ny; j++) for (int i = 1; i < Nx; i++)
            if (!solid[i+j*Nx] && !solid[(i-1)+j*Nx]) u[U(i,j)] -= k * (p[i+j*Nx] - p[(i-1)+j*Nx]);
        for (int j = 0; j < Ny; j++)
            if (!solid[(Nx-1)+j*Nx]) u[U(Nx,j)] -= k * (0.0f - p[(Nx-1)+j*Nx]);
        for (int j = 1; j < Ny; j++) for (int i = 0; i < Nx; i++)
            if (!solid[i+j*Nx] && !solid[i+(j-1)*Nx]) v[V(i,j)] -= k * (p[i+j*Nx] - p[i+(j-1)*Nx]);
    }
    void diffuse() {
        vector<float> u0 = u, v0 = v;
        float a = nu * dt / (dL*dL);
        int iters = std::min(50, 2 + (int)(10*a));
        for (int it = 0; it < iters; it++) {
            for (int j = 0; j < Ny; j++) for (int i = 1; i < Nx; i++) {
                float dn = j > 0    ? u[U(i,j-1)] : -u[U(i,j)];
                float up = j < Ny-1 ? u[U(i,j+1)] : -u[U(i,j)];
                u[U(i,j)] = (u0[U(i,j)] + a * (u[U(i+1,j)] + u[U(i-1,j)] + up + dn)) / (1 + 4*a);
            }
            for (int j = 1; j < Ny; j++) for (int i = 0; i < Nx; i++) {
                float lf = i > 0    ? v[V(i-1,j)] : -v[V(i,j)]; 
                float rt = i < Nx-1 ? v[V(i+1,j)] :  v[V(i,j)];
                v[V(i,j)] = (v0[V(i,j)] + a * (lf + rt + v[V(i,j+1)] + v[V(i,j-1)])) / (1 + 4*a);
            }
        }
    }

    void jet() {
        for (int j = 0; j < Ny; j++) {
            float y = (j + 0.5f) * dL - Ny * dL / 2.0f;     // metres from centre
            if (abs(y) < 0.02f) for (int i = 1; i < 3; i++) u[U(i,j)] = 5.0f;
        }
    }
    void forces() {
    for (int j = 0; j < Ny; j++) {
        float y = (j + 0.5f) * dL - Ny * dL / 2.0f;
        if (abs(y) < 0.02f) for (int i = 1; i < 3; i++) u[U(i,j)] += 300.0f * dt;
    }
}
    // void probe(vec2 c, float D) {
    //     static float t = 0, last = 0, prev = 0;
    //     t += dt;
    //     int i = (c.x + Nx*dL/2 + 3*D) / dL, j = (c.y + Ny*dL/2) / dL;
    //     float now = v[V(i,j)];
    //     if (prev < 0 && now >= 0) { if (last > 0) cout << "St = " << D / (t - last) / Uin << endl; last = t; }
    //     prev = now;
    // }
};
Fluid fluid;

struct Object {
    vec2 pos; float r;
    float angle = 0.0f, omega=0.0f;
    int w = 0, h = 0;
    unsigned char* img = nullptr; 

    Object(vec2 pos, float r, const char* path) : pos(pos), r(r) {
        int n; img = stbi_load(path, &w, &h, &n, 2);
        if (!img) cerr << "no png lol" << endl;
    }

    void draw () {
        float pxPerM = engine.WIDTH / (fluid.Nx * fluid.dL);
        vec2 c = pos * pxPerM;
        float s = r * pxPerM;

        glPushMatrix();
        glTranslatef(c.x, c.y, 0.0f);
        glRotatef(angle * 180.0f / M_PI, 0.0f, 0.0f, 1.0f);
        glTranslatef(-c.x, -c.y, 0.0f);
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(c.x,   c.y);
        glVertex2f(c.x-s, c.y-s);
        glVertex2f(c.x+s, c.y-s);
        glVertex2f(c.x+s, c.y+s);
        glVertex2f(c.x-s, c.y+s);
        glVertex2f(c.x-s, c.y-s);
        glEnd();
        glPopMatrix();
    }
    
    bool inside(vec2 p) {
        vec2 d = p - pos;
        d = vec2(cos(-angle)*d.x - sin(-angle)*d.y,
                sin(-angle)*d.x + cos(-angle)*d.y);
        float mpp = 2*r / std::max(w, h);              
        int px = d.x / mpp + w/2, py = h/2 - d.y / mpp; 
        if (!img || px < 0 || py < 0 || px >= w || py >= h) return false;
        unsigned char* c = img + (px + py*w) * 2;
        return c[1] > 128 && c[0] < 128;            
    }
    void block(Fluid& f) {
        for (int y = 0; y < f.Ny; y++) {
            for (int x = 0; x < f.Nx; x++) {
                vec2 cellPos = vec2((x + 0.5f) * f.dL - (f.Nx * f.dL) / 2.0f,
                                    (y + 0.5f) * f.dL - (f.Ny * f.dL) / 2.0f);
                int i = x + y*f.Nx;
                f.solid[i] = inside(cellPos);
            }
        }
    }
    void spin(float w) { omega = w; angle += w * fluid.dt; }

};
Object square(vec2(0.0f, 0.0f), 0.2f, "shapes/airfoil.png");

int main () {

    while(!glfwWindowShouldClose(engine.window)) {
        engine.run();
        square.spin(engine.rotateInput());

        // --- physiques ---
        square.block(fluid);
        //fluid.forces();
        fluid.boundaries(square.pos, square.omega);
        fluid.advect();
        fluid.diffuse();
        fluid.boundaries(square.pos, square.omega);
        fluid.project();
        // fluid.probe(square.pos, 2*square.r);

        // --- draw ---
        fluid.draw();
        // square.draw();

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

