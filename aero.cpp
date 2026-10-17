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
    int slice(int k, int n) {
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) k++;
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) k--;
        return glm::clamp(k, 0, n - 1);
    }

};
Engine engine;

struct Fluid {
    int Nx = 128, Ny = 96, Nz = 64;
    real dL = 0.01, dt = 1.0/60.0, maxVal = 2.2;
    real rho = 1.2, nu = 1.5e-5, Uin = 0.01875;

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
    void draw(int k) {
        vec3 a = -vec3(Nx, Ny, Nz) * dL / 2.0f, b = -a;
        float z = a.z + (k + 0.5f) * dL;
        glBegin(GL_QUADS);
        for (int y = 0; y < Ny; y++) for (int x = 0; x < Nx; x++) {
            vec3 s((u[U(x,y,k)] + u[U(x+1,y,k)]) / 2, (v[V(x,y,k)] + v[V(x,y+1,k)]) / 2, (w[W(x,y,k)] + w[W(x,y,k+1)]) / 2);
            vec3 col = solid[C(x,y,k)] ? vec3(1) : colorMap(glm::clamp(length(s) / (maxVal * Uin), 0.0f, 1.0f));
            glColor3f(col.r, col.g, col.b);
            float px = a.x + x*dL, py = a.y + y*dL;
            glVertex3f(px, py, z); glVertex3f(px+dL, py, z); glVertex3f(px+dL, py+dL, z); glVertex3f(px, py+dL, z);
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
};
Object sphere{vec3(0.0, 0, 0), 0.04};

int main() {
    cout << "Re = " << fluid.Uin * 2*sphere.r / fluid.nu << endl;
    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();

        // --- physiques ---
        sphere.block(fluid);
        fluid.boundaries(sphere.pos, sphere.omega);

        // --- draw ---
        static int k = fluid.Nz / 2;
        k = engine.slice(k, fluid.Nz);
        fluid.draw(k);
    }
    glfwTerminate();
    return 0;
}



