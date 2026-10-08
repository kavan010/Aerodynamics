#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <cmath>
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
    // hidden: Q / E rotate, returns radians to add this frame
    float rotateInput(float dt) {
        float speed = 1.5f; // rad/s
        float a = 0.0f;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) a += speed * dt;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) a -= speed * dt;
        return a;
    }
};
Engine engine;

struct Fluid {
    int Nx = 133, Ny=100;
    float dL = 0.01f, dt = 1.f/60.f;
    float maxVal = 1.0f;
    //     air density, viscosity, inlet wind speed
    float rho = 1.2f, nu = 1.5e-5f, Uin = 1.0f; 

    vector<float> u, v, p;
    vector<bool> solid;
    int U(int i, int j) { return i + j*(Nx+1); }
    int V(int i, int j) { return i + j*Nx; }

    Fluid (){
        u.resize((Nx+1)*Ny, 1.0f);
        v.resize(Nx*(Ny+1), 0.0f);
        p.resize(Nx*Ny, 0.0f);
        solid.resize(Nx*Ny, false);
        for (float& x : u) x = 0.5f;
    }

    // ---- drawing the fluid ----
    vec3 colorMap(float t) {
        vec3 c1(0,0,0), c2(0,0,1), c3(0,1,0), c4(1,1,0), c5(1,0,0);
        if (t < 0.25f) return mix(c1, c2, t/0.25f);
        if (t < 0.5f) return mix(c2, c3, (t-0.25f) / 0.25f);
        if (t < 0.75f) return mix(c3, c4, (t-0.5f) / 0.25f);
        return mix(c4, c5, (t-0.75f) / 0.25f);
    }
    void draw () {
        float cellW = (float)engine.WIDTH / Nx;
        float cellH = (float)engine.HEIGHT / Ny;

        for (int y = 0; y < Ny; y++) {
            for (int x = 0; x < Nx; x++) {
                float val = length(vec2(u[U(x,y)]+u[U(x+1,y)], v[V(x,y)]+v[V(x,y+1)]) * 0.5f);
                length(vec2(u[U(x,y)]+u[U(x+1,y)], v[V(x,y)]+v[V(x,y+1)]) * 0.5f);

                vec3 col = colorMap(clamp(val / maxVal, 0.0f, 1.0f));
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
    void boundaries() {
        for (int j = 0; j < Ny; j++) {
            u[U(0,j)]  = Uin;            // inlet
            u[U(Nx,j)] = u[U(Nx-1,j)];   // outlet
        }
        for (int i = 0; i < Nx; i++) {
            v[V(i,0)]  = 0.0f;           // bottom wall
            v[V(i,Ny)] = 0.0f;           // top wall
        }
        for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++) {
            if (!solid[i + j*Nx]) continue;
            u[U(i,j)] = u[U(i+1,j)] = 0.0f; // left and right faces of the solid cell
            v[V(i,j)] = v[V(i,j+1)] = 0.0f; // bottom and top faces
        }
    }
    float sample(const vector<float>& f, int w, int h, float x, float y) {
        x = clamp(x, 0.0f, w - 1.001f);
        y = clamp(y, 0.0f, h - 1.001f);
        int i = (int)x, j = (int)y;
        float fx = x - i, fy = y - j;
        float bottom = mix(f[i + j*w],     f[(i+1) + j*w],     fx);
        float top    = mix(f[i + (j+1)*w], f[(i+1) + (j+1)*w], fx);
        return mix(bottom, top, fy);
    }
    void advect() {
        vector<float> nu = u, nv = v;
        float k = dt / dL; // m/s -> cells per step
        for (int j = 0; j < Ny; j++) for (int i = 0; i <= Nx; i++) {   // u faces, at (i, j+0.5)
            float vx = u[U(i,j)];                                       // u is stored here
            float vy = sample(v, Nx, Ny+1, i - 0.5f, j + 0.5f);         // v interpolated here
            nu[U(i,j)] = sample(u, Nx+1, Ny, i - vx*k, j - vy*k);       // u at the backtraced point
        }
        for (int j = 0; j <= Ny; j++) for (int i = 0; i < Nx; i++) {   // v faces, at (i+0.5, j)
            float vx = sample(u, Nx+1, Ny, i + 0.5f, j - 0.5f);         // u interpolated here
            float vy = v[V(i,j)];                                       // v is stored here
            nv[V(i,j)] = sample(v, Nx, Ny+1, i - vx*k, j - vy*k);       // v at the backtraced point
        }
        u = nu; v = nv;
    }
    void project() {
        for (int j = 0; j < Ny; j++) u[U(0,j)] = u[U(Nx,j)] = 0.0f; // left/right walls
        for (int i = 0; i < Nx; i++) v[V(i,0)] = v[V(i,Ny)] = 0.0f; // bottom/top walls

        vector<float> div(Nx*Ny);
        for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++)
            div[i+j*Nx] = (u[U(i+1,j)] - u[U(i,j)] + v[V(i,j+1)] - v[V(i,j)]) / dL;

        float s = rho * dL*dL / dt;
        for (int it = 0; it < 100; it++)
            for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++) {
                float sum = 0.0f; int n = 0;
                if (i > 0)    { sum += p[(i-1)+j*Nx]; n++; }
                if (i < Nx-1) { sum += p[(i+1)+j*Nx]; n++; }
                if (j > 0)    { sum += p[i+(j-1)*Nx]; n++; }
                if (j < Ny-1) { sum += p[i+(j+1)*Nx]; n++; }
                p[i+j*Nx] = mix(p[i+j*Nx], (sum - s*div[i+j*Nx]) / n, 1.9f);
            }

        float k = dt / (rho * dL);
        for (int j = 0; j < Ny; j++) for (int i = 1; i < Nx; i++)
            u[U(i,j)] -= k * (p[i+j*Nx] - p[(i-1)+j*Nx]);
        for (int j = 1; j < Ny; j++) for (int i = 0; i < Nx; i++)
            v[V(i,j)] -= k * (p[i+j*Nx] - p[i+(j-1)*Nx]);
    }
    void jet() {
        for (int j = 48; j < 52; j++) for (int i = 1; i < 3; i++) u[U(i,j)] = 5.0f;
    }
};
Fluid fluid;

struct Object {
    vec2 pos; float r;
    float angle = 0.0f; // radians
    Object(vec2 pos, float r) : pos(pos), r(r) {}

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
        d = vec2(cos(-angle)*d.x - sin(-angle)*d.y,   // rotate into the square's frame
                 sin(-angle)*d.x + cos(-angle)*d.y);
        return abs(d.x) < r && abs(d.y) < r;
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
};
Object square(vec2(0.0f), 0.1f);

int main () {

    while(!glfwWindowShouldClose(engine.window)) {
        float frameDt = engine.run();
        square.angle += engine.rotateInput(frameDt);

        // --- physiques ---
        fluid.jet();
        square.block(fluid);
        fluid.advect();
        fluid.project();
        fluid.boundaries();

        // --- draw ---
        fluid.draw();
        square.draw();

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}

