#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <cmath>
using namespace glm;
using namespace std;

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
    float Lx=1.33f, Ly=1.0f, dL = 0.01f;

    vector<float> dye;
    vector<vec2> vel;

    Fluid (){
        dye.resize(Nx*Ny);
        vel.resize(Nx*Ny, vec2(0.5f, 0.0f));
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
                vec3 col = colorMap(dye[x+y*Nx]);
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
    template<typename T>
    T sample(const vector<T>&f, float x, float y) {
        x = clamp(x, 0.0f, (float)Nx - 1.001f);
        y = clamp(y, 0.0f, (float)Ny - 1.001f);

        int i = (int)x, j = (int)y;
        float fx = x - i, fy = y - j;

        T bottom = mix(f[i + j*Nx],   f[(i+1) + j*Nx], fx);
        T top    = mix(f[i+(j+1)*Nx], f[(i+1) + (j+1)*Nx], fx);
        return mix(bottom, top, fy);
    }
    template<typename T>
    void advect(vector<T>& f, float dt) {
        vector<T> out(Nx*Ny);

        for (int y = 0; y < Ny; y++) {
            for (int x = 0; x < Nx; x++) {
                vec2 v = vel[x+y*Nx];
                out[x+y*Nx] = sample(f, x - v.x*dt/dL, y-v.y*dt/dL);
            }
        }
        f = out;
    }

    // ---- wave pulses ----
    float t = 0.0f;
    void inlet(float dt) {
        t += dt;
        float freq = 1.0f;   // waves per second

        float lo = 0.05f, hi = 0.25f;   // near-black .. light blue
        float wave = 0.5f + 0.5f * sin(2.0f * M_PI * freq * t);
        float d = lo + (hi - lo) * wave;

        for (int y = 0; y < Ny; y++)
            dye[0 + y*Nx] = d;
    }
};
Fluid fluid;

struct Object {
    vec2 pos; int idx; float r;
    float angle = 0.0f; // radians
    Object(vec2 pos, int idx, float r) : pos(pos), idx(idx), r(r) {}
    float pxPerM = engine.WIDTH / fluid.Lx;
    vec2 c = pos * pxPerM;
    float s = r * pxPerM;
    void draw () {
        glPushMatrix();
        glTranslatef(c.x, c.y, 0.0f);
        glRotatef(angle * 180.0f / M_PI, 0.0f, 0.0f, 1.0f);
        glTranslatef(-c.x, -c.y, 0.0f);
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_TRIANGLE_FAN);
        if (idx == 0) {
            glVertex2f(c.x, c.y);
            for (float a = 0; a <= 6.3; a+=0.1) {
                glVertex2f(c.x + cos(a)*r, c.y + sin(a)*r);
            }
        } else {
            glVertex2f(c.x,   c.y);
            glVertex2f(c.x-s, c.y-s);
            glVertex2f(c.x+s, c.y-s);
            glVertex2f(c.x+s, c.y+s);
            glVertex2f(c.x-s, c.y+s);
            glVertex2f(c.x-s, c.y-s);
        }
        glEnd();
        glPopMatrix();
    }
    
    bool inside(vec2 p) {
        vec2 d = p - pos;
        d = vec2(cos(-angle)*d.x - sin(-angle)*d.y,   // rotate into the square's frame
                 sin(-angle)*d.x + cos(-angle)*d.y);
        if (idx == 0) return length(d) < r;
        return abs(d.x) < r && abs(d.y) < r;
    }
    void block(Fluid& f) {
        for (int y = 0; y < f.Ny; y++) {
            for (int x = 0; x < f.Nx; x++) {
                vec2 cellPos = vec2((x + 0.5f) * f.dL - f.Lx / 2.0f,
                                    (y + 0.5f) * f.dL - f.Ly / 2.0f);
                int i = x + y*f.Nx;
                if (inside(cellPos)) {
                    f.vel[i] = vec2(0.0f);
                    f.dye[i] = 0.0f;          // no memory: solid is always empty
                } else {
                    f.vel[i] = vec2(0.5f, 0.0f);
                }
            }
        }
    }
};
Object square(vec2(0, 0), 1, 0.1f);

int main () {

    while(!glfwWindowShouldClose(engine.window)) {
        float dt = engine.run();

        square.angle += engine.rotateInput(dt);
        square.block(fluid);
        fluid.advect(fluid.dye, dt);
        fluid.inlet(dt);
        fluid.draw();

        square.draw();

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}