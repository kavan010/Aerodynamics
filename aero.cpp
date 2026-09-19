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
    }
    void run() {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-WIDTH / 2.0, WIDTH / 2.0, -HEIGHT / 2.0, HEIGHT / 2.0, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
};
Engine engine;

struct Fluid {
    int Nx = 133, Ny=100;
    float Lx=1.33f, Ly=1.0f, dL = 0.01f;


    vector<float> dye = vector<float>(Nx*Ny, 0.0f);
    vector<vec2>  vel = vector<vec2>(Nx*Ny, vec2(0,0));
    vector<float> p   = vector<float>(Nx*Ny, 0.0f);

    Fluid (){
        // // init gradient
        // for (int y = 0; y < Ny; y++)
        //     for (int x = 0; x < Nx; x++)
        //         dye[x+y*Nx] = (float)(x + y) / (float)(Nx + Ny);
    }

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
    void addDye() {
        int cx = 10;
        int cy = Ny / 2;
        int radius = 5;

        for (int y = 0; y < Ny; y++) {
            for (int x = 0; x < Nx; x++) {

                float dx = x - cx;
                float dy = y - cy;

                if (dx * dx + dy * dy <= radius * radius) {
                    dye[x + y * Nx] = 1.0f;
                }
            }
        }
    }

    template<typename T>
    T sample(const vector<T>& f, float x, float y) {
        x = clamp(x, 0.0f, (float)Nx - 1.001f);
        y = clamp(y, 0.0f, (float)Ny - 1.001f);

        int i = (int)x, j = (int)y;
        float fx = x - i, fy = y - j;

        T bottom = mix(f[i + j*Nx],     f[(i+1) + j*Nx],     fx);
        T top    = mix(f[i + (j+1)*Nx], f[(i+1) + (j+1)*Nx], fx);
        return mix(bottom, top, fy);
    }
    
    template<typename T>
    void advect(vector<T>& f, float dt) {
        vector<T> out(Nx*Ny);

        for (int y = 0; y < Ny; y++) {
            for (int x = 0; x < Nx; x++) {
                vec2 v = vel[x + y*Nx];
                out[x + y*Nx] = sample(f, x - v.x*dt/dL, y - v.y*dt/dL);
            }
        }
        f = out;
    }
};
Fluid fluid;

int main () {
    float dt = 1.0f / 60.0f;
    for (vec2& v : fluid.vel) v = vec2(0.05f, 0.0f);
    fluid.addDye();
    while(!glfwWindowShouldClose(engine.window)) {
        engine.run();

        fluid.advect(fluid.vel, dt); 
        fluid.advect(fluid.dye, dt); 
        fluid.draw();

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}