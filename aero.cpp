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

    vector<float> dye;
    vector<vec2> vel;
    vector<float> p;

    Fluid (){
        dye.resize(Nx*Ny);
        vel.resize(Nx*Ny, vec2(0,0));
        for (int y = 0; y < Ny; y++)
            for (int x = 0; x < Nx; x++)
                dye[x+y*Nx] = (float)(x + y) / (float)(Nx + Ny);
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
    
};
Fluid fluid;

int main () {
    while(!glfwWindowShouldClose(engine.window)) {
        engine.run();

        fluid.draw();

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}