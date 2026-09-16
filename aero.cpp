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
        // wake up GLFW
        if (!glfwInit()) {
            cerr << "glfw didn't start" << endl;
            exit(EXIT_FAILURE);
        }

        // ask the operating system for a window
        window = glfwCreateWindow(WIDTH, HEIGHT, "subscribe to kavan xD", nullptr, nullptr);
        if (!window) {
            cerr << "no window for you" << endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        // point every future drawing command at this window
        glfwMakeContextCurrent(window);
        glewInit();

        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);
    }

    // this runs once per frame
    void run() {
        // wipe the screen to almost-black
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // put (0, 0) in the middle instead of the corner
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-WIDTH / 2.0, WIDTH / 2.0, -HEIGHT / 2.0, HEIGHT / 2.0, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
};
Engine engine;

struct Fluid {
    int N = 100;
    vector<float> density;

    Fluid() {
        density.resize(N * N);
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                density[IX(x, y)] = (float)(x + y) / (2 * N);
    }




    int IX(int x, int y) {
        return x + y * N;
    }

    vec3 colormap(float t) {
        t = clamp(t, 0.0f, 1.0f);
        vec3 c1(0, 0, 1), c2(0, 1, 1), c3(0, 1, 0), c4(1, 1, 0), c5(1, 0, 0);
        if (t < 0.25f) return mix(c1, c2, t / 0.25f);
        if (t < 0.5f)  return mix(c2, c3, (t - 0.25f) / 0.25f);
        if (t < 0.75f) return mix(c3, c4, (t - 0.5f) / 0.25f);
        return mix(c4, c5, (t - 0.75f) / 0.25f);
    }

    void draw(int screenW, int screenH) {
        float cellW = (float)engine.WIDTH / N;
        float cellH = (float)engine.HEIGHT / N;

        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                vec3 col = colormap(density[IX(x, y)]);
                glColor3f(col.r, col.g, col.b);

                float px = x * cellW - screenW / 2.0f;
                float py = y * cellH - screenH / 2.0f;

                glBegin(GL_QUADS);
                glVertex2f(px, py);
                glVertex2f(px + cellW, py);
                glVertex2f(px + cellW, py + cellH);
                glVertex2f(px, py + cellH);
                glEnd();
            }
        }
    }
};
Fluid fluid;


int main() {

    // keep running until someone closes the window
    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();

        fluid.draw(engine.WIDTH, engine.HEIGHT);

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
