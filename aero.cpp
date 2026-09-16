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

struct Particle {

    vec2 pos, vel;
    vec3 col;
    int r = 10;
    Particle(vec2 p, vec2 v, vec3 c = vec3(1, 1, 1)) : pos(p), vel(v), col(c) {}

    void draw () {
        
        glColor3f(col.r, col.b, col.g);
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(pos.x, pos.y);
        for (float a = 0; a < 6.34; a+= 0.1) {
            glVertex2f(cos(a) * r + pos.x, sin(a) * r + pos.y);
        }
        glEnd();
    }

};
vector<Particle> particles = {
    Particle(vec2(0, 0), vec2(0)),
    Particle(vec2(20, 0), vec2(0)),
    Particle(vec2(40, 0), vec2(0)),
    Particle(vec2(60, 0), vec2(0)),
};

int main() {
    Engine engine;

    // keep running until someone closes the window
    while (!glfwWindowShouldClose(engine.window)) {
        engine.run();

        for (Particle p : particles ) {
            p.draw();
        }

        glfwSwapBuffers(engine.window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
