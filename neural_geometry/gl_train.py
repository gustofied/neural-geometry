"""
gl_train.py — live training partition viewer

Watch the joint activation partition evolve as the ReLU network trains.
Straight cuts accumulate into the curved decision boundary in real time.
Drag to pan, scroll to zoom, space to pause/resume.
"""
import os, sys, ctypes
import numpy as np
import OpenGL.GL as GL
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw
import glfw.GLFW as GLFW_CONSTANTS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relu import ReLU, Softmax, CrossEntropy, Linear, Model, one_hot
from data import make_radial_bands


# ── grid computation ───────────────────────────────────────────────────────
def compute_textures(model, x_range, y_range, res=300):
    xs = np.linspace(*x_range, res)
    ys = np.linspace(*y_range, res)
    grid = np.c_[np.meshgrid(xs, ys)[0].ravel(), np.meshgrid(xs, ys)[1].ravel()]

    # logit diff
    logits = model.logits(grid)
    logit_diff = logits[:, 1] - logits[:, 0]

    # joint regions
    model.forward(grid)
    m1 = model.layers[1].x_in > 0
    m2 = model.layers[3].x_in > 0
    _, ids = np.unique(np.concatenate([m1, m2], axis=1), axis=0, return_inverse=True)

    r_lo = (ids % 256).astype(np.uint8)
    r_hi = (ids // 256).astype(np.uint8)
    region_tex = np.stack(
        [r_lo, r_hi, np.zeros_like(r_lo), np.full_like(r_lo, 255)], axis=1
    ).reshape(res, res, 4)[::-1].copy()

    logit_tex = np.clip((logit_diff + 12) / 24, 0, 1).astype(np.float32)
    logit_tex = logit_tex.reshape(res, res)[::-1].copy()

    return region_tex, logit_tex, len(np.unique(ids))


def train_epoch(model, X, Y, lr, rng, batch_size=64):
    perm = rng.permutation(len(X))
    X_s, Y_s = X[perm], Y[perm]
    running_loss = 0.0
    for i in range(0, len(X_s), batch_size):
        xb, yb = X_s[i:i+batch_size], Y_s[i:i+batch_size]
        running_loss += model.loss(xb, yb).sum()
        model.backward()
        for layer in model.layers:
            if isinstance(layer, Linear):
                layer.weights -= lr * layer.grad_w
                layer.biases  -= lr * layer.grad_b
    return running_loss / len(X)


# ── GLSL (same as gl_relu but with region tint) ───────────────────────────
VERT_SRC = """
#version 330 core
layout(location=0) in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
uniform sampler2D u_region;
uniform sampler2D u_logit;
uniform vec2      u_view_min, u_view_max, u_tex_min, u_tex_max;
in  vec2 v_uv;
out vec4 f_color;

int decode(vec2 uv) {
    vec4 c = texture(u_region, uv);
    return int(c.r * 255.0 + 0.5) + int(c.g * 255.0 + 0.5) * 256;
}

void main() {
    vec2 world = mix(u_view_min, u_view_max, v_uv);
    vec2 tuv   = (world - u_tex_min) / (u_tex_max - u_tex_min);

    if (any(lessThan(tuv, vec2(0.0))) || any(greaterThan(tuv, vec2(1.0)))) {
        f_color = vec4(0.020, 0.020, 0.032, 1.0);
        return;
    }

    int  rid = decode(tuv);
    vec2 ts  = 1.0 / vec2(textureSize(u_region, 0));

    bool on_bnd = decode(tuv + vec2( ts.x, 0.0)) != rid
               || decode(tuv + vec2(-ts.x, 0.0)) != rid
               || decode(tuv + vec2(0.0,  ts.y)) != rid
               || decode(tuv + vec2(0.0, -ts.y)) != rid;

    float logit = texture(u_logit, tuv).r;
    float d     = abs(logit - 0.5) * 2.0;
    float glow  = exp(-d * d * 22.0);

    // region tint
    float h = fract(float(rid) * 0.618033988);
    vec3 col = vec3(0.028, 0.030, 0.048) + vec3(h * 0.025, (1.0 - h) * 0.018, h * 0.02);

    // region edges
    if (on_bnd) col += vec3(0.12, 0.11, 0.10) * 0.18;

    // decision boundary
    col += vec3(0.96, 0.04, 0.28) * glow * 0.7;

    f_color = vec4(col, 1.0);
}
"""

VERT_PT_SRC = """
#version 330 core
layout(location=0) in vec2 in_pos;
layout(location=1) in vec3 in_col;
uniform vec2 u_view_min, u_view_max;
out vec3 v_col;
void main() {
    vec2 ndc     = (in_pos - u_view_min) / (u_view_max - u_view_min) * 2.0 - 1.0;
    gl_Position  = vec4(ndc, 0.0, 1.0);
    gl_PointSize = 6.0;
    v_col = in_col;
}
"""

FRAG_PT_SRC = """
#version 330 core
in  vec3 v_col;
out vec4 f_color;
void main() {
    float r = dot(gl_PointCoord - 0.5, gl_PointCoord - 0.5) * 4.0;
    float a = exp(-r * 10.0);
    if (a < 0.02) discard;
    f_color = vec4(v_col, a * 0.7);
}
"""


# ── GPU helpers ────────────────────────────────────────────────────────────
def make_program(vs, fs):
    tmp = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(tmp)
    prog = compileProgram(compileShader(vs, GL.GL_VERTEX_SHADER),
                          compileShader(fs, GL.GL_FRAGMENT_SHADER))
    GL.glBindVertexArray(0)
    GL.glDeleteVertexArrays(1, (tmp,))
    return prog


class QuadMesh:
    def __init__(self):
        v = np.array([[-1,-1],[1,-1],[1,1],[-1,-1],[1,1],[-1,1]], dtype=np.float32)
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, v.nbytes, v, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)
    def draw(self):
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)
    def destroy(self):
        GL.glDeleteVertexArrays(1, (self.vao,))
        GL.glDeleteBuffers(1, (self.vbo,))


class PointCloud:
    def __init__(self, X, y):
        cols = np.array([(1.0, 0.62, 0.0) if yi==0 else (0.0, 0.72, 0.82) for yi in y],
                        dtype=np.float32)
        data = np.hstack([X.astype(np.float32), cols])
        self.n   = len(X)
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 20, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 20, ctypes.c_void_p(8))
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)
    def draw(self):
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_POINTS, 0, self.n)
        GL.glBindVertexArray(0)
    def destroy(self):
        GL.glDeleteVertexArrays(1, (self.vao,))
        GL.glDeleteBuffers(1, (self.vbo,))


def upload_tex_rgba(arr):
    h, w = arr.shape[:2]
    tid = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0,
                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, arr.tobytes())
    for p, v in [(GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                 (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE),
                 (GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST),
                 (GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)]:
        GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tid


def upload_tex_r32f(arr):
    h, w = arr.shape
    tid = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F, w, h, 0,
                    GL.GL_RED, GL.GL_FLOAT, arr.tobytes())
    for p, v in [(GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                 (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE),
                 (GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR),
                 (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)]:
        GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tid


def update_tex_rgba(tid, arr):
    h, w = arr.shape[:2]
    GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                       GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, arr.tobytes())
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)


def update_tex_r32f(tid, arr):
    h, w = arr.shape
    GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                       GL.GL_RED, GL.GL_FLOAT, arr.tobytes())
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)


# ── run ────────────────────────────────────────────────────────────────────
def run():
    np.random.seed(42)
    X, y, _ = make_radial_bands(
        n_samples=1600, band_radii=(0.55, 1.05, 1.55, 2.05),
        band_width=0.12, xy_noise=0.02, seed=42,
    )
    Y = one_hot(y)

    net = Model([Linear(2, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 2), Softmax()],
                CrossEntropy())

    margin = 0.6
    xr = (float(X[:, 0].min() - margin), float(X[:, 0].max() + margin))
    yr = (float(X[:, 1].min() - margin), float(X[:, 1].max() + margin))
    RES = 300
    EPOCHS_PER_FRAME = 5
    MAX_EPOCH = 2000
    LR = 0.05

    cx   = float((xr[0]+xr[1])/2)
    cy   = float((yr[0]+yr[1])/2)
    half = float(max(xr[1]-xr[0], yr[1]-yr[0]) * 0.6)
    WIN  = 820

    drag_active = False
    drag_last   = (0.0, 0.0)
    paused      = [False]
    epoch       = [0]
    n_regions   = [0]
    rng         = np.random.default_rng(0)

    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WIN, WIN, "training", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    GL.glClearColor(0.020, 0.020, 0.032, 1.0)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

    prog     = make_program(VERT_SRC, FRAG_SRC)
    prog_pts = make_program(VERT_PT_SRC, FRAG_PT_SRC)
    quad     = QuadMesh()
    points   = PointCloud(X, y)

    # initial textures
    region_tex, logit_tex, nr = compute_textures(net, xr, yr, RES)
    n_regions[0] = nr
    t_region = upload_tex_rgba(region_tex)
    t_logit  = upload_tex_r32f(logit_tex)

    tex_min = np.array([xr[0], yr[0]], np.float32)
    tex_max = np.array([xr[1], yr[1]], np.float32)

    def bounds():
        return (np.array([cx-half, cy-half], np.float32),
                np.array([cx+half, cy+half], np.float32))

    def on_key(win, key, sc, action, mods):
        if action == glfw.PRESS:
            if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_SPACE:
                paused[0] = not paused[0]
            elif key == glfw.KEY_R:
                # restart training
                np.random.seed(42)
                net.layers[0].__init__(2, 64)
                net.layers[2].__init__(64, 64)
                net.layers[4].__init__(64, 2)
                epoch[0] = 0
                paused[0] = False

    def on_scroll(win, dx, dy):
        nonlocal half, cx, cy
        xpos, ypos = glfw.get_cursor_pos(win)
        lo, hi = bounds()
        mx = lo[0] + (xpos / WIN) * (hi[0] - lo[0])
        my = lo[1] + (1.0 - ypos / WIN) * (hi[1] - lo[1])
        half = float(np.clip(half * 0.9**dy, 0.05, 20.0))
        lo2, hi2 = bounds()
        cx += mx - (lo2[0] + (xpos / WIN) * (hi2[0] - lo2[0]))
        cy += my - (lo2[1] + (1.0 - ypos / WIN) * (hi2[1] - lo2[1]))

    def on_mouse_button(win, button, action, mods):
        nonlocal drag_active, drag_last
        if button == glfw.MOUSE_BUTTON_LEFT:
            drag_active = action == glfw.PRESS
            drag_last = glfw.get_cursor_pos(win)

    def on_cursor(win, xpos, ypos):
        nonlocal cx, cy, drag_last
        if drag_active:
            lo, hi = bounds()
            cx -= (xpos - drag_last[0]) * (hi[0]-lo[0]) / WIN
            cy += (ypos - drag_last[1]) * (hi[1]-lo[1]) / WIN
        drag_last = (xpos, ypos)

    glfw.set_key_callback(window, on_key)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # train
        if not paused[0] and epoch[0] < MAX_EPOCH:
            for _ in range(EPOCHS_PER_FRAME):
                if epoch[0] >= MAX_EPOCH:
                    break
                train_epoch(net, X, Y, LR, rng)
                epoch[0] += 1

            # recompute textures
            region_tex, logit_tex, nr = compute_textures(net, xr, yr, RES)
            n_regions[0] = nr
            update_tex_rgba(t_region, region_tex)
            update_tex_r32f(t_logit, logit_tex)

        # draw
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        vmin, vmax = bounds()

        GL.glUseProgram(prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t_region)
        GL.glUniform1i(GL.glGetUniformLocation(prog, "u_region"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t_logit)
        GL.glUniform1i(GL.glGetUniformLocation(prog, "u_logit"), 1)
        GL.glUniform2f(GL.glGetUniformLocation(prog, "u_view_min"), *vmin)
        GL.glUniform2f(GL.glGetUniformLocation(prog, "u_view_max"), *vmax)
        GL.glUniform2f(GL.glGetUniformLocation(prog, "u_tex_min"),  *tex_min)
        GL.glUniform2f(GL.glGetUniformLocation(prog, "u_tex_max"),  *tex_max)
        quad.draw()

        GL.glUseProgram(prog_pts)
        GL.glUniform2f(GL.glGetUniformLocation(prog_pts, "u_view_min"), *vmin)
        GL.glUniform2f(GL.glGetUniformLocation(prog_pts, "u_view_max"), *vmax)
        points.draw()

        status = "paused" if paused[0] else ("done" if epoch[0] >= MAX_EPOCH else "training")
        acc = (np.argmax(net.forward(X), axis=1) == y).mean()
        glfw.set_window_title(window,
            f"epoch {epoch[0]}/{MAX_EPOCH}  |  {n_regions[0]} regions  |  "
            f"acc {acc:.2f}  |  {status}  |  space=pause  r=restart")

        glfw.swap_buffers(window)

    quad.destroy()
    points.destroy()
    GL.glDeleteTextures(2, [t_region, t_logit])
    GL.glDeleteProgram(prog)
    GL.glDeleteProgram(prog_pts)
    glfw.terminate()


if __name__ == "__main__":
    run()
