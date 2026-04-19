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
from data import make_radial_bands
from relu import ReLU, Softmax, CrossEntropy, Linear, Model, one_hot


# ── fast grid computation ────────────────────────────────────────────────
RES = 250

def _hash_regions(m1, m2):
    """Hash activation patterns into uint32 IDs without np.unique."""
    k1 = np.zeros(m1.shape[0], dtype=np.uint64)
    k2 = np.zeros(m2.shape[0], dtype=np.uint64)
    for i in range(m1.shape[1]):
        k1 |= m1[:, i].astype(np.uint64) << np.uint64(i)
    for i in range(m2.shape[1]):
        k2 |= m2[:, i].astype(np.uint64) << np.uint64(i)
    return ((k1 * np.uint64(1315423911)) ^ (k2 * np.uint64(2654435761))).astype(np.uint32)


def _id_to_rgb(ids):
    """Hash region IDs to pastel RGB on CPU."""
    h = (ids.astype(np.float32) * 0.61803398875) % 1.0
    r = 0.30 + 0.60 * (0.5 + 0.5 * np.sin(2*np.pi*h))
    g = 0.25 + 0.60 * (0.5 + 0.5 * np.sin(2*np.pi*h + 2.1))
    b = 0.30 + 0.60 * (0.5 + 0.5 * np.sin(2*np.pi*h + 4.2))
    rgb = np.stack([r, g, b], axis=-1)
    return (255 * np.clip(rgb, 0, 1)).astype(np.uint8)


def compute_frame(model, grid, res):
    """Single forward pass, returns color texture + logit texture."""
    z1 = model.layers[0].forward(grid)
    a1 = model.layers[1].forward(z1)
    z2 = model.layers[2].forward(a1)
    a2 = model.layers[3].forward(z2)
    z3 = model.layers[4].forward(a2)

    logit_diff = z3[:, 1] - z3[:, 0]
    ids = _hash_regions(z1 > 0, z2 > 0)

    color_tex = _id_to_rgb(ids).reshape(res, res, 3)[::-1].copy()
    logit_tex = np.clip((logit_diff + 12) / 24, 0, 1).astype(np.float32)
    logit_tex = logit_tex.reshape(res, res)[::-1].copy()

    return color_tex, logit_tex


# ── GLSL ─────────────────────────────────────────────────────────────────
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
uniform sampler2D u_color;
uniform sampler2D u_logit;
uniform vec2      u_view_min, u_view_max, u_tex_min, u_tex_max;
uniform float     u_boundary_str;
in  vec2 v_uv;
out vec4 f_color;

void main() {
    vec2 world = mix(u_view_min, u_view_max, v_uv);
    vec2 tuv   = (world - u_tex_min) / (u_tex_max - u_tex_min);

    if (any(lessThan(tuv, vec2(0.0))) || any(greaterThan(tuv, vec2(1.0)))) {
        f_color = vec4(0.020, 0.020, 0.032, 1.0);
        return;
    }

    vec3 region_col = texture(u_color, tuv).rgb;
    vec2 ts = 1.0 / vec2(textureSize(u_color, 0));

    // edge detection on color difference
    vec3 cr = texture(u_color, tuv + vec2(ts.x, 0.0)).rgb;
    vec3 cl = texture(u_color, tuv - vec2(ts.x, 0.0)).rgb;
    vec3 cu = texture(u_color, tuv + vec2(0.0, ts.y)).rgb;
    vec3 cd = texture(u_color, tuv - vec2(0.0, ts.y)).rgb;
    float edge = length(cr - region_col) + length(cl - region_col)
               + length(cu - region_col) + length(cd - region_col);
    bool on_bnd = edge > 0.01;

    float logit = texture(u_logit, tuv).r;
    float d     = abs(logit - 0.5) * 2.0;
    float glow  = exp(-d * d * 22.0);

    vec3 col = region_col;

    // white region edges
    if (on_bnd) col = mix(col, vec3(0.95, 0.93, 0.88), 0.72);

    // decision boundary: thin, grows with training
    col += vec3(0.96, 0.04, 0.28) * glow * 0.28 * u_boundary_str;

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
    gl_PointSize = 5.0;
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
    f_color = vec4(v_col, a * 0.5);
}
"""


# ── GPU helpers ──────────────────────────────────────────────────────────
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


class PointCloud:
    def __init__(self, X, y):
        cols = np.array([(0.69, 0.50, 0.40) if yi==0 else (0.41, 0.60, 0.66) for yi in y],
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


def make_tex(internal, fmt, dtype, w, h, data=None, filter_=GL.GL_LINEAR):
    tid = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, dtype,
                    data.tobytes() if data is not None else None)
    for p, v in [(GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                 (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE),
                 (GL.GL_TEXTURE_MIN_FILTER, filter_),
                 (GL.GL_TEXTURE_MAG_FILTER, filter_)]:
        GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tid


def update_tex(tid, fmt, dtype, w, h, data):
    GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
    GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, dtype, data.tobytes())
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)


# ── run ──────────────────────────────────────────────────────────────────
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

    # precompute grid once
    xs = np.linspace(*xr, RES)
    ys = np.linspace(*yr, RES)
    grid = np.c_[np.meshgrid(xs, ys)[0].ravel(), np.meshgrid(xs, ys)[1].ravel()]

    cx   = float((xr[0]+xr[1])/2)
    cy   = float((yr[0]+yr[1])/2)
    half = float(max(xr[1]-xr[0], yr[1]-yr[0]) * 0.6)
    WIN  = 820

    drag_active = False
    drag_last   = (0.0, 0.0)
    paused      = [False]
    lr          = 0.05
    batch_size  = 64
    rng         = np.random.default_rng(0)
    epoch       = [0]
    perm        = rng.permutation(len(X))
    ptr         = [0]

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
    color_tex, logit_tex = compute_frame(net, grid, RES)
    t_color = make_tex(GL.GL_RGB, GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                       RES, RES, color_tex, GL.GL_NEAREST)
    t_logit = make_tex(GL.GL_R32F, GL.GL_RED, GL.GL_FLOAT,
                       RES, RES, logit_tex)

    tex_min = np.array([xr[0], yr[0]], np.float32)
    tex_max = np.array([xr[1], yr[1]], np.float32)

    last_tex_t = glfw.get_time()

    def bounds():
        return (np.array([cx-half, cy-half], np.float32),
                np.array([cx+half, cy+half], np.float32))

    def train_step(n_batches=2):
        nonlocal perm
        for _ in range(n_batches):
            if epoch[0] >= 2000:
                return
            if ptr[0] >= len(X):
                perm = rng.permutation(len(X))
                ptr[0] = 0
                epoch[0] += 1

            idx = perm[ptr[0]:ptr[0] + batch_size]
            ptr[0] += batch_size

            xb, yb = X[idx], Y[idx]
            net.loss(xb, yb)
            net.backward()
            for layer in net.layers:
                if isinstance(layer, Linear):
                    layer.weights -= lr * layer.grad_w
                    layer.biases  -= lr * layer.grad_b

    def refresh_textures():
        nonlocal color_tex, logit_tex
        color_tex, logit_tex = compute_frame(net, grid, RES)
        update_tex(t_color, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, RES, RES, color_tex)
        update_tex(t_logit, GL.GL_RED, GL.GL_FLOAT, RES, RES, logit_tex)

    def on_key(win, key, sc, action, mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE:
            paused[0] = not paused[0]

    def on_scroll(win, dx, dy):
        nonlocal half, cx, cy
        xpos, ypos = glfw.get_cursor_pos(win)
        lo, hi     = bounds()
        mx = lo[0] + (xpos / WIN) * (hi[0] - lo[0])
        my = lo[1] + (1.0 - ypos / WIN) * (hi[1] - lo[1])
        half       = float(np.clip(half * 0.9**dy, 0.05, 20.0))
        lo2, hi2   = bounds()
        cx        += mx - (lo2[0] + (xpos / WIN) * (hi2[0] - lo2[0]))
        cy        += my - (lo2[1] + (1.0 - ypos / WIN) * (hi2[1] - lo2[1]))

    def on_mouse_button(win, button, action, mods):
        nonlocal drag_active, drag_last
        if button == glfw.MOUSE_BUTTON_LEFT:
            drag_active = action == glfw.PRESS
            drag_last   = glfw.get_cursor_pos(win)

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
        now = glfw.get_time()

        if not paused[0]:
            train_step(n_batches=2)

            if now - last_tex_t > 0.08:
                refresh_textures()
                last_tex_t = now

        boundary_str = min(epoch[0] / 400.0, 1.0)
        acc = (net.predict(X) == y).mean() if epoch[0] > 0 else 0.0
        status = "paused" if paused[0] else "training"
        glfw.set_window_title(window,
            f"epoch {epoch[0]}  |  acc {acc:.2f}  |  {status}")

        vmin, vmax = bounds()
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t_color)
        GL.glUniform1i(GL.glGetUniformLocation(prog, "u_color"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t_logit)
        GL.glUniform1i(GL.glGetUniformLocation(prog, "u_logit"), 1)
        GL.glUniform2f(GL.glGetUniformLocation(prog, "u_view_min"), *vmin)
        GL.glUniform2f(GL.glGetUniformLocation(prog, "u_view_max"), *vmax)
        GL.glUniform2f(GL.glGetUniformLocation(prog, "u_tex_min"),  *tex_min)
        GL.glUniform2f(GL.glGetUniformLocation(prog, "u_tex_max"),  *tex_max)
        GL.glUniform1f(GL.glGetUniformLocation(prog, "u_boundary_str"), boundary_str)
        quad.draw()

        GL.glUseProgram(prog_pts)
        GL.glUniform2f(GL.glGetUniformLocation(prog_pts, "u_view_min"), *vmin)
        GL.glUniform2f(GL.glGetUniformLocation(prog_pts, "u_view_max"), *vmax)
        points.draw()

        glfw.swap_buffers(window)

    GL.glDeleteTextures(2, [t_color, t_logit])
    GL.glDeleteProgram(prog)
    GL.glDeleteProgram(prog_pts)
    glfw.terminate()


if __name__ == "__main__":
    run()
