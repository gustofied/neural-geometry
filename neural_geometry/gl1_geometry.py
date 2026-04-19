"""
gl1_geometry.py — ReLU region viewer

Move mouse over the field to light up the linear region under the cursor.
The decision boundary glows red. Drag to pan, scroll to zoom.
"""
import os, sys, ctypes
import numpy as np
import OpenGL.GL as GL
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw
import glfw.GLFW as GLFW_CONSTANTS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relu import build


# ── grid ───────────────────────────────────────────────────────────────────
def build_grid(model, x_range, y_range, res):
    print("  building grid…")
    xs = np.linspace(*x_range, res)
    ys = np.linspace(*y_range, res)
    grid = np.c_[np.meshgrid(xs, ys)[0].ravel(), np.meshgrid(xs, ys)[1].ravel()]

    # logit diff (before softmax)
    x = grid
    for l in model.layers[:-1]: x = l.forward(x)
    logit_diff = x[:, 1] - x[:, 0]

    # joint regions
    model.forward(grid)
    m1 = model.layers[1].x_in > 0
    m2 = model.layers[3].x_in > 0
    _, ids = np.unique(np.concatenate([m1, m2], axis=1), axis=0, return_inverse=True)

    # region texture: region ID packed into RG channels (uint16 via uint8 pair)
    r_lo = (ids % 256).astype(np.uint8)
    r_hi = (ids // 256).astype(np.uint8)
    region_tex = np.stack([r_lo, r_hi, np.zeros_like(r_lo), np.full_like(r_lo, 255)], axis=1)
    region_tex = region_tex.reshape(res, res, 4)[::-1].copy()

    # logit texture: float [0,1], 0.5 = decision boundary
    logit_tex = np.clip((logit_diff + 12) / 24, 0, 1).astype(np.float32)
    logit_tex = logit_tex.reshape(res, res)[::-1].copy()

    return ids, region_tex, logit_tex


# ── GLSL ───────────────────────────────────────────────────────────────────
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
uniform int       u_hover;
uniform vec2      u_view_min, u_view_max, u_tex_min, u_tex_max;
uniform vec2      u_auto_pos;
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
        f_color = vec4(0.022, 0.022, 0.038, 1.0);
        return;
    }

    int  rid    = decode(tuv);
    vec2 ts     = 1.0 / vec2(textureSize(u_region, 0));
    bool on_bnd = decode(tuv + vec2(ts.x, 0.0)) != rid
               || decode(tuv + vec2(0.0, ts.y)) != rid;

    float logit = texture(u_logit, tuv).r;
    float d     = abs(logit - 0.5) * 2.0;
    float glow  = exp(-d * d * 22.0);

    vec3 col = vec3(0.022, 0.025, 0.042);

    if (rid == u_hover)  col += vec3(0.22, 0.03, 0.26);
    if (on_bnd)          col += vec3(0.08, 0.14, 0.12) * 0.22;

    col += vec3(0.96, 0.04, 0.28) * glow;

    // auto-cursor dot — same pink, softer
    if (u_auto_pos.x > -999.0) {
        float dd   = length(world - u_auto_pos);
        col += vec3(0.96, 0.04, 0.28) * exp(-dd * dd * 280.0) * 0.55;
        col += vec3(0.96, 0.04, 0.28) * exp(-dd * dd * 40.0)  * 0.18;
    }

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
    gl_PointSize = 7.0;
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
    f_color = vec4(v_col, a);
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


def _upload(arr, internal, fmt, dtype, filter_=GL.GL_LINEAR):
    h, w = arr.shape[:2]
    tid  = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, dtype, arr.tobytes())
    for p, v in [(GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                 (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE),
                 (GL.GL_TEXTURE_MIN_FILTER, filter_),
                 (GL.GL_TEXTURE_MAG_FILTER, filter_)]:
        GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tid


class Renderer:
    def __init__(self, X, y, ids, region_tex, logit_tex,
                 tex_min, tex_max, grid_res, x_range, y_range):
        GL.glClearColor(0.022, 0.022, 0.038, 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

        self.prog     = make_program(VERT_SRC,    FRAG_SRC)
        self.prog_pts = make_program(VERT_PT_SRC, FRAG_PT_SRC)
        self.quad     = QuadMesh()
        self.points   = PointCloud(X, y)

        self.t_region = _upload(region_tex, GL.GL_RGBA, GL.GL_RGBA,
                                GL.GL_UNSIGNED_BYTE, GL.GL_NEAREST)

        h, w = logit_tex.shape
        self.t_logit = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.t_logit)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F, w, h, 0,
                        GL.GL_RED, GL.GL_FLOAT, logit_tex.tobytes())
        for p, v in [(GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                     (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE),
                     (GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR),
                     (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)]:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self.tex_min  = tex_min
        self.tex_max  = tex_max
        self.ids      = ids
        self.res      = grid_res
        self.x_range  = x_range
        self.y_range  = y_range
        self.hover    = -1
        self.auto_pos = np.array([-9999.0, -9999.0], np.float32)

    def world_to_region(self, wx, wy):
        tx = (wx - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        ty = (wy - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
        if not (0.0 <= tx <= 1.0 and 0.0 <= ty <= 1.0):
            return -1
        col = min(int(tx * self.res), self.res - 1)
        row = min(int(ty * self.res), self.res - 1)
        return int(self.ids[row * self.res + col])

    def draw(self, vmin, vmax):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self.prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.t_region)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "u_region"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.t_logit)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "u_logit"), 1)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "u_hover"), self.hover)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, "u_auto_pos"), *self.auto_pos)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, "u_view_min"), *vmin)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, "u_view_max"), *vmax)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, "u_tex_min"),  *self.tex_min)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, "u_tex_max"),  *self.tex_max)
        self.quad.draw()

        GL.glUseProgram(self.prog_pts)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog_pts, "u_view_min"), *vmin)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog_pts, "u_view_max"), *vmax)
        self.points.draw()

    def destroy(self):
        self.quad.destroy()
        self.points.destroy()
        GL.glDeleteTextures(2, [self.t_region, self.t_logit])
        GL.glDeleteProgram(self.prog)
        GL.glDeleteProgram(self.prog_pts)


# ── run ────────────────────────────────────────────────────────────────────
def run():
    net, X, y, _, _ = build()

    margin  = 0.6
    xr = (X[:,0].min()-margin, X[:,0].max()+margin)
    yr = (X[:,1].min()-margin, X[:,1].max()+margin)
    RES = 600
    ids, region_tex, logit_tex = build_grid(net, xr, yr, RES)
    tex_min = np.array([xr[0], yr[0]], np.float32)
    tex_max = np.array([xr[1], yr[1]], np.float32)

    cx   = float((xr[0]+xr[1])/2)
    cy   = float((yr[0]+yr[1])/2)
    half = float(max(xr[1]-xr[0], yr[1]-yr[0]) * 0.6)
    WIN  = 820
    drag_active  = False
    drag_last    = (0.0, 0.0)
    mouse_last_t = [0.0]
    data_cx      = float(X[:, 0].mean())
    data_cy      = float(X[:, 1].mean())

    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WIN, WIN, "gl1", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    ren = Renderer(X, y, ids, region_tex, logit_tex, tex_min, tex_max, RES, xr, yr)

    def bounds():
        return (np.array([cx-half, cy-half], np.float32),
                np.array([cx+half, cy+half], np.float32))

    def s2w(sx, sy):
        lo, hi = bounds()
        return lo[0]+(sx/WIN)*(hi[0]-lo[0]), lo[1]+(1-sy/WIN)*(hi[1]-lo[1])

    def on_key(win, key, sc, action, mods):
        if action == glfw.PRESS and key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            glfw.set_window_should_close(win, True)

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
        mouse_last_t[0] = glfw.get_time()
        wx, wy      = s2w(xpos, ypos)
        ren.hover   = ren.world_to_region(wx, wy)
        if drag_active:
            lo, hi  = bounds()
            cx -= (xpos - drag_last[0]) * (hi[0]-lo[0]) / WIN
            cy += (ypos - drag_last[1]) * (hi[1]-lo[1]) / WIN
        drag_last = (xpos, ypos)

    glfw.set_key_callback(window,          on_key)
    glfw.set_scroll_callback(window,       on_scroll)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window,   on_cursor)

    old_t = glfw.get_time()
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
            glfw.set_window_should_close(window, True)
        t = glfw.get_time(); dt = t - old_t; old_t = t
        if t - mouse_last_t[0] > 1.5:
            auto_wx = data_cx + 1.6 * np.sin(t * 0.31)
            auto_wy = data_cy + 1.6 * np.sin(t * 0.57)
            ren.hover    = ren.world_to_region(auto_wx, auto_wy)
            ren.auto_pos = np.array([auto_wx, auto_wy], np.float32)
        else:
            ren.auto_pos = np.array([-9999.0, -9999.0], np.float32)
        glfw.set_window_title(window, f"gl1  |  FPS {1/dt if dt>0 else 0:.0f}")
        ren.draw(*bounds())
        glfw.swap_buffers(window)

    ren.destroy()
    glfw.terminate()


if __name__ == "__main__":
    run()
