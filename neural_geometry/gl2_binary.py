"""
gl2_binary.py — MAP vs LLLA confidence field

Colors encode the label field: orange = class 0, teal = class 1.
Saturation encodes conviction — vivid near data, grey where uncertain.
MAP stays vivid everywhere. LLLA fades into grey far from training data.

Sampled posterior decision boundaries are drawn as a pink fan: tight
near training data (model is certain), fanning out where it is not.
The bright white line is the MAP boundary.

Drag to pan, scroll to zoom on cursor. Divider sweeps automatically.
"""
import os, sys, ctypes
import numpy as np
import OpenGL.GL as GL
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw
import glfw.GLFW as GLFW_CONSTANTS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bayesian import build, get_features


# ── probability grids ──────────────────────────────────────────────────────
def build_grids(model, bll, x_range, y_range, res, n_samples=200):
    print("  computing probability fields…")
    xs = np.linspace(*x_range, res)
    ys = np.linspace(*y_range, res)
    xx, yy = np.meshgrid(xs, ys)
    grid   = np.c_[xx.ravel(), yy.ravel()]

    p_map = model.forward(grid).flatten()
    phi   = get_features(model, grid)
    p_bll = bll.sample(phi, n_samples=n_samples).mean(axis=0)

    # flip: OpenGL bottom-left origin
    return (p_map.reshape(res, res)[::-1].copy().astype(np.float32),
            p_bll.reshape(res, res)[::-1].copy().astype(np.float32))


# ── posterior boundary lines ───────────────────────────────────────────────
def _contour_segs(logit_2d, xs, ys):
    """Vectorized marching squares — returns (N, 2) float32 vertex pairs for GL_LINES."""
    H, W = logit_2d.shape

    # Horizontal edges: logit[j, i] → logit[j, i+1]
    L = logit_2d[:, :-1];  R = logit_2d[:, 1:]
    h_mask = (L * R) < 0
    h_t  = np.where(h_mask, L / (L - R), 0.0)
    h_wx = xs[:-1] + h_t * (xs[1:] - xs[:-1])          # (H, W-1)
    h_wy = np.broadcast_to(ys[:, None], (H, W-1))

    # Vertical edges: logit[j, i] → logit[j+1, i]
    T = logit_2d[:-1, :];  B = logit_2d[1:, :]
    v_mask = (T * B) < 0
    v_t  = np.where(v_mask, T / (T - B), 0.0)
    v_wx = np.broadcast_to(xs[None, :], (H-1, W))
    v_wy = ys[:-1, None] + v_t * (ys[1:, None] - ys[:-1, None])  # (H-1, W)

    # Per-cell edge masks (H-1, W-1)
    top_m   = h_mask[:-1, :];  top_x,  top_y  = h_wx[:-1, :],  h_wy[:-1, :]
    bot_m   = h_mask[1:,  :];  bot_x,  bot_y  = h_wx[1:,  :],  h_wy[1:,  :]
    left_m  = v_mask[:, :-1];  left_x, left_y = v_wx[:, :-1],  v_wy[:, :-1]
    right_m = v_mask[:, 1: ];  right_x,right_y= v_wx[:, 1: ],  v_wy[:, 1: ]

    parts = []

    def seg(mask, ax, ay, bx, by):
        if not mask.any():
            return
        m = mask.ravel()
        n = m.sum()
        out = np.empty((n * 2, 2), np.float32)
        out[0::2, 0] = ax.ravel()[m];  out[0::2, 1] = ay.ravel()[m]
        out[1::2, 0] = bx.ravel()[m];  out[1::2, 1] = by.ravel()[m]
        parts.append(out)

    seg(top_m  & left_m  & ~bot_m  & ~right_m, top_x,  top_y,  left_x,  left_y)
    seg(top_m  & right_m & ~bot_m  & ~left_m,  top_x,  top_y,  right_x, right_y)
    seg(top_m  & bot_m   & ~left_m & ~right_m, top_x,  top_y,  bot_x,   bot_y)
    seg(bot_m  & left_m  & ~top_m  & ~right_m, bot_x,  bot_y,  left_x,  left_y)
    seg(bot_m  & right_m & ~top_m  & ~left_m,  bot_x,  bot_y,  right_x, right_y)
    seg(left_m & right_m & ~top_m  & ~bot_m,   left_x, left_y, right_x, right_y)
    # saddle (4 crossings) — arbitrary split
    saddle = top_m & bot_m & left_m & right_m
    seg(saddle, top_x, top_y, left_x, left_y)
    seg(saddle, bot_x, bot_y, right_x, right_y)

    return np.concatenate(parts) if parts else np.zeros((0, 2), np.float32)


def compute_boundary_lines(net, bll, x_range, y_range, res=150, n_samples=80):
    print("  sampling posterior boundaries…")
    xs = np.linspace(*x_range, res)
    ys = np.linspace(*y_range, res)
    grid = np.c_[np.meshgrid(xs, ys)[0].ravel(), np.meshgrid(xs, ys)[1].ravel()]

    phi = get_features(net, grid)           # (res², 64) — computed once
    rng = np.random.default_rng(7)

    all_segs = []
    for _ in range(n_samples):
        W = rng.normal(bll.W_mean, bll.W_std)
        b = rng.normal(bll.b_mean, bll.b_std)
        logit = (phi @ W + b).reshape(res, res)
        s = _contour_segs(logit, xs, ys)
        if len(s):
            all_segs.append(s)

    map_logit = (phi @ bll.W_mean + bll.b_mean).reshape(res, res)
    map_segs  = _contour_segs(map_logit, xs, ys)

    sampled = np.concatenate(all_segs) if all_segs else np.zeros((0, 2), np.float32)
    return sampled.astype(np.float32), map_segs.astype(np.float32)


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
uniform sampler2D u_map;
uniform sampler2D u_bll;
uniform float     u_split;
uniform vec2      u_view_min, u_view_max, u_tex_min, u_tex_max;
in  vec2 v_uv;
out vec4 f_color;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 world = mix(u_view_min, u_view_max, v_uv);
    vec2 tuv   = (world - u_tex_min) / (u_tex_max - u_tex_min);

    float p;
    if (any(lessThan(tuv, vec2(0.0))) || any(greaterThan(tuv, vec2(1.0)))) {
        p = 0.5;
    } else {
        p = (v_uv.x < u_split) ? texture(u_map, tuv).r
                                : texture(u_bll, tuv).r;
    }

    float hue = mix(0.08, 0.50, p);
    float sat = pow(abs(p - 0.5) * 2.0, 0.55) * 0.95;
    vec3 col  = hsv2rgb(vec3(hue, sat, 0.90));

    // divider glow
    float sd = abs(v_uv.x - u_split);
    col = mix(col, vec3(1.0), exp(-sd * sd * 80000.0) * 0.65);

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
    gl_PointSize = 8.0;
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

VERT_LINE_SRC = """
#version 330 core
layout(location=0) in vec2 in_pos;
uniform vec2 u_view_min, u_view_max;
void main() {
    vec2 ndc = (in_pos - u_view_min) / (u_view_max - u_view_min) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

FRAG_LINE_SRC = """
#version 330 core
uniform vec4 u_color;
out vec4 f_color;
void main() {
    f_color = u_color;
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
        cols = np.array([(1.0, 0.55, 0.0) if yi==0 else (0.0, 0.70, 0.78) for yi in y],
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


def upload_r32f(arr):
    h, w = arr.shape
    tid  = GL.glGenTextures(1)
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


class BoundaryLines:
    """Sampled posterior decision boundaries + MAP boundary in one VBO."""
    def __init__(self, sampled_verts, map_verts):
        self.n_sampled = len(sampled_verts)
        self.n_map     = len(map_verts)

        all_v = np.concatenate([sampled_verts, map_verts]) if (self.n_sampled and self.n_map) \
                else (sampled_verts if self.n_sampled else map_verts)

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, all_v.nbytes, all_v, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

    def draw_sampled(self):
        if not self.n_sampled:
            return
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.n_sampled)
        GL.glBindVertexArray(0)

    def draw_map(self):
        if not self.n_map:
            return
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_LINES, self.n_sampled, self.n_map)
        GL.glBindVertexArray(0)

    def destroy(self):
        GL.glDeleteVertexArrays(1, (self.vao,))
        GL.glDeleteBuffers(1, (self.vbo,))


class Renderer:
    def __init__(self, X, y, p_map, p_bll, sampled_segs, map_segs, tex_min, tex_max):
        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

        self.prog      = make_program(VERT_SRC,      FRAG_SRC)
        self.prog_pts  = make_program(VERT_PT_SRC,   FRAG_PT_SRC)
        self.prog_line = make_program(VERT_LINE_SRC, FRAG_LINE_SRC)
        self.quad      = QuadMesh()
        self.points    = PointCloud(X, y)
        self.boundary  = BoundaryLines(sampled_segs, map_segs)

        self.t_map = upload_r32f(p_map)
        self.t_bll = upload_r32f(p_bll)

        self.tex_min = tex_min
        self.tex_max = tex_max
        self.split   = 0.5

    def draw(self, vmin, vmax):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # confidence field
        GL.glUseProgram(self.prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.t_map)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "u_map"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.t_bll)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "u_bll"), 1)
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, "u_split"),    self.split)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, "u_view_min"), *vmin)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, "u_view_max"), *vmax)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, "u_tex_min"),  *self.tex_min)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog, "u_tex_max"),  *self.tex_max)
        self.quad.draw()

        # sampled boundary lines — additive blend so they glow where they stack
        GL.glUseProgram(self.prog_line)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog_line, "u_view_min"), *vmin)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog_line, "u_view_max"), *vmax)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        GL.glUniform4f(GL.glGetUniformLocation(self.prog_line, "u_color"),
                       0.96, 0.04, 0.28, 0.055)
        self.boundary.draw_sampled()

        # MAP boundary — bright white-pink, normal blend
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glUniform4f(GL.glGetUniformLocation(self.prog_line, "u_color"),
                       1.0, 0.90, 0.95, 0.92)
        self.boundary.draw_map()

        # data points on top
        GL.glUseProgram(self.prog_pts)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog_pts, "u_view_min"), *vmin)
        GL.glUniform2f(GL.glGetUniformLocation(self.prog_pts, "u_view_max"), *vmax)
        self.points.draw()

    def destroy(self):
        self.quad.destroy()
        self.points.destroy()
        self.boundary.destroy()
        GL.glDeleteTextures(2, [self.t_map, self.t_bll])
        GL.glDeleteProgram(self.prog)
        GL.glDeleteProgram(self.prog_pts)
        GL.glDeleteProgram(self.prog_line)


# ── run ────────────────────────────────────────────────────────────────────
def run():
    net, bll, X, y = build()

    margin = 5.5
    xc, yc = float(X[:,0].mean()), float(X[:,1].mean())
    xr = (xc - margin, xc + margin)
    yr = (yc - margin, yc + margin)

    RES = 500
    p_map, p_bll         = build_grids(net, bll, xr, yr, RES)
    sampled_segs, map_segs = compute_boundary_lines(net, bll, xr, yr)
    tex_min = np.array([xr[0], yr[0]], np.float32)
    tex_max = np.array([xr[1], yr[1]], np.float32)

    WIN         = 740
    half        = margin + 0.8
    cx, cy      = xc, yc
    drag_active = False
    drag_last   = (0.0, 0.0)

    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(WIN, WIN, "gl2", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    ren = Renderer(X, y, p_map, p_bll, sampled_segs, map_segs, tex_min, tex_max)

    def bounds():
        return (np.array([cx-half, cy-half], np.float32),
                np.array([cx+half, cy+half], np.float32))

    def on_key(win, key, sc, action, mods):
        if action == glfw.PRESS and key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            glfw.set_window_should_close(win, True)

    def on_scroll(win, dx, dy):
        nonlocal half, cx, cy
        xpos, ypos = glfw.get_cursor_pos(win)
        lo, hi     = bounds()
        mx = lo[0] + (xpos / WIN) * (hi[0] - lo[0])
        my = lo[1] + (1.0 - ypos / WIN) * (hi[1] - lo[1])
        half       = float(np.clip(half * 0.9**dy, 0.3, 20.0))
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

    glfw.set_key_callback(window,          on_key)
    glfw.set_scroll_callback(window,       on_scroll)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window,   on_cursor)

    print("  scroll=zoom-on-cursor   drag=pan   q=quit")

    old_t = glfw.get_time()
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
            glfw.set_window_should_close(window, True)
        t  = glfw.get_time(); dt = t - old_t; old_t = t
        ren.split = 0.5 + 0.44 * np.sin(t * 0.38)
        side = int(ren.split * 100)
        glfw.set_window_title(window,
            f"gl2  |  MAP {side}%  LLLA {100-side}%  |  FPS {1/dt if dt>0 else 0:.0f}")
        ren.draw(*bounds())
        glfw.swap_buffers(window)

    ren.destroy()
    glfw.terminate()


if __name__ == "__main__":
    run()
