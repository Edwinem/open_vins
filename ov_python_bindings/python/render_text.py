import moderngl
import numpy as np
from moderngl_window import resources
from moderngl_window.meta import (
    DataDescription,
    TextureDescription,
)
from moderngl_window.opengl.vao import VAO
from moderngl_window.text.bitmapped.base import BaseText, FontMeta
from pyrr import matrix44

# Shaders for the text rendering

text_vertex_shader = """
#version 330

in uint char_id;
in float xpos;
in float ypos;

uniform vec2 char_size;

out uint vs_char_id;

void main() {
	gl_Position = vec4(vec3(xpos*char_size.x, ypos*char_size.y, 0.0), 1.0);
    vs_char_id = char_id;
}
"""

text_geometry_shader = """
#version 330

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 m_proj;
uniform vec2 text_pos;
uniform vec2 char_size;

in uint vs_char_id[1];
out vec2 uv;
flat out uint gs_char_id;

void main() {
    vec3 pos = gl_in[0].gl_Position.xyz + vec3(text_pos, 0.0);

    vec3 right = vec3(1.0, 0.0, 0.0) * char_size.x / 2.0;
    vec3 up = vec3(0.0, 1.0, 0.0) * char_size.y / 2.0;

    // upper right
    uv = vec2(1.0, 1.0);
    gs_char_id = vs_char_id[0];
    gl_Position = m_proj * vec4(pos + (right + up), 1.0);
    EmitVertex();

    // upper left
    uv = vec2(0.0, 1.0);
    gs_char_id = vs_char_id[0];
    gl_Position = m_proj * vec4(pos + (-right + up), 1.0);
    EmitVertex();

    // lower right
    uv = vec2(1.0, 0.0);
    gs_char_id = vs_char_id[0];
    gl_Position = m_proj * vec4(pos + (right - up), 1.0);
    EmitVertex();

    // lower left
    uv = vec2(0.0, 0.0);
    gs_char_id = vs_char_id[0];
    gl_Position = m_proj * vec4(pos + (-right - up), 1.0);
    EmitVertex();

    EndPrimitive();
}"""

text_fragment_shader = """
#version 330

out vec4 fragColor;
uniform sampler2DArray font_texture;
in vec2 uv;
flat in uint gs_char_id;

void main()
{
    fragColor = texture(font_texture, vec3(uv, gs_char_id));
}

"""


class CustomTextWriter2D(BaseText):
    """Simple monospaced bitmapped text renderer. It interprets \n as a new line"""

    def __init__(self, ctx):
        super().__init__()

        meta = FontMeta(resources.data.load(
            DataDescription(path="bitmapped/text/meta.json")
        ))
        self._texture = resources.textures.load(
            TextureDescription(
                path="bitmapped/textures/VeraMono.png",
                kind="array",
                mipmap=True,
                layers=meta.characters,
            )
        )
        self._program = ctx.program(vertex_shader=text_vertex_shader, fragment_shader=text_fragment_shader,
                                    geometry_shader=text_geometry_shader)

        self._init(meta)

        self._string_buffer = self.ctx.buffer(reserve=1024 * 4)
        self._string_buffer.clear(chunk=b'\32')
        pos = self.ctx.buffer(data=bytes([0] * 4 * 3))

        self._vao = VAO("textwriter", mode=moderngl.POINTS)

        self._text: str = None

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value
        self._write(value)

    def _write(self, text: str):
        char_ids = np.fromiter(
            self._translate_string(text),
            dtype=np.uint32,
        )
        y_positions = np.zeros((len(text)), dtype=np.float32)
        x_positions = np.zeros((len(text)), dtype=np.float32)
        new_line_counter = 0
        x_counter = 0
        for i in range(0, len(text)):
            y_positions[i] = -new_line_counter
            x_positions[i] = x_counter
            x_counter += 1
            if text[i] == "\n":
                new_line_counter += 1
                x_counter = 0

        self._vao.buffer(y_positions, 'f', 'ypos')
        self._vao.buffer(x_positions, 'f', 'xpos')
        self._vao.buffer(char_ids, 'u', 'char_id')

    def draw(self, pos, length=-1, size=24.0):
        # Calculate ortho projection based on viewport
        vp = self.ctx.fbo.viewport
        w, h = vp[2] - vp[0], vp[3] - vp[1]
        projection = matrix44.create_orthogonal_projection_matrix(
            0,  # left
            w,  # right
            0,  # bottom
            h,  # top
            1.0,  # near
            -1.0,  # far
            dtype=np.float32,
        )

        self._texture.use(location=0)
        self._program["m_proj"].write(projection)
        self._program["text_pos"].value = pos
        self._program["font_texture"].value = 0
        self._program["char_size"].value = self._meta.char_aspect_wh * size, size
        #        self._program["line_length"].value = 20

        self._vao.render(self._program)
