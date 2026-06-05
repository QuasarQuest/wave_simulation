#version 330

// raylib built-in vertex attributes / uniforms
in vec3 vertexPosition;
in vec2 vertexTexCoord;
uniform mat4 mvp;
uniform mat4 matModel;

// texture0 is raylib's MATERIAL_MAP_ALBEDO slot; we pack the ocean fields into
// it: r = vertical height, g = choppy displacement x, b = choppy displacement z.
uniform sampler2D texture0;
uniform float uVScale;    // vertical exaggeration
uniform float uHScale;    // horizontal (choppiness) scale
uniform float uUVRepeat;  // how many times the patch tiles across the mesh

out vec3 fragWorld;
out vec2 fragUV;
out float fragHeight;

void main()
{
    // One continuous mesh; UVs repeat so the (REPEAT-wrapped) ocean texture
    // tiles seamlessly with no inter-patch edges/gaps.
    vec2 uv = vertexTexCoord * uUVRepeat;
    vec4 d = texture(texture0, uv);
    vec3 disp = vec3(d.g * uHScale, d.r * uVScale, d.b * uHScale);
    vec3 pos = vertexPosition + disp;
    fragWorld = (matModel * vec4(pos, 1.0)).xyz;
    fragUV = uv;
    fragHeight = d.r;
    gl_Position = mvp * vec4(pos, 1.0);
}
