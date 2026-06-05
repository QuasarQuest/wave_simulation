#version 330

in vec3 fragWorld;
in vec2 fragUV;
in float fragHeight;
out vec4 finalColor;

uniform sampler2D texture0;   // r=height, g=Dx, b=Dz
uniform sampler2D texture2;   // r=dH/dx, g=dH/dz  (raylib normal-map slot)
uniform vec3 uSunDir;
uniform vec3 uCamPos;
uniform vec3 uDeep;
uniform vec3 uShallow;
uniform vec3 uSky;
uniform vec3 uSunCol;
uniform float uVScale;
uniform float uSlopeScale;

void main()
{
    // Normal straight from the FFT slope fields. These are periodic, so the
    // bilinear-sampled normal is continuous across tiled patches -- no seams.
    vec2 slope = texture(texture2, fragUV).rg * uSlopeScale;
    vec3 N = normalize(vec3(-slope.x, 1.0, -slope.y));

    vec3 V = normalize(uCamPos - fragWorld);
    vec3 L = normalize(uSunDir);

    // Base water colour: deeper troughs darker, crests lighter.
    float h = clamp(fragHeight * uVScale * 0.05 + 0.5, 0.0, 1.0);
    vec3 base = mix(uDeep, uShallow, h);

    float diff = max(dot(N, L), 0.0);
    vec3 col = base * (0.25 + 0.75 * diff);

    // Fresnel sky reflection -- grazing angles reflect the sky.
    float fres = mix(0.02, 1.0, pow(1.0 - max(dot(N, V), 0.0), 5.0));
    col = mix(col, uSky, fres * 0.55);

    // Sharp sun glint.
    vec3 Hh = normalize(L + V);
    float spec = pow(max(dot(N, Hh), 0.0), 220.0);
    col += uSunCol * spec * 1.4;

    // Subtle whitecaps on steep up-facing crests. Driven by the local slope
    // (steepness) rather than raw height so it doesn't spike at patch edges.
    float steep = 1.0 - clamp(N.y, 0.0, 1.0);
    float foam = smoothstep(0.35, 0.7, steep);
    col = mix(col, vec3(0.85, 0.92, 1.0), clamp(foam, 0.0, 0.5));

    finalColor = vec4(col, 1.0);
}
