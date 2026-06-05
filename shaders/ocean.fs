#version 330

in vec3 fragWorld;
in vec2 fragUV;
in float fragHeight;
out vec4 finalColor;

uniform sampler2D texture0;   // r=height, g=Dx, b=Dz
uniform vec3 uSunDir;
uniform vec3 uCamPos;
uniform vec3 uDeep;
uniform vec3 uShallow;
uniform vec3 uSky;
uniform vec3 uSunCol;
uniform float uVScale;
uniform float uGridN;      // height-texture resolution
uniform float uWorldSize;  // world span of one patch

void main()
{
    // Normal from the periodic height texture via central differences. With
    // REPEAT wrapping this is seamless across tiled patches (no edge seams).
    float texel = 1.0 / uGridN;
    float hL = texture(texture0, fragUV - vec2(texel, 0.0)).r;
    float hR = texture(texture0, fragUV + vec2(texel, 0.0)).r;
    float hD = texture(texture0, fragUV - vec2(0.0, texel)).r;
    float hU = texture(texture0, fragUV + vec2(0.0, texel)).r;
    float dWorld = uWorldSize / uGridN;  // world distance between texels
    vec3 N = normalize(vec3(-(hR - hL) * uVScale,
                            2.0 * dWorld,
                            -(hU - hD) * uVScale));

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
