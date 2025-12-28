// Postprocessing to convert a render texture + depth map into an occlusion map.
export const OcclusionMapShader = {
  name: 'OcclusionMapShader',
  defines: {},

  vertexShader: /* glsl */ `
varying vec2 vTexCoord;

void main() {
    vTexCoord = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
  `,

  fragmentShader: /* glsl */ `
#include <packing>

precision mediump float;

uniform sampler2DArray uDepthTextureArray;
uniform mat4 uUvTransform;
uniform float uRawValueToMeters;
uniform float uAlpha;
uniform float uViewId;
uniform bool uFloatDepth;
uniform bool uIsGpuDepth;
uniform float uDepthNear;

uniform sampler2D tDiffuse;
uniform sampler2D tDepth;
uniform float cameraNear;
uniform float cameraFar;

varying vec2 vTexCoord;

float DepthArrayGetMeters(in sampler2DArray depth_texture, in vec2 depth_uv) {
  if (uIsGpuDepth) {
    float textureValue = texture(depth_texture, vec3(depth_uv.x, depth_uv.y, uViewId)).r;
    return uRawValueToMeters * uDepthNear / (1.0 - textureValue);
  }
  float textureValue = texture(depth_texture, vec3(depth_uv.x, 1.0 - depth_uv.y, uViewId)).r;
  return uRawValueToMeters * textureValue; 
}


float readOrthographicDepth( sampler2D depthSampler, vec2 coord ) {
  float fragCoordZ = texture2D( depthSampler, coord ).x;
  // See https://github.com/mrdoob/three.js/issues/23072.
  #ifdef USE_LOGDEPTHBUF
    float viewZ = 1.0 - exp2(fragCoordZ * log(cameraFar + 1.0) / log(2.0));
  #else
    float viewZ = perspectiveDepthToViewZ(fragCoordZ, cameraNear, cameraFar);
  #endif
  return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );
}

void main(void) {
  vec4 texCoord = vec4(vTexCoord, 0, 1);
  vec2 uv = texCoord.xy;
  uv.y = 1.0 - uv.y;

  vec4 diffuse = texture2D( tDiffuse, texCoord.xy );
  highp float real_depth = DepthArrayGetMeters(uDepthTextureArray, uv);
  highp float virtual_depth =
    (readOrthographicDepth(tDepth, texCoord.xy ) *
    (cameraFar - cameraNear) + cameraNear);
  gl_FragColor = vec4(step(virtual_depth, real_depth), step(0.001, diffuse.a), 0.0, 0.0);
}
`,
};
