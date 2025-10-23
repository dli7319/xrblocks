import * as THREE from 'three';

export class OcclusionMapMeshMaterial extends THREE.MeshBasicMaterial {
  uniforms: {[uniform: string]: THREE.IUniform};

  constructor(camera: THREE.PerspectiveCamera, useFloatDepth: boolean) {
    super();
    this.uniforms = {
      uDepthTexture: {value: null},
      uDepthTextureArray: {value: null},
      uViewId: {value: 0.0},
      uIsTextureArray: {value: 0.0},
      uRawValueToMeters: {value: 8.0 / 65536.0},
      cameraFar: {value: camera.far},
      cameraNear: {value: camera.near},
      uFloatDepth: {value: useFloatDepth},
      // Used for interpreting Quest 3 depth.
      uDepthNear: {value: 0},
    };
    this.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, this.uniforms);
      this.uniforms = shader.uniforms;
      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <common>',
          [
            'varying vec2 vTexCoord;',
            'varying float vVirtualDepth;',
            '#include <common>',
          ].join('\n')
        )
        .replace(
          '#include <fog_vertex>',
          [
            '#include <fog_vertex>',
            'vec4 view_position = modelViewMatrix * vec4( position, 1.0 );',
            'vVirtualDepth = -view_position.z;',
            'gl_Position = gl_Position / gl_Position.w;',
            'vTexCoord = 0.5 + 0.5 * gl_Position.xy;',
          ].join('\n')
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          'uniform vec3 diffuse;',
          [
            'uniform vec3 diffuse;',
            'uniform sampler2D uDepthTexture;',
            'uniform sampler2DArray uDepthTextureArray;',
            'uniform float uRawValueToMeters;',
            'uniform float cameraNear;',
            'uniform float cameraFar;',
            'uniform bool uFloatDepth;',
            'uniform bool uIsTextureArray;',
            'uniform float uDepthNear;',
            'uniform int uViewId;',
            'varying vec2 vTexCoord;',
            'varying float vVirtualDepth;',
          ].join('\n')
        )
        .replace(
          '#include <clipping_planes_pars_fragment>',
          [
            '#include <clipping_planes_pars_fragment>',
            `
  float DepthGetMeters(in sampler2D depth_texture, in vec2 depth_uv) {
    // Depth is packed into the luminance and alpha components of its texture.
    // The texture is in a normalized format, storing raw values that need to be
    // converted to meters.
    vec2 packedDepthAndVisibility = texture2D(depth_texture, depth_uv).rg;
    if (uFloatDepth) {
      return packedDepthAndVisibility.r * uRawValueToMeters;
    }
    return dot(packedDepthAndVisibility, vec2(255.0, 256.0 * 255.0)) * uRawValueToMeters;
  }
  float DepthArrayGetMeters(in sampler2DArray depth_texture, in vec2 depth_uv) {
    float textureValue = texture(depth_texture, vec3(depth_uv.x, depth_uv.y, uViewId)).r;
    return uRawValueToMeters * uDepthNear / (1.0 - textureValue);
  }
`,
          ].join('\n')
        )
        .replace(
          '#include <dithering_fragment>',
          [
            '#include <dithering_fragment>',
            'vec4 texCoord = vec4(vTexCoord, 0, 1);',
            'vec2 uv = vec2(texCoord.x, uIsTextureArray?texCoord.y:(1.0 - texCoord.y));',
            'highp float real_depth = uIsTextureArray ? DepthArrayGetMeters(uDepthTextureArray, uv) : DepthGetMeters(uDepthTexture, uv);',
            'gl_FragColor = vec4(step(vVirtualDepth, real_depth), 1.0, 0.0, 1.0);',
          ].join('\n')
        );
    };
  }
}
