const canvas = document.getElementById('sparkleCanvas');
const statusEl = document.getElementById('status');
const densitySlider = document.getElementById('densitySlider');
const roughnessSlider = document.getElementById('roughnessSlider');
const densityValueEl = document.getElementById('densityValue');
const roughnessValueEl = document.getElementById('roughnessValue');
const shininessSlider = document.getElementById('shininessSlider');
const shininessValueEl = document.getElementById('shininessValue');
const luminanceSlider = document.getElementById('luminanceSlider');
const chromaSlider = document.getElementById('chromaSlider');
const hueSlider = document.getElementById('hueSlider');
const luminanceValueEl = document.getElementById('luminanceValue');
const chromaValueEl = document.getElementById('chromaValue');
const hueValueEl = document.getElementById('hueValue');
const autoRotateToggle = document.getElementById('autoRotateToggle');
const autoLightRotateToggle = document.getElementById('autoLightRotateToggle');

if (!canvas || !statusEl || !densitySlider || !roughnessSlider || !densityValueEl || !roughnessValueEl || !shininessSlider || !shininessValueEl || !luminanceSlider || !chromaSlider || !hueSlider || !luminanceValueEl || !chromaValueEl || !hueValueEl || !autoRotateToggle || !autoLightRotateToggle) {
  throw new Error('Missing required DOM elements.');
}

if (!navigator.gpu) {
  statusEl.textContent = 'WebGPU is not available in this browser.';
  throw new Error('WebGPU is not available.');
}

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) {
  statusEl.textContent = 'No suitable GPU adapter found.';
  throw new Error('No suitable GPU adapter found.');
}

const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');
if (!context) {
  statusEl.textContent = 'Failed to create WebGPU context.';
  throw new Error('Failed to create WebGPU context.');
}

const tryFormats = ['rgba16float', navigator.gpu.getPreferredCanvasFormat()];
let presentationFormat = null;
let toneMappingMode = 'standard';

function tryConfigure(format, mode) {
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
    colorSpace: 'display-p3',
    toneMapping: { mode },
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
  });
}

for (const fmt of tryFormats) {
  if (!fmt) continue;
  try {
    tryConfigure(fmt, 'extended');
    presentationFormat = fmt;
    toneMappingMode = 'extended';
    break;
  } catch {
    try {
      tryConfigure(fmt, 'standard');
      presentationFormat = fmt;
      toneMappingMode = 'standard';
      break;
    } catch {
      // Try next format.
    }
  }
}

if (!presentationFormat) {
  statusEl.textContent = 'Unable to configure a WebGPU canvas format.';
  throw new Error('Unable to configure a WebGPU canvas format.');
}

const cieSampler = device.createSampler({
  addressModeU: 'clamp-to-edge',
  addressModeV: 'clamp-to-edge',
  magFilter: 'linear',
  minFilter: 'linear',
});

let cieTexture = null;
try {
  const cieTextureInfo = await loadCieTexture(device, 'cie1931xyz2e.csv');
  cieTexture = cieTextureInfo.texture;
} catch (error) {
  console.warn('Failed to load cie1931xyz2e.csv, using fallback texture.', error);
  cieTexture = createFallbackCieTexture(device, 256);
}

const shader = device.createShaderModule({
  label: 'sparkle-object-shader',
  code: /* wgsl */ `
struct Uniforms {
  resolution: vec2<f32>,
  time_sec: f32,
  auto_yaw: f32,
  glint_density: f32,
  roughness: f32,
  rotation: vec2<f32>,
  shininess_scale: f32,
  zoom_scale: f32,
  light_rotation: vec2<f32>,
  auto_light_yaw: f32,
  _pad0: f32,
  oklch: vec3<f32>,
  _pad1: f32,
};

@group(0) @binding(0) var cieSampler: sampler;
@group(0) @binding(1) var cieTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

const microfacet_roughness: f32 = 0.01;
const pixel_filter_size: f32 = 0.7;
const pi: f32 = 3.14159265358979;

const LINEAR_SRGB_TO_LINEAR_P3: mat3x3<f32> = mat3x3<f32>(
  vec3<f32>(0.82246210, 0.03319419, 0.01708263),
  vec3<f32>(0.17753790, 0.96680581, 0.07239744),
  vec3<f32>(0.00000000, 0.00000000, 0.91051993)
);

fn srgb_transfer_function(a: f32) -> f32 {
  if (a <= 0.0031308) {
    return 12.92 * a;
  }
  return 1.055 * pow(a, 1.0 / 2.4) - 0.055;
}

fn linearSRGBToLinearP3(ls: vec3<f32>) -> vec3<f32> {
  return LINEAR_SRGB_TO_LINEAR_P3 * ls;
}

fn oklab_to_linear_srgb(lab: vec3<f32>) -> vec3<f32> {
  let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;

  let l = l_ * l_ * l_;
  let m = m_ * m_ * m_;
  let s = s_ * s_ * s_;

  return vec3<f32>(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  );
}

fn oklch_to_p3(oklch: vec3<f32>) -> vec3<f32> {
  let hue = oklch.z * pi / 180.0;
  let a = oklch.y * cos(hue);
  let b = oklch.y * sin(hue);
  let linearSRGB = oklab_to_linear_srgb(vec3<f32>(oklch.x, a, b));
  return linearSRGBToLinearP3(linearSRGB);
}

struct VSOut {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );

  var out: VSOut;
  out.position = vec4<f32>(positions[idx], 0.0, 1.0);
  return out;
}

fn lambert(v: vec3<f32>) -> vec2<f32> {
  return v.xy / sqrt(1.0 + v.z);
}

fn ndf_to_disk_ggx(v: vec3<f32>, alpha: f32) -> vec3<f32> {
  let hemi = vec3<f32>(v.xy / alpha, v.z);
  let denom = dot(hemi, hemi);
  let v_disk = lambert(normalize(hemi)) * 0.5 + vec2<f32>(0.5);
  let jacobian_determinant = 1.0 / (alpha * alpha * denom * denom);
  return vec3<f32>(v_disk, jacobian_determinant);
}

fn inv_quadratic(M: mat2x2<f32>) -> mat2x2<f32> {
  let D = determinant(M);
  let A = dot(M[0] / D, M[0] / D);
  let B = -dot(M[0] / D, M[1] / D);
  let C = dot(M[1] / D, M[1] / D);
  return mat2x2<f32>(vec2<f32>(C, B), vec2<f32>(B, A));
}

fn uv_ellipsoid(uv_J: mat2x2<f32>) -> mat2x2<f32> {
  let Q = inv_quadratic(transpose(uv_J));
  let tr = 0.5 * (Q[0][0] + Q[1][1]);
  let D = sqrt(max(0.0, tr * tr - determinant(Q)));
  let l1 = tr - D;
  let l2 = tr + D;
  let v1 = vec2<f32>(l1 - Q[1][1], Q[0][1]);
  let v2 = vec2<f32>(Q[1][0], l2 - Q[0][0]);
  let n = 1.0 / sqrt(vec2<f32>(l1, l2));
  return mat2x2<f32>(normalize(v1) * n.x, normalize(v2) * n.y);
}

fn QueryLod(uv_J: mat2x2<f32>, filter_size: f32) -> f32 {
  let s0 = length(uv_J[0]);
  let s1 = length(uv_J[1]);
  return log2(max(s0, s1) * filter_size) + pow(2.0, filter_size);
}

fn shuffle(v_in: vec2<u32>) -> vec2<u32> {
  var v = v_in;
  v = v * vec2<u32>(1664525u) + vec2<u32>(1013904223u);
  v.x = v.x + v.y * 1664525u;
  v.y = v.y + v.x * 1664525u;

  v = v ^ (v >> vec2<u32>(16u));

  v.x = v.x + v.y * 1664525u;
  v.y = v.y + v.x * 1664525u;
  v = v ^ (v >> vec2<u32>(16u));
  return v;
}

fn rand(v: vec2<u32>) -> vec2<f32> {
  return vec2<f32>(shuffle(v)) * pow(0.5, 32.0);
}

fn inverse2x2(m: mat2x2<f32>) -> mat2x2<f32> {
  let a = m[0].x;
  let b = m[1].x;
  let c = m[0].y;
  let d = m[1].y;
  let det = a * d - b * c;
  let inv_det = 1.0 / det;
  return mat2x2<f32>(
    vec2<f32>( d * inv_det, -c * inv_det),
    vec2<f32>(-b * inv_det,  a * inv_det)
  );
}

fn normal(cov: mat2x2<f32>, x: vec2<f32>) -> f32 {
  return exp(-0.5 * dot(x, inverse2x2(cov) * x)) / (sqrt(determinant(cov)) * 2.0 * pi);
}

fn Rand2D(x: vec2<f32>, y: vec2<f32>, l: f32, i: u32) -> vec2<f32> {
  let ux = bitcast<vec2<u32>>(x);
  let uy = bitcast<vec2<u32>>(y);
  let ul = bitcast<u32>(l);
  return rand(((ux >> vec2<u32>(16u)) | (ux << vec2<u32>(16u))) ^ uy ^ vec2<u32>(ul) ^ vec2<u32>(i * 0x124u));
}

fn Rand1D(x: vec2<f32>, y: vec2<f32>, l: f32, i: u32) -> f32 {
  return Rand2D(x, y, l, i).x;
}

fn erf_approx(x: f32) -> f32 {
  let e = exp(-x * x);
  return sign(x) * 2.0 * sqrt((1.0 - e) / pi) * (sqrt(pi) * 0.5 + 31.0 / 200.0 * e - 341.0 / 8000.0 * e * e);
}

fn cdf(x: f32, mu: f32, sigma: f32) -> f32 {
  return 0.5 + 0.5 * erf_approx((x - mu) / (sigma * sqrt(2.0)));
}

fn integrate_interval(x: f32, size: f32, mu: f32, stdev: f32, lower_limit: f32, upper_limit: f32) -> f32 {
  return cdf(min(x + size, upper_limit), mu, stdev) - cdf(max(x - size, lower_limit), mu, stdev);
}

fn integrate_box(
  x: vec2<f32>,
  size: vec2<f32>,
  mu: vec2<f32>,
  sigma: mat2x2<f32>,
  lower_limit: vec2<f32>,
  upper_limit: vec2<f32>
) -> f32 {
  return
    integrate_interval(x.x, size.x, mu.x, sqrt(sigma[0][0]), lower_limit.x, upper_limit.x) *
    integrate_interval(x.y, size.y, mu.y, sqrt(sigma[1][1]), lower_limit.y, upper_limit.y);
}

fn compensation(x_a: vec2<f32>, sigma_a: mat2x2<f32>, res_a: f32) -> f32 {
  let containing = integrate_box(vec2<f32>(0.5), vec2<f32>(0.5), x_a, sigma_a, vec2<f32>(0.0), vec2<f32>(1.0));
  let explicitly_evaluated = integrate_box(round(x_a * res_a) / res_a, vec2<f32>(1.0 / res_a), x_a, sigma_a, vec2<f32>(0.0), vec2<f32>(1.0));
  return containing - explicitly_evaluated;
}

fn ndf(h: vec3<f32>, alpha: f32, glint_alpha: f32, uv: vec2<f32>, uv_J: mat2x2<f32>, N: f32, filter_size: f32) -> f32 {
  let res = sqrt(N);
  let x_s = uv;
  let x_a_and_d = ndf_to_disk_ggx(h, alpha);
  let x_a = x_a_and_d.xy;
  let d = x_a_and_d.z;

  let lambda = QueryLod(res * uv_J, filter_size);

  var D_filter = 0.0;

  for (var m: f32 = 0.0; m < 2.0; m = m + 1.0) {
    let l = floor(lambda) + m;

    let w_lambda = 1.0 - abs(lambda - l);
    let res_s = res * pow(2.0, -l);
    let res_a = pow(2.0, l);

    let uv_J2 = filter_size * uv_J;
    let sigma_s = uv_J2 * transpose(uv_J2);

    let sigma_a = d * pow(glint_alpha, 2.0) * mat2x2<f32>(
      vec2<f32>(1.0, 0.0),
      vec2<f32>(0.0, 1.0)
    );

    let base_i_a = clamp(round(x_a * res_a), vec2<f32>(1.0), vec2<f32>(res_a - 1.0));
    for (var j_a: i32 = 0; j_a < 4; j_a = j_a + 1) {
      let i_a = base_i_a + vec2<f32>(f32(j_a % 2), f32((j_a / 2) % 2)) - vec2<f32>(0.5);

      let base_i_s = round(x_s * res_s);
      for (var j_s: i32 = 0; j_s < 4; j_s = j_s + 1) {
        let i_s = base_i_s + vec2<f32>(f32(j_s % 2), f32((j_s / 2) % 2)) - vec2<f32>(0.5);

        let g_s = (i_s + Rand2D(i_s, i_a, l, 1u) - vec2<f32>(0.5)) / res_s;
        let g_a = (i_a + Rand2D(i_s, i_a, l, 2u) - vec2<f32>(0.5)) / res_a;

        let r = Rand1D(i_s, i_a, l, 4u);
        let roulette = smoothstep(max(0.0, r - 0.1), min(1.0, r + 0.1), w_lambda);

        D_filter = D_filter + roulette * normal(sigma_a, x_a - g_a) * normal(sigma_s, x_s - g_s) / N;
      }
    }
    D_filter = D_filter + w_lambda * compensation(x_a, sigma_a, res_a);
  }

  return D_filter * d / pi;
}

fn iTorus(ro: vec3<f32>, rd: vec3<f32>, tor: vec2<f32>) -> f32 {
  var po = 1.0;

  let Ra2 = tor.x * tor.x;
  let ra2 = tor.y * tor.y;

  let m = dot(ro, ro);
  let n = dot(ro, rd);

  {
    let h = n * n - m + (tor.x + tor.y) * (tor.x + tor.y);
    if (h < 0.0) {
      return -1.0;
    }
  }

  let k = (m - ra2 - Ra2) / 2.0;
  var k3 = n;
  var k2 = n * n + Ra2 * rd.z * rd.z + k;
  var k1 = k * n + Ra2 * ro.z * rd.z;
  var k0 = k * k + Ra2 * ro.z * ro.z - Ra2 * ra2;

  if (abs(k3 * (k3 * k3 - k2) + k1) < 0.01) {
    po = -1.0;
    let tmp = k1;
    k1 = k3;
    k3 = tmp;
    k0 = 1.0 / k0;
    k1 = k1 * k0;
    k2 = k2 * k0;
    k3 = k3 * k0;
  }

  var c2 = 2.0 * k2 - 3.0 * k3 * k3;
  var c1 = k3 * (k3 * k3 - k2) + k1;
  var c0 = k3 * (k3 * (-3.0 * k3 * k3 + 4.0 * k2) - 8.0 * k1) + 4.0 * k0;

  c2 = c2 / 3.0;
  c1 = c1 * 2.0;
  c0 = c0 / 3.0;

  let Q = c2 * c2 + c0;
  let R = 3.0 * c0 * c2 - c2 * c2 * c2 - c1 * c1;

  let h = R * R - Q * Q * Q;
  var z = 0.0;
  if (h < 0.0) {
    let sQ = sqrt(Q);
    z = 2.0 * sQ * cos(acos(R / (sQ * Q)) / 3.0);
  } else {
    let sQ = pow(sqrt(h) + abs(R), 1.0 / 3.0);
    z = sign(R) * abs(sQ + Q / sQ);
  }
  z = c2 - z;

  var d1 = z - 3.0 * c2;
  var d2 = z * z - 3.0 * c0;
  if (abs(d1) < 1.0e-4) {
    if (d2 < 0.0) {
      return -1.0;
    }
    d2 = sqrt(d2);
  } else {
    if (d1 < 0.0) {
      return -1.0;
    }
    d1 = sqrt(d1 / 2.0);
    d2 = c1 / d1;
  }

  var result = 1e20;

  var h2 = d1 * d1 - z + d2;
  if (h2 > 0.0) {
    h2 = sqrt(h2);
    var t1 = -d1 - h2 - k3;
    var t2 = -d1 + h2 - k3;
    if (po < 0.0) {
      t1 = 2.0 / t1;
      t2 = 2.0 / t2;
    }
    if (t1 > 0.0) {
      result = t1;
    }
    if (t2 > 0.0) {
      result = min(result, t2);
    }
  }

  h2 = d1 * d1 - z - d2;
  if (h2 > 0.0) {
    h2 = sqrt(h2);
    var t1 = d1 - h2 - k3;
    var t2 = d1 + h2 - k3;
    if (po < 0.0) {
      t1 = 2.0 / t1;
      t2 = 2.0 / t2;
    }
    if (t1 > 0.0) {
      result = min(result, t1);
    }
    if (t2 > 0.0) {
      result = min(result, t2);
    }
  }

  return result;
}

fn nTorus(pos: vec3<f32>, tor: vec2<f32>) -> vec3<f32> {
  return normalize(pos * (dot(pos, pos) - tor.y * tor.y - tor.x * tor.x * vec3<f32>(1.0, 1.0, -1.0)));
}

fn torusFrame(pos: vec3<f32>, tor: vec2<f32>) -> mat3x3<f32> {
  let n0 = nTorus(vec3<f32>(pos.x, pos.z, pos.y), tor);
  let n = vec3<f32>(n0.x, n0.z, n0.y);
  let t1 = normalize(vec3<f32>(pos.z, 0.0, -pos.x));
  let t2 = normalize(cross(n, t1));
  return mat3x3<f32>(t1, t2, n);
}

fn G1_GGX(n: vec3<f32>, _h: vec3<f32>, v: vec3<f32>, alpha: f32) -> f32 {
  let ndotv = abs(dot(n, v));
  let ndotv_sq = max(ndotv * ndotv, 1e-6);
  let tan_theta_sq = (1.0 - ndotv_sq) / ndotv_sq;
  let Gamma = -0.5 + 0.5 * sqrt(1.0 + alpha * alpha * tan_theta_sq);
  return 1.0 / (1.0 + Gamma);
}

fn G_GGX(n: vec3<f32>, h: vec3<f32>, light_in: vec3<f32>, light_out: vec3<f32>, alpha: f32) -> f32 {
  return G1_GGX(n, h, light_in, alpha) * G1_GGX(n, h, light_out, alpha);
}

fn brdf(alpha: f32, view: vec3<f32>, light: vec3<f32>, base: mat3x3<f32>, uv: vec2<f32>, uv_J: mat2x2<f32>) -> f32 {
  let h_sum = view + light;
  let h_len2 = dot(h_sum, h_sum);
  if (h_len2 <= 1e-8) {
    return 0.0;
  }
  let h_world = h_sum * inverseSqrt(h_len2);
  let h_local_raw = transpose(base) * h_world;
  // Stabilize NDF mapping by forcing half-vector to local +Z hemisphere.
  let h_local = normalize(vec3<f32>(h_local_raw.x, h_local_raw.y, abs(h_local_raw.z)));

  var density = uniforms.glint_density;
  if (density == 0.0) {
    density = 0.7;
  }

  let D = ndf(h_local, alpha, microfacet_roughness, uv, uv_J, 8e5 * pow(10.0, density * 6.0 - 2.0), pixel_filter_size);
  let F = mix(pow(1.0 - dot(h_world, light), 5.0), 1.0, 0.96);
  let G = G_GGX(base[2], h_world, light, view, alpha);
  let nDotV = max(abs(dot(base[2], view)), 1e-4);
  let nDotL = max(abs(dot(base[2], light)), 1e-4);
  return D * F * G / (4.0 * nDotV * nDotL);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let iResolution = uniforms.resolution;

  let fragCoord = in.position.xy;
  let uv_screen = (fragCoord / iResolution - vec2<f32>(0.5)) * vec2<f32>(1.0, iResolution.y / iResolution.x);

  let lightYaw = uniforms.auto_light_yaw + uniforms.light_rotation.y;
  let lightPitch = uniforms.light_rotation.x;
  let lightWorld = normalize(vec3<f32>(
    cos(lightYaw) * cos(lightPitch),
    sin(lightPitch),
    sin(lightYaw) * cos(lightPitch)
  ));
  var col = vec3<f32>(0.0);

  let torus = vec2<f32>(1.0, 0.4);

  let tilt = 0.8;
  let M = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, cos(tilt), sin(tilt)),
    vec3<f32>(0.0, -sin(tilt), cos(tilt))
  );

  let ro = M * vec3<f32>(2.5, 0.0, 0.0);
  let eye = normalize(-ro);
  let up = M * vec3<f32>(0.0, 1.0, 0.0);
  let right = normalize(cross(eye, up));
  let lookat = mat3x3<f32>(right, cross(right, eye), eye);

  let orthoScale = uniforms.zoom_scale;
  let ro_ray = ro + lookat[0] * uv_screen.x * orthoScale + lookat[1] * uv_screen.y * orthoScale;
  let rd = normalize(eye);

  let objectYaw = uniforms.auto_yaw + uniforms.rotation.y;
  let objectPitch = uniforms.rotation.x;
  let objectRotatePitchLocal = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, cos(objectPitch), sin(objectPitch)),
    vec3<f32>(0.0, -sin(objectPitch), cos(objectPitch))
  );
  let objectRotateYawLocal = mat3x3<f32>(
    vec3<f32>(cos(objectYaw), 0.0, -sin(objectYaw)),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(sin(objectYaw), 0.0, cos(objectYaw))
  );
  let objectRotateLocal = objectRotateYawLocal * objectRotatePitchLocal;
  let objectRotate = M * objectRotateLocal * transpose(M);
  let objectInvRotate = transpose(objectRotate);

  let ro_obj = objectInvRotate * ro_ray;
  let rd_obj = objectInvRotate * rd;
  var light = normalize(objectInvRotate * lightWorld);

  let t = iTorus(vec3<f32>(ro_obj.x, ro_obj.z, ro_obj.y), vec3<f32>(rd_obj.x, rd_obj.z, rd_obj.y), torus);

  // WGSL requires derivative calls in uniform control flow.
  let t_for_derivative = select(1.0, t, t > 0.0);
  let pos_pre = ro_obj + rd_obj * t_for_derivative;
  let texcoord_pre = vec2<f32>(
    atan2(pos_pre.x, pos_pre.z),
    atan2(pos_pre.y, length(pos_pre.xz) - torus.x)
  ) / (2.0 * pi) + vec2<f32>(0.5);
  let uv_pre = fract(texcoord_pre);
  var uv_J = mat2x2<f32>(dpdx(uv_pre), dpdy(uv_pre));
  uv_J = uv_ellipsoid(uv_J);

  if (t > 0.0) {
    let pos = ro_obj + rd_obj * t;
    let texcoord = vec2<f32>(
      atan2(pos.x, pos.z),
      atan2(pos.y, length(pos.xz) - torus.x)
    ) / (2.0 * pi) + vec2<f32>(0.5);
    let uv = fract(texcoord);
    let base = torusFrame(pos, torus);

    var roughness = uniforms.roughness;
    if (roughness == 0.0) {
      roughness = 0.3;
    }

    let alpha = 0.2 + roughness * 0.8;
    let shininess = 20.0 * uniforms.glint_density * pow(uniforms.roughness, 1.0) * pow(uniforms.shininess_scale, 2.2);

    var lcolor = max(oklch_to_p3(uniforms.oklch), vec3<f32>(0.0)) / 5.;
    let ndotl = abs(dot(base[2], light));
    var reflectance = 0.5 * (
      brdf(alpha, -rd_obj, light, base, uv, uv_J) +
      brdf(alpha, -rd_obj, -light, base, uv, uv_J)
    );

    let ndot = clamp(ndotl, 0.0, 1.0);
    let uvCell = floor(uv * 2048.0);
    let rng = fract(sin(dot(uvCell, vec2<f32>(127.1, 311.7))) * 43758.5453);
    let sampleJitter = (rng - 0.5) * 1.25;
    let sampleU = clamp(ndot - 0.25 + sampleJitter * .25, 0.0, 1.0);
    let xyz = textureSampleLevel(cieTexture, cieSampler, vec2<f32>(sampleU, 0.5), 0.0).xyz;
    let xyz_to_p3 = mat3x3<f32>(
      vec3<f32>( 2.4934969, -0.8294890,  0.0358458),
      vec3<f32>(-0.9313836,  1.7626641, -0.0761724),
      vec3<f32>(-0.4027108,  0.0236247,  0.9568845)
    );
    let lcolor2 = xyz_to_p3 * xyz;

    let lcolorLen = length(lcolor);
    let lcolorUnit = select(vec3<f32>(0.0), lcolor / lcolorLen, lcolorLen > 1e-6);
    if (ndot < 0.01) {
      reflectance /= 1.5;
    }
    col = reflectance * max(ndot, 0.1) * max(lcolorUnit * 0.75, lcolor2 * 4.) * shininess + lcolor * (1./4. + pow(ndot, 2.)) * 4.;
  }

  let linear = max(col, vec3<f32>(0.0));
  let encoded = vec3<f32>(
    srgb_transfer_function(linear.x),
    srgb_transfer_function(linear.y),
    srgb_transfer_function(linear.z)
  );
  return vec4<f32>(encoded, 1.0);
}
`,
});

const pipeline = device.createRenderPipeline({
  label: 'sparkle-object-pipeline',
  layout: 'auto',
  vertex: {
    module: shader,
    entryPoint: 'vs_main',
  },
  fragment: {
    module: shader,
    entryPoint: 'fs_main',
    targets: [{ format: presentationFormat }],
  },
  primitive: {
    topology: 'triangle-list',
  },
});

const uniformData = new Float32Array(20);
const uniformBuffer = device.createBuffer({
  size: uniformData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: cieSampler },
    { binding: 1, resource: cieTexture.createView() },
    { binding: 2, resource: { buffer: uniformBuffer } },
  ],
});

let densityControl = Number.parseFloat(densitySlider.value);
let roughnessControl = Number.parseFloat(roughnessSlider.value);
let autoRotateEnabled = autoRotateToggle.checked;
let autoLightRotateEnabled = autoLightRotateToggle.checked;
let shininessScale = Number.parseFloat(shininessSlider.value);
let oklchLuminance = Number.parseFloat(luminanceSlider.value);
let oklchChroma = Number.parseFloat(chromaSlider.value);
let oklchHue = Number.parseFloat(hueSlider.value);
let zoomScale = 6.;
let autoYawAngle = 0;
let autoLightYawAngle = 0;
let previousFrameMs = null;
if (!Number.isFinite(densityControl)) densityControl = 0.7;
if (!Number.isFinite(roughnessControl)) roughnessControl = 0.3;
if (!Number.isFinite(shininessScale)) shininessScale = 1.0;
if (!Number.isFinite(oklchLuminance)) oklchLuminance = 0.44;
if (!Number.isFinite(oklchChroma)) oklchChroma = 0.11;
if (!Number.isFinite(oklchHue)) oklchHue = 270.0;

function updateControlLabels() {
  densityValueEl.textContent = densityControl.toFixed(3);
  roughnessValueEl.textContent = roughnessControl.toFixed(3);
  shininessValueEl.textContent = shininessScale.toFixed(2);
  luminanceValueEl.textContent = oklchLuminance.toFixed(3);
  chromaValueEl.textContent = oklchChroma.toFixed(3);
  hueValueEl.textContent = `${Math.round(oklchHue)}°`;
}

densitySlider.addEventListener('input', () => {
  const v = Number.parseFloat(densitySlider.value);
  if (!Number.isFinite(v)) return;
  densityControl = Math.max(0, Math.min(1, v));
  updateControlLabels();
});

roughnessSlider.addEventListener('input', () => {
  const v = Number.parseFloat(roughnessSlider.value);
  if (!Number.isFinite(v)) return;
  roughnessControl = Math.max(0, Math.min(1, v));
  updateControlLabels();
});

shininessSlider.addEventListener('input', () => {
  const v = Number.parseFloat(shininessSlider.value);
  if (!Number.isFinite(v)) return;
  shininessScale = Math.max(0, v);
  updateControlLabels();
});

luminanceSlider.addEventListener('input', () => {
  const v = Number.parseFloat(luminanceSlider.value);
  if (!Number.isFinite(v)) return;
  oklchLuminance = Math.max(0, Math.min(1.0, v));
  updateControlLabels();
});

chromaSlider.addEventListener('input', () => {
  const v = Number.parseFloat(chromaSlider.value);
  if (!Number.isFinite(v)) return;
  oklchChroma = Math.max(0, Math.min(0.45, v));
  updateControlLabels();
});

hueSlider.addEventListener('input', () => {
  const v = Number.parseFloat(hueSlider.value);
  if (!Number.isFinite(v)) return;
  oklchHue = ((v % 360) + 360) % 360;
  updateControlLabels();
});

autoRotateToggle.addEventListener('change', () => {
  autoRotateEnabled = autoRotateToggle.checked;
});
autoLightRotateToggle.addEventListener('change', () => {
  autoLightRotateEnabled = autoLightRotateToggle.checked;
});

updateControlLabels();

const rotation = {
  dragging: false,
  dragMode: 'object',
  pointerId: null,
  lastX: 0,
  lastY: 0,
  pitch: 0,
  yaw: 0,
};
const lightRotation = {
  pitch: 0.615,
  yaw: 0.785,
};

const ROTATE_SPEED = 0.008;
const MAX_PITCH = 3.14 / 2.;

canvas.addEventListener('pointerdown', (event) => {
  rotation.dragging = true;
  rotation.dragMode = event.button === 2 ? 'light' : 'object';
  rotation.pointerId = event.pointerId;
  rotation.lastX = event.clientX;
  rotation.lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener('pointermove', (event) => {
  if (!rotation.dragging || event.pointerId !== rotation.pointerId) return;
  const dx = event.clientX - rotation.lastX;
  const dy = event.clientY - rotation.lastY;
  rotation.lastX = event.clientX;
  rotation.lastY = event.clientY;

  if (rotation.dragMode === 'light') {
    lightRotation.yaw += dx * ROTATE_SPEED;
    lightRotation.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, lightRotation.pitch + dy * ROTATE_SPEED));
  } else {
    rotation.yaw += dx * ROTATE_SPEED;
    rotation.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, rotation.pitch + dy * ROTATE_SPEED));
  }
});

canvas.addEventListener('pointerup', (event) => {
  if (event.pointerId !== rotation.pointerId) return;
  rotation.dragging = false;
  rotation.pointerId = null;
  canvas.releasePointerCapture(event.pointerId);
});

canvas.addEventListener('pointercancel', (event) => {
  if (event.pointerId !== rotation.pointerId) return;
  rotation.dragging = false;
  rotation.pointerId = null;
  canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener('contextmenu', (event) => {
  event.preventDefault();
});

canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  const factor = Math.exp(event.deltaY * 0.0012);
  zoomScale = Math.max(1e-6, zoomScale * factor);
}, { passive: false });

function resize() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width === width && canvas.height === height) return;

  canvas.width = width;
  canvas.height = height;

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'opaque',
    colorSpace: 'display-p3',
    toneMapping: { mode: 'extended' },
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
  });
}

function renderFrame(timeMs) {
  resize();
  if (previousFrameMs == null) {
    previousFrameMs = timeMs;
  }
  const dtSec = Math.max(0, (timeMs - previousFrameMs) * 0.001);
  previousFrameMs = timeMs;
  if (autoRotateEnabled) {
    autoYawAngle += dtSec * 0.2;
  }
  if (autoLightRotateEnabled) {
    autoLightYawAngle += dtSec * 0.2;
  }

  const width = canvas.width;
  const height = canvas.height;

  uniformData[0] = width;
  uniformData[1] = height;
  uniformData[2] = timeMs * 0.001;
  uniformData[3] = autoYawAngle;
  uniformData[4] = densityControl;
  uniformData[5] = roughnessControl;
  uniformData[6] = rotation.pitch;
  uniformData[7] = rotation.yaw;
  uniformData[8] = shininessScale;
  uniformData[9] = zoomScale;
  uniformData[10] = lightRotation.pitch;
  uniformData[11] = lightRotation.yaw;
  uniformData[12] = autoLightYawAngle;
  uniformData[13] = 0;
  uniformData[14] = 0;
  uniformData[15] = 0;
  uniformData[16] = oklchLuminance;
  uniformData[17] = oklchChroma;
  uniformData[18] = oklchHue;
  uniformData[19] = 0;

  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: 'clear',
      storeOp: 'store',
    }],
  });

  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();

  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(renderFrame);
}

const cfg = context.getConfiguration?.();
const configuredToneMappingMode = cfg?.toneMapping?.mode ?? toneMappingMode;
if (presentationFormat === 'rgba16float' && configuredToneMappingMode === 'extended') {
  statusEl.textContent = 'HDR output active (rgba16float + extended tone mapping).';
} else {
  statusEl.textContent = `HDR limited. format=${presentationFormat}, toneMapping=${configuredToneMappingMode}`;
}

window.addEventListener('resize', resize);
resize();
requestAnimationFrame(renderFrame);

async function loadCieTexture(gpuDevice, csvPath) {
  const response = await fetch(csvPath);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${csvPath}: ${response.status}`);
  }
  const text = await response.text();
  const rows = text
    .trim()
    .split(/\r?\n/)
    .map((line) => line.split(',').map((cell) => cell.trim()))
    .filter((parts) => parts.length >= 4)
    .map((parts) => ({
      wavelength: Number.parseFloat(parts[0]),
      X: Number.parseFloat(parts[1]),
      Y: Number.parseFloat(parts[2]),
      Z: Number.parseFloat(parts[3]),
    }))
    .filter((row) => Number.isFinite(row.wavelength) && Number.isFinite(row.X) && Number.isFinite(row.Y) && Number.isFinite(row.Z));

  if (!rows.length) {
    throw new Error(`No usable rows in ${csvPath}`);
  }

  const width = rows.length;
  const data = new Float32Array(width * 4);
  for (let i = 0; i < width; i += 1) {
    const base = i * 4;
    data[base + 0] = rows[i].X;
    data[base + 1] = rows[i].Y;
    data[base + 2] = rows[i].Z;
    data[base + 3] = 1;
  }

  const texture = gpuDevice.createTexture({
    size: { width, height: 1, depthOrArrayLayers: 1 },
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  upload1DTexture(gpuDevice, texture, width, data);
  return { texture, width };
}

function createFallbackCieTexture(gpuDevice, width) {
  const data = new Float32Array(width * 4);
  for (let i = 0; i < width; i += 1) {
    const t = i / Math.max(1, width - 1);
    const base = i * 4;
    data[base + 0] = t;
    data[base + 1] = t;
    data[base + 2] = t;
    data[base + 3] = 1;
  }
  const texture = gpuDevice.createTexture({
    size: { width, height: 1, depthOrArrayLayers: 1 },
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  upload1DTexture(gpuDevice, texture, width, data);
  return texture;
}

function upload1DTexture(gpuDevice, texture, width, data) {
  const half = float32ArrayToFloat16(data);
  const bytesPerTexel = 8; // rgba16float
  const rowSize = width * bytesPerTexel;
  const paddedRowSize = Math.ceil(rowSize / 256) * 256;
  const padded = new Uint16Array(paddedRowSize / 2);
  padded.set(half);

  gpuDevice.queue.writeTexture(
    { texture },
    padded,
    { offset: 0, bytesPerRow: paddedRowSize, rowsPerImage: 1 },
    { width, height: 1, depthOrArrayLayers: 1 }
  );
}

function float32ToFloat16(value) {
  const floatView = new Float32Array(1);
  const intView = new Uint32Array(floatView.buffer);
  floatView[0] = value;

  const x = intView[0];
  const sign = (x >> 16) & 0x8000;
  const mantissa = x & 0x7fffff;
  const exponent = (x >> 23) & 0xff;

  if (exponent === 0) return sign;
  if (exponent === 0xff) return sign | 0x7c00;

  let newExp = exponent - 127 + 15;
  if (newExp >= 0x1f) return sign | 0x7c00;
  if (newExp <= 0) {
    if (newExp < -10) return sign;
    const denorm = (mantissa | 0x800000) >> (1 - newExp);
    return sign | ((denorm + 0x1000) >> 13);
  }

  return sign | (newExp << 10) | ((mantissa + 0x1000) >> 13);
}

function float32ArrayToFloat16(src) {
  const dst = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i += 1) {
    dst[i] = float32ToFloat16(src[i]);
  }
  return dst;
}
