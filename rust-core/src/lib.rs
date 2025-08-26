use rgb::RGBA8;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();
}

fn scale_dims(w: u32, h: u32, max_side: u32) -> (u32, u32) {
    if max_side == 0 {
        return (w, h);
    }
    let m = w.max(h);
    if m <= max_side {
        return (w, h);
    }
    let s = max_side as f32 / m as f32;
    (
        ((w as f32 * s).round() as u32),
        ((h as f32 * s).round() as u32),
    )
}

#[wasm_bindgen]
pub fn preprocess_rgba_to_nchw(
    rgba: &[u8], // Uint8ClampedArray from canvas
    width: u32,
    height: u32,
    max_side: u32,     // e.g., 512
    range_m1to1: bool, // true => [-1,1], false => [0,1]
) -> js_sys::Object {
    // 1) Optional resize (nearest for minimal deps; swap for fast_image_resize if you like)
    let (tw, th) = scale_dims(width, height, max_side);
    let mut tmp = vec![RGBA8::new(0, 0, 0, 255); (tw * th) as usize];

    if tw == width && th == height {
        // copy
        for i in 0..(width * height) as usize {
            tmp[i] = RGBA8::new(rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2], 255);
        }
    } else {
        // very simple box sampling; replace with fast_image_resize for quality/perf
        let sx = width as f32 / tw as f32;
        let sy = height as f32 / th as f32;
        for y in 0..th {
            for x in 0..tw {
                let src_x = (x as f32 * sx).floor() as u32;
                let src_y = (y as f32 * sy).floor() as u32;
                let i = (src_y * width + src_x) as usize * 4;
                let j = (y * tw + x) as usize;
                tmp[j] = RGBA8::new(rgba[i], rgba[i + 1], rgba[i + 2], 255);
            }
        }
    }

    // 2) Normalize + layout to NCHW (1,3,H,W)
    let plane = (tw * th) as usize;
    let mut out = vec![0f32; plane * 3];
    if range_m1to1 {
        for p in 0..plane {
            out[p] = tmp[p].r as f32 / 127.5 - 1.0;
            out[p + plane] = tmp[p].g as f32 / 127.5 - 1.0;
            out[p + plane * 2] = tmp[p].b as f32 / 127.5 - 1.0;
        }
    } else {
        for p in 0..plane {
            out[p] = tmp[p].r as f32 / 255.0;
            out[p + plane] = tmp[p].g as f32 / 255.0;
            out[p + plane * 2] = tmp[p].b as f32 / 255.0;
        }
    }

    // Return { data: Float32Array, width: tw, height: th }
    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &"data".into(),
        &js_sys::Float32Array::from(out.as_slice()),
    )
    .unwrap();
    js_sys::Reflect::set(&obj, &"width".into(), &JsValue::from(tw)).unwrap();
    js_sys::Reflect::set(&obj, &"height".into(), &JsValue::from(th)).unwrap();
    obj
}

#[inline]
fn clamp255(v: f32) -> u8 {
    v.max(0.0).min(255.0) as u8
}

#[wasm_bindgen]
pub fn postprocess_to_rgba(
    tensor: &[f32], // e.g., (1,3,H,W) or (1,H,W,3)
    width: u32,
    height: u32,
    nchw: bool, // true if (1,3,H,W), false if (1,H,W,3)
    range_m1to1: bool,
) -> js_sys::Uint8ClampedArray {
    let w = width as usize;
    let h = height as usize;
    let plane = w * h;
    let mut out = vec![0u8; plane * 4];

    if nchw {
        let (r0, g0, b0) = (0, plane, plane * 2);
        for p in 0..plane {
            let r = tensor[r0 + p];
            let g = tensor[g0 + p];
            let b = tensor[b0 + p];
            let (r, g, b) = if range_m1to1 {
                ((r + 1.0) * 127.5, (g + 1.0) * 127.5, (b + 1.0) * 127.5)
            } else {
                (r * 255.0, g * 255.0, b * 255.0)
            };
            let i = p * 4;
            out[i] = clamp255(r);
            out[i + 1] = clamp255(g);
            out[i + 2] = clamp255(b);
            out[i + 3] = 255;
        }
    } else {
        for p in 0..plane {
            let i3 = p * 3;
            let r = tensor[i3];
            let g = tensor[i3 + 1];
            let b = tensor[i3 + 2];
            let (r, g, b) = if range_m1to1 {
                ((r + 1.0) * 127.5, (g + 1.0) * 127.5, (b + 1.0) * 127.5)
            } else {
                (r * 255.0, g * 255.0, b * 255.0)
            };
            let i = p * 4;
            out[i] = clamp255(r);
            out[i + 1] = clamp255(g);
            out[i + 2] = clamp255(b);
            out[i + 3] = 255;
        }
    }

    js_sys::Uint8ClampedArray::from(out.as_slice())
}
