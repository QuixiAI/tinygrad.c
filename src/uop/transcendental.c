/* transcendental.c - Faithful line-by-line port of reference/tinygrad/uop/transcendental.py */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "uop/uop.h"
#include "dtype/dtype.h"

// Supported dtypes for transcendental operations
#define TRANSCENDENTAL_SUPPORTED_COUNT 3

// Math constants
#define MATH_PI 3.14159265358979323846
#define MATH_E 2.71828182845904523536

// 190 bits of 2/pi used by Payne–Hanek reduction (32-bit words)
static const uint32_t TWO_OVER_PI_F[7] = {
    0x00000000u, 0x28be60dbu, 0x9391054au, 0x7f09d5f4u, 0x7d4d3770u, 0x36d8a566u, 0x4f10e410u
};

// Helper for Payne–Hanek: select TWO_OVER_PI_F[i + offset] via chained where over dynamic i
static UOp* ph_take(UOp* i_uop, int offset, DType i_dtype, DType u32v) {
    UOp* an = uop_const(u32v, 0.0);
    int max_k = (int)(sizeof(TWO_OVER_PI_F)/sizeof(TWO_OVER_PI_F[0])) - 1 - offset; // skip last per reference code
    for (int k = max_k - 1; k >= 0; --k) {
        UOp* cond = uop_cmpne(i_uop, uop_const(i_dtype, (double)k));
        UOp* val = uop_const(u32v, (double)TWO_OVER_PI_F[k + offset]);
        an = uop_where(cond, an, val);
    }
    return an;
}

// Helper for Cody–Waite: dtype-specific remainder reconstruction
static UOp* cw_reduce(UOp* x, UOp* q, UOp* qdh) {
    DType sc = dtype_scalar(&x->dtype);
    if (dtype_eq(&sc, &dtypes.float64)) {
        // Double-precision split of PI
        const double PI_A = 3.1415926218032836914;
        const double PI_B = 3.1786509424591713469e-08;
        const double PI_C = 1.2246467864107188502e-16;
        const double PI_D = 1.2736634327021899816e-24;

        UOp* d0 = uop_add(uop_mul(qdh, uop_const(x->dtype, -PI_A)), x);
        UOp* d1 = uop_add(uop_mul(q,   uop_const(x->dtype, -PI_A)), d0);
        UOp* d2 = uop_add(uop_mul(qdh, uop_const(x->dtype, -PI_B)), d1);
        UOp* d3 = uop_add(uop_mul(q,   uop_const(x->dtype, -PI_B)), d2);
        UOp* d4 = uop_add(uop_mul(qdh, uop_const(x->dtype, -PI_C)), d3);
        UOp* d5 = uop_add(uop_mul(q,   uop_const(x->dtype, -PI_C)), d4);
        UOp* qdh_q = uop_add(qdh, q);
        UOp* d6 = uop_add(uop_mul(qdh_q, uop_const(x->dtype, -PI_D)), d5);
        return d6;
    } else if (dtype_eq(&sc, &dtypes.float16)) {
        // Compute in float32 for FP16 precision, then cast back
        UOp* x32 = uop_cast_vec(x, dtypes.float32, x->dtype.count);
        UOp* q32 = uop_cast_vec(q, dtypes.float32, q->dtype.count);
        UOp* r32 = cw_reduce(x32, q32, NULL);
        return uop_cast_vec(r32, dtypes.float16, x->dtype.count);
    } else {
        // Single-precision path
        const double C0 = -3.1414794921875;
        const double C1 = -0.00011315941810607910156;
        const double C2 = -1.9841872589410058936e-09;
        const double C3 = -1.2154201256553420762e-10;
        UOp* d0 = uop_add(uop_mul(q, uop_const(x->dtype, C0)), x);
        UOp* d1 = uop_add(uop_mul(q, uop_const(x->dtype, C1)), d0);
        UOp* d2 = uop_add(uop_mul(q, uop_const(x->dtype, C2)), d1);
        UOp* d3 = uop_add(uop_mul(q, uop_const(x->dtype, C3)), d2);
        return d3;
    }
}

// *** helper functions for bit manipulation ***

int transcendental_mantissa_bits(DType* d) {
    if (!d) return 0;
    DType sc = dtype_scalar(d);
    if (dtype_eq(&sc, &dtypes.float64)) return 52;
    if (dtype_eq(&sc, &dtypes.float32)) return 23;
    if (dtype_eq(&sc, &dtypes.float16)) return 10;
    return 0;
}

int transcendental_exponent_bias(DType* d) {
    if (!d) return 0;
    DType sc = dtype_scalar(d);
    if (dtype_eq(&sc, &dtypes.float64)) return 1023;
    if (dtype_eq(&sc, &dtypes.float32)) return 127;
    if (dtype_eq(&sc, &dtypes.float16)) return 15;
    return 0;
}

int transcendental_exponent_mask(DType* d) {
    if (!d) return 0;
    DType sc = dtype_scalar(d);
    if (dtype_eq(&sc, &dtypes.float64)) return 2047;
    if (dtype_eq(&sc, &dtypes.float32)) return 255;
    if (dtype_eq(&sc, &dtypes.float16)) return 31;
    return 0;
}

// **** utils ****

// integer shift helpers: operate on integer UOps using bit shifts
UOp* transcendental_shr(UOp* x, int y) {
    return uop_shr(x, uop_const(x->dtype, (double)y));
}

UOp* transcendental_shl(UOp* x, int y) {
    return uop_shl(x, uop_const(x->dtype, (double)y));
}

// Horner polynomial evaluation for UOps
static UOp* transcendental_polyN(UOp* x, const double* coeffs, int n) {
    if (n <= 0) return uop_const(x->dtype, 0.0);
    UOp* acc = uop_const(x->dtype, coeffs[n-1]);
    for (int i = n-2; i >= 0; --i) {
        acc = uop_add(uop_mul(acc, x), uop_const(x->dtype, coeffs[i]));
    }
    return acc;
}

UOp* transcendental_rintk(UOp* d) {
    // round d:float to int away from 0
    DType out_dtype;
    
    {
        DType sc = dtype_scalar(&d->dtype);
        if (dtype_eq(&sc, &dtypes.float64)) {
        out_dtype = dtype_vec(&dtypes.int64, d->dtype.count);
        } else if (dtype_eq(&sc, &dtypes.float32)) {
        out_dtype = dtype_vec(&dtypes.int32, d->dtype.count);
        } else if (dtype_eq(&sc, &dtypes.float16)) {
        out_dtype = dtype_vec(&dtypes.int16, d->dtype.count);
        } else {
        return NULL;
        }
    }
    
    // d + (d<0.0).where(d.const_like(-0.5), d.const_like(0.5)).cast(out_dtype)
    UOp* half_neg = uop_const(d->dtype, -0.5);
    UOp* half_pos = uop_const(d->dtype, 0.5);
    UOp* cond = uop_lt(d, uop_const(d->dtype, 0.0));
    UOp* adjust = uop_where(cond, half_neg, half_pos);
    UOp* sum = uop_add(d, adjust);
    return uop_cast(sum, out_dtype);
}

UOp* transcendental_pow2if(UOp* q, const DType* float_scalar) {
    // cast(2^q, float_dtype) where q is any integer in the range of [-126, 127]
    DType out_dtype;
    
    {
        DType scq = dtype_scalar(&q->dtype);
        if (dtype_eq(&scq, &dtypes.int64)) {
        out_dtype = dtype_vec(&dtypes.float64, q->dtype.count);
        } else if (dtype_eq(&scq, &dtypes.int32)) {
        out_dtype = dtype_vec(&dtypes.float32, q->dtype.count);
        } else if (dtype_eq(&scq, &dtypes.int16)) {
        out_dtype = dtype_vec(float_scalar, q->dtype.count);
        } else {
        return NULL;
        }
    }
    
    // shl(q + exponent_bias(out_dtype), mantissa_bits(out_dtype)).bitcast(out_dtype)
    UOp* bias = uop_const(q->dtype, (double)transcendental_exponent_bias(&out_dtype));
    UOp* sum = uop_add(q, bias);
    UOp* shifted = transcendental_shl(sum, transcendental_mantissa_bits(&out_dtype));
    return uop_bitcast(shifted, out_dtype);
}

UOp* transcendental_ilogb2k(UOp* d) {
    // calculate the integer part of log2(d), where d is normalized fp value in the range of [0, +inf).
    {
        DType scd = dtype_scalar(&d->dtype);
        if (!dtype_eq(&scd, &dtypes.float16) && !dtype_eq(&scd, &dtypes.float32) && !dtype_eq(&scd, &dtypes.float64)) {
        return NULL;
        }
    }
    
    // DType to cast to
    DType int_dtype;
    {
        DType sc = dtype_scalar(&d->dtype);
        if (dtype_eq(&sc, &dtypes.float64)) {
        int_dtype = dtype_vec(&dtypes.int64, d->dtype.count);
        } else if (dtype_eq(&sc, &dtypes.float32)) {
        int_dtype = dtype_vec(&dtypes.int32, d->dtype.count);
        } else {
        int_dtype = dtype_vec(&dtypes.int16, d->dtype.count);
        }
    }
    
    // d.bitcast(int_dtype)
    UOp* dint = uop_bitcast(d, int_dtype);
    
    // -1 <= ilog2bk(d) <= 128
    // (shr(dint, mantissa_bits(d.dtype)) & exponent_mask(d.dtype)) - exponent_bias(d.dtype)
    UOp* mantissa_shift_result = transcendental_shr(dint, transcendental_mantissa_bits(&d->dtype));
    UOp* masked = uop_and(mantissa_shift_result, uop_const(dint->dtype, (double)transcendental_exponent_mask(&d->dtype)));
    UOp* bias = uop_const(dint->dtype, (double)transcendental_exponent_bias(&d->dtype));
    return uop_sub(masked, bias);
}

UOp* transcendental_ldexp3k(UOp* d, UOp* e) {
    // d*2^e. e is a number obtained by casting an integer in the range [-127, 127] to a float. d is any float number.
    {
        DType sc = dtype_scalar(&d->dtype);
        if (!dtype_eq(&sc, &dtypes.float16) && !dtype_eq(&sc, &dtypes.float32) && !dtype_eq(&sc, &dtypes.float64)) {
        return NULL;
        }
    }
    
    {
        DType sce = dtype_scalar(&e->dtype);
        if (!dtype_eq(&sce, &dtypes.float16) && !dtype_eq(&sce, &dtypes.float32) && !dtype_eq(&sce, &dtypes.float64)) return NULL;
    }
    
    DType int_dtype;
    {
        DType scd = dtype_scalar(&d->dtype);
        if (dtype_eq(&scd, &dtypes.float64)) {
            int_dtype = dtype_vec(&dtypes.int64, d->dtype.count);
        } else if (dtype_eq(&scd, &dtypes.float32)) {
            int_dtype = dtype_vec(&dtypes.int32, d->dtype.count);
        } else {
            int_dtype = dtype_vec(&dtypes.int16, d->dtype.count);
        }
    }
    
    // m1 = d.bitcast(dtype)
    UOp* m1 = uop_bitcast(d, int_dtype);
    
    // m2 = shl(e.cast(dtype), mantissa_bits(d.dtype))
    UOp* e_cast = uop_cast(e, int_dtype);
    UOp* m2 = transcendental_shl(e_cast, transcendental_mantissa_bits(&d->dtype));
    
    // (m1 + m2).bitcast(d.dtype).cast(d.dtype)
    UOp* sum = uop_add(m1, m2);
    UOp* bitcast = uop_bitcast(sum, d->dtype);
    return uop_cast(bitcast, d->dtype);
}

UOp* transcendental_ldexp2k(UOp* d, UOp* e) {
    // d*2^e. much faster than ldexp3k but risky. d > 0 and d is not denormal.
    {
        DType sc = dtype_scalar(&d->dtype);
        if (!dtype_eq(&sc, &dtypes.float16) && !dtype_eq(&sc, &dtypes.float32) && !dtype_eq(&sc, &dtypes.float64)) {
        return NULL;
        }
    }
    
    {
        DType sce = dtype_scalar(&e->dtype);
        if (!dtype_eq(&sce, &dtypes.int16) && !dtype_eq(&sce, &dtypes.int32) && !dtype_eq(&sce, &dtypes.int64)) {
        return NULL;
        }
    }
    
    // (d * pow2if(shr(e, 1), d.dtype)) * pow2if(e - shr(e, 1), d.dtype)
    UOp* e_half = transcendental_shr(e, 1);
    DType scdf = dtype_scalar(&d->dtype);
    const DType* sc_float = dtype_eq(&scdf, &dtypes.float64) ? &dtypes.float64 : (dtype_eq(&scdf, &dtypes.float32) ? &dtypes.float32 : &dtypes.float16);
    UOp* pow1 = transcendental_pow2if(e_half, sc_float);
    UOp* temp1 = uop_mul(d, pow1);
    
    UOp* e_half2 = transcendental_shr(e, 1);
    UOp* e_diff = uop_sub(e, e_half2);
    UOp* pow2 = transcendental_pow2if(e_diff, sc_float);
    
    return uop_mul(temp1, pow2);
}

UOp** transcendental_frexp(UOp* v, UOp** mantissa, UOp** exponent) {
    // frexp(v) -> (mantissa, exponent) assuming v != 0
    DType scv = dtype_scalar(&v->dtype);
    if (!dtype_eq(&scv, &dtypes.float16) && !dtype_eq(&scv, &dtypes.float32) && !dtype_eq(&scv, &dtypes.float64)) return NULL;

    *mantissa = NULL;
    *exponent = NULL;

    // m1 = masks for mantissa, m2 = masks to normalize the mantissa.
    uint64_t m1, m2;
    if (dtype_eq(&scv, &dtypes.float64)) {
        m1 = 0x000FFFFFFFFFFFFF;
        m2 = 0x3FE0000000000000;
    } else if (dtype_eq(&scv, &dtypes.float32)) {
        m1 = 0x807FFFFF;
        m2 = 0x3F000000;
    } else {
        m1 = 0x83FF;
        m2 = 0x3800;
    }

    DType uint_dtype;
    if (dtype_eq(&scv, &dtypes.float64)) {
        uint_dtype = dtype_vec(&dtypes.uint64, v->dtype.count);
    } else if (dtype_eq(&scv, &dtypes.float32)) {
        uint_dtype = dtype_vec(&dtypes.uint32, v->dtype.count);
    } else {
        uint_dtype = dtype_vec(&dtypes.uint16, v->dtype.count);
    }

    // bits = v.bitcast(uint_dtype)
    UOp* bits = uop_bitcast(v, uint_dtype);

    // exponent = shr(bits, mantissa_bits(v.dtype)) & exponent_mask(v.dtype)
    UOp* shr_result = transcendental_shr(bits, transcendental_mantissa_bits(&v->dtype));
    UOp* exponent_mask_val = uop_const(bits->dtype, (double)transcendental_exponent_mask(&v->dtype));
    *exponent = uop_and(shr_result, exponent_mask_val);

    // mantissa = ((bits & m1) | m2).bitcast(v.dtype)
    UOp* m1_const = uop_const(bits->dtype, (double)m1);
    UOp* masked_and = uop_and(bits, m1_const);
    UOp* m2_const = uop_const(bits->dtype, (double)m2);
    UOp* masked_or = uop_or(masked_and, m2_const);
    *mantissa = uop_bitcast(masked_or, v->dtype);

    // exp = exponent - exponent_bias(v.dtype) + 1
    UOp* bias = uop_const((*exponent)->dtype, (double)transcendental_exponent_bias(&v->dtype));
    UOp* bias_sub = uop_sub(*exponent, bias);
    UOp* one = uop_const(bias_sub->dtype, 1.0);
    *exponent = uop_add(bias_sub, one);

    return mantissa;
}

// *** reduction algorithms for sine ***

UOp** transcendental_payne_hanek_reduction(UOp* d, UOp** r_result, UOp** q_result) {
    // Performs Payne-Hanek Reduction for 39800.0 <= d <= +Inf
    {
        DType sc = dtype_scalar(&d->dtype);
        if (!dtype_eq(&sc, &dtypes.float16) && !dtype_eq(&sc, &dtypes.float32) && !dtype_eq(&sc, &dtypes.float64)) return NULL;
    }

    *r_result = NULL;
    *q_result = NULL;

    // intermediate dtype: float32 if base scalar is float16, else same as d
    DType scb = dtype_scalar(&d->dtype);
    DType intermediate_dtype = dtype_eq(&scb, &dtypes.float16) ? dtype_vec(&dtypes.float32, d->dtype.count) : d->dtype;

    // f, e = frexp(d)
    UOp *f = NULL, *e = NULL;
    transcendental_frexp(d, &f, &e);

    // ia = (f.cast(intermediate_dtype) * 2**32).cast_vec(uint64)
    UOp* f_inter = uop_cast(f, intermediate_dtype);
    UOp* ia_f = uop_mul(f_inter, uop_const(intermediate_dtype, 4294967296.0));
    UOp* ia = uop_cast_vec(ia_f, dtypes.uint64, d->dtype.count);

    // i = shr(e.cast_vec(uint64), 5)
    UOp* e_u64 = uop_cast_vec(e, dtypes.uint64, d->dtype.count);
    UOp* i_uop = transcendental_shr(e_u64, 5);

    // e = e.cast_vec(int32) & 31; offset_uop = 32 - e
    UOp* e_i32 = uop_and(uop_cast_vec(e, dtypes.int32, d->dtype.count), uop_const(dtype_vec(&dtypes.int32, d->dtype.count), 31));
    UOp* offset_uop = uop_sub(uop_const(dtype_vec(&dtypes.int32, d->dtype.count), 32.0), e_i32);

    // _take: an = two_over_pi_f[i + offset] via chained where over i
    DType u32v = dtype_vec(&dtypes.uint32, d->dtype.count);
    DType u64v = dtype_vec(&dtypes.uint64, d->dtype.count);
    DType i_dtype = i_uop->dtype;
    
    UOp* a0 = ph_take(i_uop, 0, i_dtype, u32v);
    UOp* a1 = ph_take(i_uop, 1, i_dtype, u32v);
    UOp* a2 = ph_take(i_uop, 2, i_dtype, u32v);
    UOp* a3 = ph_take(i_uop, 3, i_dtype, u32v);

    // _shl_lazy/_shr_lazy helpers
    // _shl_lazy(x, e): (x.cast_vec(uint64) * pow2if(e, d.dtype).cast_vec(uint64)).cast_vec(uint32)
    // _shr_lazy(x, off): (x.cast_vec(uint64) // pow2if(off, d.dtype).cast_vec(uint64)).cast_vec(uint32)
    UOp* pow_e = transcendental_pow2if(e_i32, &scb);
    UOp* pow_off = transcendental_pow2if(offset_uop, &scb);
    UOp* hi_l = uop_cast_vec(uop_mul(uop_cast_vec(a0, dtypes.uint64, d->dtype.count), uop_cast_vec(pow_e, dtypes.uint64, d->dtype.count)), dtypes.uint32, d->dtype.count);
    UOp* mi_l = uop_cast_vec(uop_mul(uop_cast_vec(a1, dtypes.uint64, d->dtype.count), uop_cast_vec(pow_e, dtypes.uint64, d->dtype.count)), dtypes.uint32, d->dtype.count);
    UOp* lo_l = uop_cast_vec(uop_mul(uop_cast_vec(a2, dtypes.uint64, d->dtype.count), uop_cast_vec(pow_e, dtypes.uint64, d->dtype.count)), dtypes.uint32, d->dtype.count);

    // divisions
    UOp* hi = uop_or(hi_l, uop_cast_vec(uop_div(uop_cast_vec(a1, dtypes.uint64, d->dtype.count), uop_cast_vec(pow_off, dtypes.uint64, d->dtype.count)), dtypes.uint32, d->dtype.count));
    UOp* mi = uop_or(mi_l, uop_cast_vec(uop_div(uop_cast_vec(a2, dtypes.uint64, d->dtype.count), uop_cast_vec(pow_off, dtypes.uint64, d->dtype.count)), dtypes.uint32, d->dtype.count));
    UOp* lo = uop_or(lo_l, uop_cast_vec(uop_div(uop_cast_vec(a3, dtypes.uint64, d->dtype.count), uop_cast_vec(pow_off, dtypes.uint64, d->dtype.count)), dtypes.uint32, d->dtype.count));

    // compute p = (ia*hi << 32) + (ia*mi) + (ia*lo >> 32)
    UOp* ia_hi = uop_mul(ia, uop_cast_vec(hi, dtypes.uint64, d->dtype.count));
    UOp* ia_mi = uop_mul(ia, uop_cast_vec(mi, dtypes.uint64, d->dtype.count));
    UOp* ia_lo = uop_mul(ia, uop_cast_vec(lo, dtypes.uint64, d->dtype.count));

    UOp* p = uop_add(transcendental_shl(ia_hi, 32), uop_add(ia_mi, transcendental_shr(ia_lo, 32)));

    // q = (p >> 62).cast_vec(int32)
    UOp* quadrant = uop_cast_vec(transcendental_shr(p, 62), dtypes.int32, d->dtype.count);
    // p = p & 0x3fffffffffffffff
    UOp* p_masked = uop_and(p, uop_const(u64v, 4611686018427387903.0));
    // r = (p.cast(intermediate_dtype) * 3.4061215800865545e-19).cast(d.dtype)
    UOp* r = uop_cast(uop_mul(uop_cast(p_masked, intermediate_dtype), uop_const(intermediate_dtype, 3.4061215800865545e-19)), d->dtype);

    // if f >= 0.5, adjust r -= pi/2, q += 1
    UOp* cond = uop_lt(f, uop_const(f->dtype, 0.5));
    UOp* r_adj = uop_sub(r, uop_const(d->dtype, MATH_PI/2.0));
    UOp* q_inc = uop_add(quadrant, uop_const(quadrant->dtype, 1.0));
    *r_result = uop_where(cond, r, r_adj);
    *q_result = uop_where(cond, quadrant, q_inc);
    return r_result;
}

UOp** transcendental_cody_waite_reduction(UOp* d, UOp** r_result, UOp** q_result) {
    // Performs Cody-Waite Reduction: computes the remainder of `d` modulo pi/2 for 0 <= abs(d) <= 39800.0
    {
        DType sc = dtype_scalar(&d->dtype);
        if (!dtype_eq(&sc, &dtypes.float16) && !dtype_eq(&sc, &dtypes.float32) && !dtype_eq(&sc, &dtypes.float64)) return NULL;
    }

    *r_result = NULL;
    *q_result = NULL;

    // Compute qdh and quadrant as in reference

    const double m_1_pi = 0.318309886183790671537767526745028724;
    const double two_pow_24 = 16777216.0; // 2**24

    // qdh = (d * (m_1_pi / 2**24)).cast_vec(int64).cast(d.dtype) * (2**24)
    UOp* scaled = uop_mul(d, uop_const(d->dtype, m_1_pi / two_pow_24));
    UOp* qdh_i64 = uop_cast_vec(scaled, dtypes.int64, d->dtype.count);
    UOp* qdh_f = uop_cast(qdh_i64, d->dtype);
    UOp* qdh = uop_mul(qdh_f, uop_const(d->dtype, two_pow_24));

    // quadrant rounding
    DType scb = dtype_scalar(&d->dtype);
    UOp* quadrant;
    if (dtype_eq(&scb, &dtypes.float64)) {
        quadrant = transcendental_rintk(uop_sub(uop_mul(d, uop_const(d->dtype, m_1_pi)), qdh));
    } else {
        quadrant = transcendental_rintk(uop_mul(d, uop_const(d->dtype, m_1_pi)));
    }

    UOp* q_cast = uop_cast(quadrant, d->dtype);
    *r_result = cw_reduce(d, q_cast, qdh);
    *q_result = uop_cast_vec(quadrant, dtypes.int32, d->dtype.count);
    return r_result;
}

// *** approximate sine on small angle. ***

UOp* transcendental_trig_poly(UOp* d, double coeff32[], int coeff32_count, double coeff64[], int coeff64_count) {
    // d * (polyN(d*d, coeff64) if d.dtype.scalar() == dtypes.float64 else polyN(d*d, coeff32))
    UOp* d_squared = uop_mul(d, d);
    
    DType scalar_type = dtype_scalar(&d->dtype);
    UOp* poly_result;
    
    if (dtype_eq(&scalar_type, &dtypes.float64) && coeff64_count > 0) {
        // Simplified polynomial evaluation for float64
        poly_result = uop_const(d_squared->dtype, coeff64[0]);
        for (int i = 1; i < coeff64_count; i++) {
            UOp* term = uop_const(d_squared->dtype, coeff64[i]);
            for (int j = 0; j < i; j++) {
                term = uop_mul(term, d_squared);
            }
            poly_result = uop_add(poly_result, term);
        }
    } else if (coeff32_count > 0) {
        // Simplified polynomial evaluation for float32
        poly_result = uop_const(d_squared->dtype, coeff32[0]);
        for (int i = 1; i < coeff32_count; i++) {
            UOp* term = uop_const(d_squared->dtype, coeff32[i]);
            for (int j = 0; j < i; j++) {
                term = uop_mul(term, d_squared);
            }
            poly_result = uop_add(poly_result, term);
        }
    } else {
        poly_result = d_squared;
    }
    
    return uop_mul(d, poly_result);
}

UOp* transcendental_sin_poly(UOp* d) {
    return transcendental_trig_poly(d, 
        (double[]){2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938, 1.0},
        5,
        (double[]){-7.97255955009037868891952e-18, 2.81009972710863200091251e-15, -7.64712219118158833288484e-13, 1.60590430605664501629054e-10,
                   -2.50521083763502045810755e-08, 2.75573192239198747630416e-06, -0.000198412698412696162806809, 0.00833333333333332974823815,
                   -0.166666666666666657414808,    1.0},
        10);
}

UOp* transcendental_ifand(UOp* q, int n) {
    return uop_cmpne(uop_and(q, uop_const(q->dtype, n)), uop_const(q->dtype, 0));
}

UOp* transcendental_sin_poly_small(UOp* d, UOp* q) {
    UOp* r = transcendental_sin_poly(d);
    UOp* neg_one = uop_const(r->dtype, -1.0);
    UOp* pos_one = uop_const(r->dtype, 1.0);
    UOp* one_and = transcendental_ifand(q, 1);
    UOp* multiplier = uop_where(one_and, neg_one, pos_one);
    return uop_mul(r, multiplier);
}

UOp* transcendental_sin_poly_large(UOp* d, UOp* q) {
    UOp* pi_half = uop_const(d->dtype, MATH_PI / 2.0);
    UOp* zero = uop_const(d->dtype, 0.0);
    UOp* one_and = transcendental_ifand(q, 1);
    UOp* adjusted_d = uop_where(one_and, pi_half, zero);
    UOp* r = transcendental_sin_poly(uop_add(d, adjusted_d));
    
    UOp* neg_one = uop_const(r->dtype, -1.0);
    UOp* pos_one = uop_const(r->dtype, 1.0);
    UOp* two_and = transcendental_ifand(q, 2);
    UOp* multiplier = uop_where(two_and, neg_one, pos_one);
    return uop_mul(r, multiplier);
}

// *** toplevel functions for xsin/xlog2/xexp2 ***

UOp* transcendental_lazy_map_numbers(UOp* x, UOp* inf, UOp* neg_inf, UOp* nan, UOp* ratio) {
    /* replace inf -> inf, -inf -> neg_inf, nan -> nan, otherwise -> ratio */
    UOp* math_inf = uop_const(x->dtype, INFINITY);
    UOp* math_neg_inf = uop_const(x->dtype, -INFINITY);

    UOp* inner = uop_where(
        uop_ne(x, x),
        nan,
        uop_where(uop_ne(x, math_neg_inf), ratio, neg_inf)
    );
    return uop_where(uop_ne(x, math_inf), inner, inf);
}

UOp* transcendental_xsin(UOp* x, bool fast, float switch_over) {
    // Implements a 1.0 ULP approximation for Ops.SIN
    UOp* d = x;
    // mask +-inf/nan as zero
    UOp* x_masked = transcendental_lazy_map_numbers(d, uop_const(d->dtype, 0.0), uop_const(d->dtype, 0.0), uop_const(d->dtype, 0.0), d);
    // x_sign = sign(x)
    UOp* x_sign = uop_where(
        uop_ne(x_masked, uop_const(d->dtype, 0.0)),
        uop_where(uop_lt(x_masked, uop_const(d->dtype, 0.0)), uop_const(d->dtype, -1.0), uop_const(d->dtype, 1.0)),
        uop_const(d->dtype, 0.0)
    );
    UOp* x_abs = uop_mul(x_masked, x_sign);

    UOp *r = NULL, *q = NULL;
    if (fast) transcendental_cody_waite_reduction(x_abs, &r, &q);
    else transcendental_payne_hanek_reduction(x_abs, &r, &q);

    UOp* result;
    if (fast) {
        result = transcendental_sin_poly_small(r, q);
    } else {
        UOp *r_small = NULL, *q_small = NULL;
        transcendental_cody_waite_reduction(x_abs, &r_small, &q_small);
        result = uop_where(
            uop_lt(x_abs, uop_const(d->dtype, (double)switch_over)),
            transcendental_sin_poly_small(r_small, q_small),
            transcendental_sin_poly_large(r, q)
        );
    }
    result = uop_mul(result, x_sign);
    return transcendental_lazy_map_numbers(d, uop_const(d->dtype, NAN), uop_const(d->dtype, NAN), uop_const(d->dtype, NAN), result);
}

UOp* transcendental_xexp2(UOp* x) {
    // Implements a 1.0 ULP approximation for Ops.EXP2
    UOp* d = x;
    UOp* x_masked = transcendental_lazy_map_numbers(d, uop_const(d->dtype, 0.0), uop_const(d->dtype, 0.0), uop_const(d->dtype, 0.0), d);
    UOp* q = transcendental_rintk(x_masked);
    UOp* s = uop_sub(x_masked, uop_cast(q, d->dtype));

    UOp* u;
    DType sc = dtype_scalar(&d->dtype);
    if (dtype_eq(&sc, &dtypes.float64)) {
        const double c[] = {0.4434359082926529454e-9, 0.7073164598085707425e-8, 0.1017819260921760451e-6, 0.1321543872511327615e-5, 0.1525273353517584730e-4,
                            0.1540353045101147808e-3, 0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1, 0.2402265069591012214e+0,
                            0.6931471805599452862e+0, 0.1000000000000000000e+1};
        u = transcendental_polyN(s, c, 12);
    } else {
        const double c[] = {0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0};
        u = transcendental_polyN(s, c, 7);
    }

    u = transcendental_ldexp2k(u, q);

    int upper, lower;
    if (dtype_eq(&sc, &dtypes.float64)) { upper = 1024; lower = -2000; }
    else if (dtype_eq(&sc, &dtypes.float32)) { upper = 128; lower = -150; }
    else { upper = 23; lower = -22; }

    UOp* u2 = uop_where(uop_ge(d, uop_const(d->dtype, (double)upper)), uop_const(d->dtype, INFINITY), u);
    u2 = uop_where(uop_lt(d, uop_const(d->dtype, (double)lower)), uop_const(d->dtype, 0.0), u2);
    return uop_where(uop_ne(d, d), uop_const(d->dtype, NAN), u2);
}

UOp* transcendental_xlog2(UOp* x) {
    // Implements a 1.0 ULP approximation for Ops.LOG2
    DType sc = dtype_scalar(&x->dtype);
    if (dtype_eq(&sc, &dtypes.float16)) {
        return uop_cast(transcendental_xlog2(uop_cast(x, dtype_vec(&dtypes.float32, x->dtype.count))), x->dtype);
    }

    UOp* d = x;
    double flt_min = dtype_eq(&sc, &dtypes.float16) ? 1e-6 : 1e-4;
    UOp* is_denormal = uop_lt(d, uop_const(d->dtype, flt_min));
    UOp* a = uop_where(is_denormal, uop_mul(d, uop_const(d->dtype, pow(2.0, 64))), d);

    UOp* e = uop_cast(transcendental_ilogb2k(uop_mul(a, uop_div(uop_const(a->dtype, 1.0), uop_const(a->dtype, 0.75)))), a->dtype);
    UOp* m = transcendental_ldexp3k(a, uop_neg(e));
    e = uop_where(is_denormal, uop_sub(e, uop_const(e->dtype, 64.0)), e);

    UOp* one = uop_const(m->dtype, 1.0);
    UOp* xv = uop_div(uop_sub(m, one), uop_add(m, one));
    UOp* x2 = uop_mul(xv, xv);

    UOp* t; UOp* s_hi; UOp* s_lo;
    if (dtype_eq(&sc, &dtypes.float64)) {
        const double c[] = {0.2211941750456081490e+0, 0.2200768693152277689e+0, 0.2623708057488514656e+0, 0.3205977477944495502e+0,
                            0.4121985945485324709e+0, 0.5770780162997058982e+0, 0.96179669392608091449};
        t = transcendental_polyN(x2, c, 7);
        s_hi = uop_add(e, uop_mul(xv, uop_const(xv->dtype, 2.885390081777926774)));
        s_lo = uop_const(e->dtype, 0.0);
    } else {
        const double c[] = {0.4374550283e+0, 0.5764790177e+0, 0.9618012905120};
        t = transcendental_polyN(x2, c, 3);
        s_hi = uop_add(e, uop_mul(xv, uop_const(xv->dtype, 2.8853900432586669922)));
        s_lo = uop_mul(xv, uop_const(xv->dtype, 3.2734474483568488616e-08));
    }
    UOp* r = uop_add(uop_mul(t, uop_mul(xv, x2)), uop_add(s_hi, s_lo));

    UOp* math_inf = uop_const(d->dtype, INFINITY);
    r = uop_where(uop_cmpne(d, math_inf), r, math_inf);
    r = uop_where(uop_lt(d, uop_const(d->dtype, 0.0)), uop_const(d->dtype, NAN), r);
    int log2_zero = dtype_eq(&sc, &dtypes.float64) ? -1087 : (dtype_eq(&sc, &dtypes.float32) ? -191 : -79);
    r = uop_where(uop_cmpne(r, uop_const(r->dtype, (double)log2_zero)), r, uop_const(r->dtype, -INFINITY));
    r = uop_where(uop_ne(d, d), uop_const(d->dtype, NAN), r);
    return uop_where(uop_cmpne(uop_recip(d), uop_const(r->dtype, -INFINITY)), r, uop_const(r->dtype, -INFINITY));
}

UOp* transcendental_xpow(UOp* base, UOp* exponent) {
    // b ** e = exp2(e * log2(|b|)) with negative base fixups
    UOp* base_abs = uop_where(uop_lt(base, uop_const(base->dtype, 0.0)), uop_neg(base), base);
    UOp* ret = transcendental_xexp2(uop_mul(exponent, transcendental_xlog2(base_abs)));

    // negative base adjustment: nan for non-integer exponent and -1 for odd exponent
    UOp* exponent_i32 = uop_cast(exponent, dtype_vec(&dtypes.int32, exponent->dtype.count));
    UOp* non_int = uop_ne(exponent, exponent_i32);
    UOp* eabs = uop_where(uop_lt(exponent, uop_const(exponent->dtype, 0.0)), uop_neg(exponent), exponent);
    UOp* eabs_i32 = uop_cast(eabs, dtype_vec(&dtypes.int32, eabs->dtype.count));
    UOp* odd = uop_cmpne(uop_remainder(eabs_i32, uop_const(eabs_i32->dtype, 2)), uop_const(eabs_i32->dtype, 0));
    UOp* adj = uop_where(non_int, uop_const(ret->dtype, NAN), uop_where(odd, uop_const(ret->dtype, -1.0), uop_const(ret->dtype, 1.0)));

    // fix 0 ** 0 = 1
    UOp* zero = uop_const(base->dtype, 0.0);
    UOp* ret1 = uop_mul(ret, uop_where(uop_lt(base, zero), adj, uop_const(ret->dtype, 1.0)));
    return uop_where(uop_and(uop_eq(base, zero), uop_eq(exponent, zero)), uop_const(ret->dtype, 1.0), ret1);
}

// *** integer division ***

DivisionMagic transcendental_magicgu(int vmax, int d) {
    // calculate m,s such that x//d == (x*m) >> s for all 0 <= x <= vmax, d>0; adapted from Hacker's Delight, Chapter 10
    DivisionMagic result = {0, 0, false};
    if (d <= 0) return result;

    int nc = ((vmax + 1) / d) * d - 1;
    int nbits = 0;
    int temp = vmax;
    while (temp > 0) { nbits++; temp >>= 1; }
    int max_s = 2 * nbits + 1;

    for (int s = 0; s <= max_s; s++) {
        unsigned __int128 two_s = ((unsigned __int128)1) << s;
        unsigned __int128 lhs = two_s;
        unsigned __int128 rhs = (unsigned __int128)nc * (unsigned __int128)(d - 1 - (int)(((two_s - 1) % d)));
        if (lhs > rhs) {
            int m = (int)((two_s + d - 1 - (int)((two_s - 1) % d)) / d);
            result.magic = m;
            result.shift = s;
            result.valid = true;
            return result;
        }
    }
    return result;
}

// local helper: unsigned variant of an integer dtype (same bitwidth)
static const DType* unsigned_variant(const DType* dt) {
    if (dtype_eq(dt, &dtypes.int8)) return &dtypes.uint8;
    if (dtype_eq(dt, &dtypes.int16)) return &dtypes.uint16;
    if (dtype_eq(dt, &dtypes.int32)) return &dtypes.uint32;
    if (dtype_eq(dt, &dtypes.int64)) return &dtypes.uint64;
    if (dtypes_is_unsigned(dt)) return dt;
    return dt; // fallback
}

static bool device_supports_dtype(const char* device, const DType* dt) {
    (void)device; // CPU backend: support integer types
    return dtypes_is_int(dt);
}

UOp* transcendental_fast_idiv(const char* device, UOp* x, int d) {
    // idiv is truncated division, but arithmetic shift is floored division, so can only do non-negative numbers!
    if (x->vmin_vmax_valid && x->vmin < 0) return NULL;

    int sign = d > 0 ? 1 : -1;
    int ad = d >= 0 ? d : -d;

    // vmax := min(x.vmax, dtypes.max(x.dtype))
    long long vmax;
    double dtype_max = dtypes_max(&x->dtype);
    if (x->vmin_vmax_valid) {
        vmax = (long long)x->vmax;
        if ((double)vmax > dtype_max) vmax = (long long)dtype_max;
    } else {
        vmax = (long long)dtype_max;
    }

    DivisionMagic ms = transcendental_magicgu((int)vmax, ad);
    if (!ms.valid) return NULL;

    // if m * vmax <= dtypes.max(x.dtype): return sign * ((x*m) >> s)
    unsigned __int128 prod = (unsigned __int128)(unsigned long long)ms.magic * (unsigned __int128)(unsigned long long)vmax;
    long double xmax = (long double)dtype_max;
    if ((long double)prod <= xmax) {
        UOp* xm = uop_mul(x, uop_const(x->dtype, (double)ms.magic));
        UOp* q = transcendental_shr(xm, ms.shift);
        UOp* sgn = uop_const(x->dtype, (double)sign);
        return uop_mul(sgn, q);
    }

    // else, try unsigned promotion of same width
    const DType* next_dtype = unsigned_variant(&x->dtype);
    if (dtypes_is_int(next_dtype) && device_supports_dtype(device, next_dtype)) {
        long double next_max = (long double)dtypes_max(next_dtype);
        if ((long double)prod <= next_max) {
            UOp* xc = uop_cast(x, *next_dtype);
            UOp* xm = uop_mul(xc, uop_const(*next_dtype, (double)ms.magic));
            UOp* q = transcendental_shr(xm, ms.shift);
            UOp* q_back = uop_cast(q, x->dtype);
            UOp* sgn = uop_const(x->dtype, (double)sign);
            return uop_mul(sgn, q_back);
        }
    }
    return NULL;
}
