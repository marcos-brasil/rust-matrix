// use typenum::{Prod, UInt, UTerm, Unsigned, B1, U1, U1024, U512};
// use typenum::{Integer, Prod, UInt, UTerm, Unsigned, B1, U512};
// use typenum::{Prod, Unsigned};
// use typenum::{U0, U1, U2, U3, U4, U5, U6, U8};
use typenum::{U1, U12, U2, U20, U25, U256, U3, U4, U5};

// use rand::rngs::ThreadRng;
// use rand_distr::{Distribution, Normal, Uniform};

// use std::ops::Mul;

// use generic_array::sequence::GenericSequence;
use generic_array::arr;
// use generic_array::{ArrayLength, GenericArray};

extern crate matrix;

use matrix::Matrix;
use std::thread;

#[test]
fn ops() {

    // let a_1 = Matrix::<U256, U256>::new();
    // println!("{:?}", a_1);
    let a = Matrix::<U2, U5>::new();
    let b = Matrix::<U2, U5>::fill(1.0);
    let c = Matrix::<U2, U5>::from_fn(|i| i as f32);
    let c_t = c.transpose();

    // println!("{:?}", c_t.dot(&c));

    let mut c1 = c.clone();

    let dia = b.diagonal::<U12, U12>();

    dia.each_i(|row, col, val| {
        if row == col {
            if row < b.len() {
                assert_eq!(val as i32, 1);
            } else {
                assert_eq!(val as i32, 0);
            }
        }
    });

    // println!("dia: {:?}", dia);

    c1.map_mut(|_, v| v * 2.0);

    let d = c.map_i(|row, col, _| 2.0 * (row + col * c.shape.0) as f32);
    let e = c.map(|_, v| v * v);
    let f = b.transpose();

    let x = Matrix::<U3, U4>::from_fn(|i| i as f32);


    let w = Matrix::<U25, U20>::from_fn(|i| i as f32);
    let w_t = w.transpose();

    let reshaped = x.reshape::<U12, U1>();

    assert_eq!(
        c1,
        Matrix {
            shape: (2, 5),
            size: 10,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32; 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        }
    );

    assert_eq!(
        reshaped,
        Matrix {
            shape: (12, 1),
            size: 12,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32; 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        }
    );

    let trimmed = x.trim::<U2, U4, U2>();

    assert_eq!(
        trimmed,
        Matrix {
            shape: (2, 4),
            size: 8,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32; 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        }
    );

    assert_eq!(
        a,
        Matrix {
            shape: (2, 5),
            size: 10,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32; 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    );

    assert_eq!(
        b.data,
        arr![f32; 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );

    assert_eq!(
        c.data,
        arr![f32; 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );

    assert_eq!(
        d.data,
        arr![f32; 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    );

    assert_eq!(
        e.data,
        arr![f32; 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0]
    );

    assert_eq!(
        f,
        Matrix {
            shape: (5, 2),
            size: 10,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32; 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
    );

    assert_eq!(
        f.dot(&b),
        Matrix {
            shape: (5, 5),
            size: 25,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32;2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        }
    );

    assert_eq!(
        c_t.dot(&c),
        Matrix {
            shape: (5, 5),
            size: 25,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32;25.0, 30.0, 35.0, 40.0, 45.0, 30.0, 37.0, 44.0, 51.0, 58.0, 35.0, 44.0, 53.0, 62.0, 71.0, 40.0, 51.0, 62.0, 73.0, 84.0, 45.0, 58.0, 71.0, 84.0, 97.0]
        }
    );

    assert_eq!(
        b.dot(&f),
        Matrix {
            shape: (2, 2),
            size: 4,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32;5.0, 5.0, 5.0, 5.0]
        }
    );

    assert_eq!(
        f.dot_fast(&b),
        Matrix {
            shape: (5, 5),
            size: 10,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32;2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        }
    );

    assert_eq!(
        c_t.dot_fast(&c).data,
        c_t.dot(&c).data
    );

    assert_eq!(
        w_t.dot_fast(&w).data,
        w_t.dot(&w).data
    );

    assert_eq!(
        b.dot_fast(&f),
        Matrix {
            shape: (2, 2),
            size: 4,
            dot_fast_partition:matrix::DOT_FAST_PARTITION,
            data: arr![f32;5.0, 5.0, 5.0, 5.0]
        }
    );
}

#[test]
fn add() {
    let b = Matrix::<U2, U5>::fill(1.0);

    assert_eq!(
        b.add(&b).data,
        arr![f32; 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    );

    let mut b_mut = b.add(&b);
    b_mut.add_mut(&b);
    assert_eq!(
        b_mut.data,
        arr![f32; 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    );

    assert_eq!(
        b.add_scalar(1.0).data,
        arr![f32; 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    );

    let mut b_mut = b.add_scalar(1.0);
    b_mut.add_scalar_mut(1.0);
    assert_eq!(
        b_mut.data,
        arr![f32; 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    );
}

#[test]
fn sub() {
    let b = Matrix::<U2, U5>::fill(1.0);

    assert_eq!(
        b.sub(&b).data,
        arr![f32; 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    let mut b_mut = b.sub(&b);
    b_mut.sub_mut(&b);
    assert_eq!(
        b_mut.data,
        arr![f32; -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    );

    assert_eq!(
        b.sub_scalar(1.0).data,
        arr![f32; 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    let mut b_mut = b.sub_scalar(1.0);
    b_mut.sub_scalar_mut(1.0);
    assert_eq!(
        b_mut.data,
        arr![f32; -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    );
}

#[test]
fn mul() {
    let b = Matrix::<U2, U5>::fill(1.0);

    assert_eq!(
        // (b + b) * (b + b)
        b.add(&b).mul(&b.add(&b)).data,
        arr![f32; 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    );

    let mut b_mut = b.add(&b);
    // (b + b) * (b + b)
    b_mut.mul_mut(&b.add(&b));

    assert_eq!(
        b_mut.data,
        arr![f32; 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    );

    assert_eq!(
        b.mul_scalar(2.0).data,
        arr![f32; 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    );

    let mut b_mut = b.mul_scalar(2.0);
    b_mut.mul_scalar_mut(2.0);
    assert_eq!(
        b_mut.data,
        arr![f32; 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    );
}

#[test]
fn div() {
    let b = Matrix::<U2, U5>::fill(1.0);

    assert_eq!(
        // (b + b) / (b + b)
        b.add(&b).div(&b.add(&b)).data,
        arr![f32; 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );

    let mut b_mut = b.add(&b);
    // (b + b) / (b + b)
    b_mut.div_mut(&b.add(&b));

    assert_eq!(
        b_mut.data,
        arr![f32; 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );

    assert_eq!(
        b.div_scalar(2.0).data,
        arr![f32; 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    );

    let mut b_mut = b.div_scalar(2.0);
    b_mut.div_scalar_mut(2.0);
    assert_eq!(
        b_mut.data,
        arr![f32; 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    );
}
