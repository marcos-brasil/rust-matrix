// use typenum::{Prod, UInt, UTerm, Unsigned, B1, U1, U1024, U512};
// use typenum::{Integer, Prod, UInt, UTerm, Unsigned, B1, U512};
// use typenum::{Prod, Unsigned};
// use typenum::{U0, U1, U2, U3, U4, U5, U6, U8};
use criterion::{black_box, Criterion};

use typenum::{U1, U12, U2, U3, U4, U5, U100, U200};

// use rand::rngs::ThreadRng;
// use rand_distr::{Distribution, Normal, Uniform};

// use std::ops::Mul;

// use generic_array::sequence::GenericSequence;
// use generic_array::arr;
// use generic_array::{ArrayLength, GenericArray};

extern crate matrix;

use matrix::Matrix;

pub fn bench_dot_product(c: &mut Criterion) {
    println!("BBBBB");
    c.bench_function("dot product", |b| {
        b.iter(|| {
        // println!("AAAAA");

            let a = Matrix::<U100, U200>::from_fn(|i| i as f32);
            let b = Matrix::<U200, U100>::from_fn(|i| (i * 2) as f32);
    
            a.dot(black_box(&b))
        });
    });


}

// criterion_group!(
//     name = benches;
//     // This can be any expression that returns a `Criterion` object.
//     config = Criterion::default().significance_level(0.1).sample_size(50);
//     targets = bench_dot_product
// );
// criterion_main!(benches);