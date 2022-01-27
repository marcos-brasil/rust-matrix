// use typenum::{Prod, UInt, UTerm, Unsigned, B1, U1, U2048, U512};
// use typenum::{Integer, Prod, UInt, UTerm, Unsigned, B1, U512};
// use typenum::{Prod, Unsigned};
// use typenum::{U0, U1, U2, U3, U4, U5, U6, U8};

use std::thread;

// use criterion::{black_box, Criterion};

use typenum::{U1, U100, U12, U2, U200, U256, U3, U4, U5, U512, U2048};

// use rand::rngs::ThreadRng;
// use rand_distr::{Distribution, Normal, Uniform};

// use std::ops::Mul;

// use generic_array::sequence::GenericSequence;
// use generic_array::arr;
// use generic_array::{ArrayLength, GenericArray};
use std::time::Instant;

extern crate matrix;

use matrix::Matrix;

pub fn bench_dot_product() {
    let builder_fast = thread::Builder::new()
        .name("dot_product".into())
        .stack_size(1024 * 1024 * 1024);

    let builder = thread::Builder::new()
        .name("dot_product".into())
        .stack_size(1024 * 1024 * 1024);

    // let bb = b.copy();
    let handler_fast = builder_fast
        .spawn(move || {
            let a1 = Matrix::<U2048, U2048>::from_fn(|i| i as f32);
            let b1 = Matrix::<U2048, U2048>::from_fn(|i| (i * 2) as f32);

            let o1 = &mut Matrix::<U2048, U2048>::new();

            let mut time: Vec<f32> = Vec::new();

            let rounds = 10;

            for _ in 0..rounds {
                let now = Instant::now();

                a1.dot_fast_out(&b1, o1);
                let elapsed = now.elapsed().as_secs_f32();

                time.push(elapsed);
                // println!("took {:?} {:.4?}", run, elapsed)
            }

            let sum = time.into_iter().fold(0.0, |acc, i| acc + i);

            println!("took fast out mean: {:.4?}", sum / (rounds as f32))
        })
        .unwrap();

        let handler = builder
        .spawn(move || {
            let a1 = Matrix::<U2048, U2048>::from_fn(|i| i as f32);
            let b1 = Matrix::<U2048, U2048>::from_fn(|i| (i * 2) as f32);
            let o1 = &mut Matrix::<U2048, U2048>::new();

            let mut time: Vec<f32> = Vec::new();

            let rounds = 10;

            for _ in 0..rounds {
                let now = Instant::now();

                a1.dot_out(&b1, o1);
                let elapsed = now.elapsed().as_secs_f32();

                time.push(elapsed);
                // println!("took {:?} {:.4?}", run, elapsed)
            }

            let sum = time.into_iter().fold(0.0, |acc, i| acc + i);

            println!("took out mean: {:.4?}", sum / (rounds as f32))
        })
        .unwrap();

    handler_fast.join().unwrap();

    handler.join().unwrap();
}
