use criterion::{ criterion_group, criterion_main, Criterion};

// #[path = "./dot_bench_2.rs"]
// extern crate benches;

mod dot_bench_2;
mod dot_bench;

pub fn bench(c: &mut Criterion) {
    dot_bench::bench_dot_product(c);
    dot_bench_2::bench_dot_product(c);
}

criterion_group!(
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.1).sample_size(50);
    targets = bench
);
criterion_main!(benches);