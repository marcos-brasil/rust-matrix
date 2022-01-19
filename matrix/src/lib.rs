// use typenum::{Prod, UInt, UTerm, Unsigned, B1, U1, U1024, U512};
// use typenum::{Integer, Prod, UInt, UTerm, Unsigned, B1, U512};
// use typenum::{assert_type_eq, Cmp, Gr, IsGreater, IsLessOrEqual, LeEq, Prod, Same, Sum, Unsigned};
// use typenum::{B1, U0};

use typenum::{Cmp, Gr, IsGreater, IsLessOrEqual, LeEq, Prod, Same, Sum, Unsigned};
use typenum::{B1, U0};

use assert_type_eq::assert_type_eq;
// macro_rules! assert_type_eq {
//     ($a:ty, $b:ty) => {
//         let _: <$a as Same<$b>>::Output;
//     };
// }

// use rand::rngs::ThreadRng;
// use rand_distr::{Distribution, Normal, Uniform};

use std::ops::{Add, Mul};

use generic_array::sequence::GenericSequence;
use generic_array::{ArrayLength, GenericArray};

#[derive(Debug, Clone)]
pub struct Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    pub shape: (usize, usize),
    pub size: usize,
    pub data: GenericArray<f32, Prod<R, C>>,
}

impl<R, C> Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    pub fn new() -> Self {
        Matrix {
            shape: (R::U32 as usize, C::U32 as usize),
            size: (R::U32 as usize) * (C::U32 as usize),
            data: GenericArray::<f32, Prod<R, C>>::default(),
        }
    }

    pub fn fill(n: f32) -> Self {
        Matrix {
            shape: (R::U32 as usize, C::U32 as usize),
            size: (R::U32 as usize) * (C::U32 as usize),
            data: GenericArray::<f32, Prod<R, C>>::generate(|_| n),
        }
    }

    pub fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> f32,
    {
        Matrix {
            shape: (R::U32 as usize, C::U32 as usize),
            size: (R::U32 as usize) * (C::U32 as usize),
            data: GenericArray::<f32, Prod<R, C>>::generate(f),
        }
    }

    pub fn id() -> Self {
        let mut out = Self::new();

        out.map_i_mut(|row, col, _| if row == col { 1.0 } else { 0.0 });

        out
    }

    pub fn from_vec(v: Vec<f32>) -> Self {
        let mut out = Self::new();

        #[allow(clippy::needless_range_loop)]
        for idx in 0..v.len() {
            out.data[idx] = v[idx]
        }

        out
    }
}

#[allow(clippy::len_without_is_empty)]
impl<R, C> Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    pub fn len(&self) -> usize {
        self.size
    }

    // there can't be a empty matrix. they are all populated with zeros by default
    pub fn is_empty() -> bool {
        false
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        let rows = self.shape.0;

        let idx = row + col * rows;
        self.data[idx]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        let rows = self.shape.0;

        let idx = row + col * rows;
        self.data[idx] = val;
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut out = Vec::new();

        for idx in 0..self.size {
            out.push(self.data[idx])
        }

        out
    }

    pub fn diagonal<R2, C2>(&self) -> Matrix<R2, C2>
    where
        R2: Unsigned + Mul<C2> + Cmp<U0> + IsGreater<U0>,
        C2: Unsigned + Mul<R2> + Cmp<U0> + IsGreater<U0>,

        Prod<R2, C2>: ArrayLength<f32>,
        Prod<C2, R2>: ArrayLength<f32>,

        Prod<R, C>: Cmp<R2> + IsLessOrEqual<R2>,
        Prod<R, C>: Cmp<C2> + IsLessOrEqual<C2>,

        LeEq<Prod<R, C>, R2>: Same<B1>,
        LeEq<Prod<R, C>, C2>: Same<B1>,

        Gr<C2, U0>: Same<B1>,
        Gr<R2, U0>: Same<B1>,
    {
        // this assertions garantee type safety.
        // that matrixes will have the minimum required sizes
        assert_type_eq!(LeEq<Prod<R, C>, R2>, B1);
        assert_type_eq!(LeEq<Prod<R, C>, C2>, B1);

        assert_type_eq!(Gr<C2, U0>, B1);
        assert_type_eq!(Gr<R2, U0>, B1);

        let mut out = Matrix::new();

        for idx in 0..self.size {
            out.set(idx, idx, self.data[idx])
        }

        out
    }

    pub fn reshape<R2, C2>(&self) -> Matrix<R2, C2>
    where
        R2: Mul<C2> + Unsigned,
        C2: Mul<R2> + Unsigned,

        R2: Cmp<U0>,
        C2: Cmp<U0>,

        R2: IsGreater<U0>,
        C2: IsGreater<U0>,

        Gr<R2, U0>: Same<B1>,
        Gr<C2, U0>: Same<B1>,

        Prod<R2, C2>: ArrayLength<f32>,
        Prod<C2, R2>: ArrayLength<f32>,

        Prod<R2, C2>: Add<U0>,
        Sum<Prod<R2, C2>, U0>: Cmp<Prod<R, C>>,

        Sum<Prod<R2, C2>, U0>: IsLessOrEqual<Prod<R, C>>,
        LeEq<Sum<Prod<R2, C2>, U0>, Prod<R, C>>: Same<B1>,
    {
        self.trim::<R2, C2, U0>()
    }

    pub fn reshape_out<'a, R2, C2>(&self, out: &'a mut Matrix<R2, C2>) -> &'a mut Matrix<R2, C2>
    where
        R2: Mul<C2> + Unsigned,
        C2: Mul<R2> + Unsigned,
        R2: Cmp<U0>,
        C2: Cmp<U0>,

        R2: IsGreater<U0>,
        C2: IsGreater<U0>,

        Gr<R2, U0>: Same<B1>,
        Gr<C2, U0>: Same<B1>,

        Prod<R2, C2>: ArrayLength<f32>,
        Prod<C2, R2>: ArrayLength<f32>,

        Prod<R2, C2>: Add<U0>,
        Sum<Prod<R2, C2>, U0>: Cmp<Prod<R, C>>,

        Sum<Prod<R2, C2>, U0>: IsLessOrEqual<Prod<R, C>>,
        LeEq<Sum<Prod<R2, C2>, U0>, Prod<R, C>>: Same<B1>,
    {
        self.trim_out::<R2, C2, U0>(out)
    }

    pub fn trim<R2, C2, E>(&self) -> Matrix<R2, C2>
    where
        E: Unsigned,

        R2: Mul<C2> + Unsigned,
        C2: Mul<R2> + Unsigned,

        R2: Cmp<U0>,
        C2: Cmp<U0>,

        R2: IsGreater<U0>,
        C2: IsGreater<U0>,

        Gr<R2, U0>: Same<B1>,
        Gr<C2, U0>: Same<B1>,

        Prod<R2, C2>: ArrayLength<f32>,
        Prod<C2, R2>: ArrayLength<f32>,

        Prod<R2, C2>: Add<E>,
        Sum<Prod<R2, C2>, E>: Cmp<Prod<R, C>>,

        Sum<Prod<R2, C2>, E>: IsLessOrEqual<Prod<R, C>>,
        LeEq<Sum<Prod<R2, C2>, E>, Prod<R, C>>: Same<B1>,
    {
        let mut out = Matrix::<R2, C2>::new();

        assert_type_eq!(LeEq<Sum<Prod<R2, C2>, E>, Prod<R, C>>, B1);
        assert_type_eq!(Gr<R2, U0>, B1);
        assert_type_eq!(Gr<C2, U0>, B1);

        let end = (R2::U32 * C2::U32) as usize;

        let offset = E::U32 as usize;

        for idx in offset..(end + offset) {
            out.data[idx - offset] = self.data[idx]
        }

        out
    }

    pub fn trim_out<'a, R2, C2, E>(&self, out: &'a mut Matrix<R2, C2>) -> &'a mut Matrix<R2, C2>
    where
        E: Unsigned,

        R2: Mul<C2> + Unsigned + IsGreater<U0>,
        C2: Mul<R2> + Unsigned + IsGreater<U0>,

        Gr<R2, U0>: Same<B1>,
        Gr<C2, U0>: Same<B1>,

        Prod<R2, C2>: ArrayLength<f32>,
        Prod<C2, R2>: ArrayLength<f32>,

        Prod<R2, C2>: Add<E>,
        Sum<Prod<R2, C2>, E>: Cmp<Prod<R, C>>,

        Sum<Prod<R2, C2>, E>: IsLessOrEqual<Prod<R, C>>,
        LeEq<Sum<Prod<R2, C2>, E>, Prod<R, C>>: Same<B1>,
    {
        // need to use this assertion.
        // the the compiler errors and suggestions helped buiild the trait restrictions.
        // But the suggestions is very berbose. then some studing and cleanup to keep the restrictions clean
        // for instance, typenum::uint::UInt<typenum::uint::UTerm, B1> is simply U1
        //
        // this is a way to inforce the type restrictions at compile time
        // R2 * C2 + E <= R * C
        // where
        //      R2 and C2 are the new matrix dimensions
        //      E is the offset
        //      R and C are the original matrix dimentions
        //
        assert_type_eq!(LeEq<Sum<Prod<R2, C2>, E>, Prod<R, C>>, B1);
        assert_type_eq!(Gr<R2, U0>, B1);
        assert_type_eq!(Gr<C2, U0>, B1);

        let end = (R2::U32 * C2::U32) as usize;

        let offset = E::U32 as usize;

        for idx in offset..(end + offset) {
            out.data[idx - offset] = self.data[idx]
        }

        out
    }
}

impl<R, C> Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    pub fn sum(&self) -> f32 {
        let mut sum = 0.0;

        for idx in 0..self.size {
            sum += self.data[idx]
        }

        sum
    }

    pub fn each<F>(&self, mut f: F)
    where
        F: FnMut(usize, f32) -> (),
    {
        for idx in 0..self.size {
            f(idx, self.data[idx])
        }
    }

    pub fn each_i<F>(&self, mut f: F)
    where
        F: FnMut(usize, usize, f32) -> (),
    {
        let rows = self.shape.0;
        let cols = self.shape.1;

        for row in 0..rows {
            for col in 0..cols {
                let idx = row + col * rows;
                f(row, col, self.data[idx])
            }
        }
    }

    pub fn map<F>(&self, mut f: F) -> Self
    where
        F: FnMut(usize, f32) -> f32,
    {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = f(idx, self.data[idx])
        }

        out
    }

    pub fn map_mut<F>(&mut self, mut f: F) -> &Self
    where
        F: FnMut(usize, f32) -> f32,
    {
        for idx in 0..self.size {
            self.data[idx] = f(idx, self.data[idx])
        }

        self
    }

    pub fn map_out<'a, F>(&self, mut f: F, out: &'a mut Matrix<R, C>) -> &'a mut Self
    where
        F: FnMut(usize, f32) -> f32,
    {
        for idx in 0..self.size {
            out.data[idx] = f(idx, self.data[idx])
        }

        out
    }

    pub fn map_i<F>(&self, mut f: F) -> Self
    where
        F: FnMut(usize, usize, f32) -> f32,
    {
        let mut out = Self::new();

        let rows = self.shape.0;
        let cols = self.shape.1;

        for row in 0..rows {
            for col in 0..cols {
                let idx = row + col * rows;
                out.data[idx] = f(row, col, self.data[idx])
            }
        }

        out
    }

    pub fn map_i_mut<F>(&mut self, mut f: F) -> &Self
    where
        F: FnMut(usize, usize, f32) -> f32,
    {
        let rows = self.shape.0;
        let cols = self.shape.1;

        for row in 0..rows {
            for col in 0..cols {
                let idx = row + col * rows;
                self.data[idx] = f(row, col, self.data[idx])
            }
        }

        self
    }

    pub fn map_i_out<'a, F>(&self, mut f: F, out: &'a mut Matrix<R, C>) -> &'a mut Self
    where
        F: FnMut(usize, usize, f32) -> f32,
    {
        let rows = self.shape.0;
        let cols = self.shape.1;

        for row in 0..rows {
            for col in 0..cols {
                let idx = row + col * rows;
                out.data[idx] = f(row, col, self.data[idx])
            }
        }

        out
    }

    // very naive dot product implementations
    // but for small matrixes is good enough
    // the important part is to avoid cache misses
    pub fn dot<C2>(&self, rhs: &Matrix<C, C2>) -> Matrix<R, C2>
    where
        R: Mul<C2> + Unsigned,
        C: Mul<C2> + Unsigned,

        C2: Mul<C> + Unsigned,
        C2: Mul<R> + Unsigned,

        Prod<R, C2>: ArrayLength<f32>,
        Prod<C2, R>: ArrayLength<f32>,

        Prod<C, C2>: ArrayLength<f32>,
        Prod<C2, C>: ArrayLength<f32>,
    {
        let mut out = Matrix::<R, C2>::new();

        let (l_rows, l_cols) = self.shape;
        let (_r_rows, r_cols) = rhs.shape;

        for idx0 in 0..l_rows {
            for idx1 in 0..r_cols {
                let mut sum = 0.0;

                for idx2 in 0..l_cols {
                    let l_item = self.data[idx0 * l_cols + idx2];
                    let r_item = rhs.data[idx2 * r_cols + idx1];

                    sum += l_item * r_item
                }

                out.data[idx0 * r_cols + idx1] = sum;
            }
        }

        out
    }

    pub fn dot_out<'a, C2>(
        &self,
        rhs: &Matrix<C, C2>,
        out: &'a mut Matrix<R, C2>,
    ) -> &'a mut Matrix<R, C2>
    where
        R: Mul<C2> + Unsigned,
        C: Mul<C2> + Unsigned,

        C2: Mul<C> + Unsigned,
        C2: Mul<R> + Unsigned,

        Prod<R, C2>: ArrayLength<f32>,
        Prod<C2, R>: ArrayLength<f32>,

        Prod<C, C2>: ArrayLength<f32>,
        Prod<C2, C>: ArrayLength<f32>,
    {
        let (l_rows, l_cols) = self.shape;
        let (_r_rows, r_cols) = rhs.shape;

        for idx0 in 0..l_rows {
            for idx1 in 0..r_cols {
                let mut sum = 0.0;

                for idx2 in 0..l_cols {
                    let l_item = self.data[idx0 * l_cols + idx2];
                    let r_item = rhs.data[idx2 * r_cols + idx1];

                    sum += l_item * r_item
                }

                out.data[idx0 * r_cols + idx1] = sum;
            }
        }

        out
    }

    pub fn transpose(&self) -> Matrix<C, R> {
        let mut out = Matrix::<C, R>::new();

        let (rows, cols) = self.shape;

        for idx0 in 0..rows {
            for idx1 in 0..cols {
                let item = self.data[idx1 + idx0 * cols];
                out.data[idx0 + idx1 * rows] = item;
            }
        }

        out
    }

    pub fn transpose_out<'a>(&self, out: &'a mut Matrix<C, R>) -> &'a mut Matrix<C, R> {
        // let mut out = Matrix::<C, R>::new();

        let (rows, cols) = self.shape;

        for idx0 in 0..rows {
            for idx1 in 0..cols {
                let item = self.data[idx1 + idx0 * cols];
                out.data[idx0 + idx1 * rows] = item;
            }
        }

        out
    }
}

impl<R, C> Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    pub fn pow_scalar(&self, rhs: f32) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx].powf(rhs)
        }

        out
    }

    pub fn pow_scalar_mut(&mut self, rhs: f32) -> &Self {
        for idx in 0..self.size {
            self.data[idx] = self.data[idx].powf(rhs)
        }

        self
    }

    pub fn pow_scalar_out<'a>(&self, rhs: f32, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx].powf(rhs)
        }

        out
    }

    pub fn pow(&self, rhs: &Self) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx].powf(rhs.data[idx])
        }

        out
    }

    pub fn pow_mut(&mut self, rhs: &Self) -> &Self {
        for idx in 0..self.size {
            self.data[idx] = self.data[idx].powf(rhs.data[idx])
        }

        self
    }

    pub fn pow_out<'a>(&self, rhs: &Self, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx].powf(rhs.data[idx])
        }

        out
    }
}

impl<R, C> Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    pub fn add_scalar(&self, rhs: f32) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx] + rhs
        }

        out
    }

    pub fn add_scalar_mut(&mut self, rhs: f32) -> &Self {
        for idx in 0..self.size {
            self.data[idx] += rhs
        }

        self
    }

    pub fn add_scalar_out<'a>(&self, rhs: f32, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx] + rhs
        }

        out
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx] + rhs.data[idx]
        }

        out
    }

    pub fn add_mut(&mut self, rhs: &Self) -> &Self {
        for idx in 0..self.size {
            self.data[idx] += rhs.data[idx]
        }

        self
    }

    pub fn add_out<'a>(&self, rhs: &Self, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx] + rhs.data[idx]
        }

        out
    }
}

impl<R, C> Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    pub fn sub_scalar(&self, rhs: f32) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx] - rhs
        }

        out
    }

    pub fn sub_scalar_mut(&mut self, rhs: f32) -> &Self {
        for idx in 0..self.size {
            self.data[idx] -= rhs
        }

        self
    }

    pub fn sub_scalar_out<'a>(&self, rhs: f32, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx] - rhs
        }

        out
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx] - rhs.data[idx]
        }

        out
    }

    pub fn sub_mut(&mut self, rhs: &Self) -> &Self {
        for idx in 0..self.size {
            self.data[idx] -= rhs.data[idx]
        }

        self
    }

    pub fn sub_out<'a>(&self, rhs: &Self, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx] - rhs.data[idx]
        }

        out
    }
}

impl<R, C> Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    pub fn mul_scalar(&self, rhs: f32) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx] * rhs
        }

        out
    }

    pub fn mul_scalar_mut(&mut self, rhs: f32) -> &Self {
        for idx in 0..self.size {
            self.data[idx] *= rhs
        }

        self
    }

    pub fn mul_scalar_out<'a>(&self, rhs: f32, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx] * rhs
        }

        out
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx] * rhs.data[idx]
        }

        out
    }

    pub fn mul_mut(&mut self, rhs: &Self) -> &Self {
        for idx in 0..self.size {
            self.data[idx] *= rhs.data[idx]
        }

        self
    }

    pub fn mul_out<'a>(&self, rhs: &Self, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx] * rhs.data[idx]
        }

        out
    }
}

impl<R, C> Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    pub fn div_scalar(&self, rhs: f32) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx] / rhs
        }

        out
    }

    pub fn div_scalar_mut(&mut self, rhs: f32) -> &Self {
        for idx in 0..self.size {
            self.data[idx] /= rhs
        }

        self
    }

    pub fn div_scalar_out<'a>(&self, rhs: f32, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx] / rhs
        }

        out
    }

    pub fn div(&self, rhs: &Self) -> Self {
        let mut out = Self::new();

        for idx in 0..self.size {
            out.data[idx] = self.data[idx] / rhs.data[idx]
        }

        out
    }

    pub fn div_mut(&mut self, rhs: &Self) -> &Self {
        for idx in 0..self.size {
            self.data[idx] /= rhs.data[idx]
        }

        self
    }

    pub fn div_out<'a>(&self, rhs: &Self, out: &'a mut Self) -> &'a mut Self {
        for idx in 0..self.size {
            out.data[idx] = self.data[idx] / rhs.data[idx]
        }

        out
    }
}

impl<R, C> Default for Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<R, C> PartialEq for Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
    fn eq(&self, rhs: &Self) -> bool {
        self.shape == rhs.shape && self.data == rhs.data
    }
}

impl<R, C> Eq for Matrix<R, C>
where
    R: Mul<C> + Unsigned,
    C: Mul<R> + Unsigned,
    Prod<R, C>: ArrayLength<f32>,
    Prod<C, R>: ArrayLength<f32>,
{
}
